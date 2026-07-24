import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
rom torch_sam import SAM
from config import *
from data import Nifti2DSliceDataset, get_train_transforms, get_val_transforms

from losses import FocalLoss
from metrics import dice_coefficient_batch

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks, _ in loader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device).squeeze(1).long()
        def closure():
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device, num_classes=NUM_CLASSES):
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    dice_sum = torch.zeros(num_classes, device=device)
    total_samples = 0
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device).squeeze(1).long()
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()
            batch_dice = dice_coefficient_batch(preds, masks, num_classes)
            dice_sum += batch_dice * images.size(0)
            total_samples += images.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    acc = correct_pixels / total_pixels
    mean_dice_per_class = dice_sum / total_samples
    mean_dice = mean_dice_per_class.mean().item()
    return epoch_loss, acc, mean_dice, mean_dice_per_class.tolist()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    full_dataset = Nifti2DSliceDataset(
        image_dir=ACDC_TRAIN_IMAGES,
        mask_dir=ACDC_TRAIN_MASKS,
        transform=get_train_transforms(IMG_SIZE),
        slice_axis=2
    )
    train_len = int(0.75 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len],
                                              generator=torch.Generator().manual_seed(25))
    train_dataset.dataset.transform = get_train_transforms(IMG_SIZE)
    val_dataset.dataset.transform = get_val_transforms(IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    if MODEL_ARCH == "pretrainedmamba":
        from models import pretrainedMambaEncUNet
        model = pretrainedMambaEncUNet(in_ch=1, out_ch=NUM_CLASSES).to(device)
    elif MODEL_ARCH == "pretrainedmambaout":
        from models import pretrainedMambaOutEncUNet
        model = pretrainedMambaOutEncUNet(in_ch=1, out_ch=NUM_CLASSES).to(device)
    elif MODEL_ARCH == "hybridmamba":
        from models import HybridMambaUNet
        model = HybridMambaUNet(in_ch=1, out_ch=NUM_CLASSES).to(device)
    elif MODEL_ARCH == "hybridmambaout":
        from models import HybridMambaOutUNet
        model = HybridMambaOutUNet(in_ch=1, out_ch=NUM_CLASSES).to(device)
    elif MODEL_ARCH == "puremamba":
        from models import PureMambaUNet
        model = PureMambaUNet(in_ch=1, out_ch=NUM_CLASSES).to(device)
    elif MODEL_ARCH == "puremambaout":
        from models import PureMambaOutUNet
        model = PureMambaOutUNet(in_ch=1, out_ch=NUM_CLASSES).to(device)
    else:
        raise ValueError(f"Unknown MODEL_ARCH: {MODEL_ARCH}")

    alpha = torch.tensor(ALPHA, device=device)
    criterion = FocalLoss(gamma=2.0, alpha=alpha)
    base_optimizer = optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer(model.parameters(), lr=LEARNING_RATE), rho=0.05)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(BEST_WEIGHTS_DIR, exist_ok=True)

    best_val_loss = float("inf")
    counter = 0
    early_stop = False

    for epoch in range(NUM_EPOCHS):
        if early_stop:
            print("Early stopping triggered.")
            break

        if epoch == UNFREEZE_EPOCH:
            if hasattr(model, 'unfreeze_encoder'):
                model.unfreeze_encoder()
                # Reinitialize optimizer only for parameters that now require gradients
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
            else:
                # For models without a frozen encoder, just lower the learning rate if desired
                for g in optimizer.param_groups:
                    g['lr'] = 1e-4

4
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_dice, dice_per_class = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Dice: {val_dice:.4f}")
        print(f"Dice per class: {[round(d,4) for d in dice_per_class]}")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"unet_epoch_{epoch+1:03d}.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(BEST_WEIGHTS_DIR, "best_model.pth"))
            print("New best model saved.")
        else:
            counter += 1
            if counter >= PATIENCE:
                early_stop = True

if __name__ == "__main__":
    main()