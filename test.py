import torch
from torch.utils.data import DataLoader
from config import *
from data import Nifti2DSliceDataset, get_val_transforms
from losses import FocalLoss
from metrics import dice_coefficient, iou_score, precision_score, recall_score, f1_score
import argparse

from config import MODEL_ARCH

def test_segmentation(model, loader, criterion, device, num_classes=NUM_CLASSES):
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    dice_scores = []
    iou_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []

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

            dice_scores.append(dice_coefficient(preds, masks, num_classes))
            iou_scores.append(iou_score(preds, masks, num_classes))
            prec_scores.append(precision_score(preds, masks, num_classes))
            rec_scores.append(recall_score(preds, masks, num_classes))
            f1_scores.append(f1_score(preds, masks, num_classes))

    dice_sum = torch.tensor(dice_scores).sum(dim=0)
    iou_sum = torch.tensor(iou_scores).sum(dim=0)
    prec_sum = torch.tensor(prec_scores).sum(dim=0)
    rec_sum = torch.tensor(rec_scores).sum(dim=0)
    f1_sum = torch.tensor(f1_scores).sum(dim=0)

    total_slices = len(loader.dataset)
    results = {
        "loss": running_loss / total_slices,
        "accuracy": correct_pixels / total_pixels,
        "dice_per_class": (dice_sum / total_slices).tolist(),
        "mean_dice": (dice_sum / total_slices).mean().item(),
        "iou_per_class": (iou_sum / total_slices).tolist(),
        "mean_iou": (iou_sum / total_slices).mean().item(),
        "precision_per_class": (prec_sum / total_slices).tolist(),
        "mean_precision": (prec_sum / total_slices).mean().item(),
        "recall_per_class": (rec_sum / total_slices).tolist(),
        "mean_recall": (rec_sum / total_slices).mean().item(),
        "f1_per_class": (f1_sum / total_slices).tolist(),
        "mean_f1": (f1_sum / total_slices).mean().item(),
    }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["acdc", "mms", "mms2"], required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


    model.load_state_dict(torch.load(os.path.join(BEST_WEIGHTS_DIR, "best_model.pth"), map_location=device))
    model.eval()

    if args.dataset == "acdc":
        img_dir, mask_dir = ACDC_TEST_IMAGES, ACDC_TEST_MASKS
    elif args.dataset == "mms":
        img_dir, mask_dir = MMS_TEST_IMAGES, MMS_TEST_MASKS
    else:
        img_dir, mask_dir = MMS2_TEST_IMAGES, MMS2_TEST_MASKS

    dataset = Nifti2DSliceDataset(img_dir, mask_dir, transform=get_val_transforms(IMG_SIZE), slice_axis=2)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    alpha = torch.tensor(ALPHA, device=device)
    criterion = FocalLoss(gamma=2.0, alpha=alpha)

    results = test_segmentation(model, loader, criterion, device)
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()