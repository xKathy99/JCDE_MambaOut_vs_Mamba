import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def visualize_unet_predictions(model, dataloader, num_classes=4, max_batches=None, max_slices=None, device="cuda"):
    model.eval()
    with torch.no_grad():
        for i, (images, masks, img_filenames) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            batch_size = images.shape[0]
            slice_limit = batch_size if max_slices is None else min(batch_size, max_slices)
            for s in range(slice_limit):
                x = images[s].unsqueeze(0).to(device)
                y_true = masks[s].squeeze().cpu()
                outputs = model(x)
                probs = F.softmax(outputs, dim=1)
                pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu()

                # Uncertainty (entropy)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                uncertainty_map = entropy.squeeze(0).cpu()

                # Error map
                diff_mask = (pred_mask != y_true).float()

                input_img = images[s].squeeze().cpu()

                fig, axs = plt.subplots(1, 5, figsize=(28,6))
                axs[0].imshow(input_img, cmap="gray")
                axs[0].set_title(f"Input: {img_filenames[s]}, slice {s}")
                axs[1].imshow(y_true, cmap="nipy_spectral")
                axs[1].set_title("Ground Truth")
                axs[2].imshow(pred_mask, cmap="nipy_spectral")
                axs[2].set_title("Prediction")
                axs[3].imshow(input_img, cmap="gray")
                axs[3].imshow(pred_mask, cmap="nipy_spectral", alpha=0.5)
                axs[3].imshow(diff_mask, cmap="Reds", alpha=0.5)
                axs[3].set_title("Overlay + Errors")
                im = axs[4].imshow(uncertainty_map, cmap="jet")
                axs[4].set_title("Uncertainty (Entropy)")
                plt.colorbar(im, ax=axs[4])
                plt.tight_layout()
                plt.show()