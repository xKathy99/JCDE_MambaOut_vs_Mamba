import pandas as pd
import torch
from torch.utils.data import DataLoader
from config import *
from data import Nifti2DSliceDataset, get_val_transforms

from losses import FocalLoss
import argparse

from config import MODEL_ARCH

def dice_coefficient_single(pred, target, num_classes, eps=1e-6):
    pred = pred.cpu()
    target = target.cpu()
    dice = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            dice.append(1.0)
        else:
            dice.append((2*inter + eps)/(union + eps))
    return dice

def test_pathology(model, loader, device, num_classes=NUM_CLASSES):
    model.eval()
    results = []
    with torch.no_grad():
        for images, masks, filenames in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device).squeeze(1).long()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            for i in range(images.size(0)):
                dice = dice_coefficient_single(preds[i], masks[i], num_classes)
                row = {"filename": filenames[i], "slice_idx": i}
                for c in range(num_classes):
                    row[f"class{c}"] = dice[c]
                results.append(row)
    return pd.DataFrame(results)

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
        info_csv = "../path/to/acdc/testing_info.csv"
        group_col = "Group"
    elif args.dataset == "mms":
        img_dir, mask_dir = MMS_TEST_IMAGES, MMS_TEST_MASKS
        info_csv = "../path/to/mms/patient_data_filled_all.csv"
        group_col = "Pathology"
    else:  # mms2
        img_dir, mask_dir = MMS2_TEST_IMAGES, MMS2_TEST_MASKS
        info_csv = "../path/to/mm2/dataset_information.csv"
        group_col = "DISEASE"

    dataset = Nifti2DSliceDataset(img_dir, mask_dir, transform=get_val_transforms(IMG_SIZE), slice_axis=2)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    df = test_pathology(model, loader, device)

    # Merge pathology info
    info = pd.read_csv(info_csv)
    if args.dataset == "acdc":
        df["PatientID"] = df["filename"].str.split("_").str[0]
        df = df.merge(info[["PatientID", group_col]], on="PatientID", how="left")
    elif args.dataset == "mms":
        df["External code"] = df["filename"].str.split("_").str[0]
        df = df.merge(info[["External code", group_col]], on="External code", how="left")
    else:
        df["SUBJECT_CODE"] = df["filename"].str.split("_").str[0]
        info["SUBJECT_CODE"] = info["SUBJECT_CODE"].astype(str)
        df = df.merge(info[["SUBJECT_CODE", group_col]], on="SUBJECT_CODE", how="left")

    grouped = df.groupby(group_col).agg(
        Count=("filename", "nunique"),
        class0=("class0", "mean"),
        class1=("class1", "mean"),
        class2=("class2", "mean"),
        class3=("class3", "mean")
    ).reset_index()
    print(grouped)

if __name__ == "__main__":
    main()