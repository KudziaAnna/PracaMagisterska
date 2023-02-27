import glob
import os
from collections import OrderedDict

import torch
from monai.transforms import (
    Compose,
    NormalizeIntensityD,
    SpacingD,
    EnsureChannelFirstD,
    LoadImageD,
    SaveImage,
    CropForegroundD,
    ToTensorD,
    AddChannelD,
    Invertd,
)

from models.ModifiedSwinUnet.modified_swin_unet import ModifiedSwinUNet

# INPUT_DIR = "/input_img"
# INPUT_DIR_META = "/input_meta"
# OUTPUT_DIR = "/output"
# FOLDS = 5
# MODEL_NAME = "model-ModifiedSwinUNet-fold-{}.pt"


INPUT_DIR = "../data/feta2022"
INPUT_DIR_META = "/input_meta"
OUTPUT_DIR = "../output"
FOLDS = 5
MODEL_NAME = "../models/model-ModifiedSwinUNet-fold-{}.pt"

PATCH_SIZE = (128, 128, 128)
CLASSES = 8
FEATURE_SIZE = 48

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_transform = Compose(
    [
        LoadImageD(keys=("image", "label")),
        EnsureChannelFirstD(keys=("image")),
        AddChannelD(keys="label"),
        SpacingD(keys=("image", "label"),
                 pixdim=(0.8, 0.8, 0.8),
                 mode="bilinear"
                 ),
        CropForegroundD(keys=("image", "label"),
                        source_key="image",
                        k_divisible=PATCH_SIZE
                        ),
        NormalizeIntensityD(keys=("image"), nonzero=True, channel_wise=True),
        ToTensorD(keys=("image", "label"))
    ]
)

post_transforms = Compose([
    Invertd(
        keys="pred",
        transform=val_transform,
        orig_keys="label",
        orig_meta_keys="label_meta_dict",
        to_tensor=True,
    ),
])


class EnsembleModel:
    def __init__(self, models):
        """
        Args:
            models:
        """
        self.models = models

    def __call__(self, x):
        preds = []
        x = x.to(device)

        with torch.no_grad():
            for m in self.models:
                pred = m(x)
                preds.append(pred)
        preds = torch.stack(preds)
        preds = torch.mean(preds, dim=0)
        return preds


def load_model():
    models = []
    for i in range(FOLDS):
        m = ModifiedSwinUNet(in_channels=1, out_channels=CLASSES, img_size=PATCH_SIZE, feature_size=FEATURE_SIZE,
                      use_checkpoint=False)
        checkpoint = torch.load(MODEL_NAME.format(i), map_location=torch.device('cpu'))

        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        m.load_state_dict(state_dict=new_state_dict)
        models.append(m)

    model = EnsembleModel(models)
    return model


def folder_inference(model, folder_path, pid):
    mri_path = glob.glob(os.path.join(folder_path, "anat", "**T2w.nii.gz"))[0]
    label_path = glob.glob(os.path.join(folder_path, "anat", "**dseg.nii.gz"))[0]
    mri_name = os.path.basename(mri_path)

    output_folder_path = os.path.join(OUTPUT_DIR, pid)
    os.makedirs(output_folder_path, exist_ok=True)

    imgs_dict = val_transform({"image": mri_path, "label": label_path})
    mri = imgs_dict["image"]
    mri = mri.unsqueeze(0).to(device)

    # label = imgs_dict["label"]
    # label = label.squeeze()

    pred = model(mri).detach().cpu()

    pred = pred.squeeze(0)
    pred = torch.argmax(pred, dim=0)
    pred = pred.unsqueeze(0)

    imgs_dict["pred"] = pred
    imgs_dict = post_transforms(imgs_dict)

    pred = imgs_dict["pred"]
    pred = pred.squeeze(0)

    SaveImage(output_dir=output_folder_path, output_postfix="dseg", separate_folder=False, channel_dim=None)(pred)
    output_path = glob.glob(os.path.join(output_folder_path, "**.nii.gz"))[0]
    output_file_name = mri_name.replace("T2w", "segmented")
    os.rename(output_path, os.path.join(output_folder_path, output_file_name))
    # SaveImage(output_dir=output_folder_path, output_postfix="label", separate_folder=False, channel_dim=None)(label)


def full_inference():
    print("Loading model...")
    model = load_model()

    print("Starting inference...")

    i= 0
    for folder in os.listdir(INPUT_DIR):
        print(f"Inferencing {folder}...")
        if folder.startswith("sub-007"):
                folder_inference(model, os.path.join(INPUT_DIR, folder), pid=folder)
        if folder.startswith("sub-009"):
                folder_inference(model, os.path.join(INPUT_DIR, folder), pid=folder)

    print("Finished")


if __name__ == '__main__':
    full_inference()
