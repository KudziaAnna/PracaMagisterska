"""Data loader for Fetal Brain MRI segmentation.
MICCAI 2022 FETA 2022 challenge."""

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    NormalizeIntensityD,
    SpacingD,
    EnsureChannelFirstD,
    LoadImageD,
    AddChannelD,
    ResizeD,
    RandRotateD,
    RandFlipD,
    RandGaussianNoiseD,
    RandZoomD,
    OrientationD,
    CropForegroundD,
    RandAdjustContrastD,
    RandSpatialCropD,
    RandGaussianSmoothD,
    SpatialPadD,
    RandScaleIntensityD,
    RandShiftIntensityD,
    ToTensorD
)


class FetalBrainMRI(Dataset):
    """
    FetalBrainMRI class.
    """

    def __init__(self,
                 data_path: str = None,
                 mode: str = None,
                 multitask: bool = None,
                 patch_size: tuple = (128, 128, 128)
                 ) -> None:
        """
        Args:
            data_path:
            mode:
        """
        self.data_path = data_path
        assert mode in ["train", "val", None]
        self.mode = mode
        self.multitask = multitask
        self.patch_size = patch_size

        self.mri_list = []
        self.seg_mask_list = []

        self.vienna_ids = []
        self.vienna_pathologies = []
        self.zurich_ids = []
        self.zurich_pathologies = []

        self.participants = pd.read_csv("../data/feta2022/participants.tsv", sep="\t")
        self.participants.replace({"Neurotypical": 0, "Pathological": 1}, inplace=True)

        patient_indice = 0

        for folder in sorted(glob.glob(self.data_path)):
            patient_id = extract_patient_id(folder)
            for filename in os.listdir(folder):
                if filename.endswith("T2w.nii.gz"):
                    pathology = self.participants.loc[
                        self.participants["participant_id"] == str(filename.split("_")[0]), "Pathology"].item()

                    if patient_id <= 80:
                        self.zurich_ids.append(patient_indice)
                        self.zurich_pathologies.append(pathology)
                    else:
                        self.vienna_ids.append(patient_indice)
                        self.vienna_pathologies.append(pathology)
                    patient_indice += 1

                    self.mri_list.append(f"{folder}/{filename}")
                if filename.endswith("dseg.nii.gz"):
                    self.seg_mask_list.append(f"{folder}/{filename}")

        assert len(self.mri_list) == len(self.seg_mask_list)

        self.train_transform = Compose(
            [EnsureChannelFirstD(keys="image"),
             AddChannelD(keys="label"),
             SpacingD(
                 keys=["image", "label"],
                 pixdim=(0.8, 0.8, 0.8),
                 mode=("bilinear", "nearest")
             ),
             CropForegroundD(
                 keys=["image", "label"],
                 source_key="image",
                 k_divisible=self.patch_size
             ),
             RandSpatialCropD(
                 keys=["image", "label"],
                 roi_size=self.patch_size,
                 random_size=False
             ),
             RandZoomD(
                 keys=["image", "label"],
                 min_zoom=0.8,
                 max_zoom=1.2,
                 mode="trilinear",
                 align_corners=True,
                 prob=0.3
             ),
             RandRotateD(
                 keys=["image", "label"],
                 range_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                 range_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                 range_z=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                 mode=["bilinear", "nearest"],
                 align_corners=True,
                 padding_mode="border",
                 prob=0.3
             ),
             RandGaussianNoiseD(keys=["image"],
                                mean=0.,
                                std=0.1,
                                prob=0.2),
             RandGaussianSmoothD(keys=["image"],
                                 sigma_x=(0.5, 1.15),
                                 sigma_y=(0.5, 1.15),
                                 sigma_z=(0.5, 1.15),
                                 prob=0.2),
             RandAdjustContrastD(keys="image",
                                 gamma=(0.7, 1.5),
                                 prob=0.3),
             RandFlipD(keys=["image", "label"],
                       prob=0.5, spatial_axis=0),
             RandFlipD(keys=["image", "label"],
                       prob=0.5, spatial_axis=1),
             RandFlipD(keys=["image", "label"],
                       prob=0.5, spatial_axis=2),
             NormalizeIntensityD(keys="image",
                                 nonzero=True,
                                 channel_wise=True),
             # RandScaleIntensityD(keys="image", factors=0.1, prob=1.0),
             # RandShiftIntensityD(keys="image", offsets=0.1, prob=1.0),
             ToTensorD(keys=["image", "label"])
             ]
        )

        self.val_transform = Compose(
            [
                # EnsureChannelFirstD(keys="image"),
                AddChannelD(keys="label"),
                # SpacingD(
                #     keys=["image", "label"],
                #     pixdim=(0.8, 0.8, 0.8),
                #     mode=("bilinear", "nearest")
                # ),
                # CropForegroundD(
                #     keys=["image", "label"],
                #     source_key="image",
                #     k_divisible=self.patch_size
                # ),
                # NormalizeIntensityD(keys="image", nonzero=True, channel_wise=True),
                ToTensorD(keys=["image", "label"])
            ]
        )

    def __getitem__(self,
                    x
                    ) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            x:
        Returns:
        """
        dict_loader = LoadImageD(keys=("image", "label"))
        data_dict = dict_loader({"image": self.mri_list[x],
                                 "label": self.seg_mask_list[x]})

        if self.mode == "train":
            data_dict = self.train_transform(data_dict)
        else:
            data_dict = self.val_transform(data_dict)

        patient_id = os.path.basename(self.mri_list[x]).split("_")[0]

        gestational_age = self.participants.loc[self.participants["participant_id"] == str(patient_id),
                                                "Gestational age"].item()
        pathology = self.participants.loc[self.participants["participant_id"] == str(patient_id), "Pathology"].item()

        return data_dict["image"], data_dict["label"], gestational_age, pathology, patient_id

    def __len__(self) -> int:
        return len(self.mri_list)


def extract_patient_id(folder):
    return int(folder.split("-")[1][:3])


if __name__ == "__main__":
    fetal_brain_MRI = FetalBrainMRI(data_path="../data/feta2022/*/anat/", mode="train")

    for x, y, z in fetal_brain_MRI:
        x, y = x.squeeze(0), y.squeeze(0)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(x[:, :, 50])
        ax2.imshow(y[:, :, 50])
        plt.show()
