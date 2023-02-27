import time
import os
import argparse
import numpy as np
from comet_ml import Experiment
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, StratifiedKFold
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from datetime import datetime
from models.UNet3D.model import UNet3D
from models.UNetr.unetr import UNETR
from models.ModifiedSwinUnet.modified_swin_unet import ModifiedSwinUNet
from data_loader import FetalBrainMRI
from utils.metric import mIOU
from utils.metric import dice_idx
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.metrics import HausdorffDistanceMetric

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Training Segmentation Network on Fetal Dataset.")
parser.add_argument("--data",
                    type=str,
                    default="../data/feta2022/*/anat",
                    help="Path to the data")
parser.add_argument("--model_name",
                    type=str,
                    default="ModifiedSwinUNet",
                    help="Type of the model.")
parser.add_argument("--multitask",
                    type=bool,
                    default=False,
                    help="if multitask learning")
parser.add_argument("--split",
                    type=int,
                    default=5,
                    help="number of splits")
parser.add_argument("--epochs",
                    type=int,
                    default=500,
                    help="Number of epochs")
parser.add_argument("--classes",
                    type=int,
                    default=8,
                    help="Number of classes in the dataset")
parser.add_argument("--patch_size",
                    type=tuple,
                    default=(128, 128, 128),
                    help="Patch size.")
parser.add_argument("--n_features",
                    type=int,
                    default=32,
                    help="Number of feature maps")
parser.add_argument("--batch_size",
                    type=int,
                    default=2,
                    help="Number of batch size")
parser.add_argument("--feature_size",
                    type=int,
                    default=48,
                    help="Number of Transformer's features.")
parser.add_argument("--lr",
                    type=float,
                    default=2e-4,
                    help="Number of learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=1e-5,
                    help="Number of weight decay")
parser.add_argument("--step_lr",
                    type=int,
                    default=300,
                    help="Learning rate step value")
parser.add_argument("--lr_gamma",
                    type=float,
                    default=0.1,
                    help="Step learning rate factor")
parser.add_argument("--optimizer",
                    type=str,
                    default="AdamW",
                    help="Type of optimizer.")
parser.add_argument("--parallel",
                    type=bool,
                    default=True,
                    help="Parallel learning.")

args = parser.parse_args()


class DummyExperiment:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def log_parameters(self, *args, **kwargs):
        pass

    def log_metric(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def log_table(self, *args, **kwargs):
        pass

    def log_current_epoch(self, *args, **kwargs):
        pass

    def log_figure(self, *args, **kwargs):
        pass

    def train(self):
        return self


TESTING_ENVIRONMENT = False

if TESTING_ENVIRONMENT:
    experiment = DummyExperiment()
else:
    experiment = Experiment("3YfcpxE1bYPCpkkg4pQ2OjQ2r", project_name="FETA2022")

experiment.log_parameters(args)

experiment.log_parameters(args)

dataset = FetalBrainMRI(data_path=args.data)

train_dataset = FetalBrainMRI(data_path=args.data,
                              mode="train")
val_dataset = FetalBrainMRI(data_path=args.data,
                            mode="val")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

kfold = KFold(n_splits=args.split, shuffle=False)
skfold = StratifiedKFold(n_splits=args.split, shuffle=False)

criterion_seg = DiceCELoss(to_onehot_y=True, softmax=True)
criterion_reg = nn.MSELoss()
hausdorff_distance = HausdorffDistanceMetric()

print("--------------------")
for fold, ((train_ids1, test_ids1), (train_ids2, test_ids2)) in enumerate(
        zip(skfold.split(dataset.zurich_ids, dataset.zurich_pathologies),
            skfold.split(dataset.vienna_ids, dataset.vienna_pathologies))):
    train_ids = np.append(train_ids1, train_ids2)
    test_ids = np.append(test_ids1, test_ids2)

    print(f"FOLD {fold}")
    print("-------------------")
    train_subsampler = SubsetRandomSampler(train_ids)
    test_subsampler = SubsetRandomSampler(test_ids)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_subsampler,
                              pin_memory=True,
                              num_workers=8)
    test_loader = DataLoader(val_dataset,
                             batch_size=args.batch_size // 2,
                             sampler=test_subsampler,
                             pin_memory=True,
                             num_workers=8)

    # Init neural network
    if args.model_name == "UNet3D":
        model = UNet3D(n_channels=1, n_classes=args.classes, n_features=args.n_features).to(device)
    elif args.model_name == "SegResNet":
        model = SegResNet(spatial_dims=3, init_filters=args.n_features, out_channels=args.classes).to(device)
    elif args.model_name == "unetr":
        model = UNETR(in_channels=1, out_channels=args.classes, img_size=(128, 128, 128))
    elif args.model_name == "ModifiedSwinUNet":
        model = ModifiedSwinUNet(in_channels=1, out_channels=args.classes, img_size=args.patch_size,
                          feature_size=args.feature_size, use_checkpoint=False)
    else:
        raise NotImplementedError(f"There are no implementation of: {args.model_name}")

    if args.parallel:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    # Init optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"There are no implementation of: {args.optimizer}")

    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, verbose=True)

    with experiment.train():

        best_dice_score = 0.0

        for epoch in range(args.epochs):
            start_time_epoch = time.time()
            print(f"Starting epoch {epoch + 1}")
            model.train()
            running_loss = 0.0
            running_jaccard = 0.0
            running_dice = 0.0

            dice_per_class_train = [0 for i in range(args.classes)]
            dice_per_class_val = [0 for i in range(args.classes)]

            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.float32)
                output_mask = model(images)
                dice_ce_loss = criterion_seg(output_mask, masks.long())
                loss = dice_ce_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                jac = mIOU(masks, output_mask, num_classes=args.classes)
                dice, dice_list = dice_idx(masks, output_mask, num_classes=args.classes)
                running_jaccard += jac.item()
                running_dice += dice.item()
                running_loss += loss.item()

                if batch_idx % 1 == 0:
                    print(" ", end="")
                    print(f"Batch: {batch_idx + 1}/{len(train_loader)}"
                          f" Loss: {loss.item():.4f}"
                          f" Jaccard: {jac.item():.4f}"
                          f" Dice Index: {dice.item():.4f}"
                          f" Time: {time.time() - start_time_epoch:.2f}s")

                for i, d in enumerate(dice_list):
                    dice_per_class_train[i] += d
            print("Training process has finished. Starting testing...")

            val_running_jac = 0.0
            val_running_dice = 0.0
            model.eval()
            with torch.no_grad():
                torch.cuda.empty_cache()
                for batch_idx, (images, masks) in enumerate(test_loader):
                    images = images.to(device=device, dtype=torch.float32)
                    masks = masks.to(device=device, dtype=torch.long)
                    output_mask = model(images)
                    jac = mIOU(masks, output_mask, num_classes=args.classes)
                    dice, dice_list = dice_idx(masks, output_mask, num_classes=args.classes)

                    val_running_jac += jac.item()
                    val_running_dice += dice.item()

                    for i, d in enumerate(dice_list):
                        dice_per_class_val[i] += d

                train_loss = running_loss / len(train_loader)

                train_jac = running_jaccard / len(train_loader)
                test_jac = val_running_jac / len(test_loader)

                train_dice = running_dice / len(train_loader)
                test_dice = val_running_dice / len(test_loader)

                save_path = f"../models/model-{args.model_name}-fold-{fold}.pt"

                if best_dice_score < test_dice:
                    torch.save(model.state_dict(), save_path)
                    best_dice_score = test_dice
                    print(f"Current best Dice score {best_dice_score}. Model saved!")

                scheduler.step()

                experiment.log_current_epoch(epoch)
                experiment.log_metric("train_jac", train_jac)
                experiment.log_metric("val_jac", test_jac)
                experiment.log_metric("train_dice", train_dice)
                experiment.log_metric("val_dice", test_dice)
                experiment.log_metric("train_loss", train_loss)

                for i, (train_dice_class, val_dice_class) in enumerate(zip(dice_per_class_train, dice_per_class_val)):
                    experiment.log_metric(f"train_dice_cl{i}", train_dice_class/len(train_loader))
                    experiment.log_metric(f"val_dice_cl{i}", val_dice_class/len(test_loader))

                print('    ', end='')

                print(f"Joint Loss: {train_loss:.4f}"
                      f" Train Jaccard: {train_jac:.4f}"
                      f" Train Dice Index: {train_dice:.4f}"
                      f" Test Jaccard: {test_jac:.4f}"
                      f" Test Dice Index: {test_dice:.4f}")

        print(f"Training finished!")
