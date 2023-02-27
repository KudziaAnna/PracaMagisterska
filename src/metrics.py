import os
import glob
import argparse
from data_loader import FetalBrainMRI
from utils.metric import mIOU
from utils.metric import dice_idx
from pymia.evaluation.metric import VolumeSimilarity
import pymia.evaluation.evaluator as eval_
from monai.transforms import ToTensorD, LoadImageD
import pymia.evaluation.writer as writer_pymia


parser = argparse.ArgumentParser(description="Training Segmentation Network on Fetal Dataset.")
parser.add_argument("--data",
                    type=str,
                    default="../data/feta2022/*/anat",
                    help="Path to the data")
parser.add_argument("--patch_size",
                    type=tuple,
                    default=(128, 128, 128),
                    help="Patch size.")

args = parser.parse_args()

transform = ToTensorD(keys=["label"])

def count_metrics():    
    id_list = []
    pathology_list = []
    ge_list = []
    dice_list = []
    iou_list = []
    hd_list = []

    dict_loader = LoadImageD(keys=("label"))
    mask_dataset = FetalBrainMRI(data_path=args.data,
                              mode="val")

    for _, mask, ge, pathology, id  in mask_dataset:
        print(id)
        id_list.append(id)
        print(pathology)
        pathology_list.append(pathology)
        print(ge)
        ge_list.append(ge)

        segmented_path = glob.glob(os.path.join("../output/", str(id), "**.nii.gz"))
        if len(segmented_path) == 0:
            continue
        else:
            segmented_path = segmented_path[0]

        data_dict = dict_loader({"label": segmented_path})
        segmented = transform(data_dict)['label']
        mask = mask.reshape((256, 256, 256))

        dice_metric, _ = dice_idx(mask, segmented, num_classes=8)
        metric = dice_metric.item()
        dice_list.append(metric)

        jac = mIOU(mask, segmented, num_classes=8)
        metric = jac.item()
        iou_list.append(metric)
        
        labels = {
              0: "BG",
              1: "CSF",
              2: "GM",
              3: "WM",
              4: "LV",
              5: "CBM",
              6: "SGM",
              7: "BS",
        }
        evaluator = eval_.SegmentationEvaluator([VolumeSimilarity()], labels)
        # evaluator = eval_.SegmentationEvaluator([VolumeSimilarity()], labels)
        evaluator.evaluate(segmented, mask, id)
        writer_pymia.CSVWriter("../results/result_VS"+str(id)+".csv").write(evaluator.results)
        evaluator.clear
        
    writer_pymia.CSVWriter("../results/result_VS.csv").write(evaluator.results)
    evaluator.clear
    rows = zip(id_list, ge_list, pathology_list, dice_list, iou_list)
    import csv
    with open("../results/result.csv", "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)



         
if __name__ == '__main__':
    count_metrics()



