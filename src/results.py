import csv
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from monai.transforms import ToTensorD, LoadImageD
from data_loader import FetalBrainMRI

def dice_per_ge():
    dice = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    jac = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    ge = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

    with open('../results/result.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            idx = round(float(row[1])) - 20
            dice[idx].append(float(row[3]))
            jac[idx].append(float(row[4]))
        
    idx = 0
    dice_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dice_std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jac_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jac_std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for d in dice:
        if len(d) == 0:
            dice_mean[idx] = 0.0
            dice_std[idx] = 0.0
        else:
            dice_mean[idx] = np.mean(d)
            dice_std[idx] = np.std(d)
        idx = idx + 1

    idx = 0
    for j in jac:
        if len(j) == 0:
            jac_mean[idx] = 0.0
            jac_std[idx] = 0.0
        else:
            jac_mean[idx] = np.mean(j)
            jac_std[idx] = np.std(j)
        idx = idx + 1

    fig, ax1 = plt.subplots(figsize =(10, 7))
    ax1.bar(ge, dice_mean, yerr = dice_std)
    ax1.set_xlabel('Gestational age (weeks)')
    ax1.set_ylabel('Mean Dice Score')
    plt.grid()
    plt.show()

    fig, ax1 = plt.subplots(figsize =(10, 7))
    ax1.bar(ge, jac_mean, yerr = jac_std)
    ax1.set_xlabel('Gestational age (weeks)')
    ax1.set_ylabel('Mean Jaccard Index')
    plt.grid()
    plt.show()

def dice_per_pathology():
    dice_pat = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    jac_pat = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    dice_non = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    jac_non = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    ge = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

    with open('../results/result.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            idx = round(float(row[1])) - 20
            if row[2] == '0':
                dice_non[idx].append(float(row[3]))
                jac_non[idx].append(float(row[4]))
            else:
                dice_pat[idx].append(float(row[3]))
                jac_pat[idx].append(float(row[4]))
        
    idx = 0
    dice_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dice_std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jac_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jac_std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    dice_mean_pat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dice_std_pat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jac_mean_pat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jac_std_pat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for d in dice_non:
        if len(d) == 0:
            dice_mean[idx] = 0.0
            dice_std[idx] = 0.0
        else:
            dice_mean[idx] = np.mean(d)
            dice_std[idx] = np.std(d)
        idx = idx + 1

    idx = 0
    for j in jac_non:
        if len(j) == 0:
            jac_mean[idx] = 0.0
            jac_std[idx] = 0.0
        else:
            jac_mean[idx] = np.mean(j)
            jac_std[idx] = np.std(j)
        idx = idx + 1

    idx = 0
    for d in dice_pat:
        if len(d) == 0:
            dice_mean_pat[idx] = 0.0
            dice_std_pat[idx] = 0.0
        else:
            dice_mean_pat[idx] = np.mean(d)
            dice_std_pat[idx] = np.std(d)
        idx = idx + 1

    idx = 0
    for j in jac_pat:
        if len(j) == 0:
            jac_mean_pat[idx] = 0.0
            jac_std_pat[idx] = 0.0
        else:
            jac_mean_pat[idx] = np.mean(j)
            jac_std_pat[idx] = np.std(j)
        idx = idx + 1

    ind = np.arange(20, 36)
    fig = plt.subplots(figsize =(10, 7))
    p1 = plt.bar(ind - 0.15, dice_mean_pat, 0.3, yerr = dice_std_pat)
    p2 = plt.bar(ind + 0.15, dice_mean, 0.3, yerr = dice_std)
    plt.xlabel('Gestational age (weeks)')
    plt.ylabel('Mean Dice Score')
    plt.xticks(ge)
    plt.legend((p1[0], p2[0]), ("Pathological", "Neurotypical"), loc = "upper right", borderaxespad=0.1)
    plt.ylim((0.0,1.15))
    plt.grid()
    plt.show()

    fig = plt.subplots(figsize =(10, 7))
    p1 = plt.bar(ind - 0.15, jac_mean_pat, 0.3, yerr = jac_std_pat)
    p2 = plt.bar(ind + 0.15, jac_mean, 0.3, yerr = jac_std)
    plt.xlabel('Gestational age (weeks)')
    plt.ylabel('Mean Jaccard Index')
    plt.xticks(ge)
    plt.legend((p1[0], p2[0]), ("Pathological", "Neurotypical"), loc = "upper right", borderaxespad=0.1)
    plt.ylim((0.0,1.15))
    plt.grid()
    plt.show()


def volume_sub():
    vs_list = []
    paths = glob.glob("../results/result_VSsub-*.csv")
    for path in paths:
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')
            i = 0
            curr_vs = 0.0
            for row in spamreader:
                if i == 0:
                    i = 1
                    continue
                curr_vs += float(row[2])
            vs_list.append(curr_vs/8)
                
    return vs_list


def volume_plot():
    dict_structures = {
        'BG' : [],
        'BS' : [],
        'CBM' : [],
        'CSF' : [],
        'GM' : [],
        'LV' : [],
        'SGM' : [],
        'WM' : [],
    }

    paths = glob.glob("../results/result_VSsub-*.csv")
    for path in paths:
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')
            i = 0
            for row in spamreader:
                if i == 0:
                    i = 1
                    continue
                dict_structures[row[1]].append(float(row[2]))

    mean = []
    std = []
    structures = ['BG', 'BS', 'CBM', 'CSF', 'GM', 'LV', 'SGM', 'WM']

    for (key, value) in dict_structures.items():
        mean.append(np.mean(value))
        std.append(np.std(value))
    
    ind = np.arange(8)
    fig = plt.subplots(figsize =(10, 7))
    plt.bar(ind, mean, 0.3, yerr = std)
    plt.xlabel('Brain structures')
    plt.ylabel('Mean Volume Similarity')
    plt.xticks(ind, structures)
    plt.grid()
    plt.show()

def volume_mean():
    vs = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    ge = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    ge_list = []

    pat_list = []
    with open('../results/result.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            pat_list.append(float(row[2]))
            ge_list.append(round(float(row[1])))


    paths = glob.glob("../results/result_VSsub-*.csv")
    j = 0 
    for path in paths:
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')
            i = 0
            for row in spamreader:
                if i == 0:
                    i = 1
                    continue
                idx = round(float(ge_list[j])) - 20
                vs[idx].append(float(row[2]))
        j = j + 1

    vs_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    vs_std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    idx = 0
    for j in vs:
        if len(j) == 0:
            vs_mean[idx] = 0.0
            vs_std[idx] = 0.0
        else:
            vs_mean[idx] = np.mean(j)
            vs_std[idx] = np.std(j)
        idx = idx + 1

    ind = np.arange(20, 36)
    fig = plt.subplots(figsize =(10, 7))
    plt.bar(ind + 0.15, vs_mean, 0.3, yerr = vs_std)
    plt.xlabel('Gestational age (weeks)')
    plt.ylabel('Mean Volume Similarity')
    plt.xticks(ge)
    plt.ylim((0.0,1.15))
    plt.grid()
    plt.show()

def volume_mean_pat():
    vs_pat = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    vs_non = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    ge = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    ge_list = []

    pat_list = []
    with open('../results/result.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            pat_list.append(float(row[2]))
            ge_list.append(round(float(row[1])))


    paths = glob.glob("../results/result_VSsub-*.csv")
    j = 0 
    for path in paths:
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')
            i = 0
            for row in spamreader:
                if i == 0:
                    i = 1
                    continue
                idx = round(float(ge_list[j])) - 20
                if  pat_list[j] == 0:
                    vs_non[idx].append(float(row[2]))
                else:
                    vs_pat[idx].append(float(row[2]))
        j = j + 1

        
    vs_mean_pat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    vs_std_pat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    vs_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    vs_std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    idx = 0
    for d in vs_pat:
        if len(d) == 0:
            vs_mean_pat[idx] = 0.0
            vs_std_pat[idx] = 0.0
        else:
            vs_mean_pat[idx] = np.mean(d)
            vs_std_pat[idx] = np.std(d)
        idx = idx + 1

    idx = 0
    for j in vs_non:
        if len(j) == 0:
            vs_mean[idx] = 0.0
            vs_std[idx] = 0.0
        else:
            vs_mean[idx] = np.mean(j)
            vs_std[idx] = np.std(j)
        idx = idx + 1

    ind = np.arange(20, 36)
    fig = plt.subplots(figsize =(10, 7))
    p1 = plt.bar(ind - 0.15, vs_mean_pat, 0.3, yerr = vs_std_pat)
    p2 = plt.bar(ind + 0.15, vs_mean, 0.3, yerr = vs_std)
    plt.xlabel('Gestational age (weeks)')
    plt.ylabel('Mean Volume Similarity')
    plt.xticks(ge)
    plt.legend((p1[0], p2[0]), ("Pathological", "Neurotypical"), loc = "upper right", borderaxespad=0.1)
    plt.ylim((0.0,1.15))
    plt.grid()
    plt.show()

    

def volume_plot_ge_pat():
    ge_list = []
    pat_list = []
    with open('../results/result.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            ge_list.append(round(float(row[1])))
            pat_list.append(float(row[2]))

    dict_structures = {
        'BG' : {
            'pat': [],
            'non' : [],
        },
        'BS' :  {
            'pat': [],
            'non' : [],
        },
        'CBM' :  {
            'pat': [],
            'non' : [],
        },
        'CSF' :  {
            'pat': [],
            'non' : [],
        },
        'GM' :  {
            'pat': [],
            'non' : [],
        },
        'LV' :  {
            'pat': [],
            'non' : [],
        },
        'SGM' :  {
            'pat': [],
            'non' : [],
        },
        'WM' :  {
            'pat': [],
            'non' : [],
        }
    }

    paths = glob.glob("../results/result_VSsub-*.csv")
    j = 0 
    for path in paths:
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')
            i = 0
            for row in spamreader:
                if i == 0:
                    i = 1
                    continue
                if  pat_list[j] == 0:
                    dict_structures[row[1]]["non"].append(float(row[2]))
                else:
                     dict_structures[row[1]]["pat"].append(float(row[2]))
        j = j + 1

    mean_pat = []
    mean_non = []
    std_pat = []
    std_non = []
    structures = ['BG', 'BS', 'CBM', 'CSF', 'GM', 'LV', 'SGM', 'WM']

    for (struct, dict_pat) in dict_structures.items():
        for (key, value) in dict_pat.items():
            if key == 'pat':
                if len(value) == 0:
                    mean_pat.append(0.0)
                    std_pat.append(0.0)
                else:
                    mean_pat.append(np.mean(value))
                    std_pat.append(np.std(value))
            else:
                if len(value) == 0:
                    mean_non.append(0.0)
                    std_non.append(0.0)
                else:
                    mean_non.append(np.mean(value))
                    std_non.append(np.std(value))
    
    ind = np.arange(8)
    fig = plt.subplots(figsize =(10, 7))
    p1 = plt.bar(ind + 0.15, mean_pat, 0.3, yerr = std_pat)
    p2 = plt.bar(ind - 0.15, mean_non, 0.3, yerr = std_non)
    plt.xlabel('Brain structures', labelpad=10)
    plt.ylabel('Mean Volume Similarity')
    plt.xticks(ind, structures)
    plt.grid()
    plt.legend((p1[0], p2[0]), ("Pathological", "Neurotypical"), loc = "upper right", borderaxespad=0.1)
    plt.ylim((0.0,1.15))
    plt.show()

def histogram_data():
    ge = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    pat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    non = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    with open('../results/result.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            idx = round(float(row[1])) - 20
            if float(row[2]) == 0:
                non[idx] = non[idx] + 1
            else:
                pat[idx] = pat[idx] + 1

    
    print(non)
    print(pat)
    fig, ax = plt.subplots()

    ax.bar(ge, non,  label='Neurotypical')
    ax.bar(ge, pat, bottom=non, label='Pathological')
    ax.set_ylabel("Number")
    ax.set_xlabel('Gestational age (weeks)')
    plt.legend()
    plt.grid()
    plt.show()

def get_min_max_id():
    min_dc_id = ''
    max_dc_id= ''
    min_dc = 1.0
    max_dc = 0.0

    min_jac_id = ''
    max_jac_id= ''
    min_jac = 1.0
    max_jac = 0.0

    min_vs_id = ''
    max_vs_id= ''
    min_vs = 1.0
    max_vs = 0.0

    vs_list = volume_sub()
    print(vs_list)

    with open('../results/result.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in spamreader:
            if float(row[3]) < min_dc:
                min_dc = float(row[3]) 
                min_dc_id = row[0]

            if float(row[4]) < min_jac:
                min_jac = float(row[4]) 
                min_jac_id = row[0]

            if vs_list[i] < min_vs:
                min_vs = vs_list[i] 
                min_vs_id = row[0]

            if float(row[3]) > max_dc:
                max_dc = float(row[3]) 
                max_dc_id = row[0]

            if float(row[4]) > max_jac:
                max_jac = float(row[4]) 
                max_jac_id = row[0]

            if vs_list[i] > max_vs:
                max_vs = vs_list[i] 
                max_vs_id = row[0]

            
            i += 1


    print("Min ds = " + str(min_dc))
    print("Min ds ID = " + str(min_dc_id))

    print("Max dc = " + str(max_dc))
    print("Max dc ID = " + str(max_dc_id))

    print("Min jac = " + str(min_jac))
    print("Min jac ID = " + str(min_jac_id))

    print("Max jac = " + str(max_jac))
    print("Max jac ID = " + str(max_jac_id))

    print("Min vs = " + str(min_vs))
    print("Min vs ID = " + str(min_vs_id))

    print("Max vs = " + str(max_vs))
    print("Max vs ID = " + str(max_vs_id))

def image_result():
    transform = ToTensorD(keys=["label"])
    dict_loader = LoadImageD(keys=("label"))

    mask_dataset = FetalBrainMRI(data_path="../data/feta2022/*/anat",
                              mode="val")
    for _, mask, ge, pathology, id  in mask_dataset:
        mask = mask.squeeze(0)
        if id == 'sub-017':
            segmented_path = os.path.join("../output/", str(id), str(id) + "_rec-mial_segmented.nii.gz")
            data_dict = dict_loader({"label": segmented_path})
            segmented = transform(data_dict)['label']

            min_seg = segmented
            min_mask = mask

            # fig, ((ax1, ax2)) = plt.subplots(2, 1)
            # ax1.imshow(segmented[:, :, 75])
            # ax1.set_title("Prediction")
            # ax2.imshow(mask[:, :, 75])
            # ax2.set_title("Original mask")
            # fig.suptitle("Jaccard = 0.70", y=0.05)
            # ax1.axis('off')
            # ax2.axis('off')
            # plt.show()

        if id == 'sub-040':
            segmented_path = os.path.join("../output/", str(id), str(id) + "_rec-mial_segmented.nii.gz")
            data_dict = dict_loader({"label": segmented_path})
            segmented = transform(data_dict)['label']

            max_seg = segmented
            max_mask = mask

            # fig, ((ax1, ax2)) = plt.subplots(2, 1)
            # ax1.imshow(segmented[:, :, 100])
            # ax1.set_title("Prediction")
            # ax2.imshow(mask[:, :, 100])
            # ax2.set_title("Original mask")
            # fig.suptitle("Jaccard = 0.84", y=0.05)
            # ax1.axis('off')
            # ax2.axis('off')
            # plt.show()
        
        if id == 'sub-047':
            segmented_path = os.path.join("../output/", str(id), str(id) + "_rec-irtk_segmented.nii.gz")
            data_dict = dict_loader({"label": segmented_path})
            segmented = transform(data_dict)['label']

            fig, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3)
            ax1.imshow(min_seg[:, :, 75])
            ax2.imshow(min_mask[:, :, 75])
            ax3.imshow(segmented[:, :, 100])
            ax3.set_title("Prediction")
            ax4.imshow(mask[:, :, 100])
            ax4.set_title("Original mask")
            ax5.imshow(max_seg[:, :, 100])
            ax6.imshow(max_mask[:, :, 100])
            fig.suptitle("Dice = 0.82                Dice = 0.86                 Dice = 0.91", y=0.05)
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            ax4.axis('off')
            ax5.axis('off')
            ax6.axis('off')
            plt.show()
            break

def plot_time():
    unet_time = [18.9, 17.0, 19.7, 18.5, 18.9, 19.1, 17.5]
    unetr_time = [6.9, 7.8, 8.6, 6.0, 9.0, 10.0, 8.1, 7.8]
    modif_time = [13.2, 8.9, 10.3, 9.9, 11.1, 12.7, 13.3, 14.7]
    
    unet_mean = np.mean(unet_time)
    unet_std = np.std(unet_time)

    unetr_mean = np.mean(unetr_time)
    unetr_std = np.std(unetr_time)

    modif_mean = np.mean(modif_time)
    modif_std = np.std(modif_time)

    model_name = ["U-Net 3D", "UNETR", "Modified \n Swin-Unet"]

    ind = [1, 1.5, 2]
    fig = plt.subplots(figsize =(10, 7))
    p1 = plt.bar(1, unet_mean, 0.3, yerr = unet_std)
    p2 = plt.bar(1.5, unetr_mean, 0.3, yerr = unetr_std)
    p3 = plt.bar(2.0, modif_mean, 0.3, yerr = modif_std)

    plt.xlabel('Model name', labelpad=10)
    plt.ylabel('Time (hours)')
    plt.xticks(ind, model_name)
    plt.grid()
    plt.ylim((0.0,22.0))
    plt.show()



if __name__ == '__main__':
    plot_time()