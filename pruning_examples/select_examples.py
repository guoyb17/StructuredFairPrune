from os import makedirs, getcwd
from os.path import join
from argparse import ArgumentParser
from shutil import copyfile
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(getcwd())
from dataloaders import fitzpatric17k_dataloader_score_v2s, isic2019_dataloader_score_v2s, celeba_dataloader_score_v2s


parser = ArgumentParser(description='Generate pruning examples')
parser.add_argument('--fitzpatrick17k', type=int, default=0, help='whether to run fitzpatrick17k part; 0: no, 1: yes; default: 0')
parser.add_argument('--isic2019', type=int, default=0, help='whether to run isic2019 part; 0: no, 1: yes; default: 0')
parser.add_argument('--celeba', type=int, default=0, help='whether to run celeba part; 0: no, 1: yes; default: 0')
parser.add_argument('--batch_size', type=int, default=32, help='batch size; default: 32')
parser.add_argument('-f', '--fair_attr', type=str, default="Male",
                    help="fairness attribute; now support: Male, Young; used for celeba")
parser.add_argument('-y', '--y_attr', type=str, default="Big_Nose",
                    help="y attribute; now support: Attractive, Big_Nose, Bags_Under_Eyes, Mouth_Slightly_Open, Big_Nose_And_Bags_Under_Eyes, Attractive_And_Mouth_Slightly_Open; used for celeba")

if __name__ == '__main__':
    args = parser.parse_args()

    # fitzpatrick17k
    if args.fitzpatrick17k == 1:
        print("[INFO] Start processing fitzpatrick17k")
        fitzpatrick17k_csv = "fitzpatrick17k/fitzpatrick17k.csv"
        fitzpatrick17k_dir = "fitzpatrick17k/dataset_images"
        fitzpatrick17_target_csv = "pruning_examples/fitzpatrick17k.csv"
        fitzpatrick17_target_dir = "pruning_examples/fitzpatrick17k"
        makedirs(fitzpatrick17_target_dir, exist_ok=True)

        target_cnt = 6 * 114 # 6 fairness groups, 114 classes
        trainloader, valloader, df = fitzpatric17k_dataloader_score_v2s(args.batch_size, 8, fitzpatrick17k_dir, fitzpatrick17k_csv, "low")
        df_selected = pd.DataFrame(columns=df.columns)

        # trainloader
        fitzpatrick17_samples = {}
        current_cnt = 0
        for _, data in enumerate(tqdm(trainloader)):
            a_labels = data["low"]
            f_labels = data["fitzpatrick"]
            for i in range(len(data["hasher"])):
                a_label = a_labels[i].item()
                f_label = f_labels[i].item()
                if f_label not in fitzpatrick17_samples and f_label != -1:
                    fitzpatrick17_samples[f_label] = {}
                if f_label in fitzpatrick17_samples and a_label not in fitzpatrick17_samples[f_label]:
                    fitzpatrick17_samples[f_label][a_label] = data["hasher"][i]
                    current_cnt += 1
            if current_cnt >= target_cnt:
                break
        for f_label in fitzpatrick17_samples.keys():
            for a_label in fitzpatrick17_samples[f_label].keys():
                df_selected = pd.concat([df_selected, df.loc[df["hasher"] == fitzpatrick17_samples[f_label][a_label]]], ignore_index=True)

        # valloader
        fitzpatrick17_samples = {}
        current_cnt = 0
        for _, data in enumerate(tqdm(valloader)):
            a_labels = data["low"]
            f_labels = data["fitzpatrick"]
            for i in range(len(data["hasher"])):
                a_label = a_labels[i].item()
                f_label = f_labels[i].item()
                if f_label not in fitzpatrick17_samples and f_label != -1:
                    fitzpatrick17_samples[f_label] = {}
                if f_label in fitzpatrick17_samples and a_label not in fitzpatrick17_samples[f_label]:
                    fitzpatrick17_samples[f_label][a_label] = data["hasher"][i]
                    current_cnt += 1
            if current_cnt >= target_cnt:
                break
        for f_label in fitzpatrick17_samples.keys():
            for a_label in fitzpatrick17_samples[f_label].keys():
                df_selected = pd.concat([df_selected, df.loc[df["hasher"] == fitzpatrick17_samples[f_label][a_label]]], ignore_index=True)

        df_selected.to_csv(fitzpatrick17_target_csv, index=False)
        for idx in range(len(df_selected)):
            img_name = df_selected.loc[df_selected.index[idx], "hasher"] + ".jpg"
            copyfile(join(fitzpatrick17k_dir, img_name), join(fitzpatrick17_target_dir, img_name))

    # isic2019
    if args.isic2019 == 1:
        print("[INFO] Start processing isic2019")
        isic2019_csv = "ISIC_2019_train/ISIC_2019_Training_Metadata.csv"
        isic2019_dir = "ISIC_2019_train/ISIC_2019_Training_Input"
        isic2019_target_csv = "pruning_examples/isic2019.csv"
        isic2019_target_dir = "pruning_examples/isic2019"
        makedirs(isic2019_target_dir, exist_ok=True)

        target_cnt = 2 * 8 # 2 fairness groups, 8 classes
        trainloader, valloader, df = isic2019_dataloader_score_v2s(args.batch_size, 8, isic2019_dir, isic2019_csv)
        df_selected = pd.DataFrame(columns=df.columns)

        # trainloader
        isic2019_samples = {}
        current_cnt = 0
        for _, data in enumerate(tqdm(trainloader)):
            a_labels = data["y_attr"]
            f_labels = data["fair_attr"]
            for i in range(len(data["img"])):
                a_label = a_labels[i].item()
                f_label = f_labels[i].item()
                if f_label not in isic2019_samples:
                    isic2019_samples[f_label] = {}
                if a_label not in isic2019_samples[f_label]:
                    isic2019_samples[f_label][a_label] = data["img"][i]
                    current_cnt += 1
            if current_cnt >= target_cnt:
                break
        for f_label in isic2019_samples.keys():
            for a_label in isic2019_samples[f_label].keys():
                df_selected = pd.concat([df_selected, df.loc[df["image"] == isic2019_samples[f_label][a_label]]], ignore_index=True)

        # valloader
        isic2019_samples = {}
        current_cnt = 0
        for _, data in enumerate(tqdm(valloader)):
            a_labels = data["y_attr"]
            f_labels = data["fair_attr"]
            for i in range(len(data["img"])):
                a_label = a_labels[i].item()
                f_label = f_labels[i].item()
                if f_label not in isic2019_samples:
                    isic2019_samples[f_label] = {}
                if a_label not in isic2019_samples[f_label]:
                    isic2019_samples[f_label][a_label] = data["img"][i]
                    current_cnt += 1
            if current_cnt >= target_cnt:
                break
        for f_label in isic2019_samples.keys():
            for a_label in isic2019_samples[f_label].keys():
                df_selected = pd.concat([df_selected, df.loc[df["image"] == isic2019_samples[f_label][a_label]]], ignore_index=True)

        df_selected.to_csv(isic2019_target_csv, index=False)
        for idx in range(len(df_selected)):
            img_name = df_selected.loc[df_selected.index[idx], "image"] + ".jpg"
            copyfile(join(isic2019_dir, img_name), join(isic2019_target_dir, img_name))

    # celeba
    if args.celeba == 1:
        print("[INFO] Start processing celeba")
        celeba_csv = "img_align_celeba/list_attr_celeba_modify.txt"
        celeba_dir = "img_align_celeba"
        celeba_target_csv = "pruning_examples/celeba_" + args.fair_attr + "_" + args.y_attr + ".csv"
        celeba_target_dir = "pruning_examples/celeba_" + args.fair_attr + "_" + args.y_attr
        makedirs(celeba_target_dir, exist_ok=True)

        target_cnt = 2 * 2 # 2 fairness groups, 2 classes
        trainloader, valloader, df = celeba_dataloader_score_v2s(args.batch_size, 8, celeba_dir, celeba_csv, args.fair_attr, args.y_attr)
        df_selected = pd.DataFrame(columns=df.columns)

        # trainloader
        celeba_samples = {}
        current_cnt = 0
        for _, data in enumerate(tqdm(trainloader)):
            a_labels = data["y_attr"]
            f_labels = data["fair_attr"]
            for i in range(len(data["img"])):
                a_label = a_labels[i].item()
                f_label = f_labels[i].item()
                if f_label not in celeba_samples:
                    celeba_samples[f_label] = {}
                if a_label not in celeba_samples[f_label]:
                    celeba_samples[f_label][a_label] = data["img"][i]
                    current_cnt += 1
            if current_cnt >= target_cnt:
                break
        for f_label in celeba_samples.keys():
            for a_label in celeba_samples[f_label].keys():
                df_selected = pd.concat([df_selected, df.loc[df["Image_Id"] == celeba_samples[f_label][a_label]]], ignore_index=True)

        # valloader
        celeba_samples = {}
        current_cnt = 0
        for _, data in enumerate(tqdm(valloader)):
            a_labels = data["y_attr"]
            f_labels = data["fair_attr"]
            for i in range(len(data["img"])):
                a_label = a_labels[i].item()
                f_label = f_labels[i].item()
                if f_label not in celeba_samples:
                    celeba_samples[f_label] = {}
                if a_label not in celeba_samples[f_label]:
                    celeba_samples[f_label][a_label] = data["img"][i]
                    current_cnt += 1
            if current_cnt >= target_cnt:
                break
        for f_label in celeba_samples.keys():
            for a_label in celeba_samples[f_label].keys():
                df_selected = pd.concat([df_selected, df.loc[df["Image_Id"] == celeba_samples[f_label][a_label]]], ignore_index=True)

        df_selected.to_csv(celeba_target_csv, index=False)
        for idx in range(len(df_selected)):
            img_name = df_selected.loc[df_selected.index[idx], "Image_Id"]
            img_id = str(int(img_name.split(".")[0]) % 5)
            makedirs(join(celeba_target_dir, img_id), exist_ok=True)
            copyfile(join(celeba_dir, img_id, img_name), join(celeba_target_dir, img_id, img_name))
