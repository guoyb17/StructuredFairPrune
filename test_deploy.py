import argparse
from tqdm import tqdm
from time import time, process_time

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp

from utils import model_backbone_v2
from dataloaders import fitzpatric17k_dataloader_score_v2, fitzpatric17k_pruning_examples, \
                        isic2019_dataloader_score_v2, isic2019_pruning_examples, \
                        celeba_dataloader_score_v2, celeba_pruning_examples


parser = argparse.ArgumentParser(description='Model Efficiency Test by Deploying on Various Devices')
parser.add_argument('-n', '--num_classes', type=int, default=114,
                    help="number of classes; used for fitzpatrick17k")
parser.add_argument('-f', '--fair_attr', type=str, default="Male",
                    help="fairness attribute; now support: Male, Young; used for celeba")
parser.add_argument('-y', '--y_attr', type=str, default="Big_Nose",
                    help="y attribute; now support: Attractive, Big_Nose, Bags_Under_Eyes, Mouth_Slightly_Open, Big_Nose_And_Bags_Under_Eyes, Attractive_And_Mouth_Slightly_Open; used for celeba")
parser.add_argument('-d', '--dataset', type=str, default="fitzpatrick17k",
                    help="the dataset to use; now support: fitzpatrick17k, celeba, isic2019")
parser.add_argument('-m', '--model_file', type=str, required=True,
                    help="Model file path")
parser.add_argument('--csv_file_name', type=str, default=None,
                    help="CSV file position")
parser.add_argument('--image_dir', type=str, default=None,
                    help="Image files directory")
parser.add_argument('-t', '--test_time', type=int, default=5,
                    help="how many rounds to test")
parser.add_argument('-o', '--output_file', type=str, required=True,
                    help="Output file path")
parser.add_argument('--pruned', type=int, default=0,
                    help="Input models type:"
                    "default: 0 (regular models);"
                    "1: FairPrune models (pre-set to zero); 2: FairPrune models (masking layers); 3: torch_pruning models; "
                    "-1: activation models")
parser.add_argument('--backbone', type=str, default="resnet18",
                    help="backbone model; now support: resnet18, vgg11, vgg11_bn")
parser.add_argument('-b', '--batch_size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--device', type=str, required=True,
                    help="Device to use; now support: cpu, cuda, mps")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.dataset == "fitzpatrick17k":
        if args.csv_file_name is None:
            csv_file_name = "fitzpatrick17k/fitzpatrick17k.csv"
        else:
            csv_file_name = args.csv_file_name
        if args.image_dir is None:
            image_dir = "fitzpatrick17k/dataset_images"
        else:
            image_dir = args.image_dir
        num_classes = args.num_classes
        if num_classes == 3:
            ctype = "high"
        elif num_classes == 9:
            ctype = "mid"
        elif num_classes == 114:
            ctype = "low"
        else:
            raise NotImplementedError
        f_attr = "skin_color_binary"
        score_size = 799
        _, _, testloader, _ = fitzpatric17k_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, ctype)
        # if args.shape_type == 1:
        #     trainloader, _, testloader, _ = fitzpatric17k_dataloader_score_v3(args.batch_size, args.workers, image_dir, csv_file_name, ctype, device)
        # else:
        #     trainloader, _, testloader, _ = fitzpatric17k_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, ctype)
    elif args.dataset == "celeba":
        if args.csv_file_name is None:
            csv_file_name = "img_align_celeba/list_attr_celeba_modify.txt"
        else:
            csv_file_name = args.csv_file_name
        if args.image_dir is None:
            image_dir = "img_align_celeba"
        else:
            image_dir = args.image_dir
        num_classes = 2
        ctype = "y_attr"
        f_attr = "fair_attr"
        score_size = 10130
        _, _, testloader, _ = celeba_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, args.fair_attr, args.y_attr, use_val_df=True)
    elif args.dataset == "isic2019":
        if args.csv_file_name is None:
            csv_file_name = "ISIC_2019_train/ISIC_2019_Training_Metadata.csv"
        else:
            csv_file_name = args.csv_file_name
        if args.image_dir is None:
            image_dir = "ISIC_2019_train/ISIC_2019_Training_Input"
        else:
            image_dir = args.image_dir
        num_classes = 8
        ctype = "y_attr"
        f_attr = "fair_attr"
        score_size = 1248
        _, _, testloader, _ = isic2019_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, use_test=True)
    else:
        raise NotImplementedError
    f = open(args.output_file, "w")
    # f.write("model,overall_accuracy,light_accuracy,dark_accuracy,diff_accuracy,light_precision,dark_precision,diff_precision,light_recall,dark_recall,diff_recall,light_F1_score,dark_F1_score,diff_F1_score,(abs_of_)EOpp0,EOpp1_abs,EOdds_abs\n")

    try:
        device = torch.device(args.device)
    except:
        f.write("[ERROR] Device not supported.\n")
        f.close()
        exit(1)

    # define the backbone model
    backbone = model_backbone_v2(num_classes, args.backbone)
    # backbone = models.vgg11(num_classes=num_classes)
    net = backbone.to(device)

    loaded_ckpt = torch.load(args.model_file, map_location=device, weights_only=True)
    print("[NOTE] Testing", args.model_file)
    if args.pruned == 3:
        if args.dataset == "fitzpatrick17k":
            example_inputs, example_labels = fitzpatric17k_pruning_examples()
        elif args.dataset == "isic2019":
            example_inputs, example_labels = isic2019_pruning_examples()
        elif args.dataset == "celeba":
            if args.fair_attr != "Male" or args.y_attr != "Big_Nose": # FIXME: hard code
                raise NotImplementedError
            example_inputs, example_labels = celeba_pruning_examples()
        example_inputs = torch.stack(example_inputs, dim=0).to(device)
        example_labels = torch.tensor(example_labels).to(device)
        DG = tp.DependencyGraph().build_dependency(net, example_inputs)
        DG.load_pruning_history(loaded_ckpt['pruning'])
        net.load_state_dict(loaded_ckpt['model'])
    elif args.pruned == 2:
        loaded_ckpt = loaded_ckpt['model_dict']
        idx = 0
        for name, module in net.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                try:
                    loaded_ckpt[name + ".weight"] = loaded_ckpt[name + ".weight"] * loaded_ckpt[name + ".weight_mask"]
                    loaded_ckpt.pop(name + ".weight_mask")
                except:
                    pass
                try:
                    loaded_ckpt[name + ".bias"] = loaded_ckpt[name + ".bias"] * loaded_ckpt[name + ".bias_mask"]
                    loaded_ckpt.pop(name + ".bias_mask")
                except:
                    pass
                idx += 1
        for key in list(loaded_ckpt.keys()):
            if key.endswith("_mask"):
                loaded_ckpt.pop(key)
        net.load_state_dict(loaded_ckpt)
    elif args.pruned == 1:
        # FairPrune loading
        net.load_state_dict(loaded_ckpt['model_dict']) # For Dewen's FairPrune models
    else:
        try:
            net.load_state_dict(loaded_ckpt['state_dict'])
        except:
            try:
                net.load_state_dict(loaded_ckpt['model_dict']) # For Dewen's FairPrune models
            except:
                try:
                    net.load_state_dict(loaded_ckpt)
                except:
                    raise NotImplementedError

    # define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss()
    # Test
    net.eval()
    correct = 0
    total = 0

    label_list = []
    y_pred_list = []
    skin_color_list = []
    average_cpu_time = 0.0
    average_process_time = 0.0
    test_time = args.test_time
    for test_id in range(test_time):
        cpu_times = 0.0
        process_times = 0.0
        for i, data in enumerate(tqdm(testloader)):
            inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
            if args.device == "cuda":
                torch.cuda.synchronize()
            elif args.device == "mps":
                torch.mps.synchronize()
            start_process = time()
            start_cpu = process_time()
            outputs = net(inputs) # Only time this line
            if args.device == "cuda":
                torch.cuda.synchronize()
            elif args.device == "mps":
                torch.mps.synchronize()
            end_cpu = process_time()
            end_process = time()
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum()
            # label_list.append(labels.detach().cpu().numpy())
            # y_pred_list.append(predicted.detach().cpu().numpy())
            # skin_color_list.append(data[f_attr].numpy())
            cpu_times += end_cpu - start_cpu
            process_times += end_process - start_process
        f.write("Test #{} - Process time: {}\n".format(test_id, process_times))
        f.write("Test #{} - CPU time: {}\n".format(test_id, cpu_times))
        average_cpu_time += cpu_times
        average_process_time += process_times
    average_cpu_time /= test_time
    average_process_time /= test_time
    f.write("Average Process time: {}\n".format(average_process_time))
    f.write("Average CPU time: {}\n".format(average_cpu_time))

    f.close()
