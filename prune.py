import os, argparse
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
# from sklearn import metrics
import torch
import torch.nn as nn
import torch_pruning as tp

from utils import model_backbone_v2, genScoreDataset, validate, genValLoader
from pruner import FairPruneImportance
from dataloaders import fitzpatric17k_dataloader_score_v2, fitzpatric17k_pruning_examples, \
                        isic2019_dataloader_score_v2, isic2019_pruning_examples, \
                        celeba_dataloader_score_v2, celeba_pruning_examples


def main(args, output_dir, f, eta_min):
    # num_workers = 0 if args.to_cuda == 1 else args.num_workers
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    num_classes = args.num_classes
    if args.dataset == "fitzpatrick17k":
        if args.csv_file_name is None:
            csv_file_name = "fitzpatrick17k/fitzpatrick17k.csv"
        else:
            csv_file_name = args.csv_file_name
        if args.image_dir is None:
            image_dir = "fitzpatrick17k/dataset_images"
        else:
            image_dir = args.image_dir
        # score_size = 799
        if num_classes == 3:
            ctype = "high"
        elif num_classes == 9:
            ctype = "mid"
        elif num_classes == 114:
            ctype = "low"
        else:
            raise NotImplementedError
        f_attr = "skin_color_binary"
        _, valloader, _, train_df = fitzpatric17k_dataloader_score_v2(args.batch_size, args.num_workers, image_dir, csv_file_name, ctype)
    elif args.dataset == "celeba":
        if args.csv_file_name is None:
            csv_file_name = "img_align_celeba/list_attr_celeba_modify.txt"
        else:
            csv_file_name = args.csv_file_name
        if args.image_dir is None:
            image_dir = "img_align_celeba"
        else:
            image_dir = args.image_dir
        # score_size = 10130
        num_classes = 2
        ctype = "y_attr"
        f_attr = "fair_attr"
        _, val_df, _, train_df = celeba_dataloader_score_v2(args.batch_size, args.num_workers, image_dir, csv_file_name, args.fair_attr, args.y_attr, True)
    elif args.dataset == "isic2019":
        if args.csv_file_name is None:
            csv_file_name = "ISIC_2019_train/ISIC_2019_Training_Metadata.csv"
        else:
            csv_file_name = args.csv_file_name
        if args.image_dir is None:
            image_dir = "ISIC_2019_train/ISIC_2019_Training_Input"
        else:
            image_dir = args.image_dir
        # score_size = 1248
        num_classes = 8
        ctype = "y_attr"
        f_attr = "fair_attr"
        _, valloader, _, train_df = isic2019_dataloader_score_v2(args.batch_size, args.num_workers, image_dir, csv_file_name, use_val=True)
    else:
        raise NotImplementedError

    if args.dataset == "fitzpatrick17k":
        example_inputs, example_labels = fitzpatric17k_pruning_examples()
    elif args.dataset == "isic2019":
        example_inputs, example_labels = isic2019_pruning_examples()
    elif args.dataset == "celeba":
        if args.fair_attr != "Male" or args.y_attr != "Big_Nose": # FIXME: hard code
            raise NotImplementedError
        example_inputs, example_labels = celeba_pruning_examples()
    else:
        raise NotImplementedError
    example_inputs = torch.stack(example_inputs, dim=0).to(device)
    example_labels = torch.from_numpy(np.asarray(torch.tensor(example_labels))).long().to(device)

    iterative_steps = args.step # progressive pruning
    model = model_backbone_v2(num_classes, args.backbone)
    if model is None:
        raise NotImplementedError
        model = torch.load(args.load_model_path, weights_only=True) # FIXME: currently, model cannot be None
    else:
        model.load_state_dict(torch.load(args.load_model_path, map_location=device, weights_only=True)["state_dict"])
    model.to(device)
    model.eval()
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == num_classes:
            ignored_layers.append(m) # DO NOT prune the final classifier!
    # Importance criteria
    global_pruning = False
    use_isomorphic = False
    if args.global_pruning == 1:
        global_pruning = True
    if args.isomorphic == 1:
        use_isomorphic = True
    '''
    -1: RandomImportance
    0: FairPruneImportance
    1: TaylorImportance
    2: MagnitudeImportance
    3: HessianImportance
    4: LAMPImportance
    5: BNScaleImportance
    '''
    if args.importance == -1:
        imp = tp.importance.RandomImportance()
    elif args.importance == 0:
        imp = FairPruneImportance(light_weight=args.light_weight, dark_weight=args.dark_weight, group_reduction=args.group_reduction, normalizer=args.normalizer)
    elif args.importance == 1:
        imp = tp.importance.TaylorImportance(group_reduction=args.group_reduction, normalizer=args.normalizer, multivariable=True if args.multivariable == 1 else False)
    elif args.importance == 2:
        imp = tp.importance.MagnitudeImportance(group_reduction=args.group_reduction, normalizer=args.normalizer)
    elif args.importance == 3:
        imp = tp.importance.HessianImportance(group_reduction=args.group_reduction, normalizer=args.normalizer)
    elif args.importance == 4:
        imp = tp.importance.LAMPImportance(group_reduction=args.group_reduction) # assert normalizer == 'lamp'
    elif args.importance == 5:
        imp = tp.importance.BNScaleImportance(group_reduction=args.group_reduction, normalizer=args.normalizer)
    else:
        raise NotImplementedError
    '''
    0: MagnitudePruner
    1: BNScalePruner
    2: GroupNormPruner
    3: GrowingRegPruner
    '''
    if args.pruner == 0:
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=args.sparsity, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,
            global_pruning=global_pruning,
            isomorphic=use_isomorphic,
        )
    elif args.pruner == 1:
        pruner = tp.pruner.BNScalePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=args.sparsity,
            ignored_layers=ignored_layers,
            global_pruning=global_pruning,
            isomorphic=use_isomorphic,
        )
    elif args.pruner == 2:
        pruner = tp.pruner.GroupNormPruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=args.sparsity,
            ignored_layers=ignored_layers,
            global_pruning=global_pruning,
            isomorphic=use_isomorphic,
        )
    elif args.pruner == 3:
        pruner = tp.pruner.GrowingRegPruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=args.sparsity,
            ignored_layers=ignored_layers,
            global_pruning=global_pruning,
            isomorphic=use_isomorphic,
        )
    else:
        raise NotImplementedError

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=eta_min, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()

    for istep in range(iterative_steps):
        print(f"[INFO] Step {istep+1}/{iterative_steps}")
        f.write(f"[INFO] Step {istep+1}/{iterative_steps}\n")
        if args.importance == 0 or args.importance == 10:
            light_score_dataset, dark_score_dataset = genScoreDataset(args, train_df, image_dir, device, False, args.max_score_size)
            lightloader = torch.utils.data.DataLoader(light_score_dataset, batch_size=args.batch_size, shuffle=False)
            darkloader = torch.utils.data.DataLoader(dark_score_dataset, batch_size=args.batch_size, shuffle=False)
        '''
        -1: RandomImportance
        0: FairPruneImportance
        1: TaylorImportance
        2: MagnitudeImportance
        3: HessianImportance
        4: LAMPImportance
        5: BNScaleImportance
        '''
        if isinstance(imp, tp.importance.HessianImportance): # 3
            # loss = F.cross_entropy(model(images), targets)
            output = model(example_inputs) 
            # compute loss for each sample
            loss = torch.nn.functional.cross_entropy(output, torch.randint(0, num_classes, (len(example_inputs),), device=device), reduction='none')
            imp.zero_grad() # clear accumulated gradients
            for l in loss:
                model.zero_grad() # clear gradients
                l.backward(retain_graph=True) # single-sample gradient
                imp.accumulate_grad(model) # accumulate g^2
        elif isinstance(imp, FairPruneImportance): # 0
            print("[NOTE] Scoring on light images...")
            for _, data in enumerate(tqdm(lightloader)):
                inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
                model.zero_grad()
                outputs = model(inputs)
                # _, predicted = torch.max(outputs.data, 1)
                loss = torch.nn.functional.cross_entropy(outputs, labels, reduction='none')
                for l in loss:
                    model.zero_grad() # clear gradients
                    l.backward(retain_graph=True) # single-sample gradient
                    imp.accumulate_grad_light(model) # accumulate g^2
                if args.debug == 1:
                    break

            print("[NOTE] Scoring on dark images...")
            for _, data in enumerate(tqdm(darkloader)):
                inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
                model.zero_grad()
                outputs = model(inputs)
                # _, predicted = torch.max(outputs.data, 1)
                loss = torch.nn.functional.cross_entropy(outputs, labels, reduction='none')
                for l in loss:
                    model.zero_grad() # clear gradients
                    l.backward(retain_graph=True) # single-sample gradient
                    imp.accumulate_grad_dark(model) # accumulate g^2
                if args.debug == 1:
                    break
        else: # -1, 1, 2, 4, 5
            logits = model(example_inputs)
            loss = criterion(logits, example_labels)
            loss.backward() # before pruner.step()

        pruner.step()

        # Verify pruning
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(
            "[INFO] Iter %d/%d, Params: %.2f M => %.2f M"
            % (istep + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
        )
        print(
            "[INFO] Iter %d/%d, MACs: %.2f G => %.2f G"
            % (istep + 1, iterative_steps, base_macs / 1e9, macs / 1e9)
        )

        print("[NOTE] Pruner applied.")
        # torch.save(model, os.path.join(output_dir, str(istep) + '.1.pth')) # without .state_dict
        torch.save({
            'model': model.state_dict(),
            'pruning': pruner.pruning_history(),
        }, os.path.join(output_dir, str(istep) + '.pruned.pkl'))
        # torch.save(tp.state_dict(model), os.path.join(output_dir, str(istep) + '.pruned.pth'))

        # finetune your model here
        if args.epochs > 0:
            print("[NOTE] Fine-tuning...")
            # macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
            best_val_acc = 0.0
            best_val_precision = 0.0
            best_val_loss = 100.0
            best_val_spd = 100.0
            best_val_eopp0_abs = 100.0
            best_val_eopp1_abs = 100.0
            best_val_eodds_abs = 100.0
            binary_classification = args.dataset == "celeba"
            for epoch in range(args.epochs):  # range(args.start_epoch, n_epoch_debug)
                print('Retrain Epoch: %d' % (epoch + 1))
                f.write('Retrain Epoch: %d\n' % (epoch + 1))

                if binary_classification:
                    light_score_dataset_s, dark_score_dataset_s, balanced_val_df = genScoreDataset(args, train_df, image_dir, device, False, args.max_score_size, val_df)
                    valloader = genValLoader(balanced_val_df, args.batch_size, args.num_workers, image_dir, args.fair_attr, args.y_attr)
                else:
                    light_score_dataset_s, dark_score_dataset_s = genScoreDataset(args, train_df, image_dir, device, False, args.max_score_size)
                lightloader_s = torch.utils.data.DataLoader(light_score_dataset_s, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
                darkloader_s = torch.utils.data.DataLoader(dark_score_dataset_s, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
                # for i, data in enumerate(tqdm(trainloader)):
                for i, (light_data, dark_data) in enumerate(tzip(lightloader_s, darkloader_s)):
                    model.train()
                    optimizer.zero_grad()
                    # prepare dataset
                    # inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
                    light_inputs, light_labels = light_data["image"].float().to(device), torch.from_numpy(np.asarray(light_data[ctype])).long().to(device)
                    dark_inputs, dark_labels = dark_data["image"].float().to(device), torch.from_numpy(np.asarray(dark_data[ctype])).long().to(device)

                    # forward & backward
                    # outputs = model(inputs)
                    light_outputs = model(light_inputs)
                    dark_outputs = model(dark_inputs)

                    # loss = criterion(outputs, labels)
                    loss = (1.0 - args.alpha) * criterion(torch.cat([light_outputs, dark_outputs]), torch.cat([light_labels, dark_labels])) + args.alpha * torch.abs(criterion(light_outputs, light_labels) - criterion(dark_outputs, dark_labels))
                    loss.backward()
                    optimizer.step()

                    if args.debug == 1:
                        break

                scheduler.step()

                # Validate
                val_acc, val_loss, val_eopp0_abs, val_eopp1_abs, val_eodds_abs, val_precision, val_spd = validate(model, valloader, criterion, device, ctype, f_attr, True)
                # if binary_classification:
                #     val_acc, val_loss, val_eopp0_abs, val_eopp1_abs, val_eodds_abs, val_precision, val_spd = validate(model, valloader, criterion, device, ctype, f_attr, True)
                # else:
                #     val_acc, val_loss, val_eopp0_abs, val_eopp1_abs, val_eodds_abs, val_spd = validate(model, valloader, criterion, device, ctype, f_attr)
                if val_acc > best_val_acc:
                    print("New best val_acc: %.02f%% -> %.02f%%" % (best_val_acc, val_acc))
                    f.write("New best val_acc: %.02f%% -> %.02f%%\n" % (best_val_acc, val_acc))
                    best_val_acc = val_acc
                    torch.save({
                        'model': model.state_dict(),
                        'pruning': pruner.pruning_history(),
                    }, os.path.join(output_dir, str(istep) + '.best_val_acc.pkl'))
                # if binary_classification:
                if val_precision > best_val_precision:
                    print("New best val_precision: %.04f -> %.04f" % (best_val_precision, val_precision))
                    f.write("New best val_precision: %.04f -> %.04f\n" % (best_val_precision, val_precision))
                    best_val_precision = val_precision
                    torch.save({
                        'model': model.state_dict(),
                        'pruning': pruner.pruning_history(),
                    }, os.path.join(output_dir, str(istep) + '.best_val_precision.pkl'))
                if val_loss < best_val_loss:
                    print("New best val_loss: %.04f -> %.04f" % (best_val_loss, val_loss))
                    f.write("New best val_loss: %.04f -> %.04f\n" % (best_val_loss, val_loss))
                    best_val_loss = val_loss
                    torch.save({
                        'model': model.state_dict(),
                        'pruning': pruner.pruning_history(),
                    }, os.path.join(output_dir, str(istep) + '.best_val_loss.pkl'))
                if val_eopp0_abs < best_val_eopp0_abs:
                    print("New best val_eopp0_abs: %.04f -> %.04f" % (best_val_eopp0_abs, val_eopp0_abs))
                    f.write("New best val_eopp0_abs: %.04f -> %.04f\n" % (best_val_eopp0_abs, val_eopp0_abs))
                    best_val_eopp0_abs = val_eopp0_abs
                    torch.save({
                        'model': model.state_dict(),
                        'pruning': pruner.pruning_history(),
                    }, os.path.join(output_dir, str(istep) + '.best_val_eopp0_abs.pkl'))
                if val_eopp1_abs < best_val_eopp1_abs:
                    print("New best val_eopp1_abs: %.04f -> %.04f" % (best_val_eopp1_abs, val_eopp1_abs))
                    f.write("New best val_eopp1_abs: %.04f -> %.04f\n" % (best_val_eopp1_abs, val_eopp1_abs))
                    best_val_eopp1_abs = val_eopp1_abs
                    torch.save({
                        'model': model.state_dict(),
                        'pruning': pruner.pruning_history(),
                    }, os.path.join(output_dir, str(istep) + '.best_val_eopp1_abs.pkl'))
                if val_eodds_abs < best_val_eodds_abs:
                    print("New best val_eodds_abs: %.04f -> %.04f" % (best_val_eodds_abs, val_eodds_abs))
                    f.write("New best val_eodds_abs: %.04f -> %.04f\n" % (best_val_eodds_abs, val_eodds_abs))
                    best_val_eodds_abs = val_eodds_abs
                    torch.save({
                        'model': model.state_dict(),
                        'pruning': pruner.pruning_history(),
                    }, os.path.join(output_dir, str(istep) + '.best_val_eodds_abs.pkl'))
                if val_spd < best_val_spd:
                    print("New best val_spd: %.04f -> %.04f" % (best_val_spd, val_spd))
                    f.write("New best val_spd: %.04f -> %.04f\n" % (best_val_spd, val_spd))
                    best_val_spd = val_spd
                    torch.save({
                        'model': model.state_dict(),
                        'pruning': pruner.pruning_history(),
                    }, os.path.join(output_dir, str(istep) + '.best_val_spd.pkl'))

            torch.save({
                'model': model.state_dict(),
                'pruning': pruner.pruning_history(),
            }, os.path.join(output_dir, str(istep) + '.finetuned.pkl'))
            # torch.save(tp.state_dict(model), os.path.join(output_dir, str(istep) + '.finetuned.pth'))
        # else:
        #     # torch.save(model, os.path.join(output_dir, str(istep) + '.2.pth')) # without .state_dict
        #     state_dict = {
        #         'model': model.state_dict(),
        #         'pruning': pruner.pruning_history(),
        #     }
        #     torch.save(state_dict, os.path.join(output_dir, str(istep) + '.pruned.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Pruning Version 2')
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")
    parser.add_argument("--load_model_path", type=str, required=True, help="path to load model or state_dict")
    parser.add_argument('--backbone', type=str, required=True, help="model type; None for loading the entire model")
    parser.add_argument('-n', '--num_classes', type=int, default=114, help="number of classes; used for fitzpatrick17k")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--epochs', type=int, default=0, help="rounds of training")
    parser.add_argument('-b', '--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--step', type=int, default=18, help="number of steps to reach the target sparsity")
    parser.add_argument('--sparsity', type=float, default=0.9, help="target sparsity")
    parser.add_argument('--eta_min_mode', type=int, default=0, help="0: ignore args.eta_min, and set eta_min as 0; 1: ignore args.eta_min, and set eta_min as lr / 2; 2: use args.eta_min")
    parser.add_argument('--eta_min', type=float, default=0., help="Only used when args.eta_min_mode == 2")
    # parser.add_argument('--image_size', type=int, default=240, help="image size") # TODO: determine whether to use this
    parser.add_argument('-d', '--dataset', type=str, default="fitzpatrick17k", help="the dataset to use; now support: fitzpatrick17k, isic2019, celeba")
    parser.add_argument('--csv_file_name', type=str, default=None, help="CSV file position")
    parser.add_argument('--image_dir', type=str, default=None, help="Image files directory")
    parser.add_argument('--pruner', type=int, default=0, help="pruner selection.\
                        0: MagnitudePruner;\
                        1: BNScalePruner;\
                        2: GroupNormPruner;\
                        3: GrowingRegPruner")
    parser.add_argument("--importance", type=int, default=0, help="importance selection.\
                        -1: RandomImportance;\
                        0: FairPruneImportance;\
                        1: TaylorImportance, GroupTaylorImportance;\
                        2: MagnitudeImportance, GroupNormImportance;\
                        3: HessianImportance, GroupHessianImportance;\
                        4: LAMPImportance;\
                        5: BNScaleImportance")
    parser.add_argument("--norm", type=int, default=2, help="norm selection; used by MagnitudeImportance")
    parser.add_argument("--global_pruning", default=0, type=int, help="0: non global pruning; 1: global pruning")
    parser.add_argument('--light_weight', required=True, type=int, help='weight for hessian list on fair_attr=0 data; original: 9')
    parser.add_argument('--dark_weight', required=True, type=int, help='weight for hessian list on fair_attr=1 data; original: -5')
    parser.add_argument('-f', '--fair_attr', type=str, default="Male", help="fairness attribute; now support: Male, Young; used for celeba")
    parser.add_argument('-y', '--y_attr', type=str, default="Big_Nose", help="y attribute; now support: Attractive, Big_Nose, Bags_Under_Eyes, Mouth_Slightly_Open, Big_Nose_And_Bags_Under_Eyes, Attractive_And_Mouth_Slightly_Open; used for celeba")
    parser.add_argument('--group_reduction', type=str, default="mean")
    parser.add_argument('--normalizer', type=str, default="mean")
    parser.add_argument('--multivariable', type=int, default=0, help="For TaylorImportance")
    parser.add_argument('--pre_load', type=int, default=0, help="Whether use pre-load datasets")
    parser.add_argument('--max_score_size', type=int, default=0, help="If > 0, use max_score_size as the score size")
    parser.add_argument('--alpha', default=0.0, type=float, help='alpha parameter; if you do not understand, do not change')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--log_file_name', type=str, required=True,
                        help="Pruning log file name; if not None, save the log to the file (in output_dir)")
    parser.add_argument('--isomorphic', type=int, default=0,
                        help="Whether to enable isomorphic pruning; 1: enable; other value (default: 0): disable")
    parser.add_argument('--debug', type=int, default=0,
                        help="Whether to enable debug mode; 1: enable; other value (default: 0): disable")

    args = parser.parse_args()

    if args.eta_min_mode == 0:
        eta_min = 0.0
    elif args.eta_min_mode == 1:
        eta_min = args.lr / 2
    else:
        eta_min = args.eta_min
    global_or_local = "global" if args.global_pruning == 1 else "local"
    summary_str = f'prune_{args.pruner}_{args.importance}_{args.backbone}_{args.dataset}_{str(args.light_weight)}_{str(args.dark_weight)}_{str(args.epochs)}_{str(args.lr)}_{str(eta_min)}_{str(args.step)}_{str(args.sparsity)}_{args.group_reduction}_{global_or_local}' # Removed: {str(image_size)}_
    output_dir = f"results/{summary_str}"
    os.makedirs(output_dir, exist_ok=True)
    f = open(os.path.join(output_dir, args.log_file_name), "w")

    try:
        main(args, output_dir, f, eta_min)
    except Exception as e:
        f.write("[ERROR] " + str(e))

    f.close()
