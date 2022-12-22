import os
import sys
import numpy as np
from torch.utils.data.sampler import Sampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import math
from torchvision import transforms
from corruption.Corruption_transform import corruption_transform
import torchvision.utils
from torch.autograd import Variable
from LMBN.utils_test import utils_test_LMBN
import torch.utils.data
import time
from LMBN.transforms_LMBN import RandomErasing_LMBN, Random2DTranslation
from CIL.augmix_transform import augmix_transform
from corruption.Corruption_transform import Masking, Compose

class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of each modalities
            color_pos, thermal_pos: positions of each identity
            batch_num_identities: batch size
    """
    def __init__(self, trainset, color_pos, thermal_pos, num_of_same_id_in_batch, batch_num_identities, dataset):
        uni_label = np.unique(trainset.train_color_label)
        self.n_classes = len(uni_label)
        N = np.maximum(len(trainset.train_color_label), len(trainset.train_thermal_label))

        for j in range(int(N / (batch_num_identities * num_of_same_id_in_batch)) + 1):
            # We choose randomly 8 identities
            batch_idx = np.random.choice(uni_label, batch_num_identities, replace=False)
            # print(f"batch idx {batch_idx}")

            for i in range(batch_num_identities):
                # We choose randomly 4 images (num_of_same_id_in_batch) for the i=8 identitities
                if dataset == "TWorld" or dataset == "RegDB":
                    sample_color = np.random.choice(color_pos[batch_idx[i]], num_of_same_id_in_batch)
                    sample_thermal = sample_color
                elif dataset == "SYSU" :
                    # Here since cameras are not colocated, we randomly form the pair of images
                    # We avoid that two blanked images get appaired thanks to degradation supervision list
                    sample_color = np.random.choice(color_pos[batch_idx[i]], num_of_same_id_in_batch)
                    sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_of_same_id_in_batch)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 30:
        lr = lr
    elif epoch >= 30 and epoch < 70:
        lr = lr * 0.1
    elif epoch >= 70:
        lr = lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr
    return lr

def empty(k):
    return([[] for i in range(k)])

# Allow to get the complement of a list slice
class SuperList(list):
    def __getitem__(self, val):
        if type(val) is slice and val.step == 'c':
            copy = self[:]
            copy[val.start:val.stop] = []
            return copy

        return super(SuperList, self).__getitem__(val)

def get_ids(Data, path):
    data = [int(Data[path].loc[i]) for i in range(Data.shape[0]) if not math.isnan(Data[path].loc[i])]
    return data

def get_path(Data, path):
    data = [Data[path][i] for i in range(Data.shape[0]) if not Data[path][i] is None]
    return data

def get_query_gallery(Data, path):
    data = [x for x in Data[path] if x != []]
    return data

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def prepare_save_folder_or_get_path_models(checkpoint_path, suffix, fold=0, save_files=True, message="", action="init_folder"):

    filtered_suffix = suffix.split("_fold_")[0] + suffix.split("_fold_")[1][1:]
    if action=="init_folder" :
        model_specific_save_folder = checkpoint_path + filtered_suffix + "/"

        model_folder = model_specific_save_folder + "/models"  # Folder for the models storage
        log_folder = model_specific_save_folder + "/log_folder"  # Folder for output logs storage
        log_file_name = log_folder + f"/log_file_{fold}.txt"
        curves_folder = model_specific_save_folder + "/curves_folder"

        if save_files:
            for folder_to_create in [model_specific_save_folder, log_folder, model_folder, curves_folder,
                                     model_specific_save_folder + "/train_params"]:
                create_folder(folder_to_create)

            with open(model_specific_save_folder + f"/train_params/train_params_fold_{fold}.txt",
                      'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

            with open(log_file_name, 'w+') as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training logs (%s) ================\n' % now)

        return model_folder, log_file_name, curves_folder
    elif action == "models_folder":
        return checkpoint_path + filtered_suffix + "/"
    elif action == "models_path":
        return checkpoint_path + filtered_suffix + "/models/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
toTensor = transforms.ToTensor()

def saveModel_displayResults(model, save_files, fold, cmc, mAP, mINP, cmc_att, mAP_att, mINP_att, net, model_folder,
                             suffix, epoch, best_epoch, log_file_name, save, best_map, best_minp, best_map_att, best_minp_att):

    if mAP > best_map:
        best_map, best_minp = mAP, mINP
        best_map_att, best_minp_att = mAP_att, mINP_att

        best_epoch = epoch
        state = {
            'net': net.state_dict(),
            'cmc': cmc,
            'mAP': mAP,
            'mINP': mINP,
            'epoch': epoch,
        }

        if save :
            torch.save(state, model_folder + "/" + suffix + '_best.t')
        else :
            print('fc : Rank-1: {:.2%} | Rank-5: {:.2%} | mAP: {:.2%}| mINP: {:.2%}'.format(cmc[0], cmc[4], mAP, mINP))
            print('att : Rank-1: {:.2%} | Rank-5: {:.2%} | mAP: {:.2%}| mINP: {:.2%}'.format(cmc_att[0], cmc_att[4], mAP_att,  mINP_att))
            print("ATTENTION WE DO NOT SAVE MODEL ")

    # Display infos
    Validation_message = 'fc : Rank-1: {:.2%} | Rank-5: {:.2%} | mAP: {:.2%}| mINP: {:.2%} \n'.format(cmc[0], cmc[4], mAP, mINP)
    Validation_message += 'att : Rank-1: {:.2%} | Rank-5: {:.2%} | mAP: {:.2%}| mINP: {:.2%}'.format(cmc_att[0], cmc_att[4], mAP_att, mINP_att)

    print(Validation_message)

    if save_files :
        with open(log_file_name, 'a') as log_file:
            log_file.write(Validation_message + "\n")

    print('Best Epoch [{}]'.format(best_epoch))

    print(f' Training {model} - fold number ({fold})')
    return best_map, best_epoch, best_minp, best_map_att, best_minp_att

def update_or_add(dico, key, value):
    if key in dico.keys():
        dico[key].append(value)
    else:
        dico.update({key:[value]})

def save_curves(dico, save_folder, epoch, fold):
    print(f"Saving training curves...")
    Training_keys, Valid_keys = [], []
    for key in dico :
        if "Validation" in key :
            Valid_keys.append(key)
        else :
            Training_keys.append(key)
    acc = 0
    for key in Training_keys :
        if "Accuracy" in key :
            acc += 1

    # Training curves
    nb_subplots_train = len(Training_keys)

    rang = 2 if nb_subplots_train >= 7 else 1
    if rang == 1 :
        Training_fig, axes = plt.subplots(int(nb_subplots_train), figsize=(12, 12), sharex=True)
        for idx, (key, ax) in enumerate(zip(Training_keys, axes)):
            ax.plot(dico[key])
            ax.set_ylabel(key)
            ax.axis(xmin=0)
            if key == "Accuracy" :
                ax.axis(ymax=100)
            if idx == nb_subplots_train - 1 :
                ax.set_xlabel("epochs")
        plt.savefig(f"{save_folder}/Training_curves_fold_{fold}.png")
    else :
        if acc > 1 : # Combined acc
            nb_subplots_train = 4

        Training_fig, axes = plt.subplots(nb_subplots_train, figsize=(12, 12), sharex=True)

        axes[0].plot(dico["Overall_loss"])
        axes[0].set_ylabel("Overall_loss")
        axes[0].axis(xmin=0)

        for idx, key in enumerate(Training_keys):
            if "Accuracy" in key :
                axes[1].plot(dico[key], label=key)
                axes[1].set_ylabel("Accuracies")
                axes[1].axis(xmin=0)
                axes[1].axis(ymin=0)
                axes[1].axis(ymax=100)
                axes[1].legend()

        for idx, key in enumerate(Training_keys):
            if "Loss" in key :
                axes[2].plot(dico[key], label=key)
                axes[2].set_ylabel("Loss")
                axes[2].axis(xmin=0)
                axes[2].set_xlabel("epochs")
                axes[2].legend()

        for idx, key in enumerate(Training_keys):
            if "Winner" in key :
                axes[3].plot(dico[key], label=key)
                axes[3].set_ylabel("Winner")
                axes[3].axis(xmin=0)
                axes[3].set_xlabel("epochs")
                axes[3].legend()

        plt.savefig(f"{save_folder}/Training_curves_fold_{fold}.png")

    ## Validation curves
    Validation_fig, axes = plt.subplots(4, figsize=(12, 8), sharex=True)
    for idx, (key, ax) in enumerate(zip(Valid_keys, axes)):
        ax.plot(dico[key])
        ax.set_ylabel(key)
        ax.axis(xmin=0, ymax=1.0)
        if idx == 3 :
            ax.set_xlabel("epochs")
    plt.savefig(f"{save_folder}/Validation_curves_fold_{fold}.png")
    print(f"Curves saved :\n{save_folder}")

# Extract features
def extract_query_or_gall_feat(query_gall_loader, n_query_gall, net, modality="VtoV", QorG="Query", model="concatenation"):
    net.eval()
    print(f'Extracting {QorG} features...')
    start = time.time()
    number_of_images=0
    all_time = 0
    ptr = 0

    pool_size={"transreid":3840, "LMBN":3584, "unimodal":512, "concatenation":1024}

    query_gall_feat_pool = np.zeros((n_query_gall, pool_size[model]))
    query_gall_feat_fc = np.zeros((n_query_gall, pool_size[model]))

    with torch.no_grad():
        for batch_idx, (input1, input2, *label) in enumerate(query_gall_loader):

            number_of_images += len(input1)
            batch_num = input1.size(0)

            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())

            if model == "transreid":
                feat_pool, feat_fc = net(input1)
            elif model == "LMBN":
                feat_pool = utils_test_LMBN(input1,device,net)
                feat_fc = feat_pool # Not to get an error
            else :
                # feat_pool, feat_fc, feat, z = net([input1, input2], modality=modality)

                feat_pool, feat_fc = net([input1, input2], model, modality)

            all_time += time.time() - start

            query_gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            query_gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()

            ptr = ptr + batch_num

        print(f"    Extracting time per image : {all_time/number_of_images}")
        print("    Extracting Time:\t {:.3f}".format(time.time() - start))
    return query_gall_feat_pool, query_gall_feat_fc, all_time / number_of_images

def transform_for_test(img_height, img_width, scenario_eval, model, CIL, XPATCH):
    transform_list = []
    transform_list.append(transforms.ToPILImage())

    if scenario_eval in ["C", "C*"]: # If star we corrupt both RGB and IR else we corrupt
        transform_list.append(corruption_transform(0, type='all'))

    if model == "transreid":
        transform_list.append(transforms.Resize([256, 128], interpolation=3))
    else :
        transform_list.append(transforms.Resize((img_height, img_width)))

    transform_list.append(transforms.ToTensor())
    transform_list.append(normalize)

    return Compose(transform_list, CIL, XPATCH, scenario_eval)

def transform_for_train(img_height, img_width, mode, model, dataset, CIL, ML_MDA, augmix, XPATCH, XREA, MASKING,
                        re_erasing, random_erasing, random_erasing_IR, masking_ratio) :

    transform_list = []

    transform_list.append(transforms.ToPILImage())

    # Use of data augmentation if training or if validation and RegDB to make the RegDB learning tougher
    use_DA = True if dataset=="RegDB" or mode == "train" else False

    if use_DA and (MASKING or ML_MDA):
        transform_list.append(Masking(ratio=masking_ratio))
    if model == "transreid":
        transform_list.append(transforms.Resize([256, 128], interpolation=3))

    if mode == "train":
        if model != "LMBN":
            transform_list.append(transforms.Pad(10))
        else :
            transform_list.append(Random2DTranslation(img_height, img_width))

        transform_list.append(transforms.RandomHorizontalFlip())

        if model != "transreid":
            transform_list.append(transforms.RandomCrop((img_height, img_width)))
        else :
            transform_list.append(transforms.RandomCrop((256, 128)))

    if use_DA: # RegDB needs augmentation to be apply on both train and valid or maxed out too easy
        if CIL or ML_MDA or augmix :
            transform_list.append(augmix_transform())

    transform_list.append(transforms.ToTensor())

    if use_DA: # RegDB needs augmentation to be apply on both train and valid or maxed out too easy
        if CIL or XREA in ["S-REA", "MS-REA"] or ML_MDA :
            transform_list.append(random_erasing)
        if XREA=="MS-REA" or ML_MDA:
            transform_list.append(random_erasing_IR)

        if CIL or XPATCH in ["S-PATCH", "MS-PATCH", "M-PATCH-SS", "M-PATCH-SD", "M-PATCH-DD"] :
            transform_list.append(re_erasing)

        if model == "transreid" and not CIL: # Using random erasing from original TransReID paper
            transform_list.append(random_erasing)
        if model == "LMBN" and not CIL: # Using Random erasing from original Light MBN paper
            transform_list.append(RandomErasing_LMBN())

    transform_list.append(normalize)

    # print(*transform_list)

    return Compose(transform_list, CIL, XPATCH)




def save_first_batch_imgs(train_loader, query_loader, dataset):
    input1, input2, classes, *_ = next(iter(train_loader))
    # Make a grid from batch
    out_RGB = torchvision.utils.make_grid(input1)
    out_IR = torchvision.utils.make_grid(input2)

    # imshow(out_RGB, title=[classes[x] for x in classes], location="last_batch_RGB.png")
    imshow(out_RGB, title=[i for i, x in enumerate(classes)], location=f"last_batch_train_RGB_{dataset}.png")
    imshow(out_IR, title=[i for i, x in enumerate(classes)], location=f"last_batch_train_IR_{dataset}.png")

    input1, input2, classes = next(iter(query_loader))
    # Make a grid from batch
    out_RGB = torchvision.utils.make_grid(input1)
    out_IR = torchvision.utils.make_grid(input2)
    imshow(out_RGB, title=[i for i, x in enumerate(classes)], location=f"last_batch_valid_RGB_{dataset}.png")
    imshow(out_IR, title=[i for i, x in enumerate(classes)], location=f"last_batch_valid__IR_{dataset}.png")

def imshow(inp, location="none.png"):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    fig = plt.figure("temp_fig")
    plt.imshow(inp)
    fig.savefig(f'last_batch_img/{location}')
    plt.pause(0.001)  # pause a bit so that plots are updated
    print(f"Batch samples saved in 'last_batch_img/{location}'")