import torch.utils.data
import torch.nn as nn
from data_loader import *
from utils import IdentitySampler, prepare_save_folder_or_get_path_models, transform_for_train, \
    saveModel_displayResults, save_curves, save_first_batch_imgs
from loss import BatchHardTripLoss
from model import Global_network
import argparse
import os
import time
from utils_train import train, valid
from optimizer import optimizer_and_params_management

from CIL.mixing_erasing import erasings
from CIL.make_loss import make_loss

from TransREID.make_model import make_model as make_model_transreid
from TransREID.loss.make_loss import make_loss as make_loss_transreid
from TransREID.solver.scheduler_factory import create_scheduler as make_scheduler_transreid

from LMBN.make_model import make_model as make_model_LMBN
from LMBN.loss.make_loss import make_loss as make_loss_LMBN
from LMBN.optim.make_optimizer import make_scheduler as make_scheduler_LMBN

parser = argparse.ArgumentParser(description='MLMDA - Training')

parser.add_argument('--model', default='concatenation', help='unimodal, concatenation, transreid, LMBN')
parser.add_argument('--fold', default='0', help='Fold number (0 to 4)')
parser.add_argument('--dataset', default='RegDB', help='dataset name (RegDB / SYSU / TWorld)')
parser.add_argument('--reid', default='BtoB', help='Type of ReID:'
                                                   'BtoB for multimodal '
                                                   'VtoV for unimodal visible'
                                                   'TtoT for unimodal thermal')

parser.add_argument('--GPU', default='0', help='GPU to use')
parser.add_argument('--workers', default='4', type=int, help='Number workers')
parser.add_argument('--lr', default='0.1', type=float, help='Learning rate')
parser.add_argument('--Batch_Size', default='32', type=int, help='32 or 64')

parser.add_argument('--data_path', default='../Datasets', help='Dataset location')
parser.add_argument('--models_path', default='../save_model', help='Checkpoints locations')

parser.add_argument('--save_model', default=True, action="store_false",  help='Save model or not')
parser.add_argument('--save_files', default=True, action="store_false",  help='Save files or not')
parser.add_argument('--save_first_batch_img', action='store_true',  help='Save first learning and validation data batch')

### Data augmentation related
parser.add_argument('--Augmix', action='store_true',  help='Augmix')
parser.add_argument('--CIL', action='store_true',  help='CIL DA')
parser.add_argument('--ML_MDA', action='store_true',  help='ML-MDA')
parser.add_argument('--XREA', default='False', help='Model trained with : '
                                                    'False'
                                                    'S-REA'
                                                    'MS-REA')
parser.add_argument('--XPATCH', default='False', help='Model trained with : '
                                                    'S-PATCH'
                                                    'MS-PATCH'
                                                    'M-PATCH-SS'
                                                    'M-PATCH-SD'
                                                    'M-PATCH-DD')
parser.add_argument('--Masking', action='store_true',  help='Model trained with masking')
parser.add_argument('--masking_ratio', default=0.5, type=float,  help='Masking appearance ratio in a pair')

args = parser.parse_args()

### Print training infos
message = '================ Training date (%s) ================\n' % time.strftime("%c")
message += '----------------- TRAINING Options ---------------\n'
for k, v in sorted(vars(args).items()):
    comment = ''
    default = parser.get_default(k)
    if v != default:
        comment = '\t \t [default: %s]' % str(default)
    message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
message += '----------------- End -------------------'
print(message)


### GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDEVICE : {device} \n")

### Check
model_list = ['unimodal', 'concatenation', 'transreid', 'LMBN']
assert args.model in model_list, f'--model should be in {model_list} but --fusion={args.model}'

### Init :
img_w, img_h = 144, 288
test_batch_size = 32
best_map, best_map1, best_map2 = 0, 0, 0
best_epoch, best_epoch1, best_epoch2 = 0, 0, 0
best_minp = 0
best_map_att, best_minp_att = 0, 0

epoch_number = 100 if args.model != "LMBN" else 140 # Respecting LMBN paper epoch number

checkpoint_path = f"{args.models_path}/{args.dataset}/"
os.makedirs(checkpoint_path, exist_ok=True)

data_path = f"{args.data_path}/{args.dataset}/"
assert os.path.isdir(data_path), f"Dataset location do not exist : {data_path} "
Curves = {"Overall_loss":[]}

assert args.Batch_Size in [32, 64], f"args.Batch_Size value {args.Batch_Size} not in :  ['32', '64'] - Need adaptations in code"
if args.Batch_Size == 32:
    batch_num_identities = 8  # 8 different identities in a batch
    num_of_same_id_in_batch = 4  # Number of image per identity in a batch
elif args.Batch_Size == 64:
    batch_num_identities = 16  # 8 different identities in a batch
    num_of_same_id_in_batch = 4  # Number of image per identity in a batch

####### DATA TRANSFORMATION RELATED #######

re_erasing, random_erasing, random_erasing_IR = erasings(args.model, args.CIL, args.XPATCH, args.XREA, args.ML_MDA)

transform_train = transform_for_train(img_h, img_w, "train", args.model, args.dataset, args.CIL, args.ML_MDA,
                              args.Augmix, args.XPATCH, args.XREA, args.Masking, re_erasing, random_erasing, random_erasing_IR, args.masking_ratio)

transform_test = transform_for_train(img_h, img_w, "valid", args.model, args.dataset, args.CIL, args.ML_MDA,
                              args.Augmix, args.XPATCH, args.XREA, args.Masking, re_erasing, random_erasing, random_erasing_IR, args.masking_ratio)

Timer1 = time.time()

######################################### Suffix

suffix = f'{args.dataset}_{args.reid}_model_{args.model}_fold_{args.fold}'

if args.CIL:
    suffix += "_CIL"
if args.Augmix or args.CIL or args.ML_MDA:  # All of those automatically use augmix per default
    suffix += "_augmix"
if args.XREA != "False":
    suffix += f"_{args.XREA}"
if args.ML_MDA:
    suffix += f"_MS-REA"
if args.XPATCH != "False":
    suffix += f"_{args.XPATCH}"
if args.Masking or args.ML_MDA:
    suffix += f"_Masking"

print(f"===> Loading Training Data")

#### Results files and folder preparation
model_folder, log_file_name, curves_folder = prepare_save_folder_or_get_path_models(checkpoint_path, suffix, fold=args.fold,
                                                            save_files=args.save_files, message=message, action="init_folder")

######################################### TRAIN SET
trainset = TrainingData(data_path, args.dataset, transform_train, args.fold)

# Get ids pos
color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

######################################### VALIDATION SET
query_img, query_label, query_cam, gall_img, gall_label, gall_cam = process_data(data_path, "valid", args.dataset, fold=args.fold)

# Gallery and query set
gallset = Prepare_set(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
queryset = Prepare_set(query_img, query_label, transform=transform_test, img_size=(img_w, img_h))
# Validation data loader
gall_loader = torch.utils.data.DataLoader(gallset, batch_size=test_batch_size, shuffle=False, num_workers=args.workers)
query_loader = torch.utils.data.DataLoader(queryset, batch_size=test_batch_size, shuffle=False, num_workers=args.workers)

# Init variables
n_class = len(np.unique(trainset.train_color_label))
n_query = len(query_label)
n_gall = len(gall_label)

######################################### Dataset statistics #########################################

print(f'------- Dataset {args.dataset} statistics -------')
print(f'     set     |  Nb ids |  Nb img    ')
print(f'  ------------------------------')
print(f'   visible  | {n_class:5d} | {len(trainset.train_color_label):8d}')
print(f'   thermal  | {n_class:5d} | {len(trainset.train_thermal_label):8d}')
print(f'  ------------------------------')
print(f'   query    | {len(np.unique(query_label)):5d} | {n_query:8d}')
print(f'   gallery  | {len(np.unique(gall_label)):5d} | {n_gall:8d}')
print(f'  ------------------------------')
print(f'   Data Loading Time:\t {time.time() - Timer1:.3f}')
print('------- End -------\n')
print("")
print('==> Building model..')

######################################### MODEL #########################################

if args.model == "transreid":
    net = make_model_transreid(n_class)
elif args.model == "LMBN":
    net = make_model_LMBN(device, n_class)
else:  # Unimodal / Concatenation
    net = Global_network(n_class, fusion_layer=8, model=args.model)

net.to(device)

######################################### TRAINING #########################################

print('==> Start Training...')

optimizer = optimizer_and_params_management(args.model, net, args.lr )

# Scheduler
if args.model == "transreid":
    scheduler = make_scheduler_transreid(optimizer)
elif args.model == "LMBN":
    scheduler = make_scheduler_LMBN(optimizer, -1)
else :
    scheduler = None

# Loss functions
criterion_tri = BatchHardTripLoss(batch_size=args.Batch_Size, margin=0.3).to(device)

if (args.Augmix or args.XPATCH!="False" or args.XREA!="False" or args.ML_MDA):
    criterion_id = make_loss(n_class)
elif args.model == "transreid":
    criterion_id = make_loss_transreid()
elif args.model == "LMBN":
    criterion_id = make_loss_LMBN(n_class)
else:
    criterion_id = nn.CrossEntropyLoss().to(device)

training_time = time.time()

for epoch in range(epoch_number):

    print('==> Preparing Data Loader...')
    # identity sampler - Give iteratively index from a randomized list of color index and thermal index
    sampler = IdentitySampler(trainset, color_pos, thermal_pos, num_of_same_id_in_batch, batch_num_identities,
                              args.dataset)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.Batch_Size, sampler=sampler, num_workers=args.workers,
                                              drop_last=True)

    if args.save_first_batch_img:  # gall_loader / query_loader
        save_first_batch_imgs(trainloader, query_loader ,args.dataset)
        args.save_first_batch_img=False

    train(epoch, args.lr, net, args.reid, trainloader, device, optimizer, criterion_id, criterion_tri,
          log_file_name, Curves, scheduler, args.model, args.save_files, args.Augmix,
          args.XPATCH, args.XREA, args.ML_MDA)

    ######################################### VALIDATION  #########################################

    # Validation done every two epochs
    if epoch > 0 and epoch % 2 == 0:
        print(f'Validation Epoch: {epoch}')

        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = valid(args.dataset, query_cam, gall_cam, query_label,
                                                           gall_label, gall_loader, query_loader, net, args.model, args.reid,
                                                            n_query, n_gall, Curves=Curves)

        best_map, best_epoch, best_minp, best_map_att,best_minp_att = \
                                        saveModel_displayResults(args.model, args.save_files, args.fold,
                                                                 cmc, mAP, mINP, cmc_att, mAP_att, mINP_att,
                                                                 net, model_folder, suffix, epoch, best_epoch,
                                                                 log_file_name, args.save_model, best_map, best_minp,
                                                                 best_map_att,best_minp_att)

        if args.save_files:
            save_curves(Curves, curves_folder, epoch, args.fold)

print(f' Training time for {args.model} model : {time.time() - training_time}')
