import torch.utils.data
import time
from data_loader import *
from model import Global_network
import argparse
from utils import prepare_save_folder_or_get_path_models, create_folder, extract_query_or_gall_feat, transform_for_test
from TransREID.make_model import make_model as make_model_transreid
from LMBN.make_model import make_model as make_model_LMBN
from evaluation import *

parser = argparse.ArgumentParser(description='MLMDA - Inference')
parser.add_argument('--model', default='concatenation', help='unimodal, concatenation, transreid, LMBN')
parser.add_argument('--dataset', default='RegDB', help='dataset name (RegDB / SYSU / TWorld)')
parser.add_argument('--reid', default='BtoB', help='Type of ReID:'
                                                   'BtoB for multimodal '
                                                   'VtoV for unimodal visible'
                                                   'TtoT for unimodal thermal')

parser.add_argument('--GPU', default='0', help='GPU to use')
parser.add_argument('--workers', default='4', type=int, help='Number workers')
parser.add_argument('--test_batch_size', default='64', type=int, help='32 or 64')

parser.add_argument('--data_path', default='../Datasets', help='Dataset location')
parser.add_argument('--models_path', default='../save_model', help='Checkpoints locations')
parser.add_argument('--results', default='./Results', help='Folder in which saving the results')

parser.add_argument('--scenario_eval', default='normal', help='Evaluation type: normal, C, C*')


parser.add_argument('--CIL', action='store_true',  help='Model trained with CIL')
parser.add_argument('--ML_MDA', action='store_true',  help='Model trained with ML_MDA')

# To test while trained with various combinations of individual DA
parser.add_argument('--Augmix', action='store_true',  help='Model trained with Augmix')

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

args = parser.parse_args()

### Init variables :
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

net = []
time_list = []

img_w, img_h = 144, 288

trials = 30
folds = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = f'{args.data_path}/{args.dataset}/'
checkpoint_path = f"{args.models_path}/{args.dataset}/"
results_folder = f"{args.results}/{args.scenario_eval}/{args.dataset}/"
os.makedirs(results_folder, exist_ok=True) # If result folder exists no pb, if not, create folders recursively

mAP_mINP_per_trial = {"mAP" : [0 for i in range(trials)], "mINP" : [0 for i in range(trials)]}
mAP_mINP_per_model = {"mAP" : [0 for i in range(folds)], "mINP" : [0 for i in range(folds)]}
mAP_mINP_per_model_pool = {"mAP" : [0 for i in range(folds)], "mINP" : [0 for i in range(folds)]}

assert os.path.isdir(checkpoint_path), f"Folder does not exist : {checkpoint_path}"

print(f"Device : {device}")

message = ''
message += '----------------- TESTING Options ---------------\n'
for k, v in sorted(vars(args).items()):
    comment = ''
    default = parser.get_default(k)
    if v != default:
        comment = '\t \t [default: %s]' % str(default)
    message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
message += '----------------- End -------------------'
print(message)

####### DATA TRANSFORMATION RELATED #######
transform_test = transform_for_test(img_h, img_w, args.scenario_eval, args.model, args.CIL, args.XPATCH)

# Test nb classes
nclass = {"RegDB" : 164, "SYSU" : 316, "TWorld" : 260}
print(f"nb class : {nclass[args.dataset]}")

trials = 1 # RegDB / TWorld on Clean data
if args.dataset == "SYSU": # SYSU anytime
    trials = 30
elif args.scenario_eval in ["C", "C*"] : # Evaluation over 10 trials for RegDB or TWorld if Corrupted data
    trials = 10

# 5 fold validation - The following loop get an average result over folds
for fold in range(folds):

    suffix = f'{args.dataset}_{args.reid}_model_{args.model}_fold_{fold}'

    if args.CIL:
        suffix += "_CIL"
    if args.Augmix or args.CIL or args.ML_MDA : # All of those automatically use augmix per default
        suffix += "_augmix"
    if args.XREA != "False":
        suffix += f"_{args.XREA}"
    if args.ML_MDA:
        suffix += f"_MS-REA"
    if args.XPATCH != "False":
        suffix += f"_{args.XPATCH}"
    if args.Masking or args.ML_MDA:
        suffix += f"_Masking"

    print('==> Resuming from checkpoint..')
    model_path = prepare_save_folder_or_get_path_models(checkpoint_path, suffix, action="models_path") + suffix + '_best.t'

    if args.model == "transreid":
        net.append(make_model_transreid(nclass[args.dataset]))
    elif args.model == "LMBN":
        net.append(make_model_LMBN(device, nclass[args.dataset]))
    else :
        net.append(Global_network(nclass[args.dataset], fusion_layer=8, model=args.model))

    checkpoint = torch.load(model_path)

    net[fold].load_state_dict(checkpoint['net'], strict=True)
    net[fold].to(device)

    print(f"Checkpoint loaded - Fold {fold}")

    model_specific_results_folder = prepare_save_folder_or_get_path_models(checkpoint_path, suffix, action="models_folder")

    for trial in range(trials):

        start = time.time()

        #Prepare query and gallery
        query_img, query_label, query_cam, gall_img, gall_label, gall_cam = process_data(data_path, "test", args.dataset, trial)

        # Gallery and query set
        gallset = Prepare_set(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
        queryset = Prepare_set(query_img, query_label, transform=transform_test, img_size=(img_w, img_h))

        # Validation data loader
        gall_loader = torch.utils.data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
        query_loader = torch.utils.data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
        nquery, ngall = len(query_label), len(gall_label)

        print('Data Loading Time:\t {:.3f}'.format(time.time() - start))

        # Extract features
        query_feat_pool, query_feat_fc, time_inference = \
            extract_query_or_gall_feat(query_loader, nquery, net = net[fold], QorG="Query", modality = args.reid, model=args.model)

        gall_feat_pool,  gall_feat_fc, _ = \
            extract_query_or_gall_feat(gall_loader, ngall, net = net[fold], QorG="Gallery", modality = args.reid, model=args.model)

        time_list.append(time_inference)
        if fold == 4 :
            mean_time_image_extraction = sum(time_list)/5

        # pool and BN feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        distmat = np.matmul( query_feat_fc, np.transpose(gall_feat_fc))

        print("==> Evaluation")
        if args.dataset == "SYSU" :
            cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)
        else :
            cmc, mAP, mINP = evaluation(-distmat, query_label, gall_label)
            cmc_pool, mAP_pool, mINP_pool = evaluation(-distmat_pool, query_label, gall_label)

        if trial == 0 and fold == 0 :
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        mAP_mINP_per_trial["mAP"][trial] += mAP
        mAP_mINP_per_trial["mINP"][trial] += mINP
        mAP_mINP_per_model["mAP"][fold] += mAP
        mAP_mINP_per_model["mINP"][fold] += mINP
        mAP_mINP_per_model_pool["mAP"][fold] += mAP_pool
        mAP_mINP_per_model_pool["mINP"][fold] += mINP_pool

        print(f'Test fold: {fold} - Test trial : {trial}')
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

model_specific_performances_folder = model_specific_results_folder + "results/"
create_folder(model_specific_performances_folder)

#Standard Deviation :
if args.dataset == "SYSU" or trials != 1:
    standard_deviation_mAP_model = np.std([mAP_mINP_per_model["mAP"][k] / trials for k in range(folds)])
    standard_deviation_mINP_model = np.std([mAP_mINP_per_model["mINP"][k] / trials for k in range(folds)])
    standard_deviation_mAP_model_pool = np.std([mAP_mINP_per_model_pool["mAP"][k] / trials for k in range(folds)])
    standard_deviation_mINP_model_pool = np.std([mAP_mINP_per_model_pool["mINP"][k] / trials for k in range(folds)])
else :
    standard_deviation_mAP_model = np.std(mAP_mINP_per_model["mAP"])
    standard_deviation_mINP_model = np.std(mAP_mINP_per_model["mINP"])
    standard_deviation_mAP_model_pool = np.std(mAP_mINP_per_model_pool["mAP"])
    standard_deviation_mINP_model_pool = np.std(mAP_mINP_per_model_pool["mINP"])

# Means
cmc = all_cmc / (folds * trials)
mAP = all_mAP / (folds * trials)
mINP = all_mINP / (folds * trials)

cmc_pool = all_cmc_pool / (folds * trials)
mAP_pool = all_mAP_pool / (folds * trials)
mINP_pool = all_mINP_pool / (folds * trials)

print('All Average:')
print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%}| mAP: {:.2%}| mINP: {:.2%} | stdmAP: {:.2%} | stdmINP {:.2%}'.format(
        cmc[0], cmc[4], mAP, mINP, standard_deviation_mAP_model, standard_deviation_mINP_model))
print('Pool:     Rank-1: {:.2%} | Rank-5: {:.2%}| mAP: {:.2%}| mINP: {:.2%} | stdmAP: {:.2%} | stdmINP {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], mAP_pool, mINP_pool, standard_deviation_mAP_model_pool, standard_deviation_mINP_model_pool))

if os.path.isfile( results_folder + f"/results.txt") :
    f_global = open( results_folder + f'/results.txt','a')
else :
    f_global = open(results_folder + f'/results.txt','w+')
    f_global.write(' , Rank-1, Rank-5, mAP, mINP, stdmAP, stdmINP\n')

if os.path.isfile(model_specific_performances_folder + f'/results.txt') :
    f_specific = open( model_specific_performances_folder + f'/results.txt','a')
else :
    f_specific = open(model_specific_performances_folder + f'/results.txt','w+')
    f_specific.write(' , Rank-1, Rank-5, mAP, mINP, stdmAP, stdmINP\n')

g = open( results_folder + f'/results_excel.txt','a') if os.path.isfile( results_folder + f"/results_excel.txt") \
    else open(results_folder + f'/results_excel.txt','w+')

data_info = f"{args.scenario_eval}_{suffix}: "
message = data_info

message +=  f'\n{mAP:.2%}±{standard_deviation_mAP_model:.2%}, ' \
            f'{mINP:.2%}±{standard_deviation_mINP_model:.2%}, ' \
            f'R1-{cmc[0]:.2%}, R5-{cmc[4]:.2%}, TIME-{mean_time_image_extraction}\n'
message +=  f'{mAP_pool:.2%}±{standard_deviation_mAP_model_pool:.2%}, ' \
            f'{mINP_pool:.2%}±{standard_deviation_mINP_model_pool:.2%}, ' \
            f'R1-{cmc_pool[0]:.2%}, R5-{cmc_pool[4]:.2%}, TIME-{mean_time_image_extraction}\n\n'

f_global.write(message)
f_specific.write(message)

g_message = data_info + \
            f"{mAP:.2%}±{standard_deviation_mAP_model:.2%} {mINP:.2%}±{standard_deviation_mINP_model:.2%} {cmc[0]:.2%} " \
            f"{mAP_pool:.2%}±{standard_deviation_mAP_model_pool:.2%} {mINP_pool:.2%}±{standard_deviation_mINP_model_pool:.2%} {cmc_pool[0]:.2%}\n"
print(g_message)
g.write( data_info + g_message)
print(f"Data results saved in : {results_folder}")

f_global.close()
f_specific.close()
g.close()
