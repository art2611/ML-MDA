import os
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as data
from utils import get_path, get_ids, get_query_gallery, progress

def TrainingData(data_path, dataset, transform, fold):
    if dataset == "SYSU":
        return(SYSUData(data_path, transform=transform, fold = fold))
    elif dataset == "RegDB":
        return(RegDBData(data_path, transform=transform, fold = fold))
    elif dataset == "TWorld":
        return(TWorldDATA(data_path, transform=transform, fold = fold))

class RegDBData(data.Dataset):
    def __init__(self, data_path, transform=None, colorIndex=None, thermalIndex=None,
                 fold = 0):

        # Load training images (path) and labels
        datas = pd.read_json('dataset_use/RegDB/train_valid_test_ids.json')

        file_label = [int(i/10) for i in range((204-40)*10)] # This way we have ordered labels from 0 class to x class

        color_img_file = get_path(datas, f"train_path_rgb_{fold}")
        thermal_img_file = get_path(datas, f"train_path_ir_{fold}")

        if data_path != "../Datasets/RegDB/" :
            color_img_file = [im_path.replace("../Datasets/RegDB/", data_path, 1) for im_path in color_img_file ]
            thermal_img_file = [im_path.replace("../Datasets/RegDB/", data_path, 1) for im_path in thermal_img_file ]

        processed_img = [[],[]]

        nb_imgs = len(color_img_file)

        # Preprocess images
        for idx, (color_im_file, thermal_im_file) in enumerate(zip(color_img_file, thermal_img_file)):
            to_process = [color_im_file, thermal_im_file]
            if idx % 50 == 0 or idx + 1 == nb_imgs: # Display loading data bar
                progress(idx+1, nb_imgs)
                if idx + 1 == nb_imgs :
                    print("")
            for i in range(len(to_process)) : # i = 0 pour RGB, 1 pour IR, 2 pour Additionnal
                to_process[i] = Image.open(to_process[i])
                to_process[i] = to_process[i].resize((144, 288), Image.ANTIALIAS)
                processed_img[i].append(np.array(to_process[i]))

        print("\n")

        # RGB
        self.train_color_image = np.array(processed_img[0])
        self.train_color_label = file_label

        # IR
        self.train_thermal_image = np.array(processed_img[1])
        self.train_thermal_label = file_label

        self.transform = transform

        # Prepare index
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        #Dataset[i] return images from both modal and the corresponding label
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img1, img2 = self.transform(img1, img2)
        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

# Get TWorld data for training
class TWorldDATA(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None, fold = 0, data_aug=False, pose_or_contour=None):

        # Load training labels
        datas = pd.read_json('dataset_use/TWorld/train_valid_test_ids.json')

        ids = datas[f"train_id_{fold}"]

        # Get list of list, each sub list containing the images location for one identity
        ids_file_RGB, ids_file_IR, pid2label_train= [], [], []

        relabel = 0
        for id in sorted(ids):
            img_dir = os.path.join(data_dir, "TV_FULL", str(id))
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                ids_file_RGB.extend(new_files)
                # Get the label from person ids
                for w in range(len(new_files)):
                    pid2label_train.append(relabel)

            relabel += 1

            img_dir = os.path.join(data_dir, "IR_8", str(id))
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                ids_file_IR.extend(new_files)

        labels = np.array(pid2label_train)

        nb_imgs = len(ids_file_RGB)

        train_color_image = self.import_images(ids_file_RGB, nb_imgs, "Visible")
        train_thermal_image = self.import_images(ids_file_IR, nb_imgs, "Thermal")

        self.train_color_label = labels
        self.train_thermal_label = labels

        # Load training images
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def import_images(self, ids, nb_imgs, label):
        print(f"===> Loading {label} Images:")
        train_image = []
        for idx, img_path in enumerate(ids):
            if idx % 100 == 0 or idx + 1 == nb_imgs:
                progress(idx+1, nb_imgs)
                if idx + 1 == nb_imgs :
                    print("")
            img = Image.open(img_path)
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_image.append(pix_array)
        return train_image

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img1, img2 = self.transform(img1, img2)
        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

# Get SYSU data for training
class SYSUData(data.Dataset):
    def __init__(self, data_path, transform=None, colorIndex=None, thermalIndex=None, fold = 0):

        datas = pd.read_json('dataset_use/SYSU/train_valid_test_ids.json')

        # Need loc[] for getting ids in the same order it was saved.
        ids = get_ids(datas, f"train_id_{fold}")
        ids = ["%04d" % int(id) for id in ids]

        ids_file_RGB, ids_file_IR= [], []
        # ### Get list of list containing images per identity

        nb_ids = len(ids)
        for idx, id in enumerate(sorted(ids)):
            if idx % 50 == 0:
                progress(idx, nb_ids)
            files_rgb, files_ir = image_list_SYSU(id, data_path)
            ids_file_RGB.extend(files_rgb)
            ids_file_IR.extend(files_ir)

        pid_container = set()
        for img_path in ids_file_IR:  # Usually, pid stored in /camx/XXXX/image_name.jpg, so we extract XXXX
            pid = int(img_path.split("cam")[1][2:6])
            pid_container.add(pid)

        # relabel
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # Should work well with both augmented or not data
        train_color_image, train_color_label  = read_imgs(ids_file_RGB, pid2label)
        train_thermal_image, train_thermal_label = read_imgs(ids_file_IR, pid2label)

        # Load training labels
        self.train_color_label = train_color_label
        self.train_thermal_label = train_thermal_label

        # Load training images
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img1, img2 = self.transform(img1, img2)
        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

# generate the idx of each person identity for instance, identity 10 have the index 100 to 109
def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)

    return color_pos, thermal_pos

# Call the corresponding dataset function to process data for the validation or the test phase
def process_data(img_dir, mode, dataset, fold=0):
    img_query, label_query, query_cam, img_gallery, label_gallery, gall_cam= (None for _ in range(6))
    if "SYSU" in dataset:
        img_query, label_query, query_cam, img_gallery, label_gallery, gall_cam = process_sysu(img_dir, mode, fold)
    elif "RegDB" in dataset :
        img_query, label_query, img_gallery, label_gallery = process_regdb(img_dir, mode, fold)
    elif "TWorld" in dataset :
        img_query, label_query, img_gallery, label_gallery = process_tworld(img_dir, mode, fold)
    return img_query, label_query, query_cam, img_gallery, label_gallery, gall_cam

def get_data_RegDB(data_path, mode, fold):
    datas = pd.read_json('dataset_use/RegDB/train_valid_test_ids.json')

    if mode == "test":
        fold = 0

    file_labels = get_ids(datas, f"{mode}_id_{fold}")
    color_image_file = get_path(datas, f"{mode}_path_rgb_{fold}")
    thermal_image_file = get_path(datas, f"{mode}_path_ir_{fold}")

    if data_path != "../Datasets/RegDB/":
        color_image_file = [data_path + im.split("../Datasets/RegDB/")[1] for im in color_image_file]
        thermal_image_file = [data_path + im.split("../Datasets/RegDB/")[1] for im in thermal_image_file]

    return file_labels, color_image_file, thermal_image_file

def process_regdb(data_path, mode, fold=0):

    file_label, color_img_file, thermal_img_file = get_data_RegDB(data_path, mode, fold)

    ids = np.unique(file_label)

    img_query, img_gallery = [], []
    label_query, label_gallery = [], []
    number_images_for_id_k = 10

    # Query and gallery are the same here as we want to compare each query image to all gallery image (LOOQ)
    for id in range(len(ids)):
        # Here we have 10 images per id for this dataset
        files_rgb = color_img_file[id*number_images_for_id_k:(id+1)*number_images_for_id_k]
        files_ir = thermal_img_file[id*number_images_for_id_k:(id+1)*number_images_for_id_k]
        files = [files_rgb, files_ir]

        for i in range(number_images_for_id_k):
            img_gallery.append([file[i] for file in files])
            img_query.append([file[i] for file in files])
            label_query.append(id)
            label_gallery.append(id)

    return img_query, np.array(label_query), img_gallery, np.array(label_gallery)

def get_data_TWorld(data_path, mode, fold_or_trial, fold):
    datas = pd.read_json('dataset_use/TWorld/train_valid_test_ids.json')
    query_gallery_datas = pd.read_json('dataset_use/TWorld/query_gallery_TWorld.json')

    if mode == "test":
        fold = 0

    #     Load Data
    ids = get_ids(datas, f"{mode}_id_{fold}")

    # Get list of list, each sub list containing the images location for one identity
    ids_file_RGB, ids_file_IR = [], []
    img_dir_init = data_path

    # We need to never use the augmented data for validating
    for id in ids:
        img_dir = os.path.join(img_dir_init, "TV_FULL/", str(id))
        if os.path.isdir(img_dir):
            # Since all images are in a same folder, we get all an id here
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir) if len(i.split("aug")) != 2])
            ids_file_RGB.append(new_files)
        img_dir = os.path.join(img_dir_init, "IR_8/", str(id))
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir) if len(i.split("aug")) != 2])
            ids_file_IR.append(new_files)

    position_list = get_query_gallery(query_gallery_datas, f"{mode}_{fold_or_trial}")

    return ids_file_RGB, ids_file_IR, position_list, ids

def process_tworld(data_path, mode, fold=0):

    fold_or_trial = int(fold)

    color_img_file, thermal_img_file, position_list, ids = get_data_TWorld(data_path, mode, fold_or_trial, fold)

    img_query, img_gallery = [], []
    label_query, label_gallery = [], []

    # Query and gallery are the same since we want to compare query to all gallery image (LOOQ)
    for id in range(len(ids)):
        # Here we have 10 images per id for this dataset
        files_rgb = color_img_file[id]
        files_ir = thermal_img_file[id]
        files = [files_rgb, files_ir]
        number_images_for_id_k = len(position_list[id])

        label_query.extend([id for _ in range(number_images_for_id_k)])
        label_gallery.extend([id for _ in range(number_images_for_id_k)])
        img_gallery.extend([file[i] for file in files] for i in position_list[id])
        img_query.extend([file[i] for file in files] for i in position_list[id])

    return img_query, np.array(label_query), img_gallery, np.array(label_gallery)

def get_data_SYSU(data_path, mode, fold_or_trial, fold):
    datas = pd.read_json('dataset_use/SYSU/train_valid_test_ids.json')
    query_gallery_datas = pd.read_json('dataset_use/SYSU/query_gallery_SYSU.json')

    if mode == "test":
        fold = 0

    ### Load Data
    ids = get_ids(datas, f"{mode}_id_{fold}")
    ids = ["%04d" % int(id) for id in ids]

    ids_file_RGB, ids_file_IR = [], []

    ### Get list of list containing images per identity
    for id in ids:
        files_rgb, files_ir = image_list_SYSU(id, data_path)
        ids_file_RGB.append(files_rgb)
        ids_file_IR.append(files_ir)

    position_list_rgb = get_query_gallery(query_gallery_datas, f"{mode}_{fold_or_trial}_rgb")
    position_list_ir = get_query_gallery(query_gallery_datas, f"{mode}_{fold_or_trial}_ir")

    return ids_file_RGB, ids_file_IR, position_list_rgb, position_list_ir, ids

# Process SYSU data for test or validation
def process_sysu(data_path, method, fold=0):

    fold_or_trial = int(fold)

    color_img_file, thermal_img_file, position_list_rgb, position_list_ir, ids = get_data_SYSU(data_path, method, fold_or_trial, fold)

    # Init var
    img_query, img_gallery = [], []
    label_query, label_gallery = [], []

    # Get the wanted query-gallery with corresponding labels
    for id in range(len(ids)):
        files_rgb = color_img_file[id]
        files_ir = thermal_img_file[id]
        # Same for RGB and IR due to pre-processed selection of positions
        number_images_for_id_k = len(position_list_rgb[id])

        # LOO Query => 1 query - remaining = gallery
        label_query.append(int(ids[id]))
        label_gallery.extend([int(ids[id]) for _ in range(number_images_for_id_k - 1)])

        img_query.append([files_rgb[position_list_rgb[id][0]], files_ir[position_list_ir[id][0]]])
        img_gallery.extend([[files_rgb[i], files_ir[j]] for i, j in zip(position_list_rgb[id][1:], position_list_ir[id][1:])])

    # Just set random distinct cams from query to gallery so that ReID is done in the evaluation function
    gall_cam = [4 for _ in range(len(img_gallery))]
    query_cam = [1 for _ in range(len(img_query))]

    return img_query, np.array(label_query), np.array(query_cam), img_gallery, np.array(label_gallery), np.array(gall_cam)

# Get all images concerning one id from the differents cameras in two distinct lists
def image_list_SYSU(id, data_path) :
    files_rgb = 0
    for k in [1,2,4,5]:
        img_dir = os.path.join(data_path, f'cam{k}', id)
        if os.path.isdir(img_dir) :
            if files_rgb == 0:
                files_rgb = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            else:
                files_rgb.extend(sorted([img_dir + '/' + i for i in os.listdir(img_dir)]))
    files_ir = 0
    for k in [3,6]:
        img_dir = os.path.join(data_path, f'cam{k}', id)
        if os.path.isdir(img_dir):
            if files_ir == 0:
                files_ir = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            else:
                files_ir.extend(sorted([img_dir + '/' + i for i in os.listdir(img_dir)]))

    return files_rgb, files_ir

class Prepare_set(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image1 = []
        test_image2 = []

        for i in range(len(test_img_file)):
            img1 = Image.open(test_img_file[i][0])
            img2 = Image.open(test_img_file[i][1])
            img1 = img1.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            img2 = img2.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array1 = np.array(img1)
            pix_array2 = np.array(img2)
            test_image1.append(pix_array1)
            test_image2.append(pix_array2)

        test_image1 = np.array(test_image1)
        test_image2 = np.array(test_image2)

        self.test_image1 = test_image1
        self.test_image2 = test_image2

        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, img2, target1 = self.test_image1[index], self.test_image2[index], self.test_label[index]
        img1, img2 = self.transform(img1, img2)
        return img1, img2, target1

    def __len__(self):
        #Should be the same len for both image 1 and image 2
        return len(self.test_image1)


def read_imgs(train_image, pid2label):
    train_img = []
    train_label = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((144, 288), Image.ANTIALIAS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        pid = int(img_path.split("cam")[1][2:6]) # Usually, pid stored in /camx/XXXX/image_name.jpg, so we extract XXXX

        pid = pid2label[pid]
        train_label.append(pid)

    return np.array(train_img), np.array(train_label)