## =============== PERFORMING IMPORTS ==============
import mxnet
from mxnet import gluon, npx
npx.set_np()
import numpy as rnp
import os, shutil, zipfile
from git import Repo
import skimage.io as io
import matplotlib.pyplot as plt
import pandas as pd
import random
import PIL.Image

## ============ LOADING THE DATA ===============

data_folder = 'ucmdata'

if not os.path.isdir(data_folder):
    print('Downloading data ...')
    Repo.clone_from('https://git.wur.nl/lobry001/ucmdata.git', data_folder)

if not os.path.isdir(data_folder + '/Images'):
    print('Extracting data ...')
    os.chdir(data_folder)
    with zipfile.ZipFile('UCMerced_LandUse.zip', 'r') as zip_ref:
        zip_ref.extractall('UCMImages')
    os.rename('UCMImages/UCMerced_LandUse/Images', 'Images')
    os.chdir('..')

for name in ['UCMImages', 'README.md', 'UCMerced_LandUse.zip']:
    file = os.path.join(data_folder, name)
    if os.path.exists(file):
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)
        print('Removed: ' + file)

UCM_images_path = os.path.join(data_folder, "Images/")
Multilabels_path = os.path.join(data_folder, "LandUse_Multilabeled.txt")

## ============= PLOTTING A TEST IMAGE ==============================

# image = io.imread('Images/golfcourse/golfcourse60.tif')
# f = plt.figure()
# plt.imshow(image)
# plt.show()

## ============== Function to split data paths in train/test for data loading =========================

def convert_and_split(parent_dir, split_ratio = 0.8):
    ''' randomly split all filenames in a parent directory into a test and a train set AND copy them to their respective directories
    parent_dir = pathname of the parent directory
    split_ratio = train/test ratio, default of 0.8 will give 80% train, 20% test
    '''
    # walk through the parent dir and find all file paths that are not hidden files
    listOfFiles = list()
    dirs = []
    for root, directories, filenames in os.walk(parent_dir):
        if not dirs: # save a list of class directories
            dirs = directories
        for filename in filenames:
            if not filename.startswith('.'):
                path = os.path.join(root, filename)
                listOfFiles += [path]
                if not path.endswith('.PNG'):
                    im = PIL.Image.open(path)
                    im.save(path.split('.')[0] + '.PNG', 'PNG', quality=100)
                    os.remove(path)

    # shuffle list of file paths and split according to the split ratio
    random.shuffle(listOfFiles)

    # create folder structure for test and train data
    for mode in ['train', 'test']:
        path = os.path.join('ucmdata', mode)
        if not os.path.exists(path):
            os.mkdir(path)
            [os.mkdir(os.path.join(path, dir)) for dir in dirs]

    for i, path in enumerate(listOfFiles):
        mode = "train" if i < round(len(listOfFiles) * split_ratio) else "test"
        file = os.path.split(path)[1]
        file_dir = ''.join(filter(lambda x: x.isalpha(), file.split('.')[0]))
        os.rename(path, os.path.join('ucmdata', mode, file_dir, file))

    return

convert_and_split(UCM_images_path)

## ============== DEFINING THE SINGLELABEL DATASET CLASS ===================
class Dataset_Singlelabel(gluon.data.Dataset):
    def __init__(self, file_paths):
        super(Dataset_Singlelabel, self).__init__()
        self.rgb_mean = rnp.array([0.485, 0.456, 0.406])
        self.rgb_std = rnp.array([0.229, 0.224, 0.225])
        self.imgs = []
        self.labels = rnp.zeros((2100, 21))
        # Create a list of all unqie classes in alphabetical order
        unique_classes = list(set([''.join(x for x in i if x.isalpha()) for i in
                                   [os.path.basename(i).split('.')[0] for i in file_paths]]))
        unique_classes.sort()
        # Define self.images
        for i, file in enumerate(file_paths):
            img = io.imread(file) / 255
            # normalize the image with mean and stdev
            img_norm = (img.astype('float32') - rnp.tile(self.rgb_mean, (img.shape[0], img.shape[1], 1))) / rnp.tile(
                self.rgb_std, (img.shape[0], img.shape[1], 1))
            self.imgs.append(img_norm)
            # Find the label from pathname and index it in the list of unique labels --> add to self.labels
            lab = ''.join(i for i in os.path.basename(file).split('.')[0] if not i.isdigit())
            self.labels[i, unique_classes.index(lab)] = 1  # append class number to labels

    def __getitem__(self, idx):
        #__getitem__ asks for the sample number idx. Since we pre-loaded the images
        #and the ground truths, we just have to return the corresponding sample.
        img = self.imgs[idx].transpose(2, 0, 1)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)

## ============== DEFINING THE MULTILABEL DATASET CLASS ===================

class Dataset_Multilabel(gluon.data.Dataset):
    def __init__(self, set_paths,label_file):
        super(Dataset_Multilabel, self).__init__()
        self.rgb_mean = rnp.array([0.485, 0.456, 0.406])
        self.rgb_std = rnp.array([0.229, 0.224, 0.225])
        self.imgs = []
        self.labels = []

        # Load labels as dataframe
        label_df = pd.read_csv(label_file, sep="\t")

        # Define self.images
        for file in set_paths:
            img = io.imread(file) / 255
            # normalize the image with mean and stdev
            img_norm = (img.astype('float32') - rnp.tile(self.rgb_mean, (img.shape[0], img.shape[1], 1))) / rnp.tile(self.rgb_std, (img.shape[0], img.shape[1], 1))
            self.imgs.append(img_norm)
            label = label_df.loc[label_df['IMAGE\LABEL'] == os.path.basename(file).split('.')[0]].values.flatten().tolist()[1:]
            self.labels.append(label)

    def __getitem__(self, idx):
        #__getitem__ asks for the sample number idx. Since we pre-loaded the images
        #and the ground truths, we just have to return the corresponding sample.
        img = self.imgs[idx].transpose(2, 0, 1)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)

## ============ DEFINE TRANSFORMATIONS ========================

def aug_transform(data, label):
    data = data.astype('float32') / 255
    augs = gluon.data.vision.transforms.Compose([
        gluon.data.vision.transforms.RandomFlipLeftRight()
    ])
    for aug in augs:
        data = aug(data)
    return data, label


def plot_mx_array(array):
    assert array.shape[2] == 3, "RGB Channel should be last"
    plt.imshow((array.clip(0, 255) / 255).asnumpy())

## =============== DEFINING DATA SETS AND LOADERS ==================

# from multiprocessing import cpu_count
# CPU_COUNT = cpu_count()
# print("Number of available CPUs: " + str(CPU_COUNT))

train_folder = 'ucmdata/train'
test_folder = 'ucmdata/test'

Dataset_Single_train = gluon.data.vision.datasets.ImageFolderDataset(train_folder, transform=aug_transform)
Dataset_Single_test = gluon.data.vision.datasets.ImageFolderDataset(test_folder, transform=aug_transform)

DataLoader_Single_train = gluon.data.DataLoader(Dataset_Single_train,
                                    batch_size=5,
                                    shuffle=True,
                                    last_batch='discard')
DataLoader_Single_test = gluon.data.DataLoader(Dataset_Single_test,
                                    batch_size=5,
                                    shuffle=True,
                                    last_batch='discard')

# for X_batch, y_batch in DataLoader_Single_test:
#     print("X_batch has shape {}, and y_batch has shape {}".format(X_batch.shape, y_batch.shape))
