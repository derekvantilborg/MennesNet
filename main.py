## =============== PERFORMING IMPORTS ==============
import mxnet
from mxnet import gluon, npx
import numpy as np
npx.set_np()
import os, shutil, zipfile
from git import Repo
import skimage.io as io
import matplotlib.pyplot as plt
import pandas as pd

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

## ============== DEFINING THE SINGLELABEL DATASET CLASS ===================
class Dataset_Singlelabel(gluon.data.Dataset):
    def __init__(self, img_folder, n_classes = 21):
        super(Dataset_Singlelabel, self).__init__()
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.imgs = []
        self.labels = []

        # Define self.images
        for sub_folder in os.listdir(img_folder):
            print("Processing {} images".format(sub_folder))
            for file in os.listdir(os.path.join(img_folder, sub_folder)):
                img = io.imread(os.path.join(img_folder, sub_folder, file)) / 255
                # normalize the image with mean and stdev
                img_norm = (img.astype('float32') - np.tile(self.rgb_mean, (img.shape[0], img.shape[1], 1))) / np.tile(
                    self.rgb_std, (img.shape[0], img.shape[1], 1))
                self.imgs.append(img_norm)
                self.labels.append(file.split('.')[0][:-2])
        #one-hot encoding of labels
        self.labels = mxnet.ndarray.one_hot(self.labels, n_classes)

    def __getitem__(self, idx):
        #__getitem__ asks for the sample number idx. Since we pre-loaded the images
        #and the ground truths, we just have to return the corresponding sample.
        img = self.imgs[idx].transpose(2, 0, 1)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)

## =============== DEFINING DATA LOADERS ==================

Dataset_Single = Dataset_Singlelabel(UCM_images_path, Multilabels_path)

from multiprocessing import cpu_count
CPU_COUNT = cpu_count()
print("Number of available CPUs: " + str(CPU_COUNT))

data_loader = gluon.data.DataLoader(Dataset_Single,
                                    batch_size=5,
                                    shuffle=True,
                                    last_batch='discard',
                                    num_workers=1)

for X_batch, y_batch in data_loader:
    print("X_batch has shape {}, and y_batch has shape {}".format(X_batch.shape, y_batch.shape))


##
# ============== DEFINING THE MULTILABEL DATASET CLASS ===================
class Dataset_Multilabel(gluon.data.Dataset):
    def __init__(self, img_folder, label_file):
        super(Dataset_Multilabel, self).__init__()
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.imgs = []
        self.labels = []

        # Load labels as dataframe
        label_df = pd.read_csv(label_file, sep="\t")

        # Define self.images
        for sub_folder in [f for f in os.listdir(img_folder) if not f.startswith('.')]:
            print("Processing {} images".format(sub_folder))
            for file in os.listdir(os.path.join(img_folder, sub_folder)):
                img = io.imread(os.path.join(img_folder, sub_folder, file)) / 255
                # normalize the image with mean and stdev
                img_norm = (img.astype('float32') - np.tile(self.rgb_mean, (img.shape[0], img.shape[1], 1))) / np.tile(self.rgb_std, (img.shape[0], img.shape[1], 1))
                self.imgs.append(img_norm)
                label = label_df.loc[label_df['IMAGE\LABEL'] == file.split('.')[0]].values.flatten().tolist()[1:]
                self.labels.append(label)


    def __getitem__(self, idx):
        #__getitem__ asks for the sample number idx. Since we pre-loaded the images
        #and the ground truths, we just have to return the corresponding sample.
        img = self.imgs[idx].transpose(2, 0, 1)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)

multilabel = Dataset_Multilabel(UCM_images_path, Multilabels_path)
##
