## =============== PERFORMING IMPORTS ==============
import mxnet
from mxnet import gluon, npx, np, autograd
from mxnet.gluon import nn
import os, shutil, zipfile
from git import Repo
import skimage.io as io
import matplotlib.pyplot as plt
import pandas as pd
import random
import PIL.Image
import matplotlib

# ! pip install d2l -q
import d2l

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

## =============== DEFINING THE NETWORK ============================

# Define the network architecture
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        # Dense will transform the input of the shape (batch size, channel,
        # height, width) into the input of the shape (batch size,
        # channel * height * width) automatically by default
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(21))

# Test the network shapes
X = mxnet.ndarray.random.uniform(shape=(5, 3, 256, 256))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

## =============== TRAINING THE NET ====================

num_epochs, lr, wd, ctx = 10, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})

num_batches, timer = len(DataLoader_Single_train), d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'train acc', 'val loss', 'val acc'])

ctx = d2l.try_all_gpus()
for epoch in range(num_epochs):
    # Training loop
    # Store training_loss, training_accuracy, num_examples, num_features
    metric = d2l.Accumulator(4)
    for i, (Xs_in, ys_in) in enumerate(DataLoader_Single_train):
        print("Training iteration: " + str(i))
        timer.start()
        Xs = gluon.utils.split_and_load(Xs_in.astype("float32"), ctx)
        ys = gluon.utils.split_and_load(ys_in.astype("float32"), ctx)
        with autograd.record():
            pys = [net(X.transpose(axes=(0, 3, 1, 2))) for X in Xs]
            ls = [loss(py, y) for py, y in zip(pys, ys)]
        for l in ls:
            l.backward()
        trainer.step(ys_in.shape[0])
        train_loss_sum = sum([float(l.sum().asnumpy()[0]) for l in ls])
        train_acc_sum = sum(d2l.accuracy(py.asnumpy(), y.asnumpy()) for py, y in zip(pys, ys))
        l, acc = train_loss_sum, train_acc_sum
        metric.add(l, acc, ys_in.shape[0], ys_in.size)
        timer.stop()
        if (i + 1) % (num_batches // 5) == 0:
            animator.add(epoch + i / num_batches,
                         (metric[0] / metric[2], metric[1] / metric[3], None, None))

    # val_acc = d2l.evaluate_accuracy_gpus(net, val_iter, split_f)
    metric_val = d2l.Accumulator(2)  # num_corrected_examples, num_examples
    for i, (Xs_in, ys_in) in enumerate(DataLoader_Single_test):
        Xs = gluon.utils.split_and_load(Xs_in.astype("float32"), ctx)
        ys = gluon.utils.split_and_load(ys_in.astype("float32"), ctx)
        pys = [net(X) for X in Xs]
        ls = [loss(py, y) for py, y in zip(pys, ys)]
        val_loss_sum = sum([float(l.sum().asnumpy()[0]) for l in ls])

        OA_val = np.sum(np.argmax(pys[0].asnumpy(), axis=1) == ys[0].asnumpy()).astype("float32") / np.prod(
            ys[0].shape)
        metric_val.add(OA_val, len(ys))

    val_acc = OA_val
    animator.add(epoch + 1, (None, None, val_loss_sum / ys_in.shape[0], val_acc))
print('loss %.3f, train acc %.3f, val acc %.3f' % (
    metric[0] / metric[2], metric[1] / metric[3], val_acc))
print('%.1f examples/sec on %s' % (
    metric[2] * num_epochs / timer.sum(), d2l.try_all_gpus()))