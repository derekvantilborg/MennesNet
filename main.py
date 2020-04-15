from mxnet import gluon, np, npx
npx.set_np()
import os
import shutil
import zipfile
from git import Repo
import zipfile
import skimage.io as io
import matplotlib.pyplot as plt

if not os.path.isdir('ucmdata'):
    print('Downloading ucmdata ...')
    Repo.clone_from('https://git.wur.nl/lobry001/ucmdata.git', 'ucmdata')

os.chdir('ucmdata')
cwd = os.getcwd()

if not os.path.isdir('Images'):
    print('Extracting ucmdata ...')
    with zipfile.ZipFile('UCMerced_LandUse.zip', 'r') as zip_ref:
        zip_ref.extractall('UCMImages')
    os.rename(cwd + '/UCMImages/UCMerced_LandUse/Images', cwd + '/Images')


for name in ['/UCMImages', '/README.md', '/UCMerced_LandUse.zip']:
    file = cwd + name
    if os.path.exists(file):
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)
        print('Removed: ' + file)

UCM_images_path = "Images/"
Multilabels_path = "LandUse_Multilabeled.txt"

# # plotting a test image
# image = io.imread('Images/golfcourse/golfcourse60.tif')
# f = plt.figure()
# plt.imshow(image)
# plt.show()

# ============== DEFINING THE DATASET CLASS ===================
# split_imgs = {}
# split_imgs["train"] = [1,3,5,7,11,13,15,17,21,23,26]
# split_imgs["val"] = [28,30,32,34,37]
# split_imgs["test"] = [2,4,6,8,10,12,14,16,20,22,24,27,29,31,33,35,38]
#
# class VaihingenDataset(gluon.data.Dataset):
#     def __init__(self, img_folder, GT_folder, split, patch_size=512):
#         super(VaihingenDataset, self).__init__()
#         self.rgb_mean = rnp.array([0.485, 0.456, 0.406])
#         self.rgb_std = rnp.array([0.229, 0.224, 0.225])
#
#         self.imgs = []
#         self.GTs = []
#
#         #Amount of overlap when defining the patches. If your dataset is big enough,
#         #you might not need to have overlaps
#         overlap = patch_size // 2
#
#         for img_index in split_imgs[split]:
#           print("Working on image " + str(img_index))
#           #Load the tile and the corresponding ground truth.
#           img = io.imread(os.path.join(img_folder, "top_mosaic_09cm_area" + str(img_index) + '.tif')) / 255
#           img = (img.astype('float32')  - rnp.tile(self.rgb_mean, (img.shape[0], img.shape[1], 1))) / rnp.tile(self.rgb_std, (img.shape[0], img.shape[1], 1))
#           GT = io.imread(os.path.join(GT_folder, "top_mosaic_09cm_area" + str(img_index) + '.tif'))
#
#           #Crop into patches, following a regularly sampled grid.
#           #i and j are defined as the center of the patch to crop.
#           for i in np.arange(patch_size//2, img.shape[0] - patch_size // 2, overlap):
#             for j in np.arange(patch_size//2, img.shape[1] - patch_size // 2, overlap):
#               #Crop the image and the ground truth into patch around (i,j) and save
#               #them in self.imgs and self.GTs arrays.
#               #For the image, note that we are taking the three channels (using ":")
#               #for the 3rd dimension, and we do the conversion to tensor.
#               i, j = int(i), int(j)
#               self.imgs.append(img[i - patch_size//2:i + patch_size // 2, j - patch_size // 2:j + patch_size // 2,:])
#               self.GTs.append(GT[i - patch_size//2:i + patch_size // 2, j - patch_size // 2:j + patch_size // 2])
#
#         print(f"Number of patches in dataset {split}: {len(self.imgs)}")
#
#
#     def __getitem__(self, idx):
#         #__getitem__ asks for the sample number idx. Since we pre-loaded the images
#         #and the ground truths, we just have to return the corresponding sample.
#         img = self.imgs[idx].transpose(2, 0, 1)
#         GT = self.GTs[idx]
#         return img, GT
#
#     def __len__(self):
#         return len(self.imgs)
