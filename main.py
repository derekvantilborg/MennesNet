from mxnet import gluon, np, npx
npx.set_np()
import os
import zipfile
from git import Repo
import zipfile

if not os.path.isdir('ucmdata'):
    print('Downloading ucmdata ...')
    Repo.clone_from('https://git.wur.nl/lobry001/ucmdata.git', 'ucmdata')

cwd = os.getcwd()
os.chdir('ucmdata')

if not os.path.isdir('UCMImages'):
    print('Extracting ucmdata ...')
    with zipfile.ZipFile('UCMerced_LandUse.zip', 'r') as zip_ref:
        zip_ref.extractall('UCMImages')

# os.rename(cwd + '/ucmdata/UCMImages/UCMerced_LandUse/Images', cwd)

# !mv UCMImages/UCMerced_LandUse/Images .
# !rm -rf UCMImages README.md  UCMerced_LandUse.zip
# !ls
#
# UCM_images_path = "Images/"
# Multilabels_path = "LandUse_Multilabeled.txt"
#
# with open(UCM_images_path + Multilabels_path) as f:
#   print(f)
#
#
# # show image
# with open("LandUse_Multilabeled.txt") as f: # The with keyword automatically closes the file when you are done
#   print(f.read())
#
# # globals()
# # os.listdir('Images/golfcourse')
#
# import skimage.io as io
# import matplotlib.pyplot as plt
#
# image = io.imread('Images/golfcourse/golfcourse60.tif')
#
# f = plt.figure()
# plt.imshow(image)
# plt.show()
