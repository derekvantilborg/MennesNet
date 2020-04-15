from mxnet import gluon, np, npx
npx.set_np()
import os
import shutil
import zipfile
from git import Repo
import zipfile

if not os.path.isdir('ucmdata'):
    print('Downloading ucmdata ...')
    Repo.clone_from('https://git.wur.nl/lobry001/ucmdata.git', 'ucmdata')

os.chdir('ucmdata')
cwd = os.getcwd()

if not os.path.isdir('UCMImages'):
    print('Extracting ucmdata ...')
    with zipfile.ZipFile('UCMerced_LandUse.zip', 'r') as zip_ref:
        zip_ref.extractall('UCMImages')
    os.rename(cwd + '/UCMImages/UCMerced_LandUse/Images', cwd + '/Images')


for name in ['UCMImages', 'README.md', 'UCMerced_LandUse.zip']:
    file = cwd + '/ucmdata/' + name
    if os.path.exists(file):
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)
        print('Removed: ' + file)

UCM_images_path = "Images/"
Multilabels_path = "LandUse_Multilabeled.txt"
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
