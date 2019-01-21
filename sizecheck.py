import os
import import skimage.io as io
img_list = os.listdir("./train")
print(img_list)

img_size_set = set()

for t in img_list:
    img = io.imread("./train/" + t)
    img_size_set.add(img.shape)

print("all image sizes: ", img_size_set)
