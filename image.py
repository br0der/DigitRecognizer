import os
from random import shuffle
from PIL import Image

def imgToVals(image):
    return [sum(val)/3/255 for val in image.getdata()]

dirs = [os.listdir(f"archive/{i}") for i in range(3)]
print("Finished finding dirs")
imgs = []
for i in range(3):
    print(f"Starting number {i}")
    for dir in dirs[i]:
        imgs.append(imgToVals(Image.open(f"archive/{i}/{dir}")) + [i])
    
shuffle(imgs)
print([len(img) for img in imgs])
print(len(imgs))