import os
import glob
from PIL import Image
import numpy as np
import json

basePath = 'D:\\Users\\Downloads\\Compressed\\'
data_dir = basePath+'val2017'

input_size = 224

def resize_and_rename(image_dir):
    i = 1
    files = glob.glob(os.path.join(image_dir, "*"))
    for file in files:
        if (i-1) % 100 == 0:
            print("Processing...."+str(i))
        im = Image.open(file)
        f, e = os.path.splitext(file)
        imResize = im.resize((input_size,input_size), Image.ANTIALIAS)
        os.remove(file)

        path = data_dir+'\\'+str(i)
        if not os.path.exists(path):
          os.mkdir(path) 
        fullPath = os.path.join(path,str(i))
        i+=1
        imResize.save(fullPath + '.png', 'PNG', quality=90)
		
resize_and_rename(data_dir)