

import torch
from torchvision import models, transforms
import os
import glob
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from elasticsearch import Elasticsearch
import shutil 

img_id = 18
query_path = '/'+str(img_id)+'/'+str(img_id)+'.png'

num_out = 10
read_dir = '/media/vatsal/681E6E471E6E0DFE/Desktop/sem3/RE-DevOps/Dataset/newdt/val2017'
out_dir = '/home/vatsal/Downloads/outputs'

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

path = read_dir + query_path

topk = 44

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
input_size = 224
def resize_and_rename(image_dir):
    i = 1
    files = glob.glob(os.path.join(image_dir, "*"))
    for file in files:
        im = Image.open(file)
        f, e = os.path.splitext(file)
        imResize = im.resize((input_size,input_size), Image.ANTIALIAS)
        os.remove(file)

        path = '/'+os.path.join(*f.split('/')[:-1])+'/'+str(i)
        if not os.path.exists(path):
          os.mkdir(path) 
        fullPath = os.path.join(path,str(i))
        i+=1
        imResize.save(fullPath + '.png', 'PNG', quality=90)'''

'''!cp -R '/content/drive/My Drive/Sample' '/content/'
#!cp -R '/content/drive/My Drive/COVID-19/Dataset_2' '/content/'
resize_and_rename(data_dir)'''

def ReLU(X):
   return np.maximum(0,X)

def CReLU(f):
    f_pos = f
    f_neg = -f
    final_f = np.concatenate((f_pos, f_neg))
    return ReLU(final_f)

def getSurrogateText(F,topK):
    surrText = ""
    count = topK
    for ele in F:
        instanceText = "RO"+str(ele)
        instanceText += (" "+instanceText)*(count-1)
        if len(surrText) == 0:
            surrText = instanceText
        else:
            surrText += (" "+instanceText)
        count -= 1
    return surrText

def F2Text(f,topK):
    rf = CReLU(f)
    perVec = np.argsort(rf)[::-1]+1
    invPer = np.argsort(perVec)+1
    after_topk = invPer[invPer <= topK]
    return getSurrogateText(after_topk,topK)

class DenseFeatures(nn.Module):
    def __init__(self,dense):
        super(DenseFeatures, self).__init__()
        self.features = nn.Sequential(*list(dense.children())[:-1])
        #self.feat_vec = nn.Sequential(*list(dense.children())[2][:-3])
        
    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        #x = torch.flatten(x, 1)
        #x= self.feat_vec(x)
        return x

dense = models.densenet121(pretrained=True)
model = DenseFeatures(dense)
model.eval()
model = model.to(device)

data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


#print(path)
image = Image.open(path)
x = data_transforms(image)
x.unsqueeze_(0)
#print(x.shape)

output = model(x).cpu().detach().numpy()
#print(output[0][:10])
surrText = F2Text(output[0],topk)
#print(surrText)


query_body = {
  "query":{
    "match":{
      "text":surrText
    }
  }
}


elastic_client = Elasticsearch(hosts=["localhost"])
a = elastic_client.search(index="cbir", body=query_body,size=num_out)
dataArr = a['hits']['hits']
dataLen = len(dataArr)
for i in range(dataLen):
	print("Score: ",dataArr[i]['_score'])
	#print("Surrogate Text: ",dataArr[i]['_source']['text'])
	read_path = read_dir+dataArr[i]['_source']['image']
	write_path = out_dir+'/'+str(i)+'.png'
	img = Image.open(read_path)
	img.save(write_path, 'PNG')
	print("Image URL: ",read_path)
	#print(write_path)
	#print()
