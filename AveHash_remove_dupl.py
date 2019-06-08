#!/usr/bin/python
# -*- coding: <encoding name> -*-
# author: Zcq in Horizon
import cv2
import  numpy as np
import time;  
import os
from PIL import Image
import shutil

start = time.time()

SIZE = 16
capFrameNum = 50
removeThres = 80
imgPath = '/home/zcq/1Horizon_Project/ImgSimilarity/data/example_duplicate/data'
savePath = '/home/zcq/1Horizon_Project/ImgSimilarity/data/example_duplicate/data_new_1/'

def aHash(img):
    img=cv2.resize(img,(SIZE,SIZE),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    s=0
    hash_str=''
    for i in range(SIZE):
        for j in range(SIZE):
            s=s+gray[i,j]
    avg=s/(SIZE*SIZE)
    for i in range(SIZE):
        for j in range(SIZE):
            if  gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'            
    return hash_str

def pHash(img):
    img=cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    img = cv2.dct(img)
    img = img[0:16,0:16]
    avg = 0
    hash_str = ''
    for i in range(16):
        for j in range(16):
            avg += img[i,j]
    avg = avg/256
    
    for i in range(16):
        for j in range(16):
            if  img[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'            
    return hash_str

def cmpHash(hash1,hash2):
    n=0
    if len(hash1)!=len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            n=n+1
    return n

imgs = []
files = os.listdir(imgPath)
for i in range(len(files)):
    if not files[i]:
        continue
    imgs.append(os.path.join(imgPath,files[i]))     
paths = imgs
paths_1=[x.split("_")[-1]+"\t"+x for x in paths if len(x.split("_")[-1].split(".")[0])<=2]
paths_2=[x for x in paths if len(x.split("_")[-1].split(".")[0])>2]
paths_1.sort()
paths_2.sort()
paths= [x.split("\t")[-1] for x in  paths_1]
paths+=paths_2
imgs = paths
    
#for i in range(len(imgs)):
#    print(imgs[i])
     
imgNum = len(imgs)
count = 0
for i in range(imgNum):
    try:
        print("\n{}st Reference img：{}".format(i,imgs[i].split('/')[-1]))
        img1 = cv2.imread(imgs[i])
        hash1 = aHash(img1)
    except ValueError: 
        break
    except IndexError:  
        break   
    remove = 0   
    for j in range(i+1,i+capFrameNum):
        try:
            img2 = cv2.imread(imgs[j-remove])
            count += 1
        except ValueError: 
            break
        except IndexError:  
            break
        hash2 = aHash(img2)
        n = cmpHash(hash1,hash2)
        n_ = (SIZE*SIZE-n)/(SIZE*SIZE) * 100
        n_ = round(n_,2)
        if n_> removeThres: 
            print('{}, Similarity：{}%,  remove:{}\t'.format(imgs[j-remove].split('/')[-1],n_,imgs[j-remove].split('/')[-1]))
            imgs.remove(imgs[j-remove])
            remove += 1
        else:
            print('{}, Similarity：{}%\t'.format(imgs[j-remove].split('/')[-1],n_))
            
end = time.time()

print('\n'+'Result:')  
for i in imgs:
    print(i.split('/')[-1])
print('\nCount：{} frame, Time of one frame: {:4.2f}ms \n'.format(count,(end-start)*1000/count))
print('Before imgNum: {}'.format(imgNum))
print('After  imgNum: {}'.format(len(imgs)))

'''
if os.path.exists(savePath):
    shutil.rmtree(savePath)
os.mkdir(savePath)

for i in imgs:
    img=Image.open(i)
    img.save(savePath+i.split('/')[-1])
print('\nSave OK')
'''
