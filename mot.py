#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import pandas as pd

from PIL import Image
from PIL import ImageFilter
import ast
import json
from glob import glob
import xml.etree.ElementTree as ET


# In[3]:


def exist_folder(folder):
    is_exist = os.path.isdir(folder)
    if is_exist:
        print('exist folder', folder)
    else:
        print('not exist folder', folder)
    return is_exist        


# In[4]:


def get_MOT_df(folder_video):    
    video = folder_video.split(os.sep)[-1]
    
    path_annot = folder_video + '/gt/gt.txt'
        
    df = pd.read_csv(path_annot, header=None, names=['frame','id','x','y','w','h','d0','d1','d2'])
    print('video', video, df.shape)
    df['video'] = video + '/img1/'    
    return df


# In[8]:


def set_file_name(df_MOT, folder_Det):
    list_img_names = []
    for frame in df_MOT.frame.values:
        img_file_name = '%#06d' % frame + '.jpg'        
        list_img_names.append(img_file_name)
    
    #df_MOT['img'] = list_img_names        
    df_MOT['filename'] = folder_Det + df_MOT['video'] + list_img_names
    
    filename_unique = df_MOT['filename'].unique()
    print('file exist check', len(filename_unique))
    for i in range(0, len(filename_unique), 100):
        if not os.path.isfile(filename_unique[i]):
            print('not exist', filename_unique[i])
            return
    
    return df_MOT


# In[15]:


def mot(folder, stride=1):
    folder_Det = folder + 'train/'
    folders_video_17 = glob(folder_Det + 'MOT17*NN')
    folders_video_20 = glob(folder_Det + 'MOT20*')
    folders_video = folders_video_17 + folders_video_20
    print('folders_video', len(folders_video))
    
    list_df_MOT = []
    for folder_video in folders_video:
        df_MOT_sub = get_MOT_df(folder_video)
        list_df_MOT.append(df_MOT_sub)
    len(list_df_MOT)
    
    df_MOT = pd.concat(list_df_MOT)
    print('df.shape', df_MOT.shape)
    
    df_MOT = set_file_name(df_MOT, folder_Det)
    print(df_MOT.head(1))

    cond_person = df_MOT.d0 == 1
    print('person class ratio', len(df_MOT), cond_person.sum()/len(df_MOT))
    df_MOT.loc[cond_person, 'cls']='person'
    
    select_cond = df_MOT.frame % stride == 0
    df_MOT_cut = df_MOT[select_cond]
    print('select by stride', len(df_MOT), len(df_MOT_cut))
    return df_MOT_cut


# folder_MOT = '/media/mvlab/469B5B3C650FBA77/data/MOT/'
# df = mot(folder_MOT, stride=100)
# df.shape

# df

# (df.x<0).sum(), (df.y<0).sum(),(df.x + df.w >1920).sum(), (df.y + df.h >1080).sum()
