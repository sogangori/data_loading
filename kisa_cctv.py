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


attributes = ['pose', 'belonging']
classes = ['bg', 'person', 'fire']
poses = ['unknown', 'stand', 'lie', 'abnormal']
belongings = ['unknown', 'no', 'yes']


# In[4]:


attr_0 = classes
attr_1 = poses
attr_2 = belongings
len(attr_0), len(attr_1), len(attr_2)


# In[5]:


2**8 * 3, 2**8 * 5


# In[6]:


padded_image_shape = (768, 1280)
image_shape = (720, 1280)
anchor_k = 9
num_classes = 30
num_classes_real = len(classes)
num_head_filter = 4 + 1 + len(classes) + len(poses) + len(belongings)
max_data_m = 10000
level_start = 3
level_end = 8
l1 = 1e-8 #1e-9

edgecolors = np.random.rand(num_classes, 3) 
edgecolors = np.minimum(edgecolors+0.1, 1.0)
path_weight = "weight/cctv_detector"


# In[7]:


num_head_filter


# ### Data load

# In[10]:


def exist_folder(folder):
    is_exist = os.path.isdir(folder)
    if is_exist:
        print('exist folder', folder)
    else:
        print('not exist folder', folder)
    return is_exist        


# In[54]:


def get_file_list(folder):
    folder_annotations = folder + 'annotation/'
    folder_images = folder + 'images/'
    
    if exist_folder(folder_annotations) and exist_folder(folder_images):
        list_csv = glob(folder_annotations + '*.csv')
        list_img = glob(folder_images + '*/*.jpg')
        
        list_img_sub = glob(folder_images + '*/*/*.jpg')
        list_img +=list_img_sub
        
        return list_img, list_csv
    else:
        return [], []


# In[55]:


def concat_annotation(list_csv):
    list_df = []
    for path_csv in list_csv:
        path_csv_split = path_csv.split(os.sep)
        video_name = path_csv_split[-1].split('.')[0]
        #print('video_name', video_name)
        df_csv = pd.read_csv(path_csv)        
        list_df.append(df_csv)
    
    df = pd.concat(list_df)    
    df = df[df.region_count > 0]    
    return df


# In[56]:


def denoise_df(df):
    col_k = len(df.columns)
    is_same_data1 = (df == df.shift(1)).sum(axis=1) == col_k 
    is_same_data2 = (df == df.shift(2)).sum(axis=1) == col_k 
    is_same_data3 = (df == df.shift(3)).sum(axis=1) == col_k 
    is_same_data = is_same_data1 | is_same_data2 | is_same_data3

    print('check same data', is_same_data1.sum(), is_same_data2.sum(), is_same_data3.sum(), is_same_data.sum())
    
    df = df[np.logical_not(is_same_data)]
    
    print('region_attributes.nunique', df.region_attributes.nunique())
    print(df.region_attributes.unique())
    
    unknown_str = poses[0]
    unknown_str = '"' + unknown_str + '"'
    
    region_att = df.region_attributes.copy()
    region_att = region_att.str.replace('"belonging":undefined', '"belonging":"no"')
    region_att = region_att.str.replace('"pose":undefined', '"pose":"stand"')
    region_att = region_att.str.replace('anbormal', 'abnormal')
    
    print('region_attributes.nunique', region_att.nunique())    
    df.region_attributes = region_att

    return df


# In[57]:


def get_image_dict(list_img):
    dict_cctv_img = dict()
    for path_img in list_img:
        img_file_name = path_img.split('/')[-1]
        dict_cctv_img[img_file_name] = path_img
    print('dict len', len(dict_cctv_img))
    #print('dict_img', dict_cctv_img)
    return dict_cctv_img


# In[58]:


def replace_to_full_path(df, dict_cctv_img):
    list_img_path = []
    for filename in df.filename:

        image_path_full = dict_cctv_img[filename]
        #print('filename', filename, 'image_path_full', image_path_full)
        list_img_path.append(image_path_full)

        if not os.path.isfile(image_path_full):
            print('not exist file', image_path_full)
            print('filename', filename)      
    df.filename = list_img_path
    return df


# In[61]:


def get_value_or_default(dic, key, default_values):
    if key in dic.keys():
        v = dic[key]
    else:
        v = default_values[0]        
    return v

def set_attribute(df):
    list_shape = []
    list_attributes = []

    for region_shape_attributes, region_attributes in zip(df.region_shape_attributes, df.region_attributes):

        region_shape_attributes = ast.literal_eval(region_shape_attributes)
        try:
            region_attributes = ast.literal_eval(region_attributes)
        except:
            print('except region_attributes', region_attributes)
            break

        x = int(region_shape_attributes['x'])
        y = int(region_shape_attributes['y'])
        width = int(region_shape_attributes['width'])
        height = int(region_shape_attributes['height'])

        class_name = 'person'
        pose = get_value_or_default(region_attributes, 'pose', poses)
        belonging = get_value_or_default(region_attributes, 'belonging', belongings)    

        list_shape.append([x,y,width,height])
        list_attributes.append([class_name, pose, belonging])    
        
    array_attributes = np.array(list_attributes)
    array_shape = np.array(list_shape)
    df['x'] = array_shape[:, 0]
    df['y'] = array_shape[:, 1]
    df['w'] = array_shape[:, 2]
    df['h'] = array_shape[:, 3]
    df['cls'] = array_attributes[:, 0]
    df['pose'] = array_attributes[:, 1]
    df['belonging'] = array_attributes[:, 2]
    df = df.drop(columns=['file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes', 'region_attributes'])
    return df


# In[62]:


def kisa_cctv(folder):
    if exist_folder(folder):    
        list_img, list_csv = get_file_list(folder)
        print('count csv', len(list_csv), 'img', len(list_img))
        
        df = concat_annotation(list_csv)
        print('df.shape', df.shape)
        df = denoise_df(df)
        
        dict_img = get_image_dict(list_img)
        df = replace_to_full_path(df, dict_img)
        df = set_attribute(df)
        #annotation = parsing_annotation(df)
        #print('annotation', len(annotation))
        
    return df#, annotation


# folder_cctv = '/media/mvlab/469B5B3C650FBA77/data/cctv test/'
# df = kisa_cctv(folder_cctv)

# In[65]:


def parsing_annotation(df):
    annotation = dict()
    for i in range(len(df)):
      
        row = df.iloc[i].values
        path_image, x0, y0, width, height, cls, pose, belonging = row
        x1 = x0 + width
        y1 = y0 + height
        if i%10000==0:
            print(i, row)            
                     
        cls_idx = classes.index(cls)        
        pose_idx = poses.index(pose)
        belonging_idx = belongings.index(belonging)        
        bbox = np.array([x0, y0, x1, y1, cls_idx, pose_idx, belonging_idx]).reshape((1, -1))        
                   
        if path_image in annotation.keys():
            pre_bbox = annotation[path_image]
            new_bbox = np.concatenate((pre_bbox, bbox), axis=0)
            #cls_bbox = np.stack(cls_bbox, 0)#.reshape([-1, 6])
            #annotation[path_image].extend(new_bbox)
            annotation[path_image] = new_bbox
        else:
            annotation[path_image] = bbox        

    return annotation

