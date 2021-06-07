#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import zipfile

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
from PIL import ImageFilter
import ast
import json
from glob import glob


# In[2]:


attributes = ['pose', 'belonging']
classes = ['bg', 'person', 'fire']
poses = ['unknown', 'stand', 'lie', 'abnormal']
belongings = ['unknown', 'no', 'yes']


# In[10]:


folder_aihub_cctv = '/media/mvlab/469B5B3C650FBA77/data/AI-hub cctv/'



# In[23]:


def exist_folder(folder):
    is_exist = os.path.isdir(folder)
    if is_exist:
        print('exist folder', folder)
    else:
        print('not exist folder', folder)
    return is_exist        


# In[32]:


def get_file_list(folder):
    folder_annotations = folder + 'annotation/'
    folder_images = folder + 'images/'
    
    if exist_folder(folder_annotations) and exist_folder(folder_images):
        list_csv = glob(folder_annotations + '*.csv')
        list_img = glob(folder_images + '*/*.jpg')    
        return list_img, list_csv
    else:
        return [], []


# In[107]:


def concat_annotation(list_csv):
    list_df = []
    for path_csv in list_csv:
        path_csv_split = path_csv.split(os.sep)
        video_name = path_csv_split[-1].split('.')[0]
        #print('video_name', video_name)
        df_csv = pd.read_csv(path_csv)
        df_csv['video'] = video_name
        list_df.append(df_csv)
    
    df = pd.concat(list_df)
    #df = df.drop(columns=['file_attributes', 'region_id'])    
    df = df[df.region_count > 0]
    df['filename'] = df.video + os.sep + df.filename
    return df


# In[115]:


def denoise_df(df):
    col_k = len(df.columns)
    is_same_data1 = (df == df.shift(1)).sum(axis=1) == col_k 
    is_same_data2 = (df == df.shift(2)).sum(axis=1) == col_k 
    is_same_data3 = (df == df.shift(3)).sum(axis=1) == col_k 
    is_same_data = is_same_data1 | is_same_data2 | is_same_data3

    print('check same data', is_same_data1.sum(), is_same_data2.sum(), is_same_data3.sum(), is_same_data.sum())
    
    df = df[np.logical_not(is_same_data)]
    
    print('region_attributes.nunique', df.region_attributes.nunique())
    for region_att in df.region_attributes.unique():
        print(region_att)
    
    unknown_str = poses[0]
    unknown_str = '"' + unknown_str + '"'
    
    region_attributes = df.region_attributes.copy()
    region_attributes = region_attributes.str.replace('undefined', unknown_str)
    region_attributes = region_attributes.str.replace('"posr":', '"pose":')
    region_attributes = region_attributes.str.replace('"abnorm",', '"abnormal",')
    
    print('region_attributes.nunique', region_attributes.nunique())    
    df.region_attributes = region_attributes

    return df


# In[116]:


def get_image_dict(list_img):
    dict_cctv_img = dict()
    for path_img in list_img:
        path_img_split = path_img.split('/')
        img_file_name = os.sep.join(path_img_split[-2:])
        dict_cctv_img[img_file_name] = path_img
    print('dict', len(dict_cctv_img))
    return dict_cctv_img


# In[117]:


def replace_to_full_path(df, dict_cctv_img):
    #VID_20210425_155818.mp4_0.jpg
    list_img_path = []
    for filename in df.filename:

        image_path_full = dict_cctv_img[filename]
        #print('filename', filename, 'image_path_full', image_path_full)
        list_img_path.append(image_path_full)

        if not os.path.isfile(image_path_full):
            print('not exist file', image_path_full)
            print('filename', filename)        

    print('len', len(list_img_path), len(df))
    df.filename = list_img_path
    return df


# ### Data preprocess

# In[118]:


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


# In[147]:


def parsing_annotation(df):
    annotation = dict()
    for i in range(len(df)):
      
        row = df.iloc[i].values
        path_image, video, x0, y0, width, height, cls, pose, belonging = row
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
            annotation[path_image] = new_bbox
        else:
            annotation[path_image] = bbox        
                
    return annotation


# In[155]:


def ai_hub_cctv(folder):
    """
    arg : path
    return df
    """
    if exist_folder(folder):    
        list_img, list_csv = get_file_list(folder)
        print('count csv', len(list_csv), 'img', len(list_img))
        
        df = concat_annotation(list_csv)
        print('df.shape', df.shape)
        df = denoise_df(df)
        
        dict_cctv_img = get_image_dict(list_img)
        df = replace_to_full_path(df, dict_cctv_img)
        df = set_attribute(df)
        #annotation = parsing_annotation(df)
        #print('annotation', len(annotation))
        
    return df#, annotation


