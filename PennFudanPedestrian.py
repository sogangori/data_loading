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
import xml.etree.ElementTree as ET


# In[2]:


folder_ped_rgb = 'PNGImages/'
folder_ped_mask = 'PedMasks/'


# In[3]:


def convert_path_rgb_to_path_mask(folder_ped, path_rgb):
    path_file_name = path_rgb.split('/')[-1]
    file_name_pre = path_file_name.split('.')[0]
    path_mask = folder_ped + folder_ped_mask + file_name_pre + '_mask' + '.png'
    if os.path.isfile(path_mask):
        return path_mask
    else:
        print('not exist file', path_mask)
        exit()


# In[4]:


def get_masked_pedestrian(path_rgbs, path_masks):
    arr_ped_list = []
    m = len(path_rgbs)
    for i in range(m):
        path_rgb = path_rgbs[i]
        path_mask = path_masks[i]
        img_rgb = Image.open(path_rgb)
        img_rgb = img_rgb.filter(ImageFilter.GaussianBlur(radius=3))
        img_mask = Image.open(path_mask)
        arr_rgb = np.array(img_rgb)
        arr_mask = np.array(img_mask)
        mask_max_v = np.max(arr_mask)
        img_h, img_w = arr_mask.shape
        #print(i, 'mask_max_v', mask_max_v, np.unique(arr_mask), 'img_h, img_w', img_h, img_w)
        if np.min(arr_rgb) < 1:
            #print('rgb min value is', np.min(arr_rgb), np.max(arr_rgb), path_rgb)
            arr_rgb = np.maximum(arr_rgb, np.ones_like(arr_rgb))
            
        arr_mask_3d = np.expand_dims((arr_mask > 0).astype(np.uint8), -1)
        arr_rgb_masked = arr_rgb * arr_mask_3d
        for j in range(1, 1 + mask_max_v):
            ped_mask_j = arr_mask == j
            ped_mask_sum_0 = np.any(ped_mask_j, axis=0)
            ped_mask_sum_1 = np.any(ped_mask_j, axis=1)
            ped_mask_sum_0 = ped_mask_sum_0.astype(np.int)
            ped_mask_sum_1 = ped_mask_sum_1.astype(np.int)
           
            x0 = np.argmax(ped_mask_sum_0)
            x1 = img_w - np.argmax(ped_mask_sum_0[::-1])
            y0 = np.argmax(ped_mask_sum_1)
            y1 = img_h - np.argmax(ped_mask_sum_1[::-1])            
            #print(ped_mask_j.shape, np.sum(ped_mask_j), x0, x1, y0, y1)            
            arr_ped_j = arr_rgb_masked[y0:y1, x0:x1]
            arr_ped_list.append(arr_ped_j)
        
        if i%100==0:print(i, len(arr_ped_list))
    return arr_ped_list


# In[5]:


def append_pedestrin_to_img(ped_arrs, img_bg, range_width = (14, 30)):
    m = len(ped_arrs)
    canvas = np.array(img_bg)
    mask_arr = np.zeros_like(canvas[:, :, 0])
    labels = []
    img_h, img_w, img_c = canvas.shape
    top_offset_y0 = int(img_h * 0.05) 
    top_offset_y1 = int(img_h * 0.9) 
    top_offset_x0 = int(img_w * 0.05) 
    top_offset_x1 = int(img_w * 0.9)          
    
    last_i = 0
    def append_object(offset_x0, offset_x1, offset_y0, offset_y1, cls, last_i, is_crop=False):
        tx = 0
        ty = 0
        for i in range(last_i, m):
            j = np.random.randint(m)
            ped_arr = ped_arrs[j]
            #print(i, ped_arr.shape)
            ped_h = ped_arr.shape[0]
            ped_h_a = 1.0
            if is_crop:
                ped_h_a = 1.0
                ped_arr = ped_arr[:ped_h//2]
                
            ped_img = Image.fromarray(ped_arr)
            ped_w = np.random.randint(range_width[0], range_width[1])
            offset_y_ratio = 1.0*(offset_y0+ty)/img_h
            ped_w = range_width[0] * (1-offset_y_ratio) + range_width[1] * offset_y_ratio
            ped_w = int(ped_w)
            ped_h = int(ped_w * ped_h_a)

            ped_img = ped_img.resize((ped_w, ped_h))
            #ped_img = ped_img.filter(ImageFilter.GaussianBlur(radius=3))
            ped_arr = np.array(ped_img)
            y0 = offset_y0 + ty + np.random.randint(10)
            y1 = y0 + ped_h
            x0 = offset_x0 + tx
            x1 = x0 + ped_w
            
            water_crop = canvas[y0:y1, x0:x1]
            ped_mask = (ped_arr > 1).astype(np.uint8)
            #print('ped_arr', ped_arr.shape, 'water_crop', water_crop.shape)
            try:
                ped_arr = ped_arr + water_crop * (1 - ped_mask)
            except:
                print('ped_arr', ped_arr.shape, 'water_crop', water_crop.shape)
                break
                
            canvas[y0:y1, x0:x1] = ped_arr
            mask_arr[y0:y1, x0:x1] = ped_mask[:, :, 0]            
            margin = 2
            bx0 = 1.0 * (x0 - margin)/img_w
            by0 = 1.0 * (y0 - margin)/img_h
            bx1 = 1.0 * (x1 + margin)/img_w
            by1 = 1.0 * (y1 + margin)/img_h
            cbbox = np.array([bx0, by0, bx1, by1, 1, 0, 0])
            labels.append(cbbox)

            tx += (x1 - x0) * 2
            if tx > offset_x1:
                tx = 0
                ty += int(ped_h*1.1)

            if offset_y0 + ty > offset_y1*0.99:
                last_i = i
                #print(m, last_i, 'break')
                break
        return last_i
    
    last_i = append_object(top_offset_x0, top_offset_x1, top_offset_y0, top_offset_y1, 3, last_i)    
    
    labels = np.stack(labels)
    mask_bool = mask_arr.astype(np.bool)
    return canvas, mask_bool, labels


# In[6]:


def synthetic_data(ped_list, list_image_bg, start_w, end_w):
    X_mask_img = []
    Y_mask_binary = []
    Y_mask_box = []
    
    for image_bg_path in list_image_bg:
        if len(X_mask_img)%10==0:
            print('synthetic', len(X_mask_img), len(list_image_bg))
        img = Image.open(image_bg_path)
        img_arr = np.array(img)
        img_with_ped, img_mask, mask_cbbox = append_pedestrin_to_img(ped_list, img_arr, range_width = (start_w, end_w))
        X_mask_img.append(img_with_ped)
        Y_mask_binary.append(img_mask)
        Y_mask_box.append(mask_cbbox)    
        
    return X_mask_img, Y_mask_binary, Y_mask_box


# In[7]:


def penn_fudan_pedestrian(folder_ped, folder_image_bg, stride=1, start_w=20, end_w=270):
        
    print(os.path.isdir(folder_ped), os.path.isdir(folder_ped+folder_ped_rgb), os.path.isdir(folder_ped+folder_ped_mask))
    
    print(os.path.isdir(folder_image_bg))
    path_rgbs = glob(folder_ped + folder_ped_rgb + '*.*')
    len(path_rgbs), path_rgbs[0], path_rgbs[-1]
    path_masks = []
    for path_rgb in path_rgbs:
        path_masks.append(convert_path_rgb_to_path_mask(folder_ped, path_rgb))
    print(len(path_rgbs), len(path_masks))
    plt.imshow(Image.open(path_masks[0]))
    plt.show()
    
    ped_list = get_masked_pedestrian(path_rgbs, path_masks)
    print('len', len(ped_list))
    p = plt.imshow(ped_list[2]) 
    plt.show()
    
    list_image_bg = glob(folder_image_bg + '/*')
    print('bg', len(list_image_bg))
    list_image_bg = list_image_bg[::stride]
    
    X_mask_img, Y_mask_binary, Y_mask_box = synthetic_data(ped_list, list_image_bg, start_w, end_w)
    return X_mask_img, Y_mask_binary, Y_mask_box


# folder_image_bg = '/media/mvlab/469B5B3C650FBA77/data/cctv test/images_bg'
# folder_ped = '/media/mvlab/469B5B3C650FBA77/data/PennFudanPed/'
# X_mask_img, Y_mask_binary, Y_mask_box = penn_fudan_pedestrian(folder_ped, folder_image_bg, 50)

# len(X_mask_img) ,len(Y_mask_binary), len(Y_mask_box)

# plt.imshow(X_mask_img[0])

# plt.imshow(Y_mask_binary[-1], cmap='gray')

# In[ ]:





# In[ ]:




