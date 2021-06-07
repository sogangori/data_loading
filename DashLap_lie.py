#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import zipfile

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import json
from glob import glob
import xml.etree.ElementTree as ET


# In[2]:


def get_filename_dict(list_jpg):
    dict_dashlap_img = dict()
    for path_jpg in list_jpg:
        img_file_name = path_jpg.split('/')[-1]

        dict_dashlap_img[img_file_name] = path_jpg 
    return dict_dashlap_img


# In[3]:


def exist_folder(folder):
    is_exist = os.path.isdir(folder)
    if is_exist:
        print('exist folder', folder)
    else:
        print('not exist folder', folder)
    return is_exist        


# In[4]:


def parse_xml(path_xml):
    tree = ET.parse(path_xml)
    root = tree.getroot()
    filename = root.findtext('filename')
    
    list_bndbox = []
    for neighbor in root.iter('bndbox'):
        #print(neighbor, neighbor.text)
        for c in list(neighbor):
            #print(c.tag, c.text)
            list_bndbox.append(c.text)
            
    bbox = np.array(list_bndbox).reshape((-1, 4)).astype(int)        
    return bbox, filename


# In[5]:


def convert_xml_to_csv(list_xml, list_jpg, stride=1):
    
    dict_dashlap_img = get_filename_dict(list_jpg)
    print('dict_dashlap_img', len(dict_dashlap_img))
    df = pd.DataFrame()
    
    for i in range(0, len(list_xml), stride):
        path_xml = list_xml[i]
        if i % 1000 == 0:
            print('path_xml',i, path_xml)

        bboxes, filename = parse_xml(path_xml)
        
        if filename in dict_dashlap_img.keys():
            if len(bboxes) > 0:
                full_path_img = dict_dashlap_img[filename]   
                for box in bboxes:
                    x, y, x1, y1 = box
                    d = {'filename':full_path_img, 'x':x, 'y':y, 'w':x1-x, 'h':y1-y, 'cls':'person'}                                
                    df = df.append(d, ignore_index=True)
            else:
                print('not exist', filename)
        else:
            print('no key', filename)

    return df


# In[6]:


def DashLap_lie(folder, stride=1):
    
    if exist_folder(folder):
        
        list_xml = glob(folder + '*anno/*.xml')
        list_jpg = glob(folder + '*/*.jpg')
        print(len(list_xml), len(list_jpg))        
        
        df = convert_xml_to_csv(list_xml, list_jpg, stride=stride)
        return df


# folder = '/media/mvlab/469B5B3C650FBA77/data/dashLab/Detector_dataset/'
# DashLap_lie(folder, stride=1000)

# In[ ]:




