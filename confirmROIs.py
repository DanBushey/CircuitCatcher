#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:16:18 2018
Confirm the rois are present in for each image. Date is taken from the *Mask.hdf files.
@author: daniel
"""
import pathlib as pl
import pandas as pd


pullDownMenu1 = {'g5L': [0.2, 0.15, 0.5] ,
                 'g5R': [0.2, 0.15, 0.5],
                 'g4L': [0.4, 0.5, 0.5],
                 'g4R': [0.4, 0.5, 0.5],
                 'g3L': [0.6, 0.2, 0],
                 'g3R': [0.6, 0.2, 0],
                 

                 'a1L': [0.3, 0.2, 0.5],
                 'a1R': [0.3, 0.2, 0.5],

                 'b1L': [0.4, 0.2, 0.5],
                 'b1R': [0.4, 0.2, 0.5],
                 
                 'b2L': [0.8, 0.5, 0],
                 'b2R': [0.8, 0.5, 0.5],

                 'bp1L': [0.4, 0.15, 0.5],
                 'bp1R': [0.4, 0.15, 0.5],
    
                 'bp2aL': [0.8, 0.5, 0],
                 'bp2aR': [0.8, 0.5, 0],
                 'bp2mpL': [0.8, 0.5, 0],
                 'bp2mpR': [0.8, 0.5, 0],
  
                 'Background' : [0.9, 0.9, 0.5]}

targetdir = '/home/daniel/Desktop/ResearchUbuntuYoga720/A61_FB_PAM_Keleman/Data'
outputdata = str(pl.Path(targetdir) / 'CheckingMaskData.xlsx')

subdirectories = [str(c) for c in pl.Path(targetdir).glob('*/') if c.is_dir()]
print(subdirectories)
#create index for pandas dataframe
index = [pl.Path(c).parts[-1] for c in subdirectories]
data = pd.DataFrame(index = index, columns = ['Path'], data = subdirectories)

#find the mask.hdf5 file in each path
data['Mask_File'] = ''
for row, drow in data.iterrows():
    data['Mask_File'].loc[row] = [str(c) for c in pl.Path(drow['Path']).glob('*Mask.hdf5')]
print(data['Mask_File'].iloc[0])
data.to_excel(outputdata)

#make columns for roi count
for roi in pullDownMenu1.keys():
    data[roi] = ''
    
#check and make sure all rois are present
for row, drow in data.iterrows():
    if len(drow['Mask_File']) != 0:
        rois = pd.read_hdf(drow['Mask_File'][0]) #get mask data
        for roi in pullDownMenu1.keys():
            data[roi].loc[row] = rois['Name'].str.count(roi).sum()
data.to_excel(outputdata)
