#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:33:30 2018

@author: daniel
"""
import pandas as pd
import pathlib as pl
import shutil
import numpy as np

maskdir = str(pl.Path.cwd())
print(maskdir)
targetdir = '/run/user/1000/gvfs/smb-share:server=busheyd-ws1,share=dr1tba/JData/A/A68_GCaMP7/A68_Data'
outputdata = str(pl.Path(maskdir) / 'transfer.xlsx')

#get list of maskdir
subdirectories = [str(c) for c in pl.Path(maskdir).glob('*/') if c.is_dir()]
index = [pl.Path(c).parts[-1][:-7] for c in subdirectories]
data = pd.DataFrame(index = index, columns = ['Path'], data = subdirectories)
print(data)

#get list of targetdir
targetlist = [str(c) for c in pl.Path(targetdir).glob('**/*') if c.is_dir()]
shorttargetlist = [str(pl.Path(i).parts[-1]) for i in targetlist]
shorttargetlist = pd.DataFrame(data=shorttargetlist)



#find files in targetdir and place mask files
data['Target'] = ''
for row, drow in data.iterrows():
    for tar in targetlist:
        tar1 = pl.Path(tar).parts[-1]
        
        if tar1 == row:
            data['Target'].loc[row] = tar
print(data['Target'])
data.to_excel(outputdata)

#find the mask.hdf5 file in each path
data['Mask_File'] = ''
for row, drow in data.iterrows():
    maskfile = [str(c) for c in pl.Path(drow['Path']).glob('*Mask.hdf5')]
    if len(maskfile) >0:
        data['Mask_File'].loc[row] = maskfile[0]
    else:
        data['Mask_File'].loc[row] = np.NaN
#remove rows that do not have a mask file
data = data.dropna(axis='index', subset=['Mask_File'])
#create path for copied mask file
data['Copy_Mask'] = ''
for row, drow in data.iterrows():
    data['Copy_Mask'].loc[row] = str(pl.Path(drow['Target']) / pl.Path(drow['Mask_File']).parts[-1])
data.to_excel(outputdata)

for row, drow in data.iterrows():
    if not pl.Path(drow['Copy_Mask']).is_file():
        shutil.copyfile(drow['Mask_File'], drow['Copy_Mask'])

#check copy
data['Copied'] = ''
for row, drow in data.iterrows():
    data['Copied'].loc[row] = pl.Path(drow['Copy_Mask']).is_file()
data.to_excel(outputdata)
print(outputdata)
