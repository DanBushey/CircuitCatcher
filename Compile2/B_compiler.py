#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:51:16 2017

@author: daniel
#combine multiple hdf5 files with  the rois results into one file
#folder should contain multiple hdf5 files
#search folder getting names of hdf5 files
#load and append each file into one list
#save the list in a file
#save a list of files into csv
"""
from projectData import *  #load data specific to the project (ie folder location)

import osDB
import pandas as pd
import os


files, index = osDB.getFileContString(path1, '.hdf5')

exceldata = pd.read_hdf(os.path.join(path1, files.values[0]), 'data') 
print(len(exceldata))
for file in files[1:]:
    newdata = pd.read_hdf(os.path.join(path1, file), 'data')
    print(len(newdata))
    exceldata = exceldata.append(newdata)
print('Total length: ', len(exceldata))

    
newfolder = os.path.join(path1, 'Analysis')
if not os.path.isdir(newfolder):
    os.mkdir(newfolder)

exceldata.reset_index(inplace = True)
exceldata.to_hdf(os.path.join(newfolder, 'Compiled_data_All.hdf5'), 'data')

drop = ['intensity_data', 'voltage']
exceldata.drop(drop, axis=1).to_excel(os.path.join(path1, 'Summary_Compiled_All.xlsx'))
