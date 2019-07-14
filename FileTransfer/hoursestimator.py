#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:18:38 2018

@author: daniel
"""
from pathlib import Path 
import pathlibDB
import pandas as pd
import numpy as np
import datetime

targetdir = '/home/daniel/Desktop/ResearchUbuntuYoga720/A61_FB_PAM_Keleman/Data'
files = pathlibDB.getDirContents(targetdir)

mask = files[files['File_Name'].str.contains('Mask.hdf5')]

timearray = mask['Modified'].values
diffarray = np.diff(timearray[1:])

mask['Time2'] = ''
for row, dseries in mask.iterrows():
    mask['Time2'].loc[row] = datetime.datetime.fromtimestamp(dseries['Modified'])
mask.sort_values(['Modified'], inplace = True)
