# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import osDB
import os
import tifffile
from skimage.draw import polygon
import skimage.measure
import ccModules
import parallelDB
import ipyparallel as ipp
#parallelDB.startLocalWait(15, 20) #(=number of engines, #number of attempts to connect to engines)
rc = ipp.Client()
dview = rc[:]

#find files
common_end = '_Mask.npy'
#targetfolder = r'/media/daniel/Seagate Backup Plus Drive1/JData/A/A08_19F01-LexA_CaMPARI/A08_Data'
targetfolder = '/data/JData/A/A12_19F01-Gal4_CaMPARI/A12 Data'
folderlist = [] #store folders that contain mask files
#get list of all folders
for root, dirs, files in os.walk(targetfolder):
    folderlist.append(root)

folders = []
for path in folderlist:
    files, index = osDB.getFileContString(path, common_end)
    if len(files) > 0:
        folders.append(path)
    
#columns in mask file
#ccModules.translateMatplotlibROI_to_PyQtFormat(folderlist['path'].iloc[0])
out = dview.map(ccModules.translateMatplotlibROI_to_PyQtFormat,  folders)
out.wait_interactive()

    

        
        
