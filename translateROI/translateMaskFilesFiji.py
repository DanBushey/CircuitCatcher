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
import ipyparallel as ipp
#parallelDB.startLocalWait(15, 20) #(=number of engines, #number of attempts to connect to engines)
rc = ipp.Client()
dview = rc[:]

#find files zip files containing the rois 
common_end = '.zip'
#targetfolder = '/run/media/busheyd/WD2/R3_vt060202_PAM_ltm_superROI/ROIs_20171220'
targetfolder = '/media/daniel/WD2/R3_vt060202_PAM_ltm_superROI/ROIs_20171220'
folderlist = [] #store folders that contain mask files
files, index = osDB.getFileContString(targetfolder, common_end)

#generate a Mask.hdf5 file for each roi
#go through each roi and translate into new row for mask file

'''
cfile = files.iloc[0]
targetfile = os.path.join(targetfolder, cfile)
out = ccModules.translateImageJROIs_to_cc(targetfile)
'''
filelist = [os.path.join(targetfolder, cfile) for cfile in files]

for pathfile in filelist:
    out = ccModules.translateImageJROIs_to_cc(pathfile)


out = dview.map(ccModules.translateImageJROIs_to_cc,  filelist)
out.wait_interactive()
out.get()

    
