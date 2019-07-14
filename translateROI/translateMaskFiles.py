 br# -*- coding: utf-8 -*-
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
#example file
ccmaskfile = r'/media/daniel/Seagate Backup Plus Drive1/HD4A-JData/A32_Saline/Ca2/20170905/20170905-A12-3-flyb_00003/20170905-A12-3-flyb_00003Mask.hdf5'
maskdata = pd.read_hdf(ccmaskfile) 
maskdata2 = np.load(ccmaskfile)

#find files
common_end = '_Mask.npy'
rois = ['Body', 'M1', 'M4', 'M8-10']
targetfolder = r'/media/daniel/Seagate Backup Plus Drive1/JData/A/A08_19F01-LexA_CaMPARI/A08_Data/2016-10-07'
folders = [] #store folders that contain mask files
folderlist = osDB.getFolders(targetfolder)
for index1, path in folderlist.iterrows():
    files, index = osDB.getFileContString(path['path'], common_end)
    if len(files) > 0:
        folders.append(path['path'])
    
#columns in mask file
fileaddon = 'Mask.hdf5'
columns = ['Name', 'Color', 'Type', 'Z:XY', 'mask_index', 'image_shape','image_file']
colors = {'Body': [0,1,0], 'M1':[1,0, 1], 'M4': [1, 1, 0], 'M8-10': [1, 0, 0]}

def translateMatplotlibROI_to_PyQtFormat(cfolder)
    #translate the old ROIs into the new format
    #cfolder = /path/ to directory with old rois
    #creates and new file with tif_name + Mask.hdf5 that can be loaded into circuit catcher
    files, index = osDB.getFileContString(cfolder, common_end)
    #create the pandas data frame to contain data
    cmaskdata = pd.DataFrame(index = range(len(files)), columns = columns)
    #fill in other columns
    #find the tif file
    tif_files, index = osDB.getFileContString(cfolder, '.tif')
    cmaskdata['image_file'] = os.path.join(cfolder, tif_files.values[0])
    img = tifffile.imread(os.path.join(cfolder, tif_files.values[0]))
    ZXY = []
    for i, cfile in enumerate(files):
        XY = np.load(os.path.join(cfolder, cfile))
        mask = np.zeros([img.shape[1], img.shape[2]] ,dtype = np.bool)
        mask[XY[0], XY[1]] =1
        mask = mask.T
        contours = skimage.measure.find_contours(mask, 0.8)[0]
        #have to down sample contours because there are too many points
        downsamplerangecontours = range(0, contours.shape[0], 15)
        contours = contours[downsamplerangecontours, :]
        print(contours.shape)
        ZXY.append( {0 : contours.tolist()})
        cmaskdata['Name'].loc[i] = cfile[:-9]
        cmaskdata['mask_index'].loc[i] = np.where(mask.flatten())
        cmaskdata['Color'].loc[i] = colors[cfile[:-9]]
    cmaskdata['Z:XY'] = ZXY
    cmaskdata['image_file'] = os.path.join(cfolder, files.values[0])
    cmaskdata['image_shape'] = pd.Series( [(img.shape[0], 1, img.shape[1],img.shape[2], 3) for x in range(len(cmaskdata))])
    cmaskdata['Type'] = 'polyArea'
    pathfile = os.path.join(cfolder, tif_files.values[0])[:-4] + fileaddon
    cmaskdata.to_hdf(pathfile, 'roi')

    

        
        