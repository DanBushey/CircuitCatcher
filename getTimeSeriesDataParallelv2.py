'''
Created on May 16, 2017

@author: Surf32
Script searches for numpy files containing roi data.
Takes the roi data and get intensity data from timeseries data with the same name.
Assumes that Mask.hdf5 only occurs once per brain.

'''
import sys
sys.path.append('/home/daniel/Desktop/ResearchUbuntuYoga720/A30_FastROI/CircuitCatcher2')
import os
import numpy as np
import pandas as pd
import sys
import shutil
import re
#folderModules = '/home/daniel/Desktop/ResearchUbuntuYoga720/A30_FastROI/CircuitCatcher'
#sys.path.append(folderModules)
import ccModules
import pathlib
import pathlibDB as pbDB
from pathlib import Path
#parallelDB.startLocalWait(13, 20) #(=number of engines, #number of attempts to connect to engines)


#search directory tree for all files containing numpy
'''
The pandas read_hdf5 cannot read of smb connected drive. Use a direct path (no smb) on sl7 if working files from nearline
'''
targetfolder = pathlib.Path.cwd().parents[0]


#targetfolder = '/media/daniel/Seagate5TB/A83/A83_data4/20190311'
print(targetfolder)
#get full directory
files = pbDB.getDirContents(targetfolder)
maskfiles = files[files['File_Name'].str.contains('Mask.hdf5')]
maskfiles = maskfiles[maskfiles['Full_Path'].str.contains('stim')]
print(maskfiles)

#search for image file
image_files = []
#dseries = maskfiles.loc[8]
for row, dseries in maskfiles.iterrows():
    HDF5files = [cfile for cfile in Path(dseries['Parent']).glob('*TimeSeries.hdf5')]
    if len(HDF5files) > 0:
        image_files.append( [HDF5files[0].parts[-1]])
    if len(HDF5files) == 0:
        Tif_files = [cfile.parts[-1] for cfile in Path(dseries['Full_Path']).glob('*.tif')]
        image_files.append( Tif_files)
maskfiles['Image_File'] = image_files
print(maskfiles)
             
#find json files for tif image stacks in mask folder that contains the registration information for individual stacks
#search for json file in the same folder that contains the mask data
json_file_name = 'jj.json'
maskfiles['Registration_JsonFile'] = ''
for row, dseries in maskfiles.iterrows():
    jsonfiles, index = ccModules.getFileContString( dseries['Parent'], json_file_name)   
    if len(jsonfiles) > 0:
        maskfiles['Registration_JsonFile'].loc[row] = os.path.join(path1, jsonfiles.values[0])
    else:
        maskfiles['Registration_JsonFile'].loc[row] = None
print(maskfiles['Registration_JsonFile'].values)
        

#maskfiles.to_excel(outputcsv)

#remove directories that already have 'ROI.jpeg'
for row, dseries in maskfiles.iterrows():
    roifile = [cfile for cfile in Path(dseries['Parent']).glob('*ROI.jpeg')]
    if len(roifile) > 0:
        maskfiles.drop(row, inplace = True)
print(maskfiles)

'''
#searching a image_file column for str within lists
list1 = []
for row, dseries in folderlist.iterrows():
    for ci in dseries['Image_files']:
        if '.hdf5' in ci:
            list1.append([row, ci])
folderlist['Image_files'].loc[244] = ['20161130-A08-1-flya_00002.tif']
'''
#maskfiles.to_excel(outputcsv)

#generate data writing into individual directories
'''
for row in range(207, len(folderlist)): 
    print(row)
    
    print(folderlist['Mask_files' ].iloc[row])
    print(folderlist['Directory' ].iloc[row])
    print(folderlist['Image_files' ].iloc[row])
    maskfile = folderlist['Mask_files' ].iloc[row]
    targetdirectory = folderlist['Directory' ].iloc[row]
    imagefile = folderlist['Image_files' ].iloc[row]
    jsonfile = folderlist['Registration_JsonFile' ].iloc[row]
    out = ccModules.getROIdata( folderlist['Mask_files' ].iloc[row], folderlist['Directory' ].iloc[row], folderlist['Image_files' ].iloc[row], folderlist['Registration_JsonFile' ].iloc[row]) 
'''

'''
output1 = dview.map(ccModules.getROIdata, maskfiles['Full_Path' ].values, maskfiles['Parent' ].values, maskfiles['Image_files' ].values, maskfiles['Registration_JsonFile' ].values )
output1.wait_interactive()
output1.get()
'''
'''
row = 10
#out = ccModules.getROIdata( maskfiles['Mask_Files' ].values[row], maskfiles['Target_directories' ].values[row], maskfiles['Registration_JsonFile' ].values[row]) 
#row = '20170713-A12-1-flyf_00002'
'''
for row in range(0, len(maskfiles)):
    print(row)
    print(maskfiles['Full_Path' ].iloc[row])
    out = ccModules.getROIdata( maskfiles['Full_Path' ].iloc[row], maskfiles['Parent' ].iloc[row], maskfiles['Image_File'].iloc[row], maskfiles['Registration_JsonFile' ].iloc[row]) 
print('Finished Generating Intensity Data')

'''
row = 0
maskfile = maskfiles['Full_Path' ].iloc[row]
targetdirectory = maskfiles['Parent' ].iloc[row]
imagefile = maskfiles['Image_File'].iloc[row]
jsonfile = maskfiles['Registration_JsonFile' ].iloc[row]



'''
