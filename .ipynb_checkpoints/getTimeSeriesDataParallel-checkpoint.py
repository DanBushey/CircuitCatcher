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
import ccModules
#import ipyparallel as ipp
#import parallelDB
#parallelDB.startLocalWait(6, 20) #(=number of engines, #number of attempts to connect to engines)
#rc = ipp.Client()
#dview = rc[:]






#search directory tree for all files containing numpy
targetfolder = r'/home/daniel/Desktop/ResearchUbuntuYoga720/A32_Saline/Ca'
json_file_name = 'dr.3x200.json'
#targetfolder = r"/groups/flyfuncconn/flyfuncconn/DanB/CircuitCatcher/jl_samplefiles/"
directories = [targetfolder]
files = []
for path, subdirs, files1 in os.walk(targetfolder):
    for cd in subdirs:
        print(cd)
        directories.append(os.path.join(path, cd))
    for cfile in files1:
        print(cfile)
        files.append(os.path.join(path, cfile))
        
#create a list of files with mask.npy in the filename
maskFileList = [cfile for cfile in files if 'Mask.hdf5' in cfile]

#find masks to corresponding consecutive acquisitions lacking mask
    #only make one mask and then reapply if the same image is being acquired
maskIndex = []
directoriesWithTargeNameAllMask = []
maskFilesMatchingDirectories = []
for cmask in maskFileList:
    print('Starting', cmask)
    #generate a string to search for other folders that mask can be applied
    #find the original image folder containing time series data
    #folder containing time series data should have the same name as gpu file
    path1, GPUfile = os.path.split(cmask)
    base2, basefolder = os.path.split(path1)
    index2 = basefolder.find('_gpuregression')
    basefolder = basefolder[:index2] 
    maskIndex.append(GPUfile)
    #GPUfile = GPUfile[:-5]
    #search for target directories with similar name
    #directoires should have either HDF5 or tif files
    directoriesWithTargetName = []
    for cd in directories:
        if basefolder in cd:
            if 'gpuregression' not in cd:
                HDF5file, index = ccModules.getFileContString(cd, '.hdf5')
                for file1 in HDF5file:
                    if file1[-10:-5].isdigit():
                        directoriesWithTargetName.append(cd)
            if not cd in directoriesWithTargetName:
                tiffiles, index = ccModules.getFileContString(cd, '.tif')
                if len(tiffiles) > 199:
                    directoriesWithTargetName.append(cd)  
    directoriesWithTargeNameAllMask.extend(directoriesWithTargetName)
    maskfiles = []  
    for i in range(len(directoriesWithTargetName)):
          maskfiles.append(cmask)
    maskFilesMatchingDirectories.extend(maskfiles)            
summaryFrame = pd.DataFrame(data={'Mask_Files' : maskFilesMatchingDirectories, 'Target_directories': directoriesWithTargeNameAllMask})

#copy mask file data into corresponding folders missing maskdata
for i, dseries in summaryFrame.iterrows():
    #make sure that a Mask.hdf5 does not already exist
    Maskfiles, index = ccModules.getFileContString(dseries['Target_directories'], 'Mask.hdf5')
    if len(Maskfiles) ==0:
        path, file = os.path.split(dseries['Mask_Files'])
        shutil.copy(dseries['Mask_Files'], os.path.join(dseries['Target_directories'], file))
        
    

#find json files for tif image stacks in mask folder that contains the registration information for individual stacks
#search for json file in the same folder that contains the mask data
summaryFrame['Registration_JsonFile'] = ''
for row in summaryFrame.index:
    tiffiles, index = ccModules.getFileContString( summaryFrame['Target_directories'].loc[row], '.tif')   
    if len(tiffiles) > 0:
        path1, file1 = os.path.split(summaryFrame['Mask_Files'].loc[row])
        jsonfiles, index = ccModules.getFileContString( path1, json_file_name)   
        if len(jsonfiles) > 0:
            summaryFrame['Registration_JsonFile'].loc[row] = os.path.join(path1, jsonfiles.values[0])
        else:
            summaryFrame['Registration_JsonFile'].loc[row] = None
    else:
        summaryFrame['Registration_JsonFile'].loc[row] = None
    

summaryFrame.to_excel(os.path.join(targetfolder, 'GenerateROIdata.xlsx'))

#generate data writing into individual directories
output1 = dview.map(ccModules.getROIdata, summaryFrame['Mask_Files' ].values, summaryFrame['Target_directories' ].values, summaryFrame['Registration_JsonFile' ].values )
output1.wait_interactive()
#row = 10
#out = ccModules.getROIdata( summaryFrame['Mask_Files' ].values[row], summaryFrame['Target_directories' ].values[row], summaryFrame['Registration_JsonFile' ].values[row]) 
parallelDB.stopLocal() 

