import os
import numpy as np
import pandas as pd
import sys
import shutil
import re
import osDB
#folderModules = '/home/daniel/Desktop/ResearchUbuntuYoga720/A30_FastROI/CircuitCatcher'
#sys.path.append(folderModules)
import ccModules
import ipyparallel as ipp
#import parallelDB
#parallelDB.startLocalWait(13, 20) #(=number of engines, #number of attempts to connect to engines)
rc = ipp.Client()
dview = rc[:]

#search directory tree for all files containing numpy
#targetfolder = r'/media/daniel/WD2/R3_vt060202_PAM_ltm_superROI'
targetfolder = r'/run/media/busheyd/WD2/R3_vt060202_PAM_ltm_superROI'

outputcsv = os.path.join(targetfolder, 'GenerateROIdata.xlsx')
json_file_name = 'dr.3x200.json'
directories = [targetfolder]
files = []
for path, subdirs, files1 in os.walk(targetfolder):
    for cd in subdirs:
        #print(cd)
        directories.append(os.path.join(path, cd))
    for cfile in files1:
        #print(cfile)
        files.append(os.path.join(path, cfile))
    
#search for directories that correspond to acquisitions - have five digits at end
pattern_search = re.compile('_\d{5,5}')
targetdirectories = [folder for folder in directories if pattern_search.findall(folder[-12:])]
del targetdirectories[-1] #remove the folder containing the rois


#create columns for a dataframe from targetdirectories
index1 = range(len(targetdirectories))
folder_names = [os.path.split(folder)[1] for folder in targetdirectories ]

#create data frame to hold information
folderlist = pd.DataFrame(index = index1, columns=['Directory'], data=targetdirectories)
folderlist['Folder_Names'] = folder_names
#folderlist.sort_index(inplace = True)
folderlist.to_excel(outputcsv)

#get list of masks
maskfolder = '/run/media/busheyd/WD2/R3_vt060202_PAM_ltm_superROI/ROIs_20171220'
maskfiles, index = osDB.getFileContString(maskfolder, 'Mask.hdf5')

#match folder to mask
#folderlist['Matching_Mask'] = ''
folderlist['Mask_files'] = ''
for cmaskfile in maskfiles:
    splits1 = cmaskfile[:-9].split('_')
    
    currentIndex=  folderlist['Folder_Names'].str.contains(splits1[0])
    for csplit in splits1[1:]:
        currentIndex = np.logical_and(currentIndex, folderlist['Folder_Names'].str.contains(csplit))
    
    for row in np.where(currentIndex.values)[0]:
        folderlist['Mask_files'].iloc[row] = os.path.join(maskfolder, cmaskfile)
        #folderlist['Mask_files'].iloc[row] = os.path.join( folderlist['Directory'].iloc[row], cmaskfile )
folderlist.to_excel(outputcsv)
    
#search for image file
folderlist['Image_files'] = ''
for row, dseries in folderlist.iterrows():
    HDF5files, index = ccModules.getFileContString(dseries['Directory'], '.hdf5')
    if len(HDF5files) > 0:
        for file1 in HDF5files:
            if file1[-10:-5].isdigit():
                folderlist['Image_files'].loc[row] = [file1]
for row, dseries in folderlist.iterrows():
    if len(folderlist['Image_files'].loc[row]) == 0:
        tiffiles, index = ccModules.getFileContString(dseries['Directory'], '.tif')
        if len(tiffiles) >= 1:
            ctiffile = []
            for tiffile in tiffiles:
                if tiffile[-9:-6].isdigit():
                    ctiffile.append(tiffile)
                folderlist['Image_files'].loc[row] = ctiffile

folderlist['Registration_JsonFile'] = ''
for row, dseries in folderlist.iterrows():
    jsonfiles, index = ccModules.getFileContString( dseries['Directory'], json_file_name)   
    if len(jsonfiles) > 0:
        folderlist['Registration_JsonFile'].loc[row] = os.path.join(path1, jsonfiles.values[0])
    else:
        folderlist['Registration_JsonFile'].loc[row] = None


folderlist.to_excel(outputcsv)
folderlist['Mask_files'].replace('', np.nan, inplace=True)
folderlist.dropna(subset = ['Mask_files'], inplace = True)

for row, dseries in folderlist.iterrows():
    roifile, index = ccModules.getFileContString(dseries['Directory'], 'ROI.jpeg')
    if len(roifile) > 0:
        folderlist.drop(row, inplace = True)

folderlist.to_excel(outputcsv)

'''
row = 0
maskfile = folderlist['Mask_files' ].iloc[row]
targetdirectory = folderlist['Directory' ].iloc[row]
imagefile = folderlist['Image_files' ].iloc[row]
jsonfile = folderlist['Registration_JsonFile' ].iloc[row]


for row in range(len(folderlist)):
    out = ccModules.getROIdata( folderlist['Mask_files' ].iloc[row], folderlist['Directory' ].iloc[row], folderlist['Image_files' ].iloc[row], folderlist['Registration_JsonFile' ].iloc[row]) 
'''
output1 = dview.map(ccModules.getROIdata, folderlist['Mask_files' ].values, folderlist['Directory' ].values, folderlist['Image_files' ].values, folderlist['Registration_JsonFile' ].values )
output1.wait_interactive()









