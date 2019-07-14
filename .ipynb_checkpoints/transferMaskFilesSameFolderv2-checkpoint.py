'''
Part of circuit catcher
1. Delete *Mask.hdf files in folders containing the tif time series data
2. Find the *Mask.hdf files in the 2color folders
3. Transfer *Mask.hdf to the time series folders


'''

import pandas as pd
import pathlibDB as pldb
from pathlib import Path
import ipyparallel as ipp
import ccModules as cc
import pathlib
import shutil


#targetfolder = str(pathlib.Path.cwd().parents[0])
targetfolder = '/media/daniel/Seagate Backup Plus Drive/A99/A99_Data2'

print(targetfolder)
#outputfolder = '/media/daniel/Seagate Backup Plus Drive/A83/A83_data3'
#outputfolder = None

#if not outputfolder is desginated targetfolder and outputfolder are the same
if not 'outputfolder' in locals():
    outputfolder = targetfolder

#get all files in targetfolder
files = pldb.getDirContents(targetfolder)
print(files)
 
#now copy the mask files into any folders with a similar name
maskfiles = files[files['File_Name'].str.contains('Mask.hdf5')]
maskfiles = maskfiles[maskfiles['File_Name'].str.contains('_2color_')]
maskfiles.drop_duplicates(subset = 'File_Name', keep=False, inplace=True)

outputfolderdir = pldb.getDirContents(outputfolder)
outputfolderdir = outputfolderdir[outputfolderdir['Directory'] == True]

#get the minimal string (BaseName) that matches mask files to the timeseries folder
''' for troubleshooting
    row =0
    dseries = maskfiles.iloc[0]
'''
maskfiles['BaseName'] =''
for row, dseries in maskfiles.iterrows():
    maskfiles['BaseName'].loc[row] = dseries['File_Name'][:-22]
print( maskfiles['BaseName'])
    
row = 524
dseries = maskfiles.loc[row]
row2 = 422
dseries2 = outputfolderdir.loc[422]
for row, dseries in maskfiles.iterrows():
    string1 = dseries['BaseName']
    print(string1)
    targetdirectories = outputfolderdir[outputfolderdir['File_Name'].str.contains(string1)]
    targetdirectories['output'] = ''
    for row2, dseries2 in targetdirectories.iterrows():
        #test to determine if mask file already exists - if it does exist do not copy maskfile
        maskfiles_present = [str(file) for file in Path(dseries2['Full_Path']).glob('*Mask.hdf5')]
        targetdirectories['output'].loc[row2] = str(Path(dseries2['Full_Path']) / dseries['File_Name'])
        if len(maskfiles_present) == 0:
            shutil.copy(dseries['Full_Path'], targetdirectories['output'].loc[row2])
print('Finished')
