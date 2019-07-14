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


targetfolder = str(pathlib.Path.cwd().parents[0])
#targetfolder = '/media/daniel/Seagate Backup Plus Drive/A83'
#targetfolder = '/media/daniel/Windows1/Users/dnabu/Desktop/ResearchYogaWindows/DataJ/A/A64_MeanGirl2/A64_data'
print(targetfolder)
#outputfolder = '/media/daniel/Seagate Backup Plus Drive/A83/A83_data3'
#outputfolder = None

#if not outputfolder is desginated targetfolder and outputfolder are the same
if outputfolder == None:
    outputfolder = targetfolder

#get all files in targetfolder
files = pldb.getDirContents(targetfolder)
print(files)
#
files = files[files['Directory'] == False]
files = files[files['File_Name'].str.contains('_gpur2')]
files = files[files['File_Name'].str.contains('Mask.hdf5')]
print(files['File_Name'])
print(files['Parent'].values)
####WARNINGING WARNING WARNING
#deleting files that are not in the colored file
'''
for row, dseries in files.iterrows():
    Path(dseries['Full_Path']).unlink()
'''
    
#get all files in targetfolder to transfer to folders containing timeseries data
files = pldb.getDirContents(targetfolder)
print(files)

 
#now copy the mask files into any folders with a similar name
maskfiles = files[files['File_Name'].str.contains('Mask.hdf5')]
outputfolderdir = pldb.getDirContents(outputfolder)
outputfolderdir = outputfolderdir[outputfolderdir['Directory'] == True]

#get the minimal string (BaseName) that matches mask files to the timeseries folder
''' for troubleshooting
    row =0
    dseries = maskfiles.iloc[0]
'''
import re
maskfiles['BaseName'] =''
for row, dseries in maskfiles.iterrows():
    index = [m.start() for m in re.finditer('_', dseries['File_Name'])]
    maskfiles['BaseName'].loc[row] = dseries['File_Name'][:index[1]]
print( maskfiles['BaseName'])
    
    
    
for row, dseries in maskfiles.iterrows():
    string1 = dseries['BaseName']
    print(string1)
    targetdirectories = outputfolderdir[outputfolderdir['File_Name'].str.contains(string1)]
    targetdirectories['output'] = ''
    for row2, dseries2 in targetdirectories.iterrows():
        #row2 = targetdirectories.index[1]
        #dseries2 = targetdirectories.loc[row2]
        targetdirectories['output'].loc[row2] = str(Path(dseries2['Full_Path']) / dseries['File_Name'])
        if not Path(targetdirectories['output'].loc[row2]).is_file():
            shutil.copy(dseries['Full_Path'], targetdirectories['output'].loc[row2])
