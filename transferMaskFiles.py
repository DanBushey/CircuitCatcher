'''
Part of circuit catcher
1. Transfer folder containing HDF image files and mask files back to original data folder.
2. Copy *Mask.hdf5 files to folder with similar name

'''

import pandas as pd
import pathlibDB as pldb
from pathlib import Path
import ipyparallel as ipp
import ccModules as cc
import pathlib
import shutil
rc = ipp.Client()
dview = rc[:]

targetfolder = str(pathlib.Path.cwd().parents[0])
#targetfolder = '/media/daniel/Windows1/Users/dnabu/Desktop/ResearchYogaWindows/DataJ/A/A64_MeanGirl2/A64_data'
print(targetfolder)
outputfolder = '/media/daniel/Seagate Backup Plus Drive/A83/A83_data3'
#outputfolder = None

#if not outputfolder is desginated targetfolder and outputfolder are the same
if outputfolder == None:
    outputfolder = targetfolder

#get all files in targetfolder
files = pldb.getDirContents(targetfolder)
print(files)

#get list of target folders
folders = files[files['File_Name'].str.contains('2color_') & files['Directory'] == True]

#transfer folders back to folders with the same date
#delete tif files in the target folder
print(folders)
folders['output_folder'] = ''
for row, dseries in folders.iterrows():
    #row = folders.index[0]
    #dseries = folders.loc[row]
    folders['output_folder'].loc[row] = str(Path(outputfolder) / dseries['Parent'][len(targetfolder)+1:] /  dseries['File_Name'])
    #make directory
    #pldb.makeTree(folders['output_folder'].loc[row])  
    #copy all files from target to output directory
    delete_files = pldb.getDirContents(folders['output_folder'].loc[row])
    for row2, dseries2 in delete_files.iterrows():
        Path(dseries2['Full_Path']).unlink()
    
    cfiles = pldb.getDirContents(dseries['Full_Path'])
    cfiles['output'] = ''
    for ccfile, dseries2 in cfiles.iterrows():
        cfiles['output'].loc[ccfile] = str(Path(outputfolder) / dseries['Parent'][len(targetfolder)+1:] /  dseries['File_Name'] / dseries2['File_Name'])
        if not Path(cfiles['output'].loc[ccfile]).is_file():
            shutil.copy2(dseries2['Full_Path'], cfiles['output'].loc[ccfile])
 
#now copy the mask files into any folders with a similar name
maskfiles = files[files['File_Name'].str.contains('Mask.hdf5')]
outputfolderdir = pldb.getDirContents(outputfolder)
outputfolderdir = outputfolderdir[outputfolderdir['Directory'] == True]

for row, dseries in maskfiles.iterrows():
    #row =0
    #dseries = maskfiles.iloc[0]
    string1 = dseries['File_Name'][:-22]
    print(string1)
    targetdirectories = outputfolderdir[outputfolderdir['File_Name'].str.contains(string1) & outputfolderdir['File_Name'].str.contains('stim30s06V')]
    targetdirectories['output'] = ''
    for row2, dseries2 in targetdirectories.iterrows():
        #row2 = targetdirectories.index[0]
        #dseries2 = targetdirectories.loc[row2]
        targetdirectories['output'].loc[row2] = str(Path(dseries2['Full_Path']) / dseries['File_Name'])
        if not Path(targetdirectories['output'].loc[row2]).is_file():
            shutil.copy(dseries['Full_Path'], targetdirectories['output'].loc[row2])
