'''
write file IntensityData.hdf5 containing the intensity data for rois
Project A57

'''
import os
import numpy as np
import pandas as pd
import ccModules
import ccModules2 as cc2
import ipyparallel as ipp
import pathlibDB as pbDB
import pathlib as pb
rc = ipp.Client()
dview = rc[:]


#search directory tree for all files containing numpy
targetfolder = r'/data/JData/A/A57_GtACR_Mi1/A57_Data'

#get full directory
files = pbDB.getDirContents(targetfolder)
maskfiles = files[files['File_Name'].str.contains('Mask.hdf5')]
print(maskfiles)

#search for corresponding Image files
maskfiles['Image_files'] = ''
for row, dseries in maskfiles.iterrows():
    HDF5files, index = ccModules.getFileContString(dseries['Parent'], 'ImageFile.hdf5')
    if len(HDF5files) > 0:
        maskfiles['Image_files'].loc[row] = str(pb.Path(dseries['Parent']) / HDF5files.values[0])
print(maskfiles['Image_files'].values)

#generate paths to outputfiles
maskfiles['Output_files'] = ''
for row, dseries in maskfiles.iterrows():
    name = str(pb.Path(dseries['Parent']).parts[-1]) + '_IntensityData.hdf5'
    pathname = str(pb.Path(dseries['Parent']) / name)
    maskfiles['Output_files'].loc[row] = pathname

maskfiles.to_csv(str(pb.Path(targetfolder) / 'generateIntensityDataFiles.csv'))

#delete rows with empty cells
maskfiles.replace('', np.nan, inplace = True)
maskfiles.dropna(inplace = True)

#generate intensity data files.hdf5
output1 = dview.map(cc2.write_Intensity_Data_File, maskfiles['Full_Path'].values, maskfiles['Image_files'].values, maskfiles['Output_files'].values)
output1.wait_interactive()
output1.get()
'''
for i in range(len(maskfiles)):
    print(i)
    mask_file = maskfiles['Full_Path'].values[i]
    image_file = maskfiles['Image_files'].values[i]
    output_file = maskfiles['Output_files'].values[i]
    cc2.write_Intensity_Data_File(mask_file, image_file, output_file)
'''

