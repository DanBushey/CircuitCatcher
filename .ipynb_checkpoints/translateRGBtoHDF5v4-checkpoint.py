'''
Creates RGB from fast time series stacked image and then deletes original tif images
Creates a single channel HDF from the 1040 nm channel
red = mean from channel 1 from 2 color image taken using fastz
green = mean from channel 2 from 2 color image taken using fastz
blue = MIP from fast time series

'''

import pandas as pd
import pathlibDB as pldb
from pathlib import Path
#import ipyparallel as ipp
import ccModules as cc
import pathlib
#rc = ipp.Client()
#dview = rc[:]

targetfolder = pathlib.Path.cwd().parents[0]
#targetfolder = '/media/daniel/Seagate Backup Plus Drive/A83'
print('targetfolder: ', targetfolder)


#get all files in targetfolder
files = pldb.getDirContents(targetfolder)
print(files)
files = files[files['Directory'] == True]
files = files[files['File_Name'].str.contains('_2color_')]
print(files['File_Name'].values)

#get and count the number of tif files in each folder
tif_files = []
tif_count = []
for row, dseries in files.iterrows():
    tif = [str(i.parts[-1]) for i in Path(dseries['Full_Path']).glob('*.tif')]
    tif_files.append(tif)
    tif_count.append(len(tif))
files['tif_count'] = tif_count
files['tif_files'] = tif_files
files = files[files['tif_count'] > 10]

#generate a name for HDF file
hdf_path = []
for row, dseries in files.iterrows():
    hdf_path.append(str(Path(dseries['Full_Path']) / str( dseries['File_Name'] + '.hdf5')))
files['Output_HDF5'] = hdf_path
files.to_excel(str(Path(targetfolder) / 'GenerateTwoColorImages.xlsx'))

print('Starting to compile 2color images')
if len(files) >0:
    #output1 = dview.map(cc.generateHDF_from_FastZ2Color, files['Output_HDF5'].tolist(),  files['Full_Path'].tolist(), files['tif_files'].tolist())
    #output1.wait_interactive()
    #print(output1.get())
    list(map(cc.generateHDF_from_FastZ2Color, files['Output_HDF5'].tolist(),  files['Full_Path'].tolist(), files['tif_files'].tolist()))
###################################################
#converting 1040 nm tif images to HDF5 file mean

targetfolder = pathlib.Path.cwd().parents[0]
#targetfolder = '/media/daniel/Seagate Backup Plus Drive/A83'
print('targetfolder: ', targetfolder)


#get all files in targetfolder
files = pldb.getDirContents(targetfolder)
print(files)
files = files[files['Directory'] == True]
files = files[files['File_Name'].str.contains('_1040nm_')]
print(files['File_Name'].values)

#get and count the number of tif files in each folder
tif_files = []
tif_count = []
for row, dseries in files.iterrows():
    tif = [str(i.parts[-1]) for i in Path(dseries['Full_Path']).glob('*.tif')]
    tif_files.append(tif)
    tif_count.append(len(tif))
files['tif_count'] = tif_count
files['tif_files'] = tif_files
files = files[files['tif_count'] > 10]

#generate a name for HDF file
hdf_path = []
for row, dseries in files.iterrows():
    hdf_path.append(str(Path(dseries['Full_Path']) / str( dseries['File_Name'] + '.hdf5')))
files['Output_HDF5'] = hdf_path
files.to_excel(str(Path(targetfolder) / 'Generate1040Images.xlsx'))

print('Starting on 1040 tif images.')
#output1 = dview.map(cc.generateHDF_from_1040, files['Output_HDF5'].tolist(),  files['Full_Path'].tolist(), files['tif_files'].tolist())
#output1.wait_interactive()
#print(output1.get())
list(map(cc.generateHDF_from_1040, files['Output_HDF5'].tolist(),  files['Full_Path'].tolist(), files['tif_files'].tolist()))
'''
for row in range(len(files)):
    output1 = cc.generateHDF_from_FastZ2Color(files['Output_HDF5'].tolist()[row], files['Full_Path'].tolist()[row], files['tif_files'].tolist()[row])
    
from dask.array.image import imreadg
import numpy as np
import matplotlib.pyplot as plt
from ccModules import normalize


row=0
index = np.where(files.index == row)
row = index[0][0]
output = files['Output_HDF5'].tolist()[row]
fastz_tif = files['tif_files'].tolist()[row]
path = files['Full_Path'].tolist()[row]



def getMeanInTime_from_TIF(path, tif_files):
    #path = full file path to tif image including image
    #tif_files = [list of .tif files to be included]
    tif_files.sort()
    sample = np.squeeze(imread(str(Path(path) / tif_files[0])).compute())
    sample = sample.astype(np.float64)
    for cfile in tif_files[1:]:
        sample = sample + np.squeeze(imread(str(Path(path) / cfile)).compute())
    sample = sample / len(tif_files)
    shape = sample.shape
    red_range = np.arange(0, shape[0], 2)
    green_range = np.arange(1, shape[0], 2)
    green = normalize(sample[green_range, :, :])
    red = normalize(sample[red_range, :, :])
    rgb = np.zeros([int(shape[0] /2), shape[1], shape[2], 3], dtype = sample.dtype)
    rgb[:, :, :, 0] = green
    rgb[:, :, :, 1] = red
    return np.squeeze(rgb)

def generateHDF_from_FastZ2Color(output, path, fastz_tif):
    #output = output path for .hdf5 file
    #path = directory where tif images are stored
    #fastz_tif = list of tif_images in path
    rgb = getMeanInTime_from_TIF(path, fastz_tif)
    #plt.imshow(np.max(rgb, axis = 0))disco
    rgb2 =np.zeros([1, rgb.shape[0], rgb.shape[1], rgb.shape[2], rgb.shape[3]], dtype = rgb.dtype)
        
    #for layer in green_range
    rgb2[0, :, :, :, 0] = rgb[:, :, :,0]
    rgb2[0, :, :, :, 1] = rgb[:, :, :,1]
    
    rgb2[rgb2 > 1] = 1
    rgb2 = np.rollaxis(rgb2, 3, 2)
    rgb2 = skimage.img_as_int(rgb2)
    hdf5_file = tables.open_file(output, mode='w')
    #filters = tables.Filters(complevel=5, complib='blosc') #faster but poor compression
    filters = tables.Filters(complevel=3, complib='zlib') #slower but better compression
    data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                  tables.Atom.from_dtype(rgb2.dtype),
                                  shape = rgb2.shape,
                                  filters=filters)
    data_storage[:] = rgb2
    data_storage.flush()
    hdf5_file.close()
    for file1 in fastz_tif:
        target = Path(path) / file1
        target.unlink()
    #remove the matfile
    for file in Path(path).glob('*.mat'):
        file.unlink()
        
def getMeanInTime_from_1040(path, tif_files):
    #path = full file path to tif image including image
    #tif_files = [list of .tif files to be included]
    tif_files.sort()
    sample = np.squeeze(imread(str(Path(path) / tif_files[0])).compute())
    sample = sample.astype(np.float64)
    for cfile in tif_files[1:]:
        sample = sample + np.squeeze(imread(str(Path(path) / cfile)).compute())
    sample = sample / len(tif_files)
    return normalize(sample)

def generateHDF_from_1040(output, path, fastz_tif):
    #output = output path for .hdf5 file
    #path = directory where tif images are stored
    #fastz_tif = list of tif_images in path
    chan2 = getMeanInTime_from_1040(path, fastz_tif)
    #plt.imshow(np.max(rgb, axis = 0))disco
    chan22 =np.zeros([1, chan2.shape[0], chan2.shape[1], chan2.shape[2]], dtype = chan2.dtype)
    chan22 [0, :, :, :] = chan2 
    
    chan22[chan22 > 1] = 1
    chan22 = np.rollaxis(chan22, 3, 2)
    chan22 = skimage.img_as_int(chan22)
    hdf5_file = tables.open_file(output, mode='w')
    #filters = tables.Filters(complevel=5, complib='blosc') #faster but poor compression
    filters = tables.Filters(complevel=3, complib='zlib') #slower but better compression
    data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                  tables.Atom.from_dtype(chan22.dtype),
                                  shape = chan22.shape,
                                  filters=filters)
    data_storage[:] = chan22
    data_storage.flush()
    hdf5_file.close()
    for file1 in fastz_tif:
        target = Path(path) / file1
        target.unlink()
    #remove the matfile
    for file in Path(path).glob('*.mat'):
        file.unlink()
'''