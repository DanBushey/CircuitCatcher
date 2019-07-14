'''
Created on May 16, 2017

@author: Surf32
Generate HDF5 files that can be read by pqtgraphTimeSeriesDB
'''
import tables
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import osDB
import os
import time
import pandas as pd
from six import string_types
from scipy.io import  loadmat
import pdb
import ipyparallel as ipp
import parallelDB
import sys
import ccModules
#parallelDB.startLocalWait(5,15)
rc = ipp.Client()
dview = rc[:]
import pathlib as pl
import pathlibDB as pldb
from dask.array.image import imread 
'''
#automatically finding the upstream folder
path1 = Path.cwd()
path1 =str([path1.parents[len(path1.parts) - i-1] for i, cfolder in enumerate(path1.parts) if cfolder=='CircuitCatcher'][0])
print('path1', path1)
'''
#folder to search
targetfolder = r'/run/media/busheyd/Seagate Backup Plus Drive/Lisha/20181024'
targetfolder = r'/media/daniel/Seagate Backup Plus Drive3/Lisha/20181024'
#targetfolder = path1
minimumNumberOfTiffFiles = 199


directoriesWithTiff = [] #store directoires with N > tif
index1 = []
files1 = []
for path in os.walk(targetfolder):
    print(path)
    tiffiles, index= osDB.getFileContString(path[0], '.tif')
    hdf5file, index = osDB.getFileContString(path[0], '.hdf5')
    if len(tiffiles) >= minimumNumberOfTiffFiles: #or len(hdf5file) > 0:
        directoriesWithTiff.append(path[0])
        index1.append(os.path.split(path[0])[1])
        files1.append(tiffiles.values)

inputdata = pd.DataFrame(index = index1, columns = ['Path', 'Tif-Files'])
inputdata["Path"] = directoriesWithTiff
inputdata['Tif-Files'] = files1
inputdata['Time'] = ''

#create a column that identifies which fly is tested
inputdata['Fly'] = ''
for row, dseries in inputdata.iterrows():
    inputdata['Fly'].loc[row] = row[9:13]

#create a new data frame to hold the image each fly will be registered against
flies = inputdata['Fly'].unique()
registration = pd.DataFrame(index=flies)
#choose a folder and create a mean image from multiple files
registration['Registration_Image'] = ''
for cfly in registration.index:
    cinputdata = inputdata[inputdata['Fly']==cfly]
    row = [cindex for cindex in cinputdata.index if '_stim_00002' in cindex][0]
    images = cinputdata['Tif-Files'].loc[row]
    images = images[-4:]
    path =  cinputdata['Path'].loc[row]
    cimage = np.squeeze(imread(str(pl.Path(path) /  images[0])))
    target = np.zeros([len(images), cimage.shape[0], cimage.shape[1], cimage.shape[2]], dtype = cimage.dtype)
    target[0, : , :, :] = cimage
    for i, ci in enumerate(images[1:]):
        target[i, :, :, :] = np.squeeze(imread(str(pl.Path(path) / ci)))
    target = target.mean(axis=0)
    output_target = pl.Path(path).parent / 'Registration' 
    pldb.mkdir(output_target)
    output_target = output_target / 'Registration.tif'
    tifffile.imsave(str(output_target), target)
    registration['Registration_Image'].loc[cfly] = str(output_target)
    
#add the correct registration image to the inputdata dataframe
inputdata['Registration_File'] = ''
for row, dseries in inputdata.iterrows():
    inputdata['Registration_File'].loc[row] = registration['Registration_Image'].loc[dseries['Fly']]
    
#inputdata.to_excel(os.path.join(targetfolder, 'TranslateTiffToHDF5.xlsx'))
output1 = dview.map(ccModules.generateHDFfileRegistration, inputdata["Path"].values, inputdata['Registration_File'].values)
#ccModules.generateHDFfile( inputdata["Path"].values[1])
output1.wait_interactive()
'''
row = inputdata.index[5]
for row in inputdata.index:
    t = ccModules.generateHDFfileRegistration(inputdata["Path"].loc[row], inputdata['Registration_File'].loc[row])

path = inputdata["Path"].loc[row]
'''
