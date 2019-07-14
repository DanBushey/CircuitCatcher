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
#import parallelDB
import sys
#folderModules = '/media/daniel/Seagate Backup Plus Drive2/JData/A/A12_19F01-Gal4_CaMPARI/A12 Data'
#sys.path.append(folderModules)
import ccModules
import ccModules2 as cc2
#parallelDB.startLocalWait(5,15)
import prairieModules as pM

rc = ipp.Client()
dview = rc[:]

#folder to search
#targetfolder = r'/data/JData/A/A18/nSyb/'
targetfolder = r'/data/JData/A/A57_GtACR_Mi1/A57_Data'
minimumNumberOfTiffFiles = 299


directoriesWithTiff = [] #store directoires with N > tif
index1 = []
files1 = []
stdev = []
for path in os.walk(targetfolder):
    print(path)
    tiffiles, index= osDB.getFileContString(path[0], '.tif')
    #hdf5file, index = osDB.getFileContString(path[0], '.hdf5')
    if len(tiffiles) >= minimumNumberOfTiffFiles: 
        #if os.path.getsize(os.path.join(path[0], tiffiles.values[0])) > 10**8:
        directoriesWithTiff.append(path[0])
        index1.append(os.path.split(path[0])[1])
        files1.append(tiffiles.values)
        #check if stdev file already exists
        path1, file1 = os.path.split(path[0])
        if os.path.isfile(os.path.join(path[0], file1 + "_STDEV.hdf5")):
            stdev.append(1)
        else:
            stdev.append(0)

inputdata = pd.DataFrame(index = index1, columns = ['Path', 'Tif-Files'])
inputdata["Path"] = directoriesWithTiff
inputdata['Tif-Files'] = files1
inputdata['STDEV.hd5_Exists'] =stdev

    
    
#inputdata.to_excel(os.path.join(targetfolder, 'TranslateTiffToHDF5.xlsx'))
#inputdata=inputdata[inputdata['STDEV.hd5_Exists'] == 0]
inputdata.to_csv(os.path.join(targetfolder, 'toHDF5.xlsx'))
#inputdata.to_excel(os.path.join(targetfolder, 'TranslateTiffToHDF5.xlsx'))
#path = inputdata["Path"].values[0]
output1 = dview.map(pM.generateHDFfileSingleTimeSeries, inputdata["Path"].values )
output1.wait_interactive()
print(output1.get())
#
'''
i=13
print(inputdata.index[i])
pM.generateHDFfileSingleTimeSeries(inputdata["Path"].iloc[i])
'''


