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
folderModules = '/data/JData/A/A30FastROI/CircuitCatcher'
sys.path.append(folderModules)
import ccModules
#parallelDB.startLocalWait(5,15)
rc = ipp.Client()
dview = rc[:]

#folder to search
#targetfolder = r'/data/JData/A/A18/nSyb/'
targetfolder = r'/data/JData/A/A58_MeanGirl/DefiningROIS/nSyb'
minimumNumberOfTiffFiles = 199


directoriesWithTiff = [] #store directoires with N > tif
index1 = []
files1 = []
for path in os.walk(targetfolder):
    print(path)
    tiffiles, index= osDB.getFileContString(path[0], '.tif')
    hdf5file, index = osDB.getFileContString(path[0], '.hdf5')
    if len(tiffiles) >= minimumNumberOfTiffFiles or len(hdf5file) > 0:
        directoriesWithTiff.append(path[0])
        index1.append(os.path.split(path[0])[1])
        files1.append(tiffiles.values)

inputdata = pd.DataFrame(index = index1, columns = ['Path', 'Tif-Files'])
inputdata["Path"] = directoriesWithTiff
inputdata['Tif-Files'] = files1
inputdata['Time'] = ''

    
    
#inputdata.to_excel(os.path.join(targetfolder, 'TranslateTiffToHDF5.xlsx'))
output1 = dview.map(ccModules.generateHDFfile, inputdata["Path"].values )
#ccModules.generateHDFfile( inputdata["Path"].values[1])
output1.wait_interactive()
