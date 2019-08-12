'''
Created on May 16, 2017

@author: Surf32
Generate HDF5 files that can be read by pqtgraphTimeSeriesDB
'''
import tables
import tifffile
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from six import string_types
from scipy.io import  loadmat
import pdb
import matplotlib
import matplotlib.pyplot as plt
import skimage.measure
import skimage.exposure
import scipy.ndimage
from six import string_types
from scipy.io import  loadmat
import pyqtgraph
import json
import ast
import matplotlib.patches
#import osDB
import scipy.ndimage as nd
from skimage.draw import polygon
import pdb
import read_roi
#pdb.set_trace()
import ccModules2 as cc2
from dask.array.image import imread 
from skimage import data
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from pathlib import Path
import skimage


def getTifffilesMetadata(file):
    #file = path to tif image file
    #return dictionary{each image metata data dictionary{tag}}
    with tifffile.TiffFile(file) as tif:
        imgs = {}
        for i, page in enumerate(tif):
            t={}
            for ctag in page.tags.values():
                if isinstance(ctag.value, int):
                    
                    t[ctag.name] = ctag.value
                elif isinstance(ctag.value, tuple):
                    
                    t[ctag.name] = ctag.value
                else:
                    string1= ctag.value.decode('utf-8')
                    for line in  string1.split('\n'):
                        index = line.find('=')
                        if index > 0:
                            cname = line[:index-1]
                            t[cname] = line[index+2:]
                    tagname = str
            imgs[i] =t
    return imgs

def getStartTimes(imagemetadata):
    time = []
    for ckey in imagemetadata.keys():
        time.append(float(imagemetadata[ckey]['frameTimestamps_sec']))
    return time
        

def getEndTimes(imagemetadata):
    time1 = []
    for ckey in imagemetadata.keys():
        time1.append(float(imagemetadata[ckey]['frameTimestamps_sec']))
    period = np.median(np.diff(time1))
    t = time1[1:].append(time1[-1] + period)
    return time1

def generateHDFfile(path):
    path2, file1 = os.path.split(path)
    hdf5_path = os.path.join(path, file1 + ".hdf5")
    if not os.path.isfile(hdf5_path):
        #get the tiff files
        tiffiles, index= getFileContString(path, '.tif')
        
        sample_data = tifffile.imread(os.path.join(path, tiffiles.values[0]))
    
        hdf5_file = tables.open_file(hdf5_path, mode='w')

        #filters = tables.Filters(complevel=5, complib='blosc') #faster but poor compression
        filters = tables.Filters(complevel=3, complib='zlib') #slower but better compression
        shape=(len(tiffiles), sample_data.shape[0], sample_data.shape[1], sample_data.shape[2], 3)
        data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                      tables.Atom.from_dtype(sample_data.dtype),
                                      shape = shape,
                                      filters=filters)
        timeStart = hdf5_file.create_carray(hdf5_file.root, 'timeStart',
                                      tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                      shape = (len(tiffiles), 1),
                                      filters=filters)
        timeEnd = hdf5_file.create_carray(hdf5_file.root, 'timeEnd',
                                      tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                      shape = (len(tiffiles), 1),
                                      filters=filters)
        #get json_file containing registration data from gpur2
        jsonfiles, index= getFileContString(path, 'dr.3x350.json')
        if len(jsonfiles) > 0:
            with open(os.path.join(path, jsonfiles.values[0])) as json_data:
                d = json.load(json_data)
            json_data = np.asarray(d, dtype = np.int)

        start_time  = time.time()
        min1 = [] #record min values
        max1 = [] #record max values
        stackmetadata = {}
        for i, cfile in enumerate( np.sort(tiffiles)):
            if 'json_data' in locals():
                img = applyTransformation(os.path.join(path, cfile), json_data[time1])
            else:
                img = tifffile.imread(os.path.join(path, cfile))
            img = np.rollaxis(img, 2, 1)
            data_storage[i, :, :, :, 0] = img
            data_storage[i, :, :, :, 1] = img
            data_storage[i, :, :, :, 2] = img
            tags1 = getTifffilesMetadata(os.path.join(path, cfile))
            stackmetadata[i] = tags1
            timeStart[i] = float(tags1[0]['frameTimestamps_sec'])
            if i != 0:
                timeEnd[i-1] = float(tags1[0]['frameTimestamps_sec'])
            data_storage.flush()
            timeStart.flush()
            timeEnd.flush()
            min1.append(np.min(img))
            max1.append(np.max(img))
            

        

        ##need to add estimated final end time for last stack
        timeEnd[i] = getEndTimes(tags1)[-1]
        timeEnd.flush()
        data_storage2 = hdf5_file.create_carray(hdf5_file.root, 'minmax',
                                      tables.Atom.from_dtype(sample_data.dtype),
                                      shape = (2, 1),
                                      filters=filters)
        data_storage2[0] = np.min(min1)
        data_storage2[1] = np.max(max1)
        #get voltage data
        matfiles, index = getFileContString(path, 'stim.mat')
        if len(matfiles) > 0:
            stimdata = loadmat(os.path.join(path, matfiles.values[0]))
            stimdata = stimdata['AOBuffer']
        
            
            voltage_storage = hdf5_file.create_carray(hdf5_file.root, 'voltage',
                                          tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                          shape = stimdata.shape,
                                          filters=filters)
            voltage_storage[:] = stimdata
            voltage_storage.flush()

        
        #add stackmetadata to hdf5 file
        #first convert stackmetadata to pandas dataframe
        columns1 = ['Stack', 'Image']
        columns1.extend(list(stackmetadata[0][0].keys()))
        metaframe = pd.DataFrame(index =[], columns =columns1)
        for cstack in stackmetadata.keys():
            for cimg in stackmetadata[cstack].keys():
                ser1 = pd.Series(index=columns1)
                ser1['Stack'] = cstack
                ser1['Image'] = cimg
                for ctag in stackmetadata[cstack][cimg].keys():
                    ser1[ctag] = str(stackmetadata[cstack][cimg][ctag])
                metaframe = metaframe.append(ser1, ignore_index=True)
        metaframe.to_hdf(hdf5_path, '/metadata')
        '''
        how to read meta data
        hdf5_file = pd.read_hdf(hdf5_path, '/metadata', mode='r')
        '''
        
        
        for i, cfile in enumerate( tiffiles):
            os.remove(os.path.join(path, cfile))
        if os.path.isfile(os.path.join(path, matfiles.values[0])):
            os.remove(os.path.join(path, matfiles.values[0]))
            
                #save a file with maximum intensity image (MIP)
        MIP_Path = os.path.join(os.path.join(path, file1 + "_MIP.hdf5"))
        MIPfile = tables.open_file(MIP_Path, mode='w')        
        
        shape=(1, sample_data.shape[1], sample_data.shape[2], sample_data.shape[3], 3)
        MIP_storage = MIP_file.create_carray(MIPfile.root, 'data',
                                      tables.Atom.from_dtype(img.dtype),
                                      shape = shape,
                                      filters=filters)
        for z in range(data_storage.shape[1]):
            MIP_storage[0, z, :, :, :] = np.max(data_storage[:, z, :, :], axis = 0)
            MIP_storage.flush()
        hdf5_file.close()
        MIPfile.close()
        

def generateHDFfileSingleTimeSeries(path, type='standard_deviation'):
    #designed for single plane images (not volume timeseries)
    #path = folder when image data is located
    #type = timeseries, 
        #standard_deviation=red and blue channels , MIP = green
        #timeseries = timeseries with standard deviation in channels red and blue with intensity changes over time in green (dim =z, time, x, y c), stdv = standard deviation all channels dim = (z=0, time=0, , x,y, c)
    path2, file1 = os.path.split(path)
    
    if not os.path.isfile(os.path.join(path, file1 + ".hdf5")) or not os.path.isfile(os.path.join(path, file1 + "_STDEV.hdf5")):
        tiffiles, index= getFileContString(path, '.tif')
        img_mat = tifffile.imread(os.path.join(path, tiffiles.values[0]))
        #img_mat = img_mat[:300]
        if img_mat.shape[0] > 1000:
            
            #filters = tables.Filters(complevel=5, complib='blosc') #faster but poor compression
            filters = tables.Filters(complevel=3, complib='zlib') #slower but better compression
            if type == 'timeseries':
                hdf5_path = os.path.join(path, file1 + ".hdf5")
                hdf5_file = tables.open_file(hdf5_path, mode='w')
                shape=(img_mat.shape[0], len(tiffiles), img_mat.shape[1], img_mat.shape[2], 3)
                data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                          tables.Atom.from_dtype(img_mat.dtype),
                                          shape = shape,
                                          filters=filters)
                timeStart = hdf5_file.create_carray(hdf5_file.root, 'timeStart',
                                          tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                          shape = (img_mat.shape[0], 1),
                                          filters=filters)
                timeEnd = hdf5_file.create_carray(hdf5_file.root, 'timeEnd',
                                          tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                          shape = (img_mat.shape[0], 1),
                                          filters=filters)
            
                start_time  = time.time()
                img_mat = np.rollaxis(img_mat, 2, 1)
                data_storage[:, 0, :, :, 1] = img_mat
                standard_deviation = skimage.exposure.adjust_gamma(np.std(img_mat,axis = 0), 0.3)
                standard_deviation = (standard_deviation - np.min(standard_deviation)) / np.max(standard_deviation)*np.max(img_mat)
                data_storage[:, 0, :, :, 0] = standard_deviation.astype(img_mat.dtype)
                data_storage[:, 0, :, :, 2] = standard_deviation.astype(img_mat.dtype)
                #data_storage[:, 0, :, :, 0] = np.max(img_mat,axis = 0)
                #data_storage[:, 0, :, :, 2] = np.max(img_mat,axis = 0)
                data_storage.flush()
                tags1 = getTifffilesMetadata(os.path.join(path,  tiffiles.values[0]))
                rate = float(tags1[0]['frameNumbers']) * (1/float(tags1[0]['SI.hRoiManager.scanFrameRate']))
                timeStart[:, 0] = np.arange( img_mat.shape[0] )*rate
                timeEnd[:, 0] = np.append(timeStart[1:], (rate+timeStart[-1]))
                timeStart.flush()
                timeEnd.flush()
                os.remove(os.path.join(path, tiffiles.values[0]))
                data_storage2 = hdf5_file.create_carray(hdf5_file.root, 'minmax',
                                              tables.Atom.from_dtype(img_mat.dtype),
                                              shape = (2, 1),
                                              filters=filters)
                data_storage2[0] = np.min(img_mat)
                data_storage2[1] = np.max(img_mat)
                #get voltage data
                matfiles, index = getFileContString(path, 'stim.mat')
                if len(matfiles) > 0:
                    stimdata = loadmat(os.path.join(path, matfiles.values[0]))
                    stimdata = stimdata['AOBuffer']
                else:
                    stimdata =np.nan
                os.remove(os.path.join(path, matfiles.values[0]))
                voltage_storage = hdf5_file.create_carray(hdf5_file.root, 'voltage',
                                              tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                              shape = stimdata.shape,
                                              filters=filters)
                voltage_storage[:] = stimdata
                voltage_storage.flush()
                hdf5_file.close()
                #add timeseries metadata to hdf5 file
                #first convert metadata to pandas dataframe
                columns1 = ['Stack', 'Image']
                columns1.extend(list(tags1[0].keys()))
                metaframe = pd.DataFrame(index =[], columns =columns1)
                for cimg in tags1.keys():
                    ser1 = pd.Series(index=columns1)
                    ser1['Image'] = cimg
                    for ctag in tags1[cimg].keys():
                        ser1[ctag] = str(tags1[cimg][ctag])
                    metaframe = metaframe.append(ser1, ignore_index=True)
                metaframe.to_hdf(hdf5_path, '/metadata')
            elif type == 'standard_deviation':
                hdf5_path = os.path.join(path, file1 + "_STDEV.hdf5")
                hdf5_file = tables.open_file(hdf5_path, mode='w')
                shape=(1, 1, img_mat.shape[1], img_mat.shape[2], 3)
                data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                          tables.Atom.from_dtype(img_mat.dtype),
                                          shape = shape,
                                          filters=filters)
                timeStart = hdf5_file.create_carray(hdf5_file.root, 'timeStart',
                                          tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                          shape = (img_mat.shape[0], 1),
                                          filters=filters)
                timeEnd = hdf5_file.create_carray(hdf5_file.root, 'timeEnd',
                                          tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                          shape = (img_mat.shape[0], 1),
                                          filters=filters)
            
                start_time  = time.time()
                img_mat1 = np.rollaxis(img_mat, 2, 1)
                
                standard_deviation = skimage.exposure.adjust_gamma(np.std(img_mat1,axis = 0), 0.3)
                MIP = skimage.exposure.adjust_gamma(img_mat1.max(axis=0).astype(img_mat.dtype), 0.3)
                MIP = (MIP - np.min(MIP))
                #histogram matach the standard_deviation and MIP images
                standard_deviation = standard_deviation / np.max(standard_deviation) * np.max(MIP)
                
                #standard_deviation = (img_mat1 - np.min(img_mat1)) / np.max(img_mat1)*np.max(img_mat1)
                data_storage[:, 0, :, :, 0] = standard_deviation.astype(img_mat.dtype)
                data_storage[:, 0, :, :, 1] = MIP.astype(img_mat.dtype)
                data_storage[:, 0, :, :, 2] = standard_deviation.astype(img_mat.dtype)
                #data_storage[:, 0, :, :, 0] = np.max(img_mat,axis = 0)
                #data_storage[:, 0, :, :, 2] = np.max(img_mat,axis = 0)
                data_storage.flush()
                tags1 = getTifffilesMetadata(os.path.join(path,  tiffiles.values[0]))
                if 'SI.hRoiManager.scanFrameRate' in tags1[0].keys(): #depending on the scanimage versions used to acquire the tags vary
                    rate = float(tags1[0]['frameNumbers']) * (1/float(tags1[0]['SI.hRoiManager.scanFrameRate']))
                else:
                    scanFrameRate = tags1[999]['scanimage.SI.hRoiManager.scanFrameRate']
                    frameNumbers = tags1[999]['frameNumbers']
                    rate = (float(frameNumbers)/1000) * (1/float(scanFrameRate))
                timeStart[:, 0] = np.arange( img_mat.shape[0] )*rate
                timeEnd[:, 0] = np.append(timeStart[1:], (rate+timeStart[-1]))
                timeStart.flush()
                timeEnd.flush()
                #os.remove(os.path.join(path, tiffiles.values[0]))
                data_storage2 = hdf5_file.create_carray(hdf5_file.root, 'minmax',
                                              tables.Atom.from_dtype(img_mat.dtype),
                                              shape = (2, 1),
                                              filters=filters)
                data_storage2[0] = np.min(img_mat)
                data_storage2[1] = np.max(img_mat)
                #get voltage data
                matfiles, index = getFileContString(path, 'stim.mat')
                if len(matfiles) > 0:
                    stimdata = loadmat(os.path.join(path, matfiles.values[0]))
                    stimdata = stimdata['AOBuffer']
                else:
                    stimdata =np.nan
                #os.remove(os.path.join(path, matfiles.values[0]))
                voltage_storage = hdf5_file.create_carray(hdf5_file.root, 'voltage',
                                              tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                              shape = stimdata.shape,
                                              filters=filters)
                voltage_storage[:] = stimdata
                voltage_storage.flush()
                hdf5_file.close()
                #add timeseries metadata to hdf5 file
                #first convert metadata to pandas dataframe
                columns1 = ['Stack', 'Image']
                columns1.extend(list(tags1[0].keys()))
                metaframe = pd.DataFrame(index =[], columns =columns1)
                for cimg in tags1.keys():
                    ser1 = pd.Series(index=columns1)
                    ser1['Image'] = cimg
                    for ctag in tags1[cimg].keys():
                        ser1[ctag] = str(tags1[cimg][ctag])
                    metaframe = metaframe.append(ser1, ignore_index=True)
                metaframe.to_hdf(hdf5_path, '/metadata')

            '''
            how to read meta data
            hdf5_file = pd.read_hdf(hdf5_path, '/metadata', mode='r')
            '''
def getFileContString(targetdir, string1):
    #code.interact(local=locals())
    filelist=getFileList(targetdir)
    if len(filelist) > 0:
        indx=filelist['file'].str.contains(string1)
        filenames=filelist['file'][indx]
        indx2=np.where(indx)
    else:
        filenames = filelist['file']
        indx2 = ([],)
    return filenames, indx2

def getFileList(targetdir):
    #get the list of files in the targetdir 
    list1 = os.listdir(targetdir)
    filelist=[]
    dirlist=[]
    for int in list1:
        fulldir = targetdir + "/" + int
        #print fulldir
        if os.path.isfile(fulldir):
            dirlist.append(fulldir)
            filelist.append(int)
            
    full_list=pd.DataFrame({'file': filelist, 'path': dirlist})        
    #full_list.to_csv(outputdir + '\\Folderlist.csv')
    return full_list


def findOccurences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]


def applyTransformation(tiffile, ct):
    imgO = tifffile.imread(tiffile)
    
    if ct[0] != 0:
        if ct[0] < 0:
            imgO[:, :, :ct[0]] = imgO[:, :, -ct[0]:]
            imgO[:, :, ct[0]:] = 0 #imgO[:, :, :-ct[0]]
        else:
            imgO[:, :, ct[0]:] = imgO[:, :, :-ct[0]] ##
            imgO[:, :, :ct[0]] = 0 #imgO[:, :, -ct[0]:]
            
 
    if ct[1] != 0:
        if ct[1] < 0:
            imgO[:, :ct[1], :] = imgO[:, -ct[1]:, :]
            imgO[:, ct[1]:, :] = 0 #imgO[:, :-ct[1], :]
        else:
            imgO[:, :-ct[1], :] = imgO[:, ct[1]:, :]
            imgO[:, -ct[1]:, :] = 0 #imgO[:, :ct[1], :]   
    if ct[2] != 0:
        if ct[2] < 0:
            imgO[:ct[2], :, :] = imgO[-ct[2]:, :, :]
            imgO[ct[2]:, :, :] = 0 #imgO[:-ct[2], :, :]
        else:
            imgO[:-ct[2], :, :] = imgO[ct[2]:, :, :]
            imgO[-ct[2]:, :, :] = 0 #imgO[:ct[2], :, :]

    return imgO

def getROIdata(maskfile, targetdirectory, imagefile, jsonfile):
    #targetdirectory = summaryFrame['Target_directories' ].values[2]
    path1, name1 = os.path.split(targetdirectory)
    #maskfile = hdf5 file containing mask data generated from gui
    #targetdirectory = directory continaining image data
    
    maskdata = pd.read_hdf(maskfile) #get mask data
    if len(maskdata) > 0:
        #if gui was done using pyqt4 then the name is comming through as a PyQt4.QtCore.QString and need to convert to normal string because excel write cannot handle pyqt4 strings
        colordict = {'g5': [255, 0, 0], 'bp2': [0, 255, 0]}
        
        for row in maskdata.index:
            maskdata['Name'].loc[row] = str(maskdata['Name'].loc[row])
            if maskdata['Name'].loc[row] in list(colordict.keys()):
                maskdata['Color'].loc[row] = pyqtgraph.mkColor(colordict[maskdata['Name'].loc[row]])
        #change colors that are equal to 0 (white) to a different color
        for row in maskdata.index:
            if not isinstance(maskdata['Color'].loc[row], list) and not isinstance(maskdata['Color'].loc[row], tuple):
                color1 = pyqtgraph.QtGui.QColor.getRgbF(maskdata['Color'].loc[row])
                if color1[0] == 1 and color1[1] == 1 and color1[2] == 1:
                    newcolor = (np.random.randint(255, dtype=np.int), np.random.randint(255, dtype=np.int), np.random.randint(255, dtype=np.int))
                    maskdata['Color'].loc[row] = pyqtgraph.mkColor(newcolor)
              
          
        #get image and intensity data depending on file type
        if '.hdf5' in imagefile[0]:
            maskdata, timeStamp, stimdata, MIPimg, shape, offset = getROIDataFromHDF5(os.path.join(targetdirectory, imagefile[0]), maskdata)
            fileFrame = pd.DataFrame(index = [0], columns=[targetdirectory], data = imagefile)
        else:
            maskdata, timeStamp, stimdata, MIPimg, shape, offset = getROIDataFromTiff(targetdirectory, imagefile, maskdata, jsonfile)
            #tiffiles = imagefile
            
            fileFrame = pd.DataFrame(index = range(len(imagefile)), columns=[targetdirectory], data = imagefile)

        #maskdata.to_hdf(os.path.join(targetdirectory, 'test.hdf5'), 'intensity_data')
        #save intensity data in hdf5 file
        hdf5file = os.path.join(targetdirectory, name1 +'_IntensityData.hdf5')
        if os.path.isfile(hdf5file):
          os.remove(hdf5file)
        maskdata.to_hdf(hdf5file, 'intensity_data')
        hdf5_fileOpen = tables.open_file(hdf5file, mode='a')
        filters = tables.Filters(complevel=3, complib='zlib')
        voltage_storage = hdf5_fileOpen.create_carray(hdf5_fileOpen.root, 'voltage', tables.Atom.from_dtype(np.dtype('float16'), dflt=0), shape = stimdata.shape, filters=filters)
        voltage_storage[:] = stimdata
        voltage_storage.flush()
        time_storage = hdf5_fileOpen.create_carray(hdf5_fileOpen.root, 'timeStamp', tables.Atom.from_dtype(np.dtype('float16'), dflt=0), shape = timeStamp.shape, filters=filters)
        time_storage[:] = timeStamp
        time_storage.flush()
        offset = np.asarray(offset)
        offset_storage = hdf5_fileOpen.create_carray(hdf5_fileOpen.root, 'offset', tables.Atom.from_dtype(np.dtype('float16'), dflt=0), shape = offset.shape, filters=filters)
        offset_storage[:] = offset
        offset_storage.flush()
        hdf5_fileOpen.close()
        #save path.file from which data was derived in the hdf5
        fileFrame.to_hdf(hdf5file, 'image_files')
        '''
        ##test hdf5 file
        maskdata2 = pd.read_hdf(hdf5file, 'intensity_data')
        HDF5_file = tables.open_file(hdf5file, mode='r')
        stimdata2 = HDF5_file.root.voltage
        timeStamp2 = HDF5_file.root.timeStamp
        '''
        
        #create a matplotlib figure save the data
        #plt.close('all')
        fig1=plt.figure(figsize=(10,8))

        #add a plot change intensity over time series
        ax2 = fig1.add_axes([0.1, 0.1, 0.8, 0.4])

        for i, roi in enumerate(maskdata['mask_index'].index):
            if isinstance(maskdata['Color'].loc[roi], list) or isinstance(maskdata['Color'].loc[roi], tuple):
                ax2.plot(timeStamp, np.mean(maskdata['intensity'].loc[roi], axis=0), color = maskdata['Color'].loc[roi], alpha =0.5)
            else:
                ax2.plot(timeStamp, np.mean(maskdata['intensity'].loc[roi], axis=0), color = pyqtgraph.QtGui.QColor.getRgbF(maskdata['Color'].loc[roi])[:3])
        #add stim times to plot
        if not np.isnan(stimdata.all()):
            stimTimeRange = np.array(range(len(stimdata))) /100
            ax21 = ax2.twinx()
            ax21.set_ylabel('Voltage')
            ax21.plot(stimTimeRange, stimdata, color = 'r', alpha =0.4, linestyle = '--')
        
        
        ax2.set_ylabel('Mean Intensity')
        ax2.set_xlabel('Time (s)')
            #add MIP brain image over timeseries and z
        ax1 = fig1.add_axes([0.01, 0.5, 0.9, 0.5])
        ax1.imshow(np.max(MIPimg, axis = 0), cmap = 'gray')
        #outline individual rois
        
        for i, roi in enumerate(maskdata['mask_index'].index):
            maskimage = np.zeros((shape[1], shape[2], shape[3]))
            index = shape[1]*shape[2]*shape[3]
            if maskdata['mask_index'].loc[roi][0].shape[0] != 0: #roi must include at least one pixel
                maskimage.reshape(index)[maskdata['mask_index'].loc[roi][0]] = 1
                maskimage = np.max(maskimage, axis =0)
                #plt.imshow(maskimage, cmap = 'gray')
                contours = skimage.measure.find_contours(maskimage, 0.8)
                
                for n, contour in enumerate(contours):
                    contour = np.array(contour, dtype = np.int)
                    #ax1.plot(contour[:, 1], contour[:, 0], linewidth = 2, color = colors[i])
                    if isinstance(maskdata['Color'].loc[roi], list) or isinstance(maskdata['Color'].loc[roi], tuple):
                        ax1.plot(contour[:, 1], contour[:, 0], linewidth = 2, color = maskdata['Color'].loc[roi])
                    else:
                        ax1.plot(contour[:, 1], contour[:, 0], linewidth = 2, color = pyqtgraph.QtGui.QColor.getRgbF(maskdata['Color'].loc[roi])[:3])
                
                lbl = scipy.ndimage.label(maskimage)
                indexC = scipy.ndimage.center_of_mass(maskimage)
                if isinstance(maskdata['Color'].loc[roi], list) or isinstance(maskdata['Color'].loc[roi], tuple):
                    ax1.text(indexC[1], indexC[0], maskdata['Name'].loc[roi], color = maskdata['Color'].loc[roi])
                else:
                    ax1.text(indexC[1], indexC[0], maskdata['Name'].loc[roi], color = pyqtgraph.QtGui.QColor.getRgbF(maskdata['Color'].loc[roi])[:3])
                #plot lines connecting rois with plotted timeseries data
                endonPlot = (float(timeStamp[-1]), np.mean(maskdata['intensity'].loc[roi], axis=0)[-1])
                if isinstance(maskdata['Color'].loc[roi], list) or isinstance(maskdata['Color'].loc[roi], tuple):
                    con = matplotlib.patches.ConnectionPatch(xyA=endonPlot, xyB=(indexC[1], indexC[0]), coordsA = "data", coordsB = "data", axesA=ax2, axesB=ax1, color=maskdata['Color'].loc[roi], linewidth=1, alpha =0.5)
                else:
                    con = matplotlib.patches.ConnectionPatch(xyA=endonPlot, xyB=(indexC[1], indexC[0]), coordsA = "data", coordsB = "data", axesA=ax2, axesB=ax1, color=pyqtgraph.QtGui.QColor.getRgbF(maskdata['Color'].loc[roi])[:3], linewidth=1, alpha =0.5)
                ax2.add_artist(con)
                ax2.plot(endonPlot[0], endonPlot[1], 'ro', markersize =2)
                ax1.plot(indexC[1], indexC[0], 'ro', markersize =2)
        ylim = ax2.get_ylim()
        #ax2.text(0, (ylim[1]+ylim[0])/2, maskfile)
            
        ax1.axis('off')
        ax2.set_zorder(1)
        ax21.set_zorder(2)
        
        #save and then close fiture
        savefig = os.path.join(targetdirectory, name1 + '_ROI.jpeg')
        fig1.savefig(savefig)
        plt.close(fig1)
        #pdb.set_trace()
            
        #export timeseries data to an excel
        excelfile = os.path.join(targetdirectory, name1 + '_MeanROIIntensity.xlsx')
        writer = pd.ExcelWriter(excelfile, engine='xlsxwriter')
        #if names were derived from pyqt4 then in pyqt4 str format; must convert to str format to save in excel file
        names = []
        for row in maskdata.index:
          names.append(str(maskdata['Name'].loc[row]))
        intensityDataframe = pd.DataFrame(index = names, columns=['{0:.2f}'.format(float(i)) for i in timeStamp])
        #intensityDataframe = pd.DataFrame(index = maskdata.index, columns=timeStamp.flatten().tolist())
        for i, roi in enumerate(maskdata['mask_index'].keys()):
            intensityDataframe.iloc[i] = np.mean(maskdata['intensity'].loc[roi], axis =0)
        intensityDataframe.transpose().to_excel(writer, "MeanValues")
        #also add a sheet with voltage information
        #voltageDataFrame = pd.DataFrame(index = ['Voltage'], columns = np.arange(len(stimdata))/100, data = stimdata.reshape(1, len(stimdata)).astype(np.float))
        #voltageDataFrame.to_excel(writer, "Voltage")
        writer.save()
    
def getROIDataFromTiff(targetdirectory, tiffiles, maskdata, jsonfile):
    #need a sample image image to determine dimensions
    sampleimg = cc2.load_tif(os.path.join(targetdirectory, tiffiles[0]))
    maskdata['intensity'] = ''

    print('how')
    #generate an image that will be used in figure = include a MIP for each stack in the timeseries
    MIPimg = np.zeros((len(tiffiles), sampleimg.shape[1], sampleimg.shape[2]), dtype = sampleimg.dtype)
    
    #create a column to hold flattened intensity values
    
    #reiterate through each 3d image gettin the masked region, time stack taken
    timeStamp = [] #keep a record of the second each stack was started
    offset = []
    #get json_file containing registration data from gpur2
    if jsonfile != None:
        if os.path.isfile(jsonfile):
            with open(jsonfile) as json_data:
                d = json.load(json_data)
            json_data = np.asarray(d, dtype = np.int)
    if len(tiffiles) > 1: #loading a timeseries where each time point is saved as a single file that includes a volume
        for roi in maskdata['mask_index'].keys():
        #stack dimensions = roi size (number of pixels) x number of stacks in timeseries
            maskdata['intensity'].loc[roi]= np.zeros((len(maskdata['mask_index'][roi][0]), len(tiffiles)), dtype = sampleimg.dtype)
        tiffiles.sort()
        for time1, imgfile in enumerate(tiffiles):
            if 'json_data' in locals():
                img = applyTransformation(os.path.join(targetdirectory, imgfile), json_data[time1])
            else:
                img = tifffile.imread(os.path.join(targetdirectory, imgfile))
            #get end time for each stack
            #tags1 has the same number of entries as images in the stack (tags1[0] == first image)
            tags1 = getTifffilesMetadata(os.path.join(targetdirectory, imgfile))
            #if time1 != 0:
            #    timeStamp.append( float(tags1[0]['frameTimestamps_sec']))
            for roi in maskdata['mask_index'].keys():
                maskdata['intensity'].loc[roi][:, time1] = img.flatten()[ maskdata['mask_index'].loc[roi][0]]
            MIPimg[time1, :, :] = np.max(img, axis =0) #saving matplotlib figure
            #get last time point for last image
            timeStamp.append(getEndTimes(tags1)[-1])
            #with older tif files there is a problem in getting the time so need another method
        
        timeStamp=np.array(timeStamp).reshape(len(tiffiles),1) #reshape to (200,1) because this is what HDF5 is saving
        
        #getting offset, the offset is the same for all layers in a stack, also among the stacks in a timeseries the the offset is constant
        offset = ast.literal_eval(tags1[0]['SI.hScan2D.channelOffsets'].replace(' ' , ','))
    else: #loading from a single file that contains a timeseries that includes only 2D taken from same plane. Image.shape = [time, x, y]

        if 'json_data' in locals():
            img = applyTransformation(os.path.join(targetdirectory, tiffiles[0]), json_data[time1])
        else:
            img = cc2.load_tif(os.path.join(targetdirectory, tiffiles[0]))
        for roi in maskdata['mask_index'].keys():
        #stack dimensions = roi size (number of pixels) x number of stacks in timeseries
            maskdata['intensity'].loc[roi]= np.zeros((len(maskdata['mask_index'][roi][0]), img.shape[0]), dtype = sampleimg.dtype)
        #get end time for each stack
        #tags1 has the same number of entries as images in the stack (tags1[0] == first image)
        tags1 = getTifffilesMetadata(os.path.join(targetdirectory, tiffiles[0]))
        for roi in maskdata['mask_index'].keys():
            for time1 in range(img.shape[0]):
                maskdata['intensity'].loc[roi][:, time1] = img[time1, :, :].flatten()[ maskdata['mask_index'].loc[roi][0]]
        MIPimg[0, :, :] = np.max(img, axis =0) #saving matplotlib figure
        #get last time point for last image
        timeStamp = np.arange(0, getEndTimes(tags1)[-1], getEndTimes(tags1)[-1]/img.shape[0])

        
        #getting offset, the offset is the same for all layers in a stack, also among the stacks in a timeseries the the offset is constant
        if 'SI.hScan2D.channelOffsets' in list(tags1[0].keys()): #key found in bigtif
            offset = ast.literal_eval(tags1[0]['SI.hScan2D.channelOffsets'].replace(' ' , ','))
        else: #tag name found in older tif versions
            offset = ast.literal_eval(tags1[0]['scanimage.SI.hChannels.channelOffset'].replace(' ' , ','))

    #get stimulation data from file
    matfiles, index = getFileContString(targetdirectory, 'stim.mat')
    if len(matfiles) > 0:
        stimdata = loadmat(os.path.join(targetdirectory, matfiles.values[0]))
        stimdata = stimdata['AOBuffer']
    else:
        stimdata =np.empty(img.shape[0])
        
    shape = (len(tiffiles), sampleimg.shape[0], sampleimg.shape[1], sampleimg.shape[2])
    return maskdata, timeStamp, stimdata, MIPimg, shape, offset


    
def getROIDataFromHDF5(targetfile, maskdata):
    #pathfile = os.path.join(cd, file1)
    
    HDF5_file = tables.open_file(targetfile, mode='r')

    #access timeseries data
    timeseries = HDF5_file.root.data
    
    maskdata['intensity'] = ''
    for roi in maskdata['mask_index'].keys():
        #stack dimensions = roi size (number of pixels) x number of stacks in timeseries
        maskdata['intensity'].loc[roi]= np.zeros((len(maskdata['mask_index'][roi][0]), timeseries.shape[0]), dtype = timeseries.dtype)
         
    #generate an image that will be used in figure = include a MIP for each stack in the timeseries
    MIPimg = np.zeros(( timeseries.shape[0], timeseries.shape[2], timeseries.shape[3]), dtype = timeseries.dtype)
    for time1 in range(timeseries.shape[0]):
        stack = timeseries[time1, :, :, :]
        stack = np.rollaxis(stack, 2, 1)
        #stack = np.rot90(stack, 1)
        #plt.imshow(np.max(stack, axis =0))
        MIPimg[time1, :, :] = np.max(stack, axis =0) #saving matplotlib figure
        for roi in maskdata['mask_index'].keys():
            maskdata['intensity'].loc[roi][:, time1] = stack.flatten()[ maskdata['mask_index'].loc[roi][0]]
    
    #get time data
    timeStamp = HDF5_file.root.timeEnd
    timeStamp = timeStamp[:]

    #get stimulation data from file
    stimdata = HDF5_file.root.voltage
    stimdata = stimdata[:]
    shape = timeseries.shape
    HDF5_file.close()
    metadata = pd.read_hdf(targetfile, 'metadata')
    offset = ast.literal_eval(metadata['SI.hScan2D.channelOffsets'].iloc[0].replace(' ' , ','))
    return maskdata, timeStamp, stimdata, MIPimg, shape, offset

def getSummaryImages(img_files, targetdir):
    #img_files = pandas dataframe with row = individual images and columns name = where images can be found
    #need to determine whether the files are hdf5 or tif
    #img_files = seriesdata['TimeSeries_Image_Files']
    #targetdir = seriesdata['Path']
        
    if 'Image.hdf5' in img_files[0]:
        #sometimes path file name change so need to search again for #####.hdf5 file

        HDF5_file = tables.open_file(os.path.join(targetdir, img_files[0]), mode='r')

        #pdb.set_trace()
        if len(HDF5_file.root.data.shape) < 5:
            timeseries = HDF5_file.root.data[:, : , :,:]
        else:
            imeseries = HDF5_file.root.data[:, : , :,:, 1 ]
        stdImg = np.std(timeseries, axis = 0)
        #images were rotated for correct orientation in pyqtgraph
        stdImg = stdImg.max(axis=0)
        stdImg = stdImg.T
        MIP = timeseries.max(axis=0)
        MIP = MIP.max(axis=0)
        MIP = MIP.T
     
    elif '.tif' in img_files[0]:
        #determine if json file exists for transformation
        jsonfiles, index = getFileContString( targetdir, 'dr.3x350.json') 
        if len(jsonfiles) >0:
            with open(os.path.join(targetdir, jsonfile)) as json_data:
                d = json.load(json_data)
            json_data = np.asarray(d, dtype = np.int)
            firstimage = applyTransformation(os.path.join(targetdir,img_files[0]), json_data[0])
            timeseries = np.zeros((len(img_files), firstimage.shape[0], firstimage.shape[1], firstimage.shape[2]), dtype = firstimage.dtype)
            timeseries[0, :, :, :] = firstimage
            for row in range(0, len(img_files)):
                timeseries[row, :, :, :] = applyTransformation(os.path.join(targetdir, img_files[img_files.columns[0]].iloc[row]), json_data[row])
        else:
            
            firstimage = np.squeeze(imread(os.path.join(targetdir, img_files[0])))
            SUM = np.zeros((firstimage.shape[0], firstimage.shape[1], firstimage.shape[2]))
            MIP =  np.zeros([2, firstimage.shape[0], firstimage.shape[1], firstimage.shape[2]])
            for cfile in img_files:
                cimg =imread(os.path.join(targetdir, cfile)).compute()
                SUM = SUM + cimg
                MIP[1] = cimg
                MIP[0] = MIP.max(axis=0)
            MEAN = SUM / len(img_files)
            SQUARE = np.zeros((firstimage.shape[0], firstimage.shape[1], firstimage.shape[2]))
            for cfile in img_files:
                cimg =imread(os.path.join(targetdir, cfile)).compute() 
                SQUARE = SQUARE + np.square(cimg - MEAN)
            MEAN_SQUARE = SQUARE / len(img_files)
            STANDARD_DEVIATION = np.sqrt(MEAN_SQUARE)
        MIP = MIP[0].squeeze()
        MIP = MIP.max(axis=0)
        STANDARD_DEVIATION = np.squeeze(STANDARD_DEVIATION)
        STANDARD_DEVIATION = STANDARD_DEVIATION.max(axis=0) 
        #MIP = MIP.T
        #STANDARD_DEVIATION = STANDARD_DEVIATION.T

    return MIP, STANDARD_DEVIATION


    

def makeTimeSeriesFig(dataframeposition, seriesdata, outputfolder, rois):
    #dataframeposition = row in dataframe containing all combined excel sheets 
    #seriesdata = row from exceldata frame now a series
    #outputfolder = location to place fig
    #seriesdata = exceldata.loc[row]
    #row = exceldata.index[0]
    #plt.close('all')
    img_files = seriesdata['TimeSeries_Image_Files']
    MIP, stdImg = getSummaryImages(img_files, seriesdata['Path'])
    fig1=plt.figure(figsize=(10,8))
    #get voltage and timestamp
    hdf5 = tables.open_file(seriesdata['Intensity_Data_File'])
    voltage = hdf5.root.voltage[:]
    timeStamp = hdf5.root.timeStamp[:]
    hdf5.close()
    intensity_data = pd.read_hdf(seriesdata['Intensity_Data_File'], 'intensity_data')
    ## generate title for page
    ax1 = fig1.add_axes([0.01, 0.95, 0.9, 0.5])
    title2 = '{0:04d}'.format(dataframeposition) + '-' + '%04d' % seriesdata['No.'] + '-' + seriesdata['Sample Name'] + ' Genotype: ' + str(seriesdata['Genotype'])
    ax1.text(0,0, title2)
    ax1.axis('off') 

    #img_files = pd.read_hdf(seriesdata['Intensity_Data_File'], 'image_files')
    
    
    ## generate a series of axes with the signal from each roi
    grouped = intensity_data.groupby(['Name'], axis=0).groups 
    #determine how many and size of each axis
    numRois = len(grouped)
    if numRois > 0:
        ysize = 0.8 / numRois
    else:
        ysize = 0.8
    
    #add axis and plot
    for i, cg in enumerate(grouped):
        ax2=fig1.add_axes([0.1, 0.1+(i)*ysize, 0.5, ysize])
        for row in grouped[cg]: #if roi is found in the given roi color scheme use this color otherwise use a default color
            if intensity_data['Name'].loc[row] in rois.keys():
                color1 = rois[intensity_data['Name'].loc[row]]
            else:
                color1 = [1, 0, 1]
            ax2.plot(timeStamp, np.mean(intensity_data['intensity'].loc[row], axis=0), color = color1)
            #remove axis if no the first
            if i !=0:
                ax2.axes.get_xaxis().set_visible(False)
            ax2.set_ylabel('Raw Intensity ' + cg)
            ax21 = ax2.twinx()
            ax21.set_ylabel('LED Power (V)')
            stimTimesRange = np.array(range(len(voltage)))/100.0
            ax21.plot(stimTimesRange,  voltage, color = 'r', linestyle = '--', alpha=0.4)
            #ax2.set_ylim(np.min(seriesdata['RawTimeSeries'][seriesdata['RawTimeSeries'] != 0]), np.max(seriesdata['RawTimeSeries']))
            #ax2.set_xlim(0, 205)
            if i == 0:
                ax2.set_xlabel('Time (s)')
            #ax2.set_title('Raw Intensity Traces')
    
    # show standard deviation image 
    ## get image standard deviation and MIP
    ## need to load either 
    #get color image
    #get color image
    color = generateMIPfromHDFcolor(seriesdata['Two_Colored_Image'])
    color = skimage.exposure.adjust_gamma(color, 0.4)
    ax3 = fig1.add_axes([0.63, 0.5, 0.4, 0.4])
    #ax3.imshow(stdImg, cmap='Greys_r')
    ax3.imshow(color)
    ax3.set_aspect('equal')
    ax3.axis('off') 
    #ax3.set_title('Standard Deviation over time')
    ax3.set_title('Increased laser power 2 Channel Image')
    
    #Generate a MIP + STDEV color image
    '''
    all1 = np.hstack([MIP, stdImg])
    max1 = np.max(all1)
    min1 = np.min(all1)
    '''
    
    ax4 = fig1.add_axes([0.63, 0.05, 0.4, 0.4])
    ax4.imshow(MIP, cmap='Greys_r')
    ax4.set_aspect('equal')
    ax4.axis('off') 
    ax4.set_title('MIP over time')
    
    
 
    #outline rois
    for roi in intensity_data.index:
        #roi=intensity_data.index[7]
        mask1 = np.zeros(intensity_data['image_shape'].loc[roi][1:-1]).flatten()
        if len(intensity_data['mask_index'].loc[roi][0] ) >0 : #roi must contain more than one pixel
            mask1[intensity_data['mask_index'].loc[roi] ]=1
            #mask1 = np.flipud(mask1)
            mask1 = mask1.reshape(intensity_data['image_shape'].loc[roi][1:-1])
            mask1 = np.sum(mask1, axis = 0)
            mask1[mask1 > 0 ] = 1
            contours = skimage.measure.find_contours(mask1, 0.8)
            if intensity_data['Name'].loc[roi] in rois.keys():
                color1 = rois[intensity_data['Name'].loc[roi]]
            else:
                color1 = [1, 0, 1]
            for n, contour in enumerate(contours):
                ax3.plot(contour[:, 1], contour[:, 0], linewidth = 2, color = color1)
                ax4.plot(contour[:, 1], contour[:, 0], linewidth = 2, color = color1)
            lbl = scipy.ndimage.label(mask1)
            indexC = scipy.ndimage.center_of_mass(mask1)
            ax3.text(indexC[1], indexC[0], intensity_data['Name'].loc[roi], color=color1)
            ax4.text(indexC[1], indexC[0], intensity_data['Name'].loc[roi], color=color1)
    #plt.show()
    '''
    # standard deviation with mask applied
    stack = np.zeros(intensity_data['image_shape'].loc[roi][1:4]).flatten()
    timeseries = np.zeros((intensity_data['image_shape'].loc[roi][1]*intensity_data['image_shape'].loc[roi][2]*intensity_data['image_shape'].loc[roi][3], intensity_data['intensity'].iloc[0].shape[1]), dtype=intensity_data['intensity'].iloc[0].dtype)
    for roi in intensity_data.index:
        stack[intensity_data['mask_index'].loc[roi][0]] = np.std(intensity_data['intensity'].loc[roi],axis = 1)
        timeseries[intensity_data['mask_index'].loc[roi][0]] = intensity_data['intensity'].loc[roi]
    ax4 = fig1.add_axes([0.8, 0.5, 0.2, 0.4])
    ax4.imshow(np.max(stack.reshape(intensity_data['image_shape'].loc[roi][1:4]), axis=0), cmap='Greys_r')
    ax4.set_aspect('equal')
    ax4.axis('off') 
    
    #create plot show normalized intensity values
    ax5 = fig1.add_axes([0.08, 0.1, 0.4, 0.35])
    if intensity_data['Name'].str.contains('Background').any():
        background = np.median(intensity_data['intensity'].loc[intensity_data['Name'].str.contains('Background')][0], axis=0)
    for row in intensity_data.index:
        data = np.median(intensity_data['intensity'].loc[row], axis=0)
        if 'background' in locals():
            data = data - background
        data = data / np.median(data)      
        ax5.plot(timeStamp[10:], data[10:], color = rois[intensity_data['Name'].loc[row]], alpha = 0.5)
    ax51 = ax5.twinx()
    ax51.set_ylabel('LED Power (V)')
    stimTimesRange = np.array(range(len(voltage)))/100.0
    ax51.plot(stimTimesRange,  voltage, color = 'r', linestyle = '--', alpha=0.4)
    if 'background' in locals():
        ax5.set_title("Normalized Data - Background ROI")
    else:
        ax5.set_title("Normalized Data")

    #ax2.set_ylim(np.min(seriesdata['RawTimeSeries'][seriesdata['RawTimeSeries'] != 0]), np.max(seriesdata['RawTimeSeries']))
    ax2.set_xlim(0, 205)
    ax2.set_ylabel('Fluorescence')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Raw Intensity Traces')
    
    #add MIP images comparing times during stimulation
    #find were stimulation occurred
    voltage[voltage > 0] = 1
    stimStart = np.where(np.diff(voltage, axis=0)==1)[0]+1 # start for each stim period
    stimEnd = np.where(np.diff(voltage, axis=0)==-1)[0] # end for each stim period
    ax4 = fig1.add_axes([0.6, 0.1, 0.4, 0.4])
    periodToCompare = -1
    baselineperiod = (stimEnd[periodToCompare] -  stimStart[periodToCompare]) / 100
    startB = stimStart[periodToCompare]/100.0 - baselineperiod
    startBi = int(np.where(timeStamp >= startB)[0][0])
    endB = stimStart[periodToCompare]/100.0
    endBi = int(np.where(timeStamp <= endB)[0][-1])
    start1 = stimStart[periodToCompare]/100.0
    start1i = int( np.where(timeStamp >= start1)[0][0])
    end1 = stimEnd[periodToCompare]/100.0 + baselineperiod
    end1i= int(np.where(timeStamp <= end1)[0][-1])
    timeseries = timeseries.reshape(intensity_data['image_shape'].loc[roi][1], intensity_data['image_shape'].loc[roi][2], intensity_data['image_shape'].loc[roi][3], intensity_data['intensity'].iloc[0].shape[1])
    imgC = np.concatenate((np.max(np.mean(timeseries[:,:,:, startBi:endBi], axis=3), axis=0), np.max(np.mean(timeseries[:, :, :, start1i:end1i], axis=3), axis=0)), 1)
    ax4.imshow(imgC, cmap='Greys_r')
    ax4.axis('off') 
    ax4.set_aspect('equal')
    ax4.set_title('Compare sum intensity levels over ' + str(int(np.floor(stimStart[periodToCompare]/100.0 - baselineperiod))) + ':' + str(int(np.floor(stimStart[periodToCompare]/100.0))) + ' and ' +  str(int(np.ceil(stimStart[periodToCompare]/100.0))) +  ':' + str(int(np.ceil(stimEnd[periodToCompare]/100.0+ baselineperiod ))))
    '''
    #save figure
    path3, foldername = os.path.split(seriesdata['Paths'])
    fig1.savefig(os.path.join(outputfolder, '{0:04d}'.format(dataframeposition) + '_' + str(seriesdata['Genotype']) + '_' + '{0:04d}'.format(int(seriesdata['No.'])) + '-' + foldername + '.jpeg'))
    plt.close(fig1)

def compileTimeSeriesData(seriesdata):
    hdf5 = tables.open_file(seriesdata['Intensity_Data_File'])
    seriesdata['voltage'] = hdf5.root.voltage[:]
    seriesdata['timestamp'] = hdf5.root.timeStamp[:]
    hdf5.close()
    intensity_data = pd.read_hdf(seriesdata['Intensity_Data_File'], 'intensity_data')
    dict1 = intensity_data.to_dict()
    dict1.pop("Color")
    #convert intensity into mean values for all pixels
    #remove rois consisting of only one pixel
    intensityFrame = pd.DataFrame(dict1)
    index2 =[1 if len(ar[0]) > 1 else 0 for ar in intensityFrame['mask_index'].values]
    intensityFrame = intensityFrame[np.array(index2, dtype=np.bool)]
    #merge rois with the same name and average over all pixels
    groups = intensityFrame.groupby(['Name']).groups
    newFrame = pd.DataFrame(columns = intensityFrame.columns, index = list(groups.keys()))
    for roi, dframe in intensityFrame.groupby(['Name']):
        newFrame['Name'].loc[roi] = roi
        newFrame['Type'].loc[roi] = dframe['Type'].values
        newFrame['image_file'].loc[roi] = dframe['image_file'].iloc[0]
        newFrame['image_shape'].loc[roi] = dframe['image_shape'].iloc[0]
        newFrame['intensity'].loc[roi] = np.mean(np.vstack(dframe['intensity']), axis =0)
        newFrame['mask_index'].loc[roi] = dframe['mask_index'].values

    seriesdata['intensity_data'] = newFrame.to_dict()
    
    return seriesdata


class fluorescent2():
    #used to study fluorescent data
    # data = shape[animals , times]
    def __init__(self, data, start, response, timeStamp1):
        self.data = data            # data is fluorescent data (trial, time)
        self.start = start         #start = when stimulation started, I subtract 1 because I do not want to include the actual stim period
        self.response = response    #expected response either 'neg' (reduced fluorescence) or 'pos' (increased response to stimulus
        self.timeStamp=timeStamp1
        
    def deltaFF(self):
        data1 = self.data
        preF = np.mean(data1[:, 0 : self.start], axis=1)
        dF = data1.T - preF
        dFF = dF/preF
        #print(dFF.shape)
        for row in range(dFF.shape[1]):
            dFF[:,row] = nd.filters.gaussian_filter1d(dFF[:,row], 1)
        return dFF.T 
    
    def deltaF(self):
        data1 = self.data
        preF = np.mean(data1[:, 0 : self.start], axis=1)
        dF = data1.T - preF
        #print(dFF.shape)
        for row in range(dF.shape[1]):
            dF[:,row] = nd.filters.gaussian_filter1d(dF[:,row], 1)
        return dF.T

    def removeSign(self):
        data1 = self.deltaFF()
        if self.response is 'neg':
            data1[data1>0] = np.NaN
        elif self.response is 'pos':
            data1[data1<0] = np.NaN
        else:
            data1 = np.absolute(data1)
        return data1
    
    def removeSign_dF(self):
        data1 = self.deltaF()
        if self.response is 'neg':
            data1[data1>0] = np.NaN
        elif self.response is 'pos':
            data1[data1<0] = np.NaN
        else:
            data1 = np.absolute(data1)
        return data1
    
    def Max(self):
        '''
        data1 = self.removeSign()
        return np.nanmax(np.absolute(data1[:, self.start:]), axis = 1) 
        '''
        
        data1 = self.removeSign()
        data1[np.isnan(data1)] = 0
        data1 = np.abs(data1)
        #data1.sort(axis=1)
        #max1 = np.nanmedian(data1[:, -10:], axis =1)
        max1 = np.nanmax(data1, axis=1)
        
        #if array is all NaN then max=0, convert such values to Nan
        max1[max1 == 0] = np.nan
        #pdb.set_trace()
        return max1
    
    def Max_dF(self):
        '''
        data1 = self.removeSign()
        return np.nanmax(np.absolute(data1[:, self.start:]), axis = 1) 
        '''
        #pdb.set_trace()
        data1 = self.removeSign_dF()
        data1[np.isnan(data1)] = 0
        data1 = np.abs(data1)
        #data1.sort(axis=1)
        #max1 = np.nanmedian(data1[:, -10:], axis =1)
        max1 = np.nanmax(data1, axis=1)
        #if array is all NaN then max=0, convert such values to Nan
        max1[max1 == 0] = np.nan
        return max1
        
          
    
    def Mean(self):
        data1 = self.removeSign()
        return np.nanmean(np.absolute(data1[:, self.start:]), axis = 1)
    
    def SNR(self): #signal to noise ratio of the deltaF/F signal
        #data1 = self.removeSign()
        #return np.nanmax(np.absolute(data1[:, :self.start:]), axis=1) / np.nanstd(np.absolute(data1[:, :self.start]), axis =1)
        return self.Max() / self.baseline_SD()
    
    def SNR3(self): #signal to noise ratio of the deltaF/F signal
        #data1 = self.removeSign()
        data1 = self.removeSign()
        #return np.nanmax(np.absolute(data1[:, :self.start:]), axis=1) / np.nanstd(np.absolute(data1[:, :self.start]), axis =1)
        #pdb.set_trace()
        return self.Max() / np.nanmean(data1[:, :self.start], axis =1)

    def SNR2(self): #signal to noise ratio of the raw intensity values
        data1 = self.data
        if self.response is 'neg':
            snr = np.min(data1[:, self.start:], axis = 1) / np.nanstd(data1[:, :self.start], axis =1)
        elif self.response is 'pos':
            snr = np.max(data1[:, self.start:], axis = 1) / np.nanstd(data1[:, :self.start], axis =1)
        return snr
        
    def SNR4(self): #signal to noise ratio of the raw intensity - Background values
        data1 = self.backgroundCorrected()
        if self.response is 'neg':
            snr = np.min(data1[:, self.start:], axis = 1) / np.nanstd(data1[:, :self.start], axis =1)
        elif self.response is 'pos':
            snr = np.max(data1[:, self.start:], axis = 1) / np.nanstd(data1[:, :self.start], axis =1)
        return snr

    def SNR_dF(self): #signal to noise ratio of the deltaF/F signal
        #data1 = self.removeSign_dF()
        #return np.nanmax(np.absolute(data1[:, :self.start:]), axis=1) / np.nanstd(np.absolute(data1[:, :self.start]), axis =1)
        return self.Max_dF() / self.baseline_SD_dF()
    
    def delay2Max(self):
        data1 = self.removeSign()
        #print(data1.shape)
        t1=np.zeros(data1.shape[0])
        for i in range(self.data.shape[0]):
            if not np.all(np.isnan(data1[i, self.start:])):
                t1[i] = self.timeStamp[np.nanargmax(np.absolute(data1[i, self.start:]))] -self.timeStamp[0] 
            else:
                t1[i] = np.NaN
            
        return t1
    
    def Median(self):
        data1 = self.removeSign()
        return np.nanmedian(np.absolute(data1[:, self.start:]), axis = 1)

    def baseline_intensity(self):
        return np.mean(self.data[:, :self.start])

    def baseline_SD(self):
        data1 = self.removeSign()
        return np.nanstd(data1[:, :self.start], axis =1)

    def baseline_SD_dF(self):
        data1 = self.removeSign_dF()
        return np.nanstd(data1[:, :self.start], axis =1)
        
    def baseline_SD_Raw(self):
        data1 = self.data
        return np.nanstd(data1[:, :self.start], axis =1)


class fluordF():
    #used to study fluorescent data
    # data = shape[animals , times]
    def __init__(self, data, background, start, response, timeStamp1):
        self.data = data            # data is fluorescent data (trial, time)
        self.background = background        #background should be the background from region with endogenous fluorescence
        self.start = start          #start = when stimulation started
        self.response = response    #expected response either 'neg' (reduced fluorescence) or 'pos' (increased response to stimulus
        self.timeStamp1=timeStamp1
        
    def deltaF(self):
        data1 = self.backgroundCorrected()
        preF = np.mean(data1[:, 0 : self.start], axis=1)
        dF = data1.T - preF
        #dFF = dF/preF
        #print(dFF.shape)
        for row in range(dF.shape[1]):
            dF[:,row] = nd.filters.gaussian_filter1d(dF[:,row], 1)
        return dF.T 
    
        
    def backgroundCorrected(self):
        corrected = self.data - self.background
        corrected[corrected < 0 ] =0
        return corrected
        
    def removeSign(self):
        data1 = self.deltaF()
        if self.response is 'neg':
            data1[data1>0] = np.NaN
        elif self.response is 'pos':
            data1[data1<0] = np.NaN
        else:
            data1 = np.absolute(data1)
        return data1
    
    def Max(self):
        '''
        data1 = self.removeSign()
        return np.nanmax(np.absolute(data1[:, self.start:]), axis = 1) 
        '''
        #pdb.set_trace()
        data1 = self.removeSign()
        data1 = np.absolute(data1[:, self.start:])
        data1[np.isnan(data1)] = 0
        data1.sort(axis=1)
        max1 = np.nanmedian(data1[:, -10:], axis =1)
        #if array is all NaN then max=0, convert such values to Nan
        max1[max1 == 0] = np.nan
        return max1
        
          
    
    def Mean(self):
        data1 = self.removeSign()
        return np.nanmean(np.absolute(data1[:, self.start:]), axis = 1)
    
    def SNR(self): #signal to noise ratio
        data1 = self.removeSign()
        return np.nanmax(np.absolute(data1[:, self.start:]), axis = 1) / np.nanstd(np.absolute(data1[:, :self.start]), axis =1)
    
    def delay2Max(self):
        data1 = self.removeSign()
        #print(data1.shape)
        t1=np.zeros(data1.shape[0])
        for i in range(self.data.shape[0]):
            if not np.all(np.isnan(data1[i, self.start:])):
                t1[i] = self.timeStamp1[np.nanargmax(np.absolute(data1[i, self.start:]))] -self.timeStamp1[0] 
            else:
                t1[i] = np.NaN
            
        return t1
    
    def Median(self):
        data1 = self.removeSign()
        return np.nanmedian(np.absolute(data1[:, self.start:]), axis = 1)

    def baseline_SD(self):
        return np.nanstd(np.absolute(data1[:, :self.start]), axis =1)


        
def confirmROIs(path, roilist):
    #confirm that all the rois have been selected for each sample
    #dataseries where each row as the path/file to each intensity_data file
        #dataseries = exceldata['Intensity_Data_File']
    #roilist = list of all the rois that should be selected
        #roilist = list(rois.keys())
    intensity_data = pd.read_hdf(path, 'intensity_data')
    #count the occurrences
    count = {}
    for roi in roilist:
        count[roi] = np.sum(intensity_data['Name'].str.contains(roi))
    return count    


def translateMatplotlibROI_to_PyQtFormat(cfolder):
    common_end = '_Mask.npy'
    fileaddon = 'Mask.hdf5'
    columns = ['Name', 'Color', 'Type', 'Z:XY', 'mask_index', 'image_shape','image_file']
    colors = {'Body': [0,1,0], 'M1':[1,0, 1], 'M4': [1, 1, 0], 'M8-10': [1, 0, 0]}
    #translate the old ROIs into the new format
    #cfolder = /path/ to directory with old rois
    #creates and new file with tif_name + Mask.hdf5 that can be loaded into circuit catcher
    files, index = getFileContString(cfolder, common_end)
    #create the pandas data frame to contain data
    cmaskdata = pd.DataFrame(index = range(len(files)), columns = columns)
    #fill in other columns
    #find the tif file
    tif_files, index = getFileContString(cfolder, '.tif')
    cmaskdata['image_file'] = os.path.join(cfolder, tif_files.values[0])
    img = tifffile.imread(os.path.join(cfolder, tif_files.values[0]))
    ZXY = []
    for i, cfile in enumerate(files):
        XY = np.load(os.path.join(cfolder, cfile))
        mask = np.zeros([img.shape[1], img.shape[2]] ,dtype = np.bool)
        mask[XY[0], XY[1]] =1
        mask = mask.T
        contours = skimage.measure.find_contours(mask, 0.8)[0]
        #have to down sample contours because there are too many points
        downsamplerangecontours = range(0, contours.shape[0], 15)
        contours = contours[downsamplerangecontours, :]
        print(contours.shape)
        ZXY.append( {0 : contours.tolist()})
        cmaskdata['Name'].loc[i] = cfile[:-9]
        #mask = mask.T
        cmaskdata['mask_index'].loc[i] = np.where(mask.T.flatten())
        cmaskdata['Color'].loc[i] = colors[cfile[:-9]]
    cmaskdata['Z:XY'] = ZXY
    cmaskdata['image_file'] = os.path.join(cfolder, files.values[0])
    cmaskdata['image_shape'] = pd.Series( [(img.shape[0], 1, img.shape[1],img.shape[2], 3) for x in range(len(cmaskdata))])
    cmaskdata['Type'] = 'polyArea'
    pathfile = os.path.join(cfolder, tif_files.values[0])[:-4] + fileaddon
    cmaskdata.to_hdf(pathfile, 'roi')

def translateImageJROIs_to_cc(targetfile, img_shape = [45, 512, 512], colors=None):
    #targetfile = path/roi.zip as created by imageJ
    rois = read_roi.read_roi_zip(targetfile)
    if colors == None:
        rgb = getColors(len(rois))
        colorDict = dict(zip(rois.keys(), rgb))
    fileaddon = 'Mask.hdf5'
    columns = ['Name', 'Color', 'Type', 'Z:XY', 'mask_index', 'image_shape','image_file']
    cmaskdata = pd.DataFrame(index = range(len(rois)), columns = columns)
    #fill in other columns
    ZXY = []
    for i, croi in enumerate(rois.keys()):
        #get x y coordinates for polygon
        cmaskdata.loc[i] = ''
        x = rois[croi]['x']
        y = rois[croi]['y']
        layer = int(croi[:4])
        ZXY.append( {layer : np.array([x, y]).transpose().tolist()})
        #get binary mask
        mask1 = np.zeros([img_shape[1], img_shape[2]] ,dtype = np.bool)
        xx, yy = polygon(y, x)
        mask1[xx, yy]=1
        #plt.imshow(mask1)
        mask3 = np.zeros(img_shape, dtype = np.bool)
        mask3[layer-1, :, :] = mask1
        cmaskdata['mask_index'].loc[i] = np.where(mask3.flatten())
    
        cmaskdata['Name'].loc[i] = croi
        cmaskdata['Color'].loc[i] = colorDict[croi]
    cmaskdata['Z:XY'] = ZXY
    cmaskdata['image_file'] = ''
    cmaskdata['image_shape'] = [img_shape] * len(rois)
    cmaskdata['Type'] = 'polyArea'
    path, file = os.path.split(targetfile)
    pathfile = os.path.join(path, file[:-4] + fileaddon)
    cmaskdata.to_hdf(pathfile, 'roi')

    
    
def getColors(num):
    #get num of distinct colors
    cm = plt.get_cmap('gist_rainbow')
    cNorm = matplotlib.colors.Normalize(vmin = 0, vmax=num-1)
    scalarMap = matplotlib.cm.ScalarMappable(norm = cNorm, cmap = cm)
    return [scalarMap.to_rgba(i) for i in range(num)]
    
    
def normalize(array):
    array = array.astype(np.float64)
    array = (array - np.min(array)) 
    array = array / np.max(array)
    return array

def generateHDF_from_2colorTif(output, RGB_image, TimeSeries_Path = None):
    #RGB_image = full file path to tif image including image
    #output = output path for .hdf5 file
    #if os.path.isfile(RGB_image):
    #get image data from colored image - high laser power taken after timeseries
    img = imread(RGB_image).compute()
    #create an rgb image
    shape = img.shape
    rgb = np.zeros([1, int(shape[1] /2), shape[2], shape[3], 3], dtype = np.float64)
    red_range = np.arange(0, shape[1], 2)
    green_range = np.arange(1, shape[1], 2)
    green = normalize(img[:, green_range, :, :])
    red = normalize(img[:, red_range, :, :])
    #get timeseries data
    if os.path.isdir(TimeSeries_Path):
        tif_files, index = getFileContString(TimeSeries_Path, '.tif')
        blue = getMaxInTime_from_TIF(TimeSeries_Path, tif_files.tolist())
        blue = normalize(blue)
    
        
    #for layer in green_range
    if 'green' in locals():
        rgb[:, :, :, :, 0] = green
        rgb[:, :, :, :, 1] = red
    if 'blue' in locals():
        rgb[:, :, :, :, 2] = blue[:42, :, :]
    
    rgb[rgb > 1] = 1
    rgb = np.rollaxis(rgb, 3, 2)
    rgb = skimage.img_as_int(rgb)
    hdf5_file = tables.open_file(output, mode='w')
    #filters = tables.Filters(complevel=5, complib='blosc') #faster but poor compression
    filters = tables.Filters(complevel=3, complib='zlib') #slower but better compression
    data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                  tables.Atom.from_dtype(rgb.dtype),
                                  shape = rgb.shape,
                                  filters=filters)
    data_storage[:] = rgb
    data_storage.flush()
    hdf5_file.close()

def generate2ColorArray(path):
    #path = full path to .tif file
    #return 2color array of shape [time, z, x, y, rgb]
    img = imread(path).compute()
    shape = img.shape
    rgb = np.zeros([1, int(shape[1] /2), shape[2], shape[3], 3], dtype = np.float64)
    red_range = np.arange(0, shape[1], 2)
    green_range = np.arange(1, shape[1], 2)
    green = normalize(img[:, green_range, :, :])
    red = normalize(img[:, red_range, :, :])
    rgb[:, :, :, :, 0] = green
    rgb[:, :, :, :, 1] = red
    return rgb

def generateMIPfromHDFcolor(path):
    #path = ///.2color_z_#####.HDF5 or from .tif file
    if '.hdf5' in path:
        HDF5_file = tables.open_file(path, mode='r')
        color = HDF5_file.root.data[:, : , :,:]
        color = np.transpose(np.squeeze(color), [0, 2,1, 3])
    else:
        color = generate2ColorArray(path)
    color = np.squeeze(color)
    color = color.max(axis = 0)
    color = skimage.img_as_ubyte(color)
    return color

def getMaxInTime_from_TIF(path, tif_files):
    #path = full file path to tif image including image
    #tif_files = [list of .tif files to be included]
    tif_files.sort()
    sample = np.squeeze(imread(str(Path(path) / tif_files[0])).compute())
    img = np.zeros([2, sample.shape[0], sample.shape[1],sample.shape[2]], dtype=sample.dtype)
    img[0] = sample
    for cfile in tif_files[1:]:
        sample = np.squeeze(imread(str(Path(path) / cfile)).compute())
        img[1] = sample
        img[0] = np.max(img, axis =0)
    return np.squeeze(img[0])

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


def generateHDF_from_2colorTif2(output, path2color, TimeSeries_Path = None):
    #output = output path for .hdf5 file
    #path2color =  path to directory containing multiple images with 2 channels at high laser power
    #TimeSeries_path = path to directorying containing times series data with single channel
    #get list of tif files
    tif_files, index  = getFileContString(path2color, '.tif')
    rgb = getMeanInTime_from_TIF(path2color, tif_files.tolist())
    trgb = np.zeros([1, rgb.shape[0], rgb.shape[1], rgb.shape[2], 3], dtype= rgb.dtype)
    trgb[0] =rgb
    #plt.imshow(skimage.img_as_ubyte(np.max(rgb, axis=0)))
    #plt.imshow(skimage.img_as_ubyte(np.max(rgb[:, :, :, 0], axis=0)))
    #plt.imshow(skimage.img_as_ubyte(np.max(rgb[:, :, :, 1], axis=0)))
    #get timeseries data
    if os.path.isdir(TimeSeries_Path):
        tif_files, index = getFileContString(TimeSeries_Path, '.tif')
        blue = getMaxInTime_from_TIF(TimeSeries_Path, tif_files.tolist())
        blue = normalize(blue)
    #for layer in green_range
    if 'blue' in locals():
        trgb[:, :, :, :, 2] = blue[:, :, :]
    
    trgb[trgb > 1] = 1
    trgb = np.rollaxis(trgb, 3, 2)
    #plt.imshow(skimage.img_as_ubyte(np.squeeze(np.max(trgb, axis=1))))
    trgb = skimage.img_as_int(trgb)
    hdf5_file = tables.open_file(output, mode='w')
    #filters = tables.Filters(complevel=5, complib='blosc') #faster but poor compression
    filters = tables.Filters(complevel=3, complib='zlib') #slower but better compression
    data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                  tables.Atom.from_dtype(rgb.dtype),
                                  shape = trgb.shape,
                                  filters=filters)
    data_storage[:] = trgb
    data_storage.flush()
    hdf5_file.close()


def generateHDF_MIP_from_tif(path, tif_files, output,):
    #path = full file path to tif image including image
    #output = output path for .hdf5 file
    #tif_files = [list of .tif files to be included]
    tif_files.sort()
    if os.path.isfile(path):
        sample = np.squeeze(cc.imread(str(Path(path) / tif_files[0])).compute())
        img = np.zeros([len(tif_files), sample.shape[0], sample.shape[1],sample.shape[2]], dtype=sample.dtype)
        img[0] = sample
        for i, cfile in enumerate(tif_files[1:]):
            img[i+1] = cc.imread(str(Path(path) / cfile)).compute()
        img = np.max(img, axis = 0)
        img = np.rollaxis(img, 2,1 )
        #filters = tables.Filters(complevel=5, complib='blosc') #faster but poor compression
        hdf5_file = tables.open_file(output, mode='w')
        filters = tables.Filters(complevel=3, complib='zlib') #slower but better compression
        data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                      tables.Atom.from_dtype(img.dtype),
                                      shape = [1, img.shape[0], img.shape[1], img.shape[2]],
                                      filters=filters)
        data_storage[:] = img
        data_storage.flush()
        hdf5_file.close()
        
        
    
    else:
        return 'path is not a file'
    


def generateHDFfileRegistration(path, target_image):
    path2, file1 = os.path.split(path)
    hdf5_path = os.path.join(path, file1 + "Image.hdf5")
    if not os.path.isfile(hdf5_path):
        #get the tiff files
        tiffiles, index= getFileContString(path, '.tif')
        
        sample_data = tifffile.imread(os.path.join(path, tiffiles.values[0]))
    
        hdf5_file = tables.open_file(hdf5_path, mode='w')

        #filters = tables.Filters(complevel=5, complib='blosc') #faster but poor compression
        filters = tables.Filters(complevel=3, complib='zlib') #slower but better compression
        shape=(len(tiffiles), sample_data.shape[0], sample_data.shape[1], sample_data.shape[2])
        data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                      tables.Atom.from_dtype(sample_data.dtype),
                                      shape = shape,
                                      filters=filters)
        timeStart = hdf5_file.create_carray(hdf5_file.root, 'timeStart',
                                      tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                      shape = (len(tiffiles), 1),
                                      filters=filters)
        timeEnd = hdf5_file.create_carray(hdf5_file.root, 'timeEnd',
                                      tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                      shape = (len(tiffiles), 1),
                                      filters=filters)

        start_time  = time.time()
        min1 = [] #record min values
        max1 = [] #record max values
        stackmetadata = {}
        target_image = np.squeeze(imread(target_image))
        #tiffiles = tiffiles[:4]
        for i, cfile in enumerate( np.sort(tiffiles)):
            img = np.squeeze(imread(os.path.join(path, cfile)))
            shift, error, diffphase = register_translation(target_image, img)
            offset_image = fourier_shift(np.fft.fftn(img), shift)
            offset_image = np.fft.ifftn(offset_image)
            img = np.rollaxis(offset_image.real, 2, 1)
            data_storage[i, :, :, :] = img
            #data_storage[i, :, :, :, 1] = img
            #data_storage[i, :, :, :, 2] = img
            tags1 = getTifffilesMetadata(os.path.join(path, cfile))
            stackmetadata[i] = tags1
            timeStart[i] = float(tags1[0]['frameTimestamps_sec'])
            if i != 0:
                timeEnd[i-1] = float(tags1[0]['frameTimestamps_sec'])
            data_storage.flush()
            timeStart.flush()
            timeEnd.flush()
            min1.append(np.min(img))
            max1.append(np.max(img))
            

        

        ##need to add estimated final end time for last stack
        timeEnd[i] = getEndTimes(tags1)[-1]
        timeEnd.flush()
        data_storage2 = hdf5_file.create_carray(hdf5_file.root, 'minmax',
                                      tables.Atom.from_dtype(sample_data.dtype),
                                      shape = (2, 1),
                                      filters=filters)
        data_storage2[0] = np.min(min1)
        data_storage2[1] = np.max(max1)
        #get voltage data
        matfiles, index = getFileContString(path, 'stim.mat')
        if len(matfiles) > 0:
            stimdata = loadmat(os.path.join(path, matfiles.values[0]))
            stimdata = stimdata['AOBuffer']
        
            
            voltage_storage = hdf5_file.create_carray(hdf5_file.root, 'voltage',
                                          tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                          shape = stimdata.shape,
                                          filters=filters)
            voltage_storage[:] = stimdata
            voltage_storage.flush()

        
        #add stackmetadata to hdf5 file
        #first convert stackmetadata to pandas dataframe
        columns1 = ['Stack', 'Image']
        columns1.extend(list(stackmetadata[0][0].keys()))
        metaframe = pd.DataFrame(index =[], columns =columns1)
        for cstack in stackmetadata.keys():
            for cimg in stackmetadata[cstack].keys():
                ser1 = pd.Series(index=columns1)
                ser1['Stack'] = cstack
                ser1['Image'] = cimg
                for ctag in stackmetadata[cstack][cimg].keys():
                    ser1[ctag] = str(stackmetadata[cstack][cimg][ctag])
                metaframe = metaframe.append(ser1, ignore_index=True)
        metaframe.to_hdf(hdf5_path, '/metadata')
        '''
        how to read meta data
        hdf5_file = pd.read_hdf(hdf5_path, '/metadata', mode='r')
        '''
        
        '''
        for i, cfile in enumerate( tiffiles):
            os.remove(os.path.join(path, cfile))
        if os.path.isfile(os.path.join(path, matfiles.values[0])):
            os.remove(os.path.join(path, matfiles.values[0]))
        '''
                #save a file with maximum intensity image (MIP)
        MIP_Path = os.path.join(os.path.join(path, file1 + "_MIP.hdf5"))
        MIPfile = tables.open_file(MIP_Path, mode='w')        
        #pdb.set_trace()
        shape=(1, sample_data.shape[0], sample_data.shape[1], sample_data.shape[2])
        MIP_storage = MIPfile.create_carray(MIPfile.root, 'data',
                                      tables.Atom.from_dtype(img.dtype),
                                      shape = shape,
                                      filters=filters)
        for z in range(data_storage.shape[1]):
            MIP_storage[0, z, :, :] = np.max(data_storage[:, z, :, :], axis = 0)
            MIP_storage.flush()
        hdf5_file.close()
        MIPfile.close()
        
    
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


def generateHDFfromStackedTimeSeries(outputTime_Series, output_MIPtime, path, fastz_tif):

    sample_data = np.squeeze(imread(str(Path(path) / fastz_tif[0])).compute())

    hdf5_file = tables.open_file(outputTime_Series, mode='w')

    #filters = tables.Filters(complevel=5, complib='blosc') #faster but poor compression
    filters = tables.Filters(complevel=3, complib='zlib') #slower but better compression
    shape=(len(fastz_tif), sample_data.shape[0], sample_data.shape[1], sample_data.shape[2])
    data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                  tables.Atom.from_dtype(sample_data.dtype),
                                  shape = shape,
                                  filters=filters)
    timeStart = hdf5_file.create_carray(hdf5_file.root, 'timeStart',
                                  tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                  shape = (len(fastz_tif), 1),
                                  filters=filters)
    timeEnd = hdf5_file.create_carray(hdf5_file.root, 'timeEnd',
                                  tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                  shape = (len(fastz_tif), 1),
                                  filters=filters)

    min1 = [] #record min values
    max1 = [] #record max values
    stackmetadata = {}
    for i, cfile in enumerate( np.sort(fastz_tif)):
        print(str(Path(path) / cfile))
        img = np.squeeze(imread(str(Path(path) / cfile)).compute())
        shift, error, diffphase = register_translation(sample_data, img)
        offset_image = fourier_shift(np.fft.fftn(img), shift)
        offset_image = np.fft.ifftn(offset_image)
        img = np.rollaxis(offset_image, 2, 1)
        data_storage[i, :, :, :] = img
        tags1 = getTifffilesMetadata(os.path.join(path, cfile))
        stackmetadata[i] = tags1
        timeStart[i] = float(tags1[0]['frameTimestamps_sec'])
        if i != 0:
            timeEnd[i-1] = float(tags1[0]['frameTimestamps_sec'])
        data_storage.flush()
        timeStart.flush()
        timeEnd.flush()
        min1.append(np.min(img))
        max1.append(np.max(img))

    ##need to add estimated final end time for last stack
    timeEnd[i] = getEndTimes(tags1)[-1]
    timeEnd.flush()
    data_storage2 = hdf5_file.create_carray(hdf5_file.root, 'minmax',
                                  tables.Atom.from_dtype(sample_data.dtype),
                                  shape = (2, 1),
                                  filters=filters)
    data_storage2[0] = np.min(min1)
    data_storage2[1] = np.max(max1)
    #get voltage data
    matfiles, index = getFileContString(path, 'stim.mat')

    stimdata = loadmat(os.path.join(path, matfiles.values[0]))
    stimdata = stimdata['AOBuffer']


    voltage_storage = hdf5_file.create_carray(hdf5_file.root, 'voltage',
                                  tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                  shape = stimdata.shape,
                                  filters=filters)
    voltage_storage[:] = stimdata
    voltage_storage.flush()



    #add stackmetadata to hdf5 file
    #first convert stackmetadata to pandas dataframe
    columns1 = ['Stack', 'Image']
    columns1.extend(list(stackmetadata[0][0].keys()))
    metaframe = pd.DataFrame(index =[], columns =columns1)
    for cstack in stackmetadata.keys():
        for cimg in stackmetadata[cstack].keys():
            ser1 = pd.Series(index=columns1)
            ser1['Stack'] = cstack
            ser1['Image'] = cimg
            for ctag in stackmetadata[cstack][cimg].keys():
                ser1[ctag] = str(stackmetadata[cstack][cimg][ctag])
            metaframe = metaframe.append(ser1, ignore_index=True)
    metaframe.to_hdf(outputTime_Series, '/metadata')
    '''
    #how to read meta data
    #hdf5_file = pd.read_hdf(hdf5_path, '/metadata', mode='r')
    '''
    
    #save a file with maximum intensity image (MIP)
    MIPfile = tables.open_file(output_MIPtime, mode='w')        

    shape=(1, sample_data.shape[0], sample_data.shape[1], sample_data.shape[2])
    MIP_storage = MIPfile.create_carray(MIPfile.root, 'data',
                                  tables.Atom.from_dtype(data_storage.dtype),
                                  shape = shape,
                                  filters=filters)
    MIP_storage[0, :, :, :] = np.max(data_storage, axis = 0)
    MIP_storage.flush()

    MIPfile.close()
    hdf5_file.close()
    #delete the tif files
    for cfile in fastz_tif:
        path_file = Path(path) / cfile
        path_file.unlink()
    
