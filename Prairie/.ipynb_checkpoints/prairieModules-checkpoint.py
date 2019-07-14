import numpy as np
import pandas as pd
from skimage.draw import polygon #used in roi creation
import dask
import dask.array as da
from dask import delayed
from numba import jit
import os
import ccModules as cc1
from dask.array.image import imread
import tables
import skimage
import pdb
import xml.etree.ElementTree as ET


def generateHDFfileSingleTimeSeries(path):
    #designed for single plane images (not volume timeseries) acquired using the prairie scope
    #path = folder when image data is located
    #path = '/data/JData/A/A57_GtACR_Mi1/A57_Data/20180601_A57/20180601-62-3-flyh-002'
    path2, file1 = os.path.split(path)
    
    if not os.path.isfile(os.path.join(path, file1 + ".hdf5")) or not os.path.isfile(os.path.join(path, file1 + "_STDEV.hdf5")):
        tiffiles, index= cc1.getFileContString(path, '.tif')
        img_mat = imread(os.path.join(path, tiffiles.values[0]))
        #img_mat = img_mat[:300]

        
        #filters = tables.Filters(complevel=5, complib='blosc') #faster but poor compression
        filters = tables.Filters(complevel=3, complib='zlib') #slower but better compression

        hdf5_path = os.path.join(path, file1 + ".hdf5")
        hdf5_file = tables.open_file(hdf5_path, mode='w')
        shape=(len(tiffiles), 1, img_mat.shape[1], img_mat.shape[2], 3)
        data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                  tables.Atom.from_dtype(img_mat.dtype),
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


        img_mat = np.rollaxis(img_mat, 2, 1)
        data_storage[0, 0, :, :, 1] = np.squeeze(img_mat.compute())
        #load the rest of the images
        for z, cfile in enumerate(tiffiles.values[1:]):
            print(z)
            print(cfile)
            img_mat = imread(os.path.join(path, cfile))
            img_mat = np.rollaxis(img_mat, 2, 1)
            data_storage[z+1, 0, :, :, 0] = np.squeeze(img_mat.compute())
        
        standard_deviation = skimage.exposure.adjust_gamma(np.std(data_storage[:, 0 ,: , :, 0],axis = 0), 0.3)
        standard_deviation = (standard_deviation - np.min(standard_deviation)) / np.max(standard_deviation)*np.max(img_mat)
        data_storage[:, 0, :, :, 1] = standard_deviation.compute().astype(img_mat.dtype)
        data_storage[:, 0, :, :, 2] =  data_storage[:, 0, :, :, 1]
        #data_storage[:, 0, :, :, 0] = np.max(img_mat,axis = 0)
        #data_storage[:, 0, :, :, 2] = np.max(img_mat,axis = 0)
        data_storage.flush()
        #write in time data
        ##find the correct xml file containing time data
        ##should have same name as targetfolder + .xml
        folder, path = os.path.split(path)
        timefile, index= cc1.getFileContString(path, folder.values[0] + '.xml')
        relativeTime, absoluteTime = getTime(timefile.values[0])
        timeStart[:, 0] = relativeTime
        timeEnd[:, 0] = absoluteTime
        timeStart.flush()
        timeEnd.flush()
        #write in voltage
        voltfile, index= cc1.getFileContString(path, '_VoltageOutput_001.xml')
        xml_volt_file = os.path.join(path, voltfile.values[0])
        stimdata = getVoltage(xml_volt_file)
        voltage_storage = hdf5_file.create_carray(hdf5_file.root, 'voltage',
                                              tables.Atom.from_dtype(np.dtype('float16'), dflt=0),
                                              shape = stimdata.shape,
                                              filters=filters)
        voltage_storage[:] = stimdata
        voltage_storage.flush()
        hdf5_file.close()

        data_storage2 = hdf5_file.create_carray(hdf5_file.root, 'minmax',
                                      tables.Atom.from_dtype(img_mat.dtype),
                                      shape = (2, 1),
                                      filters=filters)
        data_storage2[0] = np.min(data_storage)
        data_storage2[1] = np.max(data_storage)
        hdf5_file.close()
        #add timeseries metadata to hdf5 file
        #first convert metadata to pandas dataframe



        '''
        how to read meta data
        hdf5_file = pd.read_hdf(hdf5_path, '/metadata', mode='r')
        '''

    
def write_Intensity_Data_File(mask_file, image_file, output_file):
    '''
    mask_file = roi data file as created by circuit catcher
    image_file = image file either as tif or hdf5
    '''
    #load roi data
    mask_frame = pd.read_hdf(mask_file)
    #load image intensity data
    ## determine what type of file holds image data
    if isinstance(image_file, str):
        if '.hdf5' in mask_file:
            timeseries = read_TimeSeries_HDF5(image_file).getIntensityData()
            
    #create empty arrays for each roi to hold intensity data
    mask_frame['intensity'] = ''
    for roi, dseries in mask_frame.iterrows():
        mask_frame['intensity'].loc[roi] = np.zeros(timeseries.shape[0])

    for time1 in range(timeseries.shape[0]):
        stack = timeseries[time1, :, :, :, 0]
        stack = np.rollaxis(stack, 2, 1)
        for roi, dseries in mask_frame.iterrows():
            #pdb.set_trace()
            mask_frame['intensity'].loc[roi][time1] = np.mean(stack.flat[mask_frame['mask_index'].loc[roi][0]])
    #save intensity_data
    mask_frame.to_hdf(output_file, 'intensity_data')
    '''
    #save other data
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
    
def getTime(xml_timefile):
    tree = ET.parse(xml_timefile)
    root = tree.getroot()
    absoluteTime = []
    relativeTime = []
    for ctime in root.iter('Frame'):
        absoluteTime.append(ctime.attrib['absoluteTime'])
        relativeTime.append(ctime.attrib['relativeTime']) 
    return(relativeTime, absoluteTime)

def getVoltage(xml_file, target_element = 'amber LED', time = 135 ):
    '''
    xml_file = path to xml file
    target_element = led used in stimulation
    time = total time in s for experiment
    return matrix similar to scanimage with intervales every 1 centisecond and equaling voltage
    '''
    tree = ET.parse(targetfile)
    root = tree.getroot()
    #the xml file is divided into waveforms. Each waveform contains a different stimulus.
    findwaves= root.findall('Waveform')   
    #find the waveform with the target stimulus
    targetwaveform =[]    
    for wave in findwaves:
        name1 = wave.findall('Name')
        #print(name1)
        for n in name1:
            if n.text == target_element:
                targetwaveform.append(wave)
    #get the stimulation data for each stimulus period in the waveform
    wavef = targetwaveform[0].findall('WaveformComponent_PulseTrain')
    timevariables = ['FirstPulseDelay', 'PulsePotentialStart', 'PulseCount', 'PulseWidth', 'PulseSpacing']
    timedict = {}
    for v in timevariables:
        timedict[v] = []
    for wave in wavef:
        for v in timevariables:
            vx = wave.findall(v)
            timedict[v].append(vx[0].text)
            
    frame1 = pd.DataFrame(timedict) 
    #create an index start stop
    frame1['start'] = ''
    frame1['stop'] = ''
    for row in range(len(frame1)):
        sum1 = frame1['FirstPulseDelay'].iloc[:row+1].as_matrix().astype(np.int64).sum()
        frame1['start'].iloc[row] = sum1 -1
        duration = int(frame1['PulseCount'].iloc[row]) * (int(frame1['PulseWidth'].iloc[row]) +int(frame1['PulseSpacing'].iloc[row]))
        frame1['stop'].iloc[row] = duration + frame1['start'].iloc[row] -1
    #generate a matrix with the same time s *100 (cs) as scanimage outputs
    matrix = np.zeros(time*100)
    for row, dseries in frame1.iterrows():
        matrix[int(frame1['start'].loc[row]/10) : int(frame1['stop'].loc[row]/10)] = frame1['PulsePotentialStart'].loc[row]
    return matrix