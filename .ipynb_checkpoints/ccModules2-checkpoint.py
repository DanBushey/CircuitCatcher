import numpy as np
import pandas as pd
from skimage.draw import polygon #used in roi creation
import dask
import dask.array as da
from dask import delayed
from numba import jit
import pathlib
import ccModules
import pdb
import matplotlib.pyplot as plt
from dask.array.image import imread 



def convertZXYtoMask(ZXY, image_shape):
    '''
    convert X Y Z points as drawn by a polygon into an index given the image_shape
    XYZ = dictionary with keys as layers and data as XY points
    img_shape = image shape generated by image.shape
    '''
    mask = np.zeros([image_shape[1], image_shape[3], image_shape[2]], dtype = np.bool) #believe I messed up x and y coordinates and now must reverse here to make compatible with later roi scripts
    for layer in ZXY.keys():
        if len(ZXY[layer]) == 1: #roi consist has only one X,Y point
            mask[layer, int(np.rint(ZXY[layer][0][0])), int(np.rint(ZXY[layer][0][1]))] = 1
        else:
            xy =  ZXY[layer]
            xy = np.vstack(xy)
            xx, yy = polygon(xy[: , 0], xy[: , 1])
            mask[layer,  yy, xx] = 1
    mask = np.where(mask.flatten())
    return mask
    



def pyqt_set_trace():
    '''Set a tracepoint in the Python debugger that works with Qt'''
    from PyQt5.QtCore import pyqtRemoveInputHook
    import pdb
    import sys
    pyqtRemoveInputHook()
    # set up the debugger
    debugger = pdb.Pdb()
    debugger.reset()
    # custom next to get outside of function scope
    debugger.do_next(None) # run the next command
    users_frame = sys._getframe().f_back # frame where the user invoked `pyqt_set_trace()`
    debugger.interaction(users_frame, None)



@jit
def getIntensity(mask_index, image):
    '''
    get raw fluorescence data over each frame in timeseries
    mask_index = index of the desired pixels in image
    image = np.array containing image data
    '''
    image = da.from_array(image, chunks=10000000)
    #get minimum square that includes mask, pulling out minimum sequence reduces time
    mask = generateMask(mask_index, image.shape)
    coord = getMaskRange(mask_index, image.shape)
    mask = mask[coord[0]:coord[1]+1, coord[2]:coord[3]+1, coord[4]:coord[5]+1]
    index2 = np.where(mask)
    #get minimum image coordinates
    image1 = image[:, coord[0]: coord[1]+1, coord[4] : coord[5]+1, coord[2] : coord[3]+1, 0]
    image1 = np.rollaxis(image1, 3,2)
    image1 = image1.compute()
    intensity_data = []
    for time in range(image1.shape[0]):
        intensity_data.append(image1[time, :, :, :][index2].mean())
    return intensity_data

    
def loadTimeSeries(frameTiffFiles):
    '''
    Based on
    http://dask.pydata.org/en/latest/array-creation.html
    loads timeseries data tif image files into an array
    '''


def generateMask(mask_index, image_shape):
    '''
    generate a binary 3d mask corresponding to the roi
    mask_index = index from flattened image
    image_shape = t, z,x,y, c image shape
    '''
    mask = np.zeros(image_shape[1:4], dtype = np.bool)
    mask.flat[mask_index] = True
    return mask
    
def getMaskRange(mask_index, image_shape):
    '''
    get max and min values for z y x
    mask_index = index from flattened image
    image_shape = t, z,x,y, c image shape
    '''
    mask = generateMask(mask_index, image_shape)
    non_zero = np.where(mask ==1)
    max_x = np.max(non_zero[1])
    max_y = np.max(non_zero[2])
    min_x = np.min(non_zero[1])
    min_y = np.min(non_zero[2])
    max_z = np.max(non_zero[0])
    min_z = np.min(non_zero[0])
    return [min_z, max_z, min_x, max_x, min_y, max_y]
    
def cropTZXYCimage(image, coordTZXYC):
    '''
    crop an image using z, x, y coordinates (coord)
    image = t, z, x y, c
    coord = [min_z, max_z, min_x, max_x, min_y, max_y]
    '''
    return image[coordZXY[0]:coordZXY[1]+1,coordZXY[2]:coordZXY[3]+1,coordZXY[4]:coordZXY[5]+1,coordZXY[6]:coordZXY[7]+1,coordZXY[8]:coord[9]+1]
    
def createBranch(list1, trunk):
    #list1 = ['string1', 'string2'] consecutives each branch progressing from trunk
    #trunk = 'path' known path to add branches
    trunk = pathlib.Path(trunk)
    if not trunk.is_dir():
        raise RuntimeError('trunk is not a directory')
    for clist in list1:
        trunk = trunk / clist
        if not trunk.is_dir():
            trunk.mkdir()
    return str(trunk)

def plot_traces(ax1, data, selection, prestart, start, stop):
    #ax1 = axes handle
    #data = pandas data frame containing data
    #selection = dictionary containing key=column attribute = to be selected, ie 'roi' : ['M1', 'M4'] = this selects roi M1 and M4 for plotting
    #prestart = time before stimulation in s
    #start = time (s) when stimulation occurs
    #stop = end of trace (s)
    t = 1

def selectInFrame(dataframe, selection):
    #returns a pandas dataframe that contains the selection
    #dataframe = pandas data frame
    #selection = dict('column' : 'selection') where the dictionary contains the column and instance to be selected
    cframe = dataframe.copy()
    for ckey in selection.keys():
        cframe = cframe[cframe[ckey] == selection[ckey]]
    return cframe

class intensityDataFrame():
    def __init__(self,dataframe):
        self.dataframe = dataframe

    def getVoltage(self):
        voltage = np.hstack(self.dataframe['voltage'])
        voltage = np.median(voltage, axis = 1)
        return voltage

    def getTimeStamp(self):
        #pdb.set_trace()
        timeStamp = np.hstack(self.dataframe['timestamp'])
        timeStamp = np.mean(timeStamp, axis =1)
        return timeStamp

    #function get intensity data from pandas group
    def getIntensityData(self):
        raw_intensity = np.vstack(self.dataframe['intensity'].values)
        return raw_intensity

    def getName(self):
        name = []
        for row, dSeries in self.dataframe.iterrows():
            name.append(str(dSeries['No.']) +'-'+ dSeries['Sample Name'])
        return(name)

    #used to match two dataframes for sample name and no
    def matchingFrame(self, allFrame, roi):
        #allFrame = entire data frame
        #remove NaN from No. column by replacing with 1
        self.dataframe = self.dataframe.fillna(1)
        allFrame = allFrame.fillna(1)
        output = pd.DataFrame(columns=allFrame.columns)
        for row, dseries in self.dataframe.iterrows():
            No = allFrame['No.'] == dseries['No.']
            Name =  allFrame['Sample Name'] == dseries['Sample Name']
            roilist = allFrame['roi'] == roi
            stimprotocol = allFrame['Stim Protocol'] == dseries['Stim Protocol']
            output = output.append( allFrame[No.values & Name.values & roilist.values & stimprotocol.values])
        return output

    def getMatchingBackground(self, original_dataframe):
        return self.matchingFrame(original_dataframe, 'Background')

    def getdFF(self, prestart, start, stop, exceldata, response):
        #prestart = time in s during protocol
        #start = time in s when stimulation occurs
        #stop = time in s for end of the current stimulation sequence
        #exceldata = entire dataframe including background
        #response = indicates whether a positive (increase) or negative (decrease) in fluoresence is expected
        timeStamp = self.getTimeStamp()
        prestart = np.argmax(timeStamp > prestart)
        start = np.argmax(timeStamp > start)
        stop = np.argmax(timeStamp > stop)
        raw_intensity = self.getIntensityData()
        if stop ==0: # raw_intensity[:, prestart : stop] > len(raw_intensity)
            stop = raw_intensity.shape[1]
        xrange1 = timeStamp[prestart : stop]
        matchBackground = self.getMatchingBackground(exceldata)
        background_intensity = np.vstack(matchBackground['intensity'].values)
        xrange1 = timeStamp[prestart : stop]
        dFF = ccModules.fluorescent(raw_intensity[:, prestart : stop], background_intensity[:, prestart : stop], start-prestart, response,  xrange1)
        return dFF
    
    def subtractBackground(self, exceldata):
        #pdb.set_trace()
        #return intensity data with the background subtracted
        corrected = np.vstack(self.dataframe['intensity'].values) - np.vstack(self.getMatchingBackground(exceldata)['intensity'])
        corrected[corrected < 0 ] =0
        correctedFrame = self.dataframe.copy()
        correctedFrame['intensity'] =corrected.tolist()
        return correctedFrame
    
def pullIntensityData(dataframe):
    '''
    takes intensity data dictionary found in exceldata and adds each roi as a separate row in the file
    '''
    cdataframe = pd.DataFrame(columns = dataframe.columns)
    for row, dseries in dataframe.iterrows():
        intensity_data=pd.DataFrame(dseries['intensity_data'])
        for rowI, dseriesIntensity in intensity_data.iterrows():
            for ccolumn in intensity_data.columns:
                dseries[ccolumn] = dseriesIntensity[ccolumn]
                #dseries.drop('intensity_data', inplace = True)
            cdataframe = cdataframe.append(dseries)
    cdataframe = cdataframe.rename(columns = {'Name': 'roi'})
    cdataframe.drop(['intensity_data'], axis = 1, inplace=True)
    cdataframe.reset_index(inplace=True)
    return cdataframe



def getSpacingSize(number, margin1, margin2, spacing):
    #get spacing for graphs
    return (1- margin1 - margin2 - spacing*(number))/(number)


def getXYSScoordinates(xnum, ynum, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, bottommargin, spacingL, spacingT):
    #returns the [x, y, xdistance, ydistance] for current axis position
    return [ leftmargin +(xnum)*(graphsizeX + spacingL), 1-((ynum+1)*(graphsizeY+spacingT))-topmargin, graphsizeX, graphsizeY]


class voltage():
    def __init__(self, voltage):
        self.voltage = voltage
        
    def getVoltageStopStarts(self):
        voltageDiff = np.diff(self.voltage, axis = 0)
        stimStarts = (np.where(voltageDiff > 0 )[0] +1) / 100
        stimStops = (np.where(voltageDiff < 0 )[0] ) / 100
        return stimStarts, stimStops    

    def plot(self, ax):
        stimStarts, stimStops = self.getVoltageStopStarts()
        for int1 in zip(stimStarts, stimStops):
            ax.fill_between(np.arange(int1[0], int1[1]), 0, self.voltage[int(int1[1]*100)], color = (1, 0.2, 0.1), alpha=0.3)
        ax.set_ylim([0, np.max(self.voltage)*1.2])

def load_tif(tif_file):
    #tif_file = full path to tif image
    #will convert all images into anto an array with the dimensions: z, y, x
    #tif_file = '/media/daniel/Windows1/Users/dnabu/Desktop/ResearchYogaWindows/DataJ/A/A70_NO/20180824/FlyA/NOC7_MIP/MIP_20180824-A70-FlyA_Baseline_00001.tif'
    image = np.squeeze(imread(tif_file))
    if len(image.shape) ==2: #assume that the image is a single slice
        imgZXY = np.zeros([1, image.shape[0], image.shape[1]], dtype=image.dtype)
        imgZXY[0,:,:] = image
    else:
        imgZXY = image
    return imgZXY
        
