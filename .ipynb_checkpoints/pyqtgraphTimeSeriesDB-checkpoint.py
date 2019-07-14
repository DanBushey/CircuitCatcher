'''
#modified from imageView
Created on Mar 26, 2017

@author: Surf32
'''
# -*- coding: utf-8 -*-
from __future__ import division
#from blaze import nan
import pandas as pd
import loadImageDialogue
import tifffile
import skimage
import ccModules2 as cc2

"""
ImageView.py -  Widget for basic image dispay and analysis
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.

Widget used for displaying 2D or 3D data. Features:
  - float or int (including 16-bit int) image display via ImageItem
  - zoom/pan via GraphicsView
  - black/white level controls
  - time slider for 3D data sets
  - ROI plotting
  - Image normalization through a variety of methods
"""
import os
import numpy as np
import pdb
#pdb.set_trace() 
import pyqtgraph as pg
import tables


from pyqtgraph.Qt import QtCore, QtGui, USE_PYSIDE
if USE_PYSIDE:
    print('pyside')
    from pyqtgraph.imageview.ImageViewTemplate_pyside import *
else:
    #from pyqtgraph.imageview.ImageViewTemplate_pyqtDB import *
    from ImageViewTemplate_pyqtDB import *
    
#from pyqtgraph.graphicsItems.ImageItem import *
from pyqtgraph_ImageItemDB import *
from pyqtgraph.graphicsItems.ROI import *
from pyqtgraph.graphicsItems.LinearRegionItem import *
from pyqtgraph.graphicsItems.InfiniteLine import *
from pyqtgraph.graphicsItems.ViewBox import *
from pyqtgraph.graphicsItems.GradientEditorItem import addGradientListToDocstring
from pyqtgraph import ptime as ptime
from pyqtgraph import debug as debug
from pyqtgraph.SignalProxy import SignalProxy
from pyqtgraph import getConfigOption
from skimage.draw import polygon #used in roi creation

try:
    from bottleneck import nanmin, nanmax
except ImportError:
    from numpy import nanmin, nanmax



class TimeSeriesView(QtGui.QWidget):
    """
    Widget used for display and analysis of image data.
    Implements many features:
    
    * Displays 2D and 3D image data. For 3D data, a z-axis
      slider is displayed allowing the user to select which frame is displayed.
    * Displays histogram of image data with movable region defining the dark/light levels
    * Editable gradient provides a color lookup table 
    * Frame slider may also be moved using left/right arrow keys as well as pgup, pgdn, home, and end.
    * Basic analysis features including:
    
        * ROI and embedded plot for measuring image values across frames
        * Image normalization / background subtraction 
    
    Basic Usage::
    
        imv = pg.ImageView()
        imv.show()
        imv.setImage(data)
        
    **Keyboard interaction**
    
    * left/right arrows step forward/backward 1 frame when pressed,
      seek at 20fps when held.
    * up/down arrows seek at 100fps
    * pgup/pgdn seek at 1000fps
    * home/end seek immediately to the first/last frame
    * space begins playing frames. If time values (in seconds) are given 
      for each frame, then playback is in realtime.
    """
    sigTimeChanged = QtCore.Signal(object, object)
    sigProcessingChanged = QtCore.Signal(object)
    change111 = QtCore.pyqtSignal(object)

    
    def __init__(self, parent=None, name="ImageView", view=None, imageItem=None, pullDownMenu = None, image_file_list = None, output_file = None, *args):
        """
        By default, this class creates an :class:`ImageItem <pyqtgraph.ImageItem>` to display image data
        and a :class:`ViewBox <pyqtgraph.ViewBox>` to contain the ImageItem. 
        
        ============= =========================================================
        **Arguments** 
        parent        (QWidget) Specifies the parent widget to which
                      this ImageView will belong. If None, then the ImageView
                      is created with no parent.
        name          (str) The name used to register both the internal ViewBox
                      and the PlotItem used to display ROI data. See the *name*
                      argument to :func:`ViewBox.__init__() 
                      <pyqtgraph.ViewBox.__init__>`.
        view          (ViewBox or PlotItem) If specified, this will be used
                      as the display area that contains the displayed image. 
                      Any :class:`ViewBox <pyqtgraph.ViewBox>`, 
                      :class:`PlotItem <pyqtgraph.PlotItem>`, or other 
                      compatible object is acceptable.
        imageItem     (ImageItem) If specified, this object will be used to
                      display the image. Must be an instance of ImageItem
                      or other compatible object.
        ============= =========================================================
        
        Note: to display axis ticks inside the ImageView, instantiate it 
        with a PlotItem instance as its view::
                
            pg.ImageView(view=pg.PlotItem())
        """
        QtGui.QWidget.__init__(self, parent, *args)
        self.selectLastROI = False #detects whether the shift key is being pressed
        self.currentRowSelectedinTable = None
        self.markerRadius = 15 #circle radius marking single points/rois
        self.clicked = []
        self.image_Designation = None
        self.levelMax = 4096
        self.levelMin = 0
        self.name = name
        self.image = None
        self.axes = {}
        self.imageDisp = None
        self.ui = Ui_Form() #pyqtgraph.imageview.ImageViewTemplate_pyqt
        self.ui.setupUi(self)
        self.scene = self.ui.graphicsView.scene()
        #get pulldownmenu information
        self.pullDownMenu = pullDownMenu
        #user can pass a list of image files to be scored
        self.image_file_list = image_file_list
        self.ignoreTimeLine = False
        self.ignoreZLine = False
        self.output_file = output_file #file location to save list of images not yet scored (list = image_file_list
        
        if view is None:
            self.view = ViewBox()
        else:
            self.view = view
        
        self.ui.graphicsView.setCentralItem(self.view)
        self.view.setAspectLocked(True)
        self.view.invertY()
        
        if imageItem is None:
            self.imageItem = ImageItem()
        else:
            self.imageItem = imageItem
        self.imageview = self.view.addItem(self.imageItem)
        self.currentTime = 0
        self.currentLayer = 0 #layer in z axis
        
        #relay x y coordinates within image
        self.view.scene().sigMouseMoved.connect(self.mouseMoved)
        self.view.scene().sigMouseClicked.connect(self.mouseClicked)
        #self.view.scene().mouseReleaseEvent(self.releaseEvent)
        #self.view.scene().mouseReleaseEvent(self.mouseReleased)
        #generate roi list
        self.croi = [] #holdes x,y,z points for roi
        self.ccroi = [] #hold pg roi for croi
        self.aroicolumns = ['Name', 'Color', 'Type', 'Z:XY', 'mask_index', 'image_file', 'image_shape']
        self.aroi = pd.DataFrame(columns=self.aroicolumns) #place to store all rois, each roi is passed as dictionary with each index = currentLayer
        self.aaroi = {} #place to store pg of arois in self.currentLayer
        self.button1 = 'off'
        
        self.ui.histogram.setImageItem(self.imageItem)
        
        self.menu = None
        
        self.ui.normGroup.hide()

        #self.normRoi = PlotROI(10)
        #self.normRoi.setPen('y')
        #self.normRoi.setZValue(20)
        #self.view.addItem(self.normRoi)
        #self.normRoi.hide()
        #self.roiCurve = self.ui.roiPlot.plot()
        
        #setting up time axis 
        self.timeLine = InfiniteLine(0, movable=True)
        self.timeLine.setPen((255, 255, 0, 200))
        #self.timeLine.setPen((255, 255, 200, 0))
        self.timeLine.setZValue(1)
        self.ui.timePlot.addItem(self.timeLine)
        self.ui.splitter.setSizes([self.height()-35, 35])
        self.ui.timePlot.hideAxis('left')
        
        #setting up z axis
        self.ui.splitter.setSizes([self.height(), 10])
        self.ui.zPlot.plot()
        self.zLine = InfiniteLine(0, movable=True, angle = 0)
        #self.zLine.setBounds([0, 42])
        self.zLine.setPen((255, 255, 0, 200))
        self.zLine.setZValue(self.currentLayer )
        self.ui.ztext.setText(str(self.currentLayer))
        
        self.ui.zPlot.addItem(self.zLine)
        self.ui.zPlot.hideAxis('bottom')
        
        
        
        self.keysPressed = {}
        self.playTimer = QtCore.QTimer()
        self.playRate = 0
        self.lastPlayTime = 0
        
        self.normRgn = LinearRegionItem()
        self.normRgn.setZValue(0)
        self.ui.timePlot.addItem(self.normRgn)
        self.normRgn.hide()
            
        ## wrap functions from view box
        for fn in ['addItem', 'removeItem']:
            setattr(self, fn, getattr(self.view, fn))

        ## wrap functions from histogram
        for fn in ['setHistogramRange', 'autoHistogramRange', 'getLookupTable', 'getLevels']:
            setattr(self, fn, getattr(self.ui.histogram, fn))

        self.timeLine.sigPositionChanged.connect(self.timeLineChanged)
        self.zLine.sigPositionChanged.connect(self.zLineChanged)
        self.ui.saveROIBtn.clicked.connect(self.saveROI)
        self.ui.loadBtn.clicked.connect(self.loadImageDialogue)
        #self.ui.normBtn.toggled.connect(self.normToggled)
        self.ui.menuBtn.clicked.connect(self.menuClicked)
        self.ui.normDivideRadio.clicked.connect(self.normRadioChanged)
        self.ui.normSubtractRadio.clicked.connect(self.normRadioChanged)
        self.ui.normOffRadio.clicked.connect(self.normRadioChanged)
        #self.ui.normROICheck.clicked.connect(self.updateNorm)
        #self.ui.normFrameCheck.clicked.connect(self.updateNorm)
        #self.ui.normTimeRangeCheck.clicked.connect(self.updateNorm)
        #radio button for navigation and roi selection
        self.ui.navRadio.clicked.connect(self.NavigationStatus)
        self.ui.radioSinglePoint.clicked.connect(self.radioSinglePoint)
        self.ui.radioAreaROI.clicked.connect(self.radioAreaROI)
        self.ui.radioPolygon.clicked.connect(self.radioPolygon)
        self.ui.radioEdit.clicked.connect(self.radioEdit)
        
        self.playTimer.timeout.connect(self.timeout)
        
        #self.normProxy = SignalProxy(self.normRgn.sigRegionChanged, slot=self.updateNorm)
        #self.normRoi.sigRegionChangeFinished.connect(self.updateNorm)
        
        self.ui.timePlot.registerPlot(self.name + '_ROI')
        self.view.register(self.name)
        
        self.noRepeatKeys = [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown]
        
        self.setAcceptDrops(True)
        if self.image_file_list == None:
            self.placeHolderPath=os.getcwd()
        else:
            self.placeHolderPath = image_file_list[0]
        #cc2.pyqt_set_trace()
        self.zLineChanged()
        #self.setDragDropMode(QAbstractItemView.InternalMove)
        
    def importimage(self, imagefile):
        
        #output1 = (output1 - output1.min()) / output1.max()
        #output1 [output1 > 1] =1
        #return skimage.img_as_ubyte(tifffile.imread(imagefile))
        
        #print('isfile?', os.path.isfile(str(imagefile)))
        #print(imagefile)
        return np.squeeze(tifffile.imread(str(imagefile)))
        
        #self.roiClicked() ## initialize roi plot to correct shape / visibility
    def loadImageDialogue(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, 'Open Image File', directory = self.placeHolderPath)
        #fileName = QtGui.QFileDialog.getOpenFileName(self, 'Open Image File', directory = self.placeHolderPath)
        #print('****************************checkingfilename')
        #print(type(fileName))
        #print(fileName)
        #these if are necessary because the QtGui.QFileDialog.getOpenFileName from Qt4 produces a different output than QT5
        if isinstance(fileName, tuple):
            self.img_file = fileName[0]
            self.loadImage(fileName[0])
        else:
            self.img_file = str(fileName)
            self.loadImage(str(fileName))
        
        
        
    def loadTimeSeries(self, folder1):
        files, index = self.getFileContString(folder1, '.tif')
        img1 = self.importimage(os.path.join(folder1, files[0]))
        imgStack = np.zeros([len(files), img1.shape[0], img1.shape[1], img1.shape[2]], dtype = np.uint8)
        imgStack[0, :, :, :] = self.normalizeImage(img1)
        for i, cfile in enumerate(files[1:].values):
            imgStack[i+1, :, :, :] = self.normalizeImage(self.importimage(os.path.join(folder1, cfile)))
        return imgStack
    
    def getFileContString(self, targetdir, string1):
        filelist=self.getFileList(targetdir)
        if len(filelist) > 0:
            indx=filelist['file'].str.contains(string1)
            filenames=filelist['file'][indx]
            indx2=np.where(indx)
        else:
            filenames = filelist['file']
            indx2 = ([],)
        return filenames, indx2
            
    def getFileList(self, targetdir):
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
    
    def normalizeImageIntensity(self, image):
        image = image-np.min(image)
        return image / np.max(image) * np.iinfo(image.dtype).max
        
    def loadImage(self, pathfile):
        if hasattr(self, 'fileROI'):
            #print("Saving previous roi before loading has started")
            self.saveROI()
        #print('loadimagestarted-3-----------------------------------------------------------------')
        #print(pathfile)
        if self.image_file_list == None:
            self.placeHolderPath, filename = os.path.split(str(pathfile))
        else:
            #change next placeholder to next image in the file list
            index = [i for i, file1 in enumerate(self.image_file_list) if file1 == pathfile]
            if index[0] + 1 <= len(self.image_file_list):
                self.placeHolderPath = self.image_file_list[index[0]+1]
            else:
                self.placeHolderPath, filename = os.path.split(str(pathfile))
            #saving a file containing list of remaining images to scoreROIlist
            frame = pd.read_excel(self.output_file, index_col=0)
            index = [i for i, file1 in enumerate(frame['Image_File'].values) if file1 == pathfile]
            frame = frame.drop(frame.index[index[0]])
            frame.to_excel(self.output_file)
        path1, filename1 = os.path.split(str(pathfile))
        lb = loadImageDialogue.ImageChoice(workingdirectory = path1, filename=filename1)
        returnCode=lb.exec_()
        value = lb.GetValue()
        self.currentRowSelectedinTable = None
        
        channels = self.channelsToAdd(value) #which channels will include image data
        if not '.hdf5' in pathfile:
            #load first image
            image = cc2.load_tif(pathfile)
            if value.loadtimeseries !=2:
                
                imgRGB = np.zeros([1, image.shape[0], image.shape[1], image.shape[2], 3], dtype=image.dtype)
                
                for cc in channels:
                    imgRGB[0,:,:,:, cc] = image
            else:
                files, index = self.getFileContString(self.placeHolderPath, '.tif')
                imgRGB = np.zeros([len(files), image.shape[0], image.shape[1], image.shape[2], 3], dtype=image.dtype)
                #filename = r'C:\Users\Surf32\Desktop\ResearchDSKTOP\DataJ\A\A30_FastROI\SampleDataComplete\testArray.npy'
                #imgRGB = np.memmap(filename, mode = 'w+', shape = (len(files), image.shape[0], image.shape[1], image.shape[2], 3), dtype=image.dtype)
                for cc in channels:
                    imgRGB[0, :, :, :, cc] = image
                for i, cfile in enumerate(files[1:].values):
                    image = self.importimage(os.path.join(self.placeHolderPath, cfile))
                    for cc in channels:
                        imgRGB[i+1, :, :, :, cc] = image
            imgRGB = np.rollaxis(imgRGB, 3, 2)
        elif 'hdf5' in pathfile:
            imgRGB_file = tables.open_file(pathfile, mode='r')
            #pdb.set_trace()
            if imgRGB_file.__contains__('/minmax'):
                self.minmax = imgRGB_file.root.minmax[:]
            imgRGB = imgRGB_file.root.data
            size1 = imgRGB.shape[0]
            for dim in imgRGB.shape[1:]:
                size1 = size1 * dim
            if size1 < 350*512*512:
                #print('Loading HDF5 file into memory')
                imgRGB = imgRGB[:]
            else:
                imgRGB.size =size1 
        
        #add secondary image to all times in the empty rgb channels
        if value.pathToSecondaryImage != None:
            imgRGB = self.normalizeImageIntensity(imgRGB)
            #print('loading secondary image',value.pathToSecondaryImage )
            image = self.importimage( value.pathToSecondaryImage)
            image = np.rollaxis(image, 2, 1)
            image = self.normalizeImageIntensity(image)
            channels2 = np.setdiff1d([0,1,2], channels)
            for ch in channels2:
                for time in range(imgRGB.shape[0]):
                    imgRGB[time, :, :, :, ch] = image
        
        #load mask data if it exists
        self.fileROI = value.pathToMaskFile
        #populate self.aroi pandas data frame
        #pdb.set_trace()
        if os.path.isfile(self.fileROI):
            #pdb.set_trace()
            #pdb.set_trace()
            self.aroi = pd.read_hdf(self.fileROI, 'roi')
            #dropColumns = ['mask_index', 'image_shape', 'image_file']
            #self.aroi = self.aroi.drop(dropColumns, axis=1)
            #pdb.set_trace()
            #print(self.aroi)
        else:
            self.aroi = pd.DataFrame(columns=self.aroicolumns)
        
        self.setImage(imgRGB,  axes = {'t': 0, 'z':1, 'x':3, 'y':2, 'c':4}, fileROI= self.fileROI)
        self.change111.emit('RoiChanged')
        
        #print('Finished loadImage')
        

    def channelsToAdd(self, value):
        channelsToAdd =[]
        if value.r ==2:
            channelsToAdd.append(0)
        if value.g ==2:
            channelsToAdd.append(1)
        if value.b ==2:
            channelsToAdd.append(2)
        return channelsToAdd
    
    def dragEnterEvent(self, event):
        event.accept()
        
    def dropEvent(self, event):
        #print('DropEvent detected')
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                print('path', path)
            elif os.path.isdir(path):
                print('dir', path)
        self.img_file = str(path)
        self.loadImage(path)
        
    
    def saveROI(self):
        #global imgStack2, roi, w1, folder
        #print(imgStack.shape)


        '''
        csave = self.aroi.copy()
        csave['mask_index'] = ''
        for croi in self.aroi.index:
            mask = np.zeros([self.image.shape[1], self.image.shape[3], self.image.shape[2]], dtype = np.bool) #believe I messed up x and y coordinates and now must reverse here to make compatible with later roi scripts
            for layer in self.aroi['Z:XY'].loc[croi].keys():
                if len(self.aroi['Z:XY'].loc[croi][layer]) == 1:
                    mask[layer, int(np.rint(self.aroi['Z:XY'].loc[croi][layer][0][0])), int(np.rint(self.aroi['Z:XY'].loc[croi][layer][0][1]))] = 1
                else:
                    xy =  self.aroi['Z:XY'].loc[croi][layer]
                    xy = np.vstack(xy)
                    xx, yy = polygon(xy[:,0], xy[:,1])
                    #print(mask.shape)
                    #print(np.max(xx))
                    #print(np.max(yy))
                    mask[layer,  yy, xx] = 1
            csave['mask_index'].loc[croi] = np.where(mask.flatten())
            #get data for roi
            #image3 = np.rollaxis(self.image[:, :, :, :, 1], 0, 4)
            #image3 = image3.reshape((image3.shape[0] * image3.shape[1]  * image3.shape[2] , image3.shape[3] ))
            #intensity_data[croi] = image3[ mask_index[croi]]
        csave['image_shape'] = [self.image.shape]* len(self.aroi)
        csave['image_file'] = self.img_file
        #savedict1 = {'image_shape': self.image.shape, 'Z:XY' : self.aroi['Z:XY'].to_dict(), 'mask_index' : mask_index, 'intensity_data' : intensity_data, 'Color': self.aroi['Color'].to_dict(), 'Type': self.aroi['Type'].to_dict()}
        #savedict1 = {'image_file': self.img_file, 'image_shape': self.image.shape, 'Z:XY' : self.aroi['Z:XY'].to_dict(), 'mask_index' : mask_index, 'Color': self.aroi['Color'].to_dict(), 'Type': self.aroi['Type'].to_dict()}
        #np.save(self.fileROI, savedict1)
        
        csave.to_hdf(self.fileROI, 'roi')
        '''
        self.aroi.to_hdf(self.fileROI, 'roi')
        
    def mouseMoved(self, pos):
        #print(evt[0])
        #print("Image position:", self.imageItem.mapFromScene(pos))

        #if self.mouseButton1 == 'on':
        #print('ButtonState:', QtCore.Qt.NoButton)
        #print('Moving Mouse')
        #print(pos.button())
        #print(pos.button())
        #pos = pos.scenePos()
        
        if self.ui.radioAreaROI.isChecked():
            #if self.button1 == 'on':
            #print('length', len(self.croi), len(self.croi)==1, self.croi)
            #print('button', len(self.view.scene().clickEvents), len(self.view.scene().clickEvents) == 0, self.view.scene().clickEvents)
            #this section was added because on Yoga720 pen coming onto screen with touching sometimes triggers a click event
            if len(self.view.scene().clickEvents) == 0 and len(self.croi)==1:
                #print('de')
    
                self.croi = []
            if self.view.scene().clickEvents:
                
                if self.view.scene().clickEvents[0].button() == 1:
                    if self.view .sceneBoundingRect().contains(pos):
                        self.clicked.append(self.view.scene().clickEvents)
                        #the following selects only clicks within the image
                        mousePoint = self.view.mapSceneToView(pos)
                        index = int(mousePoint.x())
                        if index > 0 and index < self.image.shape[self.axes['x']]:
                            #label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
                            #print(mousePoint.x())
                            #print(mousePoint.y())
                            #return(mousePoint.x(), mousePoint.y())
                            #if self.ccroi:
                            #    self.view.removeItem( self.ccroi) #removed roi generated from previous movements
                            #self.croi.append([mousePoint.x(), mousePoint.y(), self.currentLayer])#crois is the current roi being selected
                            #
                            #adding individual segments is very fast and only want to add segments if more than a certain amount from last spot
                            if len(self.croi) < 1:
                                distance =10000
                            else:
                                distance = np.sqrt((self.croi[-1][0]-mousePoint.x())**2 + (self.croi[-1][1]-mousePoint.y())**2)
                            if distance > 6:
                                self.croi.append([mousePoint.x(), mousePoint.y()])

                                if len(self.croi) > 1:
                                    self.ccroi.append(pg.LineSegmentROI(self.croi[-2:]))
                                    #self.ccroi.append(pg.LineROI(self.croi[-2], self.croi[-1], 1))
                                    self.view.addItem( self.ccroi[-1])
                                #print('appending roi coordinates')
                                #print(self.croi) #crois is the current roi

                else:
                    #print(length3, self.croi)
                    if self.croi:
                        if len(self.croi) == 1:
                            self.croi = []
            else:
                if self.croi:
                    #if self.ccroi:
                    #    self.view.removeItem( self.ccroi) #remove the roi built has the mouse was dragged
                    #generate a novel key for dictionary holding rois
                    #testname = 0
                    #while testname in self.aroi.keys():
                    #    testname += 1
                        
                    #self.aroi[testname] = {self.currentLayer : self.croi}
                    if len(self.croi) > 1: # having problems with 1 pixel rois forming with pen and this will select against such roi
                        self.createROI(pg.PolyLineROI(self.croi, closed= True), 'polyArea')
                    else:
                        if self.ccroi:
                            for cccroi in self.ccroi:
                                self.view.removeItem(cccroi)
                                self.croi = []
                    #self.aaroi[testname] =pg.PolyLineROI(self.croi, closed= True)
                    #self.view.addItem(self.aaroi[testname])  
                    #self.croi = []
                    #self.ccroi =[]
                    #print('transfered current roi (croi) to list for all rois (aroi)')     
                    
    def createROI(self, newroi, type):
        if self.ccroi:
            #print('wwwwwwww')
            #print(self.ccroi)
            if isinstance(self.ccroi, list):
                for cccroi in self.ccroi:
                    self.view.removeItem( cccroi)
            else:
                self.view.removeItem(self.ccroi)
        #aroi is the roi class
        print('currentRowSelectedinTable', self.currentRowSelectedinTable)
        if self.currentRowSelectedinTable == None:
            '''
            continue adding roi to selection or change roi if roi already chosen for current z-layer
            '''
            testname = 0
            while testname in self.aroi.index:
                testname += 1
            #testname = str(testname)
            #[Color', 'Type', 'Z:XY']
            #following picks the roi name
            #roi name is first roi in the pullDownMenu dictionary if self.aroi is empty
            #roi name is the next roi in the pullDownMenu dictionary if more than one roi in self.aroi
            #roi name is the last key in the pullDownMenu dictionary if the number of rows in self.aroi equals or more than the number of keys in the pullDownMenu
            if len(self.aroi) == 0:
                roiname =  list(self.pullDownMenu.keys())[0]
            elif len(self.aroi) > 0 and not len(self.aroi) >= len(self.pullDownMenu):
                index = [i for i, name in enumerate(self.pullDownMenu.keys()) if name == self.aroi['Name'].iloc[-1]] #find the last roi used
                if len(index) > 0 : #test wheter name occured in pre-derived names for rois 
                    if index[0]+1 <= len(self.pullDownMenu.keys())-1: #test whether index exceeds list of rois
                        roiname = list(self.pullDownMenu.keys())[index[0]+1]
                    else:
                        roiname = list(self.pullDownMenu.keys())[index[0]]
                else: #last roi was not in the pre-derived names for rois
                    roiname = list(self.pullDownMenu.keys())[-1]
            elif len(self.aroi) >= len(self.pullDownMenu):
                roiname = list(self.pullDownMenu.keys())[-1]
            add_dict = {'Name' : roiname,
                        'Color' : self.pullDownMenu[roiname],
                        'Z:XY'  : {self.currentLayer : self.croi},
                        'Type' : type,
                        'mask_index' : cc2.convertZXYtoMask({self.currentLayer : self.croi}, self.image.shape),
                        'image_file' : self.img_file,
                        'image_shape' : self.image.shape,
                        }
            self.aroi.loc[testname] = pd.Series(add_dict)
            '''            
            self.aroi.loc[testname] = ''
            #print(self.aroi)
            #print(testname)
            #cc2.pyqt_set_trace()
            self.aroi['Name'].loc[testname] = roiname
            self.aroi['Color'].loc[testname] = self.pullDownMenu[roiname]
            self.aroi['Z:XY'].loc[testname] = {self.currentLayer : self.croi}
            self.aroi['Type'].loc[testname] = type
            self.aroi['mask_index'].loc[testname] = cc2.convertZXYtoMask({self.currentLayer : self.croi}, self.image.shape)
            self.aroi['image_file'].loc[testname] = self.img_file
            self.aroi['image_shape'].loc[testname] = self.image.shape
            
            
            #self.aroi.loc[testname] = [roiname, self.pullDownMenu[roiname], type, {self.currentLayer : self.croi}] 
            '''
        else:
            #if roi already exists and was changed
            testname = self.aroi.index[self.currentRowSelectedinTable]
            currentroi = self.aroi['Z:XY'].loc[testname]
            currentroi[self.currentLayer] = self.croi
            self.aroi['mask_index'].loc[testname] = cc2.convertZXYtoMask(currentroi, self.image.shape)
            
        newroi.key = testname
        newroi.sigRegionChanged.connect(self.updateRegion)
        if testname in self.aaroi.keys():
            self.view.removeItem( self.aaroi[testname]) 
        self.aaroi[testname] = newroi
        self.aaroi[testname].setPen(color=[i*255 for i in self.aroi['Color'].loc[testname]], width = 3)
        self.view.addItem(self.aaroi[testname]) 
        self.croi = []
        self.ccroi =[]
        #print('transfered current roi (croi) to list for all rois (aroi)')
        self.change111.emit('RoiChanged')
    
    def updateRegion(self,roi):
        #print('updateregion')
        #print(roi)
        
        coord = roi.getLocalHandlePositions()
        #reorganize the coordinates
        
        #the coord has both pyqt points and pyQT5 points
        t = [[XY.x(), XY.y()] for N, XY in coord]
        #old = self.aroi['Z:XY'].loc[roi.key][self.currentLayer] 
        self.aroi['Z:XY'].loc[roi.key][self.currentLayer] = t
        self.aroi['mask_index'].loc[roi.key] = cc2.convertZXYtoMask(self.aroi['Z:XY'].loc[roi.key], self.image.shape)
        

    def mouseClicked(self, pos):
        #print('mouseClicked')
        #print(QtCore.Qt.NoButton)
        #print(pos.button())
        #self.view.scene()
        if self.view .sceneBoundingRect().contains(pos.scenePos()):
            #the following selects only clicks within the image
            mousePoint = self.view.mapSceneToView(pos.scenePos())
            index = int(mousePoint.x())
            if index > 0 and index < self.image.shape[self.axes['x']]:
                if self.ui.radioSinglePoint.isChecked():
                    mousePoint = self.view.mapSceneToView(pos.scenePos())
                    #print('adding single point to roi list')
                    #self.aroi.append([mousePoint.x(), mousePoint.y()])
                    #testname = 0
                    #while testname in self.aroi.keys():
                    #    testname += 1
                    self.croi.append([mousePoint.x(), mousePoint.y()])            
                    #self.aroi[testname] = {self.currentLayer : [mousePoint.x(), mousePoint.y()]}
                    self.createROI(pg.CircleROI(self.circlePosition([mousePoint.x(), mousePoint.y()]), [self.markerRadius,self.markerRadius], pen = (4,9)), 'singlepoint')
                    #self.aaroi[testname] = pg.CircleROI(self.circlePosition([mousePoint.x(), mousePoint.y()]), [self.markerRadius,self.markerRadius], pen = (4,9))
                    #self.view.addItem(self.aaroi[testname])  
                    
                
                elif self.ui.radioPolygon.isChecked():
                    mousePoint = self.view.mapSceneToView(pos.scenePos())
                    if   pos.button()  ==   1:
                        #print('adding point to polygon')
                        self.croi.append([mousePoint.x(), mousePoint.y()])
                        if self.ccroi:
                            self.view.removeItem( self.ccroi) #removed roi generated from previous movements
                        self.ccroi = pg.PolyLineROI(self.croi, closed = False)
                        self.view.addItem( self.ccroi)  
        
                    elif pos.button() == 2:
                        #print('finished polygon and adding to roi list')
                        self.createROI(pg.PolyLineROI(self.croi, closed= True), 'polygon')
                



    def setImage(self, img, autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True, image_Designation=None, fileROI=None):
        """
        Set the image to be displayed in the widget.
        
        ================== ===========================================================================
        **Arguments:**
        img                (numpy array) the image to be displayed. See :func:`ImageItem.setImage` and
                           *notes* below.
        xvals              (numpy array) 1D array of z-axis values corresponding to the third axis
                           in a 3D image. For video, this array should contain the time of each frame.
        autoRange          (bool) whether to scale/pan the view to fit the image.
        autoLevels         (bool) whether to update the white/black levels to fit the image.
        levels             (min, max); the white and black level values to use.
        axes               Dictionary indicating the interpretation for each axis.
                           This is only needed to override the default guess. Format is::
                       
                               {'t':0, 'z':1, 'x':2, 'y':3, 'c':4};
        
        pos                Change the position of the displayed image
        scale              Change the scale of the displayed image
        transform          Set the transform of the displayed image. This option overrides *pos*
                           and *scale*.
        autoHistogramRange If True, the histogram y-range is automatically scaled to fit the
                           image data.
        ================== ===========================================================================

        **Notes:**        
        
        For backward compatibility, image data is assumed to be in column-major order (column, row).
        However, most image data is stored in row-major order (row, column) and will need to be
        transposed before calling setImage()::
        
            imageview.setImage(imagedata.T)nnn
            
        This requirement can be changed by the ``imageAxisOrder``
        :ref:`global configuration option <apiref_config>`.
        
        """
        self.image_Designation = image_Designation 
        profiler = debug.Profiler()
        
        if hasattr(img, 'implements') and img.implements('MetaArray'):
            img = img.asarray()
        '''
        if not isinstance(img, np.ndarray):
            required = ['dtype', 'max', 'min', 'ndim', 'shape', 'size']
            if not all([hasattr(img, attr) for attr in required]):
                raise TypeError("Image must be NumPy array or any object "
                                "that provides compatible attributes/methods:\n"
                                "  %s" % str(required))
        '''
        self.image = img # image array
        self.fileROI = fileROI
        self.imageDisp = None
        
        profiler()
        
        if axes is None:
            x,y = (0, 1) if self.imageItem.axisOrder == 'col-major' else (1, 0)
            
            if img.ndim == 2:
                self.axes = {'t': None, 'x': x, 'y': y, 'c': None}
            elif img.ndim == 3:
                # Ambiguous case; make a guess
                if img.shape[2] <= 4:
                    self.axes = {'t': None, 'z': None, 'x': x, 'y': y, 'c': 2}
                else:
                    self.axes = {'t': 0, 'z': None, 'x': x+1, 'y': y+1, 'c': None}
            elif img.ndim == 4:
                # Even more ambiguous; just assume the default
                if img.shape[2] <= 4:
                    self.axes = {'t': 0, 'z': None, 'x': x+2, 'y': y+2, 'c': 4}
                else: 
                    self.axes = {'t': 0, 'z': 1, 'x': x+2, 'y': y+2, 'c': None}
            elif img.ndim == 5:
                self.axes = {'t': 0, 'z': 1, 'x': x+2, 'y': y+2, 'c': 4}
            else:
                raise Exception("Can not interpret image with dimensions %s" % (str(img.shape)))
        elif isinstance(axes, dict):
            self.axes = axes.copy()
        elif isinstance(axes, list) or isinstance(axes, tuple):
            self.axes = {}
            for i in range(len(axes)):
                self.axes[axes[i]] = i
        else:
            raise Exception("Can not interpret axis specification %s. Must be like {'t': 2, 'x': 0, 'y': 1} or ('t', 'x', 'y', 'c')" % (str(axes)))
        
        for x in ['t', 'z', 'x', 'y', 'c']:
            self.axes[x] = self.axes.get(x, None)
        axes = self.axes
        
        #set min amd max values for time range
        if axes['t'] is not None:
            if hasattr(img, 'xvals'):
                try:
                    self.tVals = img.xvals(axes['t'])
                except:
                    self.tVals = np.arange(img.shape[axes['t']])
            else:
                self.tVals = np.arange(img.shape[axes['t']])
        
        #set min amd max values for z range
        if axes['z'] is not None:
            if hasattr(img, 'xvals'):
                try:
                    self.zVals = img.xvals(axes['z'])
                except:
                    self.zVals = np.arange(img.shape[axes['z']])
            else:
                self.zVals = np.arange(img.shape[axes['z']])

        
        profiler()
        
        self.currentTime = 0
        self.currentLayer =0
        self.updateImage(autoHistogramRange=autoHistogramRange)
        if levels is None and autoLevels:
            self.autoLevels()
        if levels is not None:  ## this does nothing since getProcessedImage sets these values again.
            self.setLevels(*levels)
          
        #if self.ui.roiBtn.isChecked():
        #    self.roiChanged()

        profiler()
        #set max and min values for time 
        if self.axes['t'] is not None:
            #self.ui.roiPlot.show()
            self.ui.timePlot.setXRange(self.tVals.min(), self.tVals.max())
            self.timeLine.setValue(0)
            #self.ui.timePlot.setMouseEnabled(False, False)
            if len(self.tVals) > 1:
                start = self.tVals.min()
                stop = self.tVals.max() + abs(self.tVals[-1] - self.tVals[0]) * 0.02
            elif len(self.tVals) == 1:
                start = self.tVals[0] - 0.5
                stop = self.tVals[0] + 0.5
            else:
                start = 0
                stop = 1
            for s in [self.timeLine, self.normRgn]:
                s.setBounds([start, stop])
            #print(start)
            #print(stop)
        #else:
            #self.ui.timePlot.hide()
            
                #set max and min values for time 
        if self.axes['z'] is not None:
            #self.ui.timePlot.show()
            self.ui.zPlot.setYRange(self.zVals.min(), self.zVals.max())
            self.ui.zPlot.setMouseEnabled(x = False, y = False) #setting to false prevents mouse interaction with x and y coordinates
            self.zLine.setValue(0)
            #self.ui.timePlot.setMouseEnabled(False, False)
            if len(self.zVals) > 1:
                start = self.zVals.min()
                stop = self.zVals.max() + abs(self.zVals[-1] - self.zVals[0]) * 0.02
            elif len(self.zVals) == 1:
                start = self.zVals[0] - 0.5
                stop = self.zVals[0] + 0.5
            else:
                start = 0
                stop = 1
            for s in [self.zLine, self.normRgn]:
                s.setBounds([start, stop])
            #print('zline value')
            #print(self.zLine.value())
            #print('start')
            #print(start)
            #print(stop)
        #else:
            #self.ui.timePlot.hide()
        profiler()
        
        self.imageItem.resetTransform()
        if scale is not None:
            self.imageItem.scale(*scale)
        if pos is not None:
            self.imageItem.setPos(*pos)
        if transform is not None:
            self.imageItem.setTransform(transform)
        
        profiler()

        if autoRange:
            self.autoRange()
        #self.roiClicked()

        profiler()
        self.view.setMouseEnabled(x=False, y=False) #turns panning and magnifcation off
        self.zLineChanged()
        
        
    def NavigationStatus(self):
        #print('navigation button pressed')
        if self.ui.navRadio.isChecked():
            self.view.setMouseEnabled(x=True, y=True)
        else:
            self.view.setMouseEnabled(x=False, y=False)
            
    def radioSinglePoint(self):
        self.view.setMouseEnabled(x=False, y=False)
        
    def radioAreaROI(self):
        self.view.setMouseEnabled(x=False, y=False)
    
    def radioPolygon(self):
        self.view.setMouseEnabled(x=False, y=False)
    
    def radioEdit(self):
        self.view.setMouseEnabled(x=False, y=False)


    def clear(self):
        self.image = None
        self.imageItem.clear()
        
    def play(self, rate):
        """Begin automatically stepping frames forward at the given rate (in fps).
        This can also be accessed by pressing the spacebar."""
        #print "play:", rate
        self.playRate = rate
        if rate == 0:
            self.playTimer.stop()
            return
            
        self.lastPlayTime = ptime.time()
        if not self.playTimer.isActive():
            self.playTimer.start(16)

            
    def autoLevels(self):
        """Set the min/max intensity levels automatically to match the image data."""
        self.setLevels(self.levelMin, self.levelMax)


    def setLevels(self, min, max):
        """Set the min/max (bright and dark) levels."""
        self.ui.histogram.setLevels(min, max)

    def autoRange(self):
        """Auto scale and pan the view around the image such that the image fills the view."""
        image = self.getProcessedImage()
        self.view.autoRange()

        
    def getProcessedImage(self):
        """Returns the image data after it has been processed by any normalization options in use.
        This method also sets the attributes self.levelMin and self.levelMax 
        to indicate the range of data in the image."""
        if self.imageDisp is None:
            image = self.normalize(self.image)
            self.imageDisp = image
            self.levelMin, self.levelMax = list(map(float, self.quickMinMax(self.imageDisp)))
            
        return self.imageDisp

        
    def close(self):
        """Closes the widget nicely, making sure to clear the graphics scene and release memory."""
        self.ui.timePlot.close()
        self.ui.graphicsView.close()
        self.scene.clear()
        del self.image
        del self.imageDisp
        super(ImageView, self).close()
        self.setParent(None)
    
    def mouseReleaseEvent(self, ev):
        print('mouseReleaseEvent')

        
    def keyPressEvent(self, ev):
        print('keyPressEvent')
        print(ev.key())
        #print ev.key()
        if ev.key() == QtCore.Qt.Key_Space:
            print('spacekey pressed')
            print('AllButtons:', QtCore.Qt.RightButton)
            #self.defineROI(ev)
            '''
            if self.playRate == 0:
                fps = (self.getProcessedImage().shape[0]-1) / (self.tVals[-1] - self.tVals[0])
                self.play(fps)
                #print fps
            else:
                self.play(0)
            ev.accept()
            '''
        
        elif ev.key() == 87: #detects w key is pressed and moves up
            if self.zLine.value() + 1 <= self.zLine.maxRange[1]:
                self.zLine.setValue(self.zLine.value()+1)
                self.zLineChanged()
        elif ev.key() == 83: #detects s key is pressed and moves down
            if self.zLine.value() - 1 >= self.zLine.maxRange[0]:
                self.zLine.setValue(self.zLine.value() -1)
                self.zLineChanged()
        elif ev.key() == 65: #detects "a" key is pressed and selects last cell for a volume
            self.selectLastROI = 'SelectLastCell'
            self.change111.emit('RoiChanged')
            self.selectLastROI = False
        elif ev.key() == 68: #detects "d" key is pressed and removes any selection
            self.selectLastROI = 'RemoveSelection'
            self.change111.emit('RoiChanged')
            self.selectLastROI = False
        elif ev.key() == 88: #detects "x" key pressed and executes delete selected row/ROI
            self.selectLastROI = 'DeleteRow'
            self.change111.emit('RoiChanged')
            self.selectLastROI = False
        elif ev.key() == 70: #detects "f" deletes selection from the roi currently selected
            self.selectLastROI = 'DeleteROIinZ'
            self.change111.emit('RoiChanged')
            self.selectLastROI = False
        elif ev.key() == QtCore.Qt.Key_Home:
            print('Key_Home')
            self.setCurrentIndex(0)
            self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_End:
            print('Key_End')
            self.setCurrentIndex(self.getProcessedImage().shape[0]-1)
            self.play(0)
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            print('noRepeatKeys')
            if ev.key() == QtCore.Qt.Key_Right:
                if self.timeLine.value() + 1 <= self.timeLine.maxRange[1]:
                    self.timeLine.setValue(self.timeLine.value()+1)
                    self.timeLineChanged()
            if ev.key() == QtCore.Qt.Key_Left:
                if self.timeLine.value() - 1 >= self.timeLine.maxRange[0]:
                    self.timeLine.setValue(self.timeLine.value()-1)
                    self.timeLineChanged()
            if ev.key() == QtCore.Qt.Key_Up:
                if self.zLine.value() + 1 <= self.zLine.maxRange[1]:
                    self.zLine.setValue(self.zLine.value()+1)
                    self.zLineChanged()
            if ev.key() == QtCore.Qt.Key_Down:
                if self.zLine.value() - 1 >= self.zLine.maxRange[0]:
                    self.zLine.setValue(self.zLine.value() -1)
                    self.zLineChanged()

            '''
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()
            '''
        else:
            QtGui.QWidget.keyPressEvent(self, ev)
    
    def defineROI(self, ev):
        print('startedROI')
        x=[]
        y=[]
        #pos = pos[0]  ## using signal proxy turns original arguments into a tuple
        while ev.key() == QtCore.Qt.Key_Space:
                    x.append(self.mouseMoved()[0])
                    y.append(self.mouseMoved()[1])
        print('all x y points:', x, y)
                

    def keyReleaseEvent(self, ev):
        if ev.key() in [QtCore.Qt.Key_Space, QtCore.Qt.Key_Home, QtCore.Qt.Key_End]:
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            try:
                del self.keysPressed[ev.key()]
            except:
                self.keysPressed = {}
            self.evalKeyState()
        else:
            QtGui.QWidget.keyReleaseEvent(self, ev)
        
    def evalKeyState(self):
        if len(self.keysPressed) == 1:
            key = list(self.keysPressed.keys())[0]
            if key == QtCore.Qt.Key_Right:
                self.play(20)
                self.jumpFrames(1)
                self.lastPlayTime = ptime.time() + 0.2  ## 2ms wait before start
                                                        ## This happens *after* jumpFrames, since it might take longer than 2ms
            elif key == QtCore.Qt.Key_Left:
                self.play(-20)
                self.jumpFrames(-1)
                self.lastPlayTime = ptime.time() + 0.2
            elif key == QtCore.Qt.Key_Up:
                self.play(-100)
            elif key == QtCore.Qt.Key_Down:
                self.play(100)
            elif key == QtCore.Qt.Key_PageUp:
                self.play(-1000)
            elif key == QtCore.Qt.Key_PageDown:
                self.play(1000)
        else:
            self.play(0)
        
    def timeout(self):
        now = ptime.time()
        dt = now - self.lastPlayTime
        if dt < 0:
            return
        n = int(self.playRate * dt)
        if n != 0:
            self.lastPlayTime += (float(n)/self.playRate)
            if self.currentTime+n > self.image.shape[0]:
                self.play(0)
            self.jumpFrames(n)
        
    def setCurrentIndex(self, ind):
        """Set the currently displayed frame index."""
        self.currentTime = np.clip(ind, 0, self.getProcessedImage().shape[self.axes['t']]-1)
        
        self.ignoreTimeLine = True
        self.timeLine.setValue(self.tVals[self.currentTime])
        self.ignoreTimeLine = False
        #self.currentLayer = np.clip(ind, 0, self.getProcessedImage().shape[self.axes['z']]-1)
        #self.zLine.setValue(self.zVals[self.currentLayer])
        self.updateImage()
        #self.ignoreZLine = False


    def jumpFrames(self, n):
        """Move video frame ahead n frames (may be negative)"""
        if self.axes['t'] is not None:
            self.setCurrentIndex(self.currentTime + n)


    def normRadioChanged(self):
        self.imageDisp = None
        self.updateImage()
        self.autoLevels()
        #self.roiChanged()
        self.sigProcessingChanged.emit(self)
    '''
    def updateNorm(self):
        if self.ui.normTimeRangeCheck.isChecked():
            self.normRgn.show()
        else:
            self.normRgn.hide()
        
        if self.ui.normROICheck.isChecked():
            self.normRoi.show()
        else:
            self.normRoi.hide()
        
        if not self.ui.normOffRadio.isChecked():
            self.imageDisp = None
            self.updateImage()
            self.autoLevels()
            #self.roiChanged()
            self.sigProcessingChanged.emit(self)
    '''
    def normToggled(self, b):
        self.ui.normGroup.setVisible(b)
        self.normRoi.setVisible(b and self.ui.normROICheck.isChecked())
        self.normRgn.setVisible(b and self.ui.normTimeRangeCheck.isChecked())

    def hasTimeAxis(self):
        return 't' in self.axes and self.axes['t'] is not None
    '''
    def roiClicked(self):
        showtimePlot = False
        if self.ui.roiBtn.isChecked():
            showtimePlot = True
            self.roi.show()
            #self.ui.timePlot.show()
            self.ui.timePlot.setMouseEnabled(True, True)
            self.ui.splitter.setSizes([self.height()*0.6, self.height()*0.4])
            self.roiCurve.show()
            self.roiChanged()
            self.ui.timePlot.showAxis('left')
        else:
            self.roi.hide()
            self.ui.timePlot.setMouseEnabled(False, False)
            self.roiCurve.hide()
            self.ui.timePlot.hideAxis('left')
            
        if self.hasTimeAxis():
            showtimePlot = True
            mn = self.tVals.min()
            mx = self.tVals.max()
            self.ui.timePlot.setXRange(mn, mx, padding=0.01)
            self.timeLine.show()
            self.timeLine.setBounds([mn, mx])
            self.ui.timePlot.show()
            if not self.ui.roiBtn.isChecked():
                self.ui.splitter.setSizes([self.height()-35, 35])
        else:
            self.timeLine.hide()
            #self.ui.timePlot.hide()
            
        self.ui.timePlot.setVisible(showRoiPlot)
    '''
    '''
    def roiChanged(self):
        if self.image is None:
            return
            
        image = self.getProcessedImage()
        if image.ndim == 2:
            axes = (0, 1)
        elif image.ndim == 3:
            axes = (1, 2)
        else:
            return
        
        data, coords = self.roi.getArrayRegion(image.view(np.ndarray), self.imageItem, axes, returnMappedCoords=True)
        if data is not None:
            while data.ndim > 1:
                data = data.mean(axis=1)
            if image.ndim == 3:
                self.roiCurve.setData(y=data, x=self.tVals)
            else:
                while coords.ndim > 2:
                    coords = coords[:,:,0]
                coords = coords - coords[:,0,np.newaxis]
                xvals = (coords**2).sum(axis=0) ** 0.5
                self.roiCurve.setData(y=data, x=xvals)
    '''

    def quickMinMax(self, data):
        """
        Estimate the min/max values of *data* by subsampling.
        """
        '''
        while data.size > 1e6:
            ax = np.argmax(data.shape)
            sl = [slice(None)] * data.ndim
            sl[ax] = slice(None, None, 100000)
            sl = tuple(sl)
            data = data[sl]
        return nanmin(data), nanmax(data)
        '''
        #cc2.pyqt_set_trace()
        #if data.size > 1e6:
        if hasattr(self, 'minmax'): #included in hdf5 files
            return self.minmax[0], self.minmax[1]
        else:
            interval = data.size / 1e6
            ax = np.argmax(data.shape)
            sl = [slice(None)] * data.ndim
            sl[ax] = slice(None, None, int(np.ceil(interval)))
            sl = tuple(sl)
            data = data[sl]
            min1 = nanmin(data)
            max1 = nanmax(data)
            if min1 < 0:
                min1 =0
            if max1 > 2**16:
                max1 = 2**16
            return min1, max1
    
        #return 0, 500


    def normalize(self, image):
        """
        Process *image* using the normalization options configured in the
        control panel.
        
        This can be repurposed to process any data through the same filter.
        """
        if self.ui.normOffRadio.isChecked():
            return image
            
        div = self.ui.normDivideRadio.isChecked()
        norm = image.view(np.ndarray).copy()
        #if div:
            #norm = ones(image.shape)
        #else:
            #norm = zeros(image.shape)
        if div:
            norm = norm.astype(np.float32)
            
        if self.ui.normTimeRangeCheck.isChecked() and image.ndim == 3:
            (sind, start) = self.timeIndex(self.normRgn.lines[0])
            (eind, end) = self.timeIndex(self.normRgn.lines[1])
            #print start, end, sind, eind
            n = image[sind:eind+1].mean(axis=0)
            n.shape = (1,) + n.shape
            if div:
                norm /= n
            else:
                norm -= n
                
        if self.ui.normFrameCheck.isChecked() and image.ndim == 3:
            n = image.mean(axis=1).mean(axis=1)
            n.shape = n.shape + (1, 1)
            if div:
                norm /= n
            else:
                norm -= n
            
        if self.ui.normROICheck.isChecked() and image.ndim == 3:
            n = self.normRoi.getArrayRegion(norm, self.imageItem, (1, 2)).mean(axis=1).mean(axis=1)
            n = n[:,np.newaxis,np.newaxis]
            #print start, end, sind, eind
            if div:
                norm /= n
            else:
                norm -= n
                
        return norm

        
    def timeLineChanged(self):
        #(ind, time) = self.timeIndex(self.ui.timeSlider)
        if self.ignoreTimeLine:
            return
        self.play(0)
        (ind, time) = self.timeIndex(self.timeLine)
        if ind != self.currentTime:
            self.currentTime = ind
            self.updateImage()
        #self.timeLine.setPos(time)
        #self.emit(QtCore.SIGNAL('timeChanged'), ind, time)
        self.sigTimeChanged.emit(ind, time)
        print('real time and ind')
        print(time)
        print(ind)
        
    def zLineChanged(self):
        print('zLineChanged')
        #(ind, time) = self.timeIndex(self.ui.timeSlider)
        if self.ignoreZLine:
            return
        (ind, time) = self.layerIndex(self.zLine)
        print('ind')
        print(ind)
        print('time')
        print(time)
        print('zLineValue')
        print(self.zLine.value())
        if ind != self.currentLayer:
            self.currentLayer = ind
            self.updateImage()
        #self.timeLine.setPos(time)
        #self.emit(QtCore.SIGNAL('timeChanged'), ind, time)
        self.sigTimeChanged.emit(ind, time)
        #remove old rois
        for croi in self.aaroi.keys():
            self.view.removeItem( self.aaroi[croi]) 
        self.aaroi = {}
        #change layer number in textbox
        self.ui.ztext.setText(str(self.currentLayer))
        #re-draw rois for the new layer
        for ckey in self.aroi.index:
            croi = self.aroi['Z:XY'].loc[ckey]
            if self.currentLayer in croi.keys():
                if len(self.aroi['Z:XY'].loc[ckey][self.currentLayer]) == 1:
                    self.aaroi[ckey] = pg.CircleROI(self.circlePosition(self.aroi['Z:XY'].loc[ckey][self.currentLayer][0]), [self.markerRadius, self.markerRadius])
                else:
                    self.aaroi[ckey] = pg.PolyLineROI(self.aroi['Z:XY'].loc[ckey][self.currentLayer], closed= True)
                #set the current line color
                #depending on roi may have saved as qcolor or rgb in the roi dataframe 
                if isinstance(self.aroi['Color'].loc[ckey], list):
                    #color1 = pg.QtGui.QColor.setRgb(self.aroi['Color'].loc[ckey][0]*255, self.aroi['Color'].loc[ckey][1]*255, self.aroi['Color'].loc[ckey][2]*255)
                    color1 = pg.mkColor((self.aroi['Color'].loc[ckey][0]*255, self.aroi['Color'].loc[ckey][1]*255, self.aroi['Color'].loc[ckey][2]*255))
                
                else:
                    color1 = self.aroi['Color'].loc[ckey]
                print('Color set in zlinechanged', color1)
                self.aaroi[ckey].setPen(color=color1, width = 3)
                self.aaroi[ckey].key = ckey
                #self.aaroi[ckey].roi = ckey
                self.view.addItem(self.aaroi[ckey])  
                self.aaroi[ckey].sigRegionChanged.connect(self.updateRegion)

                
    def circlePosition(self, xy):
        x= xy[0]-self.markerRadius/2
        y=xy[1] - self.markerRadius/2
        return [x, y]

    def updateImage(self, autoHistogramRange=True):
        ## Redraw image on screen
        if self.image is None:
            return
            
        image = self.getProcessedImage()
        
        if autoHistogramRange:
            self.ui.histogram.setHistogramRange(self.levelMin, self.levelMax)
        
        # Transpose image into order expected by ImageItem
        if self.imageItem.axisOrder == 'col-major':
            axorder = ['t', 'z', 'x', 'y', 'c']
        else:
            axorder = ['t', 'z', 'y', 'x', 'c']
        axorder = [self.axes[ax] for ax in axorder if self.axes[ax] is not None]
        #transpose comment out for timeseries data
        #image = image.transpose(axorder)
        
            
        # Select time index
        if self.axes['t'] is not None:
            self.ui.timePlot.show()
            image = image[self.currentTime, self.currentLayer]
        self.imageItem.updateImage(image)
            
            
    def timeIndex(self, slider):
        ## Return the time and frame index indicated by a slider
        if self.image is None:
            return (0,0)
        
        t = slider.value()
        
        xv = self.tVals
        if xv is None:
            ind = int(t)
        else:
            if len(xv) < 2:
                return (0,0)
            totTime = xv[-1] + (xv[-1]-xv[-2])
            inds = np.argwhere(xv < t)
            if len(inds) < 1:
                return (0,t)
            ind = inds[-1,0]
        return ind, t

    def layerIndex(self, slider):
        ## Return the time and frame index indicated by a slider
        if self.image is None:
            return (0,0)
        
        t = slider.value()
        
        xv = self.zVals
        if xv is None:
            ind = int(t)
        else:
            if len(xv) < 2:
                return (0,0)
            totTime = xv[-1] + (xv[-1]-xv[-2])
            inds = np.argwhere(xv < t)
            if len(inds) < 1:
                return (0,t)
            ind = inds[-1,0]
        return ind, t

    def getView(self):
        """Return the ViewBox (or other compatible object) which displays the ImageItem"""
        return self.view

        
    def getImageItem(self):
        """Return the ImageItem for this ImageView."""
        return self.imageItem

        
    def getRoiPlot(self):
        """Return the ROI PlotWidget for this ImageView"""
        return self.ui.timePlot

       
    def getHistogramWidget(self):
        """Return the HistogramLUTWidget for this ImageView"""
        return self.ui.histogram


    def export(self, fileName):
        """
        Export data from the ImageView to a file, or to a stack of files if
        the data is 3D. Saving an image stack will result in index numbers
        being added to the file name. Images are saved as they would appear
        onscreen, with levels and lookup table applied.
        """
        img = self.getProcessedImage()
        if self.hasTimeAxis():
            base, ext = os.path.splitext(fileName)
            fmt = "%%s%%0%dd%%s" % int(np.log10(img.shape[0])+1)
            for i in range(img.shape[0]):
                self.imageItem.setImage(img[i], autoLevels=False)
                self.imageItem.save(fmt % (base, i, ext))
            self.updateImage()
        else:
            self.imageItem.save(fileName)

            
    def exportClicked(self):
        fileName = QtGui.QFileDialog.getSaveFileName()
        if fileName == '':
            return
        self.export(fileName)
        
    def buildMenu(self):
        self.menu = QtGui.QMenu()
        self.normAction = QtGui.QAction("Normalization", self.menu)
        self.normAction.setCheckable(True)
        self.normAction.toggled.connect(self.normToggled)
        self.menu.addAction(self.normAction)
        self.exportAction = QtGui.QAction("Export", self.menu)
        self.exportAction.triggered.connect(self.exportClicked)
        self.menu.addAction(self.exportAction)
        
    def menuClicked(self):
        if self.menu is None:
            self.buildMenu()
        self.menu.popup(QtGui.QCursor.pos())

    def setColorMap(self, colormap):
        """Set the color map. 

        ============= =========================================================
        **Arguments**
        colormap      (A ColorMap() instance) The ColorMap to use for coloring 
                      images.
        ============= =========================================================
        """
        self.ui.histogram.gradient.setColorMap(colormap)


    @addGradientListToDocstring()
    def setPredefinedGradient(self, name):
        """Set one of the gradients defined in :class:`GradientEditorItem <pyqtgraph.graphicsItems.GradientEditorItem>`.
        Currently available gradients are:   
        """
        self.ui.histogram.gradient.loadPreset(name)
