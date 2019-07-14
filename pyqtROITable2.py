'''
Created on May 9, 2017

@author: Surf32
'''
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import functools
import ccModules2 as cc2



class roiTableWidget(QtGui.QTableWidget):
    '''
    classdocs
    '''
    newitemselected = QtCore.pyqtSignal(object)

    def __init__(self,  graphscene=None, colors=None, intensity_plot = None, *args, **kwds):
        '''
        Constructor
        '''
        QtGui.QTableWidget.__init__(self, *args)
        self.setRowCount(4)
        self.setColumnCount(1)
        self.setItem(0,0, QtGui.QTableWidgetItem('Test'))
       
        #self.itemPressed.connect(self.itemSelected)
        #self.cellPressed.connect(self.itemSelected)
        self.itemSelectionChanged.connect(self.itemSelected) #detects if row in table is selected
        self.graphscene = graphscene
        graphscene.change111.connect(self.changeROI) #detects change in roi from Timeseries data
        self.colors = colors
        self.actDict={}
        self.makeMenu()
        self.currentRow = None
        self.currentCol = None
        self.setData()
        self.itemChanged.connect(self.changeName) # triggered if user inputs data into cell
        self.changeNameActive = 1
        self.intensity_plot = intensity_plot
        
    def changeName(self):
        
        print('Changing ROI name')
        
        if self.currentRow != None and self.changeNameActive == 1:
            cname = self.item(self.currentRow, 0)
            print(cname.text())
            if self.graphscene.aroi['Name'].iloc[self.currentRow] != cname.text():
                #self.graphscene.aroi.iloc[self.currentRow].rename(str(cname.text()))
                #make sure text is a string because PyQt4 is passing it as PyQT4 string
                self.graphscene.aroi['Name'].iloc[self.currentRow]  = str(cname.text())
                print(self.graphscene.pullDownMenu.keys())
                if cname.text() in self.graphscene.pullDownMenu.keys():
                    self.graphscene.aroi['Color'].iloc[self.currentRow]  = self.graphscene.pullDownMenu[cname.text()]
                    col = pg.mkColor((self.graphscene.aroi['Color'].iloc[self.currentRow][0]*255, self.graphscene.aroi['Color'].iloc[self.currentRow][1]*255, self.graphscene.aroi['Color'].iloc[self.currentRow][2]*255))
                    if self.graphscene.aroi.index[self.currentRow] in self.graphscene.aaroi.keys(): #if item is not currently in layer it will not be in aaroi must check
                        self.graphscene.aaroi[self.graphscene.aroi.index[self.currentRow]].setPen(color=col, width = 3) #self.graphscene.aaroi is an empty dictionary in some cases?? and causes program to crash
                    self.item(self.currentRow, 1).setBackground(col)
                else:
                    self.graphscene.pullDownMenu[cname.text()] = list(pg.QtGui.QColor.getRgbF(self.item(self.currentRow, 1).background().color()))
                    self.graphscene.aroi['Color'].iloc[self.currentRow] = list(pg.QtGui.QColor.getRgbF(self.item(self.currentRow, 1).background().color()))
                print('Changed graphscene.aroi', self.graphscene.aroi.index)
            if cname.text() not in self.graphscene.pullDownMenu:
                print('newtype')
                print(type(cname.text()))
                self.graphscene.pullDownMenu[cname.text()] = [0.3,0.3, 0.3]
                #print('pulldownmenu', self.pullDownMenu)
                self.makeMenu()

    
    def makeMenu(self):
        self.remPullDownMenu()
        if self.currentColumn == 0:
            for item1 in list(self.graphscene.pullDownMenu.keys()):
                #pdb.set_trace()
                actionEdit = QtGui.QAction(item1, self)
                actionEdit.triggered.connect(functools.partial(self.addItemAction, item1))
                self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
                self.addAction(actionEdit)
                self.actDict[item1]=actionEdit
        elif self.currentColumn == 1:
            if self.colors != None:
                for item1 in list(self.graphscene.pullDownMenu.keys()):
                    #pdb.set_trace()
                    actionEdit = QtGui.QAction(item1, self)
                    actionEdit.triggered.connect(functools.partial(self.addItemAction, item1))
                    self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
                    self.addAction(actionEdit)
                    self.actDict[item1]=actionEdit
                    
    
    def addItemAction(self, i1):
        print('additem triggered')
        self.setItem(self.currentRow, self.currentColumn, QtGui.QTableWidgetItem(i1))
        #self.store[self.segmentation.columnnames[self.currentColumn]][self.currentRow]=i1
        
        
    def remPullDownMenu(self):
        for key in self.actDict:
            self.removeAction(self.actDict[key])
        
        
    def itemSelected(self):
        if self.selectedItems():
            for currentTable in self.selectedItems():
                print('column', currentTable.column())
                print('row', currentTable.row())  
                self.currentColumn = currentTable.column()
                self.currentRow = currentTable.row()
                self.graphscene.currentRowSelectedinTable = currentTable.row()
            self.remPullDownMenu()
            self.makeMenu()
            if self.currentColumn == 1:
                col = QtGui.QColorDialog.getColor()
                self.graphscene.aroi['Color'].iloc[self.currentRow] = list(pg.QtGui.QColor.getRgbF(col)[0:3])
                self.graphscene.pullDownMenu[self.item(self.currentRow, 0).text()] = list(pg.QtGui.QColor.getRgbF(col)[0:3])
                print(pg.QtGui.QColor.getRgbF(col))
                self.item(self.currentRow, self.currentColumn).setBackground(col)
                #set roi line color
                #only set if current roi is being displayed in current layer
                print('Changing line color')
                print(self.graphscene.aroi.index)
                print(self.graphscene.aaroi.keys())
                if self.graphscene.aroi.index[self.currentRow] in self.graphscene.aaroi.keys():
                    print('Line color in current layer')
                    self.graphscene.aaroi[self.graphscene.aroi.index[self.currentRow]].setPen(color=col, width = 3)
            elif self.currentColumn == 2:
                #plot the current row
                color1 = self.graphscene.aroi['Color'].iloc[self.currentRow]
                color1 = np.array(color1) * 255
                color1 = color1.tolist()
                intensitydata = cc2.getIntensity(self.graphscene.aroi['mask_index'].iloc[self.currentRow], self.graphscene.image)
                self.intensity_plot.plot(intensitydata, pen = pg.mkPen(color1, width=2))
                #go roi region select in table
                ## get the top z layer
                zlayers = list(self.graphscene.aroi['Z:XY'].iloc[self.currentRow].keys())
                z = zlayers[0]
                print('sending to z layer', z)
                self.graphscene.zLine.setValue(z+1)
                self.graphscene.zLineChanged()
                if self.graphscene.aroi.index[self.currentRow] in self.graphscene.aaroi.keys():
                    print('Line color in current layer')
                    self.graphscene.aaroi[self.graphscene.aroi.index[self.currentRow]].setPen(color=color1, width = 10, style=QtCore.Qt.DotLine, alpha = 0.5)
                #cc2.pyqt_set_trace()
        else:
            #self.currentRow = None
            #self.graphscene.currentRowSelectedinTable = None
            print('Nothing selected')
            #self.intensity_plot.clear()
            self.My_clear_Selection()
        self.newitemselected.emit('RoiChanged')
                
    def setData(self): 
        self.changeNameActive = 0
        columns = ('Name', 'Type', 'Plot')
        self.setHorizontalHeaderLabels(columns)
        self.setColumnCount(len(columns))
        self.setRowCount(len(self.graphscene.aroi))
        for i, roi_key in enumerate(self.graphscene.aroi.index):
            Name = self.graphscene.aroi['Name'].loc[roi_key]
            if isinstance(Name, int):
                Name = str(Name)
            
            self.setItem(i, 0, QtGui.QTableWidgetItem(Name ))
            #for i2, ccol in enumerate(columns[1:]):
            self.setItem(i, 1, QtGui.QTableWidgetItem(self.graphscene.aroi['Type'].loc[roi_key] ))
            if isinstance(self.graphscene.aroi['Color'].loc[roi_key], list):
                color1 = pg.mkColor((self.graphscene.aroi['Color'].loc[roi_key][0]*255, self.graphscene.aroi['Color'].loc[roi_key][1]*255, self.graphscene.aroi['Color'].loc[roi_key][2]*255))
            else:
                color1 = self.graphscene.aroi['Color'].loc[roi_key]
            self.item(i, 1).setBackground(color1)
            self.setItem(i, 2, QtGui.QTableWidgetItem(' '))
            
        self.resizeColumnsToContents()
        self.changeNameActive = 1
    
    def changeROI(self):
        if self.graphscene.selectLastROI == 'SelectLastCell': #quick key "a"
            lastrow = len(self.graphscene.aroi)
            self.setCurrentCell(lastrow -1,0)
        elif self.graphscene.selectLastROI == 'RemoveSelection': #quick key "d"
            print('shift key pressed')
            #cc2.pyqt_set_trace()
            self.My_clear_Selection()
        elif self.graphscene.selectLastROI == 'DeleteRow': #quick key "x"
            print('Delete Selected Row = key x')
            self.deleteROI()
        elif self.graphscene.selectLastROI == 'DeleteROIinZ': #quick key "x"
            print('Delete Selected roi in current z = key f')
            self.deleteCurrentROIFunct()
        else:
            self.setData()
            

    def My_clear_Selection(self):
        self.clearSelection()
        self.intensity_plot.clear()
        self.currentColumn = None
        self.currentRow = None
        self.graphscene.currentRowSelectedinTable = None
        
    def deleteCurrentROIFunct(self):
        print('delete ROI in current layer only')
        if self.selectedItems():
            #pdb.set_trace()
            if self.graphscene.aroi.index[self.currentRow] in self.graphscene.aaroi.keys():
                self.graphscene.view.removeItem(self.graphscene.aaroi[self.graphscene.aroi.index[self.currentRow]])
                del self.graphscene.aaroi[self.graphscene.aroi.index[self.currentRow]]
            if self.graphscene.currentLayer in self.graphscene.aroi['Z:XY'].loc[self.graphscene.aroi.index[self.currentRow]] .keys():
                #self.graphscene.aroi['Z:XY'].loc[self.graphscene.aroi.index[self.currentRow]] = self.graphscene.aroi['Z:XY'].loc[self.graphscene.aroi.index[self.currentRow]].pop(self.graphscene.currentLayer)
                self.graphscene.aroi['Z:XY'].loc[self.graphscene.aroi.index[self.currentRow]].pop(self.graphscene.currentLayer, None)
                #ZXY = self.graphscene.aroi['Z:XY'].loc[self.graphscene.aroi.index[self.currentRow]]
                #ZXY.pop(self.graphscene.currentLayer, None)
                #self.graphscene.aroi['Z:XY'].loc[self.graphscene.aroi.index[self.currentRow]] = ZXY
                self.setData()

    def deleteROI(self):
        if self.selectedItems():
            if self.graphscene.aroi.index[self.currentRow] in self.graphscene.aaroi.keys():
                self.graphscene.view.removeItem(self.graphscene.aaroi[self.graphscene.aroi.index[self.currentRow]])
                del self.graphscene.aaroi[self.graphscene.aroi.index[self.currentRow]]
                self.graphscene.aroi = self.graphscene.aroi.drop(self.graphscene.aroi.index[self.currentRow])
                self.setData()
                self.My_clear_Selection()
            else:
                print('Row is not selected')
