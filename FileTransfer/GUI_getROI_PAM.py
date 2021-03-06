# -*- coding: utf-8 -*-
"""
DB edited this code to test
This example demonstrates the use of pyqtgraph's dock widget system.

The dockarea system allows the design of user interfaces which can be rearranged by
the user at runtime. Docks can be moved, resized, stacked, and torn out of the main
window. This is similar in principle to the docking system built into Qt, but 
offers a more deterministic dock placement API (in Qt it is very difficult to 
programatically generate complex dock arrangements). Additionally, Qt's docks are 
designed to be used as small panels around the outer edge of a window. Pyqtgraph's 
docks were created with the notion that the entire window (or any portion of it) 
would consist of dockable components.

"""



import pyqtgraph as pg
import pyqtgraphTimeSeriesDB as tsDB
from pyqtgraph.Qt import QtCore, QtGui
#import pyqtgraph.console
#import numpy as np
#import os
#import tifffile
#import numpy as np
#import matplotlib
#import time
import pyqtROITable2
from pyqtgraph.dockarea import *
import pandas as pd

#Example of how to set color maskdata['Color'].loc[row] = pyqtgraph.mkColor(colordict[maskdata['Name'].loc[row]])
pullDownMenu1 = {'g5L': [0.2, 0.15, 0.5] ,
                 'g5R': [0.2, 0.15, 0.5],
                 'g4L': [0.4, 0.5, 0.5],
                 'g4R': [0.4, 0.5, 0.5],
                 'g3L': [0.6, 0.2, 0],
                 'g3R': [0.6, 0.2, 0],
                 

                 'a1L': [0.3, 0.2, 0.5],
                 'a1R': [0.3, 0.2, 0.5],

                 'b1L': [0.4, 0.2, 0.5],
                 'b1R': [0.4, 0.2, 0.5],
                 
                 'b2L': [0.8, 0.5, 0],
                 'b2R': [0.8, 0.5, 0.5],

                 'bp1L': [0.4, 0.15, 0.5],
                 'bp1R': [0.4, 0.15, 0.5],
    
                 'bp2aL': [0.8, 0.5, 0],
                 'bp2aR': [0.8, 0.5, 0],
                 'bp2mpL': [0.8, 0.5, 0],
                 'bp2mpR': [0.8, 0.5, 0],
  
                 'Background' : [0.9, 0.9, 0.5]}
#targetfilelist
excelfile = 'R:\leiz\I\4cVA\cVA_DON\scoreROIlist.xlsx'
'''
image_file_list = pd.read_excel(excelfile)
image_file_list = image_file_list['Image_File'].tolist()
output_file = 'R:\leiz\I\4cVA\cVA_DON\scoreROIlist.xlsx'
'''
image_file_list = None
output_file = None


app = QtGui.QApplication([])
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1000,500)
win.setWindowTitle('ROI Selection Tool')

## Create docks, place them into the window one at a time.
## Note that size arguments are only a suggestion; docks will still have to
## fill the entire dock area and obey the limits of their internal widgets.
d1 = Dock("Image-Dock1", size=(800, 800))     ## give this dock the minimum possible size
d4 = Dock("ROI-Table")
area.addDock(d1)  ## place d5 at left edge of d1
area.addDock(d4, 'right', d1) 


#add image widget
w1 = tsDB.TimeSeriesView(pullDownMenu = pullDownMenu1, image_file_list = image_file_list, output_file = output_file )
d1.addWidget(w1)



# add table filled with roi
table = pyqtROITable2.roiTableWidget(graphscene = w1)


#button to delete roi
delteBtn = QtGui.QPushButton('Delete')
deselectROIBtn = QtGui.QPushButton('Remove Selection')
deleteCurrentROI = QtGui.QPushButton('Del Cur-Layer Sel')
tableDockLayout = pg.LayoutWidget()
tableDockLayout.addWidget(table, row = 0, col =0, rowspan=4, colspan =3)
tableDockLayout.addWidget(delteBtn, row = 5, col =1, colspan=1) 
tableDockLayout.addWidget(deselectROIBtn, row = 5, col =0, colspan=1) 
tableDockLayout.addWidget(deleteCurrentROI, row = 5, col =2, colspan=1) 
#table.setData(data)
d4.addWidget(tableDockLayout)
def deleteROI():
    #pdb.set_trace()
    print('delete')
    global table
    global w1
    #pdb.set_trace()
    if table.selectedItems():
        if w1.aroi.index[table.currentRow] in w1.aaroi.keys():
            w1.view.removeItem(w1.aaroi[w1.aroi.index[table.currentRow]])
            del w1.aaroi[w1.aroi.index[table.currentRow]]
        w1.aroi = w1.aroi.drop(w1.aroi.index[table.currentRow])
        table.setData()
    else:
        print('Row is not selected')

def deleteCurrentROIFunct():
    print('delete ROI in current layer only')
    global table
    global w1
    if table.selectedItems():
        #pdb.set_trace()
        if w1.aroi.index[table.currentRow] in w1.aaroi.keys():
            w1.view.removeItem(w1.aaroi[w1.aroi.index[table.currentRow]])
            del w1.aaroi[w1.aroi.index[table.currentRow]]
        if w1.currentLayer in w1.aroi['Z:XY'].loc[w1.aroi.index[table.currentRow]] .keys():
            #w1.aroi['Z:XY'].loc[w1.aroi.index[table.currentRow]] = w1.aroi['Z:XY'].loc[w1.aroi.index[table.currentRow]].pop(w1.currentLayer)
            w1.aroi['Z:XY'].loc[w1.aroi.index[table.currentRow]].pop(w1.currentLayer, None)
            #ZXY = w1.aroi['Z:XY'].loc[w1.aroi.index[table.currentRow]]
            #ZXY.pop(w1.currentLayer, None)
            #w1.aroi['Z:XY'].loc[w1.aroi.index[table.currentRow]] = ZXY
            table.setData()
    else:
        print('Row is not selected')

    

def deselectROI():
    print('deselect')
    global table
    table.clearSelection()
        
        
delteBtn.clicked.connect(deleteROI)
deselectROIBtn.clicked.connect(deselectROI)
deleteCurrentROI.clicked.connect(deleteCurrentROIFunct)


win.show()



## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
