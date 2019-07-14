# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ImageViewTemplate.ui'
#
# Created: Thu May  1 15:20:40 2014
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from pyqtgraph.Qt import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(726, 588)
        self.gridLayout_3 = QtGui.QGridLayout(Form)
        self.gridLayout_3.setMargin(0)
        self.gridLayout_3.setSpacing(0)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        
        self.splitter = QtGui.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        
        self.layoutWidget = QtGui.QWidget(self.splitter)
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        
        #image view
        self.graphicsView = GraphicsView(self.layoutWidget)
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        self.gridLayout.addWidget(self.graphicsView, 0, 1, 3, 4)
        
        #intensity histogram
        self.histogram = HistogramLUTWidget(self.layoutWidget)
        self.histogram.setObjectName(_fromUtf8("histogram"))
        #self.histogram.setMinimumSize(QtCore.QSize(10, 40))
        self.histogram.setMaximumSize(QtCore.QSize(100,1000))
        self.gridLayout.addWidget(self.histogram, 0, 5, 1, 1)
        
        #save roi
        self.saveROIBtn = QtGui.QPushButton(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.saveROIBtn.sizePolicy().hasHeightForWidth())
        self.saveROIBtn.setSizePolicy(sizePolicy)
        self.saveROIBtn.setCheckable(True)
        self.saveROIBtn.setObjectName(_fromUtf8("saveROIBtn"))
        self.gridLayout.addWidget(self.saveROIBtn, 1, 5, 1, 1)
        
        #load new image dalogue
        self.loadBtn = QtGui.QPushButton(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.loadBtn.sizePolicy().hasHeightForWidth())
        self.loadBtn.setSizePolicy(sizePolicy)
        self.loadBtn.setCheckable(True)
        self.loadBtn.setObjectName(_fromUtf8("loadBtn"))
        self.gridLayout.addWidget(self.loadBtn, 4, 5, 1, 1)       
        
        self.menuBtn = QtGui.QPushButton(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.menuBtn.sizePolicy().hasHeightForWidth())
        self.menuBtn.setSizePolicy(sizePolicy)
        self.menuBtn.setObjectName(_fromUtf8("menuBtn"))
        self.gridLayout.addWidget(self.menuBtn, 2, 5, 1, 1)
        
        #add a radio button to turn off/on navigation with the mouse
        self.navRadio = QtGui.QRadioButton(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.menuBtn.sizePolicy().hasHeightForWidth())
        self.navRadio.setSizePolicy(sizePolicy)
        self.navRadio.setObjectName(_fromUtf8("navRadio"))
        self.gridLayout.addWidget(self.navRadio, 3, 1, 1, 1)
        
        #add radio button to select different rois
        #self.roiBox = QtGui.QButtonGroup()
        self.radioSinglePoint = QtGui.QRadioButton(self.layoutWidget)
        #self.roiBox.addButton(self.radioSinglePoint)
        #sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(1)
        #sizePolicy.setHeightForWidth(self.menuBtn.sizePolicy().hasHeightForWidth())
        #self.roiBox.setSizePolicy(sizePolicy)
        self.radioSinglePoint.setObjectName(_fromUtf8("radioSinglePoint"))
        self.gridLayout.addWidget(self.radioSinglePoint, 3, 2, 1, 1)
        
        #add roi button to select areas
        self.radioAreaROI = QtGui.QRadioButton(self.layoutWidget)
        self.radioAreaROI.setObjectName(_fromUtf8("radioAreaROI"))
        self.gridLayout.addWidget(self.radioAreaROI, 3, 3, 1, 1)
        
        #add radio button to select plygon
        self.radioPolygon = QtGui.QRadioButton(self.layoutWidget)
        self.radioPolygon.setObjectName(_fromUtf8("radioPolygon"))
        self.gridLayout.addWidget(self.radioPolygon, 3, 4, 1, 1)
        
        #add radio button to edit rois
        self.radioEdit = QtGui.QRadioButton(self.layoutWidget)
        self.radioEdit.setObjectName(_fromUtf8("radioEdit"))
        self.gridLayout.addWidget(self.radioEdit, 3, 5, 1, 1)
        
        #add a textbox indicating where in z location
        self.ztext = QtGui.QLineEdit(self.layoutWidget)
        self.ztext.setObjectName(_fromUtf8("ztext"))
        self.ztext.setMinimumSize(QtCore.QSize(10, 40))
        self.ztext.setMaximumSize(QtCore.QSize(100,100))
        self.gridLayout.addWidget(self.ztext, 3, 0, 1, 1)
        '''
        #box forms but cannot get in the correct
        self.radioGroup = QtGui.QGroupBox(Form)
        #self.radioGroup = QtGui.QGroupBox(self.layoutWidget)
        self.radioGroup.setObjectName(_fromUtf8("radioGroup"))
        self.gridLayoutR = QtGui.QGridLayout(self.radioGroup)
        #self.gridLayoutR = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayoutR.setMargin(0)
        self.gridLayoutR.setSpacing(0)
        self.gridLayoutR.setObjectName(_fromUtf8("gridLayoutR"))
        self.navRadio = QtGui.QRadioButton(self.radioGroup)
        self.navRadio.setObjectName(_fromUtf8("navRadio"))
        self.gridLayoutR.addWidget(self.navRadio, 0, 0, 1, 1)
        self.singleRadio = QtGui.QRadioButton(self.radioGroup)
        self.singleRadio.setChecked(False)
        self.singleRadio.setObjectName(_fromUtf8("singleRadio"))
        self.gridLayoutR.addWidget(self.singleRadio, 0, 1, 1, 1)
        self.labelR = QtGui.QLabel(self.radioGroup)
        #self.gridLayout.addWidget(self.labelR, 4, 5, 1, 1)
        
        #this works better but radio buttons are on top of each other
        self.radioGroup = QtGui.QGroupBox(self.layoutWidget)
        self.navRadio = QtGui.QRadioButton(self.radioGroup, 0, 0)
        self.navRadio.setChecked(True)
        self.navRadio.setObjectName(_fromUtf8("navRadio"))
        self.singleRadio = QtGui.QRadioButton(self.radioGroup, 0,1)
        self.singleRadio.setChecked(False)
        self.gridLayout.addWidget(self.radioGroup, 4, 5, 1, 1)
        '''
        
        
        #self.layoutWidget
        #self.roiPlot = PlotWidget(self.splitter)
        self.timePlot = PlotWidget(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.timePlot.sizePolicy().hasHeightForWidth())
        self.timePlot.setSizePolicy(sizePolicy)
        self.timePlot.setMinimumSize(QtCore.QSize(0, 40))
        self.timePlot.setMaximumSize(QtCore.QSize(1000, 50))
        self.timePlot.setObjectName(_fromUtf8("roiPlot"))
        self.gridLayout.addWidget(self.timePlot, 4, 0, 1, 4)
        
        #self.zPlot = PlotWidget(self.splitter)
        self.zPlot = PlotWidget(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.timePlot.sizePolicy().hasHeightForWidth())
        self.zPlot.setSizePolicy(sizePolicy)
        #self.zPlot.setMinimumSize(QtCore.QSize(10, 40))
        #self.zPlot.setMaximumSize(QtCore.QSize(50,1000))
        self.zPlot.setObjectName(_fromUtf8("zPlot"))
        self.gridLayout.addWidget(self.zPlot, 0, 0, 3, 1)

        
        #Radio buttons to normalize data
        self.gridLayout_3.addWidget(self.splitter, 0, 0, 1, 1)
        self.normGroup = QtGui.QGroupBox(Form)
        self.normGroup.setObjectName(_fromUtf8("normGroup"))
        self.gridLayout_2 = QtGui.QGridLayout(self.normGroup)
        self.gridLayout_2.setMargin(0)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.normSubtractRadio = QtGui.QRadioButton(self.normGroup)
        self.normSubtractRadio.setObjectName(_fromUtf8("normSubtractRadio"))
        self.gridLayout_2.addWidget(self.normSubtractRadio, 0, 2, 1, 1)
        self.normDivideRadio = QtGui.QRadioButton(self.normGroup)
        self.normDivideRadio.setChecked(False)
        self.normDivideRadio.setObjectName(_fromUtf8("normDivideRadio"))
        self.gridLayout_2.addWidget(self.normDivideRadio, 0, 1, 1, 1)
        self.label_5 = QtGui.QLabel(self.normGroup)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 1)
        self.label_3 = QtGui.QLabel(self.normGroup)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_2.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_4 = QtGui.QLabel(self.normGroup)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_2.addWidget(self.label_4, 2, 0, 1, 1)
        self.normROICheck = QtGui.QCheckBox(self.normGroup)
        self.normROICheck.setObjectName(_fromUtf8("normROICheck"))
        self.gridLayout_2.addWidget(self.normROICheck, 1, 1, 1, 1)
        self.normXBlurSpin = QtGui.QDoubleSpinBox(self.normGroup)
        self.normXBlurSpin.setObjectName(_fromUtf8("normXBlurSpin"))
        self.gridLayout_2.addWidget(self.normXBlurSpin, 2, 2, 1, 1)
        self.label_8 = QtGui.QLabel(self.normGroup)
        self.label_8.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.gridLayout_2.addWidget(self.label_8, 2, 1, 1, 1)
        self.label_9 = QtGui.QLabel(self.normGroup)
        self.label_9.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.gridLayout_2.addWidget(self.label_9, 2, 3, 1, 1)
        self.normYBlurSpin = QtGui.QDoubleSpinBox(self.normGroup)
        self.normYBlurSpin.setObjectName(_fromUtf8("normYBlurSpin"))
        self.gridLayout_2.addWidget(self.normYBlurSpin, 2, 4, 1, 1)
        self.label_10 = QtGui.QLabel(self.normGroup)
        self.label_10.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.gridLayout_2.addWidget(self.label_10, 2, 5, 1, 1)
        self.normOffRadio = QtGui.QRadioButton(self.normGroup)
        self.normOffRadio.setChecked(True)
        self.normOffRadio.setObjectName(_fromUtf8("normOffRadio"))
        self.gridLayout_2.addWidget(self.normOffRadio, 0, 3, 1, 1)
        self.normTimeRangeCheck = QtGui.QCheckBox(self.normGroup)
        self.normTimeRangeCheck.setObjectName(_fromUtf8("normTimeRangeCheck"))
        self.gridLayout_2.addWidget(self.normTimeRangeCheck, 1, 3, 1, 1)
        self.normFrameCheck = QtGui.QCheckBox(self.normGroup)
        self.normFrameCheck.setObjectName(_fromUtf8("normFrameCheck"))
        self.gridLayout_2.addWidget(self.normFrameCheck, 1, 2, 1, 1)
        self.normTBlurSpin = QtGui.QDoubleSpinBox(self.normGroup)
        self.normTBlurSpin.setObjectName(_fromUtf8("normTBlurSpin"))
        self.gridLayout_2.addWidget(self.normTBlurSpin, 2, 6, 1, 1)
        self.gridLayout_3.addWidget(self.normGroup, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.saveROIBtn.setText(_translate("Form", "SaveROI", None))
        self.loadBtn.setText(_translate("Form", "Load Image", None))
        self.menuBtn.setText(_translate("Form", "Menu", None))
        self.navRadio.setText(_translate("Form", "Navigate", None))
        self.radioSinglePoint.setText(_translate("Form", "Single ROI", None))
        self.radioAreaROI.setText(_translate("Form", "Area ROI", None))
        self.radioPolygon.setText(_translate("Form", "Poly ROI", None))
        self.radioEdit.setText(_translate("Form", "Edit ROIs", None))
        self.normGroup.setTitle(_translate("Form", "Normalization", None))
        self.normSubtractRadio.setText(_translate("Form", "Subtract", None))
        self.normDivideRadio.setText(_translate("Form", "Divide", None))
        self.label_5.setText(_translate("Form", "Operation:", None))
        self.label_3.setText(_translate("Form", "Mean:", None))
        self.label_4.setText(_translate("Form", "Blur:", None))
        self.normROICheck.setText(_translate("Form", "ROI", None))
        self.label_8.setText(_translate("Form", "X", None))
        self.label_9.setText(_translate("Form", "Y", None))
        self.label_10.setText(_translate("Form", "T", None))
        self.normOffRadio.setText(_translate("Form", "Off", None))
        self.normTimeRangeCheck.setText(_translate("Form", "Time range", None))
        self.normFrameCheck.setText(_translate("Form", "Frame", None))

from pyqtgraph.widgets.HistogramLUTWidget import HistogramLUTWidget
from pyqtgraph.widgets.GraphicsView import GraphicsView
from pyqtgraph.widgets.PlotWidget import PlotWidget



class Ui_FormPlotROI(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(726, 588)
        
        self.splitter = QtGui.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        
        self.layoutWidget = QtGui.QWidget(self.splitter)
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        
        
        self.plotwidget = PlotWidget(self.layoutWidget)
        self.plotwidget.setObjectName(_fromUtf8("plotView"))
        self.gridLayout.addWidget(self.plotwidget, 0, 1, 3, 3)
        