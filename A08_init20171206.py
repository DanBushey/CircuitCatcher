'''
Created on Dec 13, 2016

@author: Surf32
used to load data from the A08_Ramp_data
'''
import platform
import pandas as pd
import os
from pathlib import Path

if 'Windows' in platform.platform():
    path1 = r'C:\Users\Surf32\Desktop\ResearchDSKTOP\DataJ\A\A08_19F01-LexA_CaMPARI\A08_Data\2017-40-02'
    #path1 = os.path.normpath(path1)

else:
    #path1 = '/data/JData/A/A08_19F01-LexA_CaMPARI/A08_Data/2017-04-02'
    path1 = '/media/daniel/Seagate Backup Plus Drive2/JData/A/A08_19F01-LexA_CaMPARI/A08_Data'

saveFig = str(Path(path1) / 'Analysis20171206')
if not os.path.isdir(saveFig):
    os.mkdir(saveFig)

projectname = 'A08_'
ExcelFile = str(Path(path1) / "A08_20171115LaminaOnly.xlsx")
outputcsv = str(Path(path1) / saveFig / "SummaryOutput.csv")
save_data = str(Path(path1) / saveFig / projectname) + 'Data20171206.hdf5'


transgene = {} 
transgene ['A12-1'] ={'sensor' : 'GCaMP6s', 'trigger': 'lexAOP2-CsChrimson', 'LexA' : '27G06-LexA (L1)', 'GAL4' : '19F01-GAL4 (Mi1)', 'response' : 'neg', 'color' : (0, 0.5, 0), 'rois' : ['Body', 'M1', 'M4', 'M8-10', 'Background']}
transgene ['A12-3']  ={'sensor' : 'coCaMPARI', 'trigger': 'lexAOP2-CsChrimson', 'LexA' : '27G06-LexA (L1)', 'GAL4' : '19F01-GAL4 (Mi1)', 'response' : 'pos', 'color' : (0.3, 0.4, 0.6), 'rois' : ['Body', 'M1', 'M4', 'M8-10', 'Background']}
transgene ['A12-6']  ={'sensor' : 'coCaMPARI-T394A', 'trigger': 'lexAOP2-CsChrimson', 'LexA' : '27G06-LexA (L1)', 'GAL4' : '19F01-GAL4 (Mi1)', 'response' : 'pos', 'color' : (0.7, 1, 1), 'rois' : ['Body', 'M1', 'M4', 'M8-10', 'Background']}
transgene ['A12-9']  ={'sensor' : 'ASAP2s', 'trigger': 'lexAOP2-CsChrimson', 'LexA' : '27G06-LexA (L1)', 'GAL4' : '19F01-GAL4 (Mi1)', 'response' : 'pos', 'color' : (0.9, 1, 1), 'rois' : ['Body', 'M1', 'M4', 'M8-10', 'Background']}
transgene ['A32-2']  ={'sensor' : 'UAS-GCaMP6s', 'trigger': 'UAS-Chrimson', 'LexA' : '59C10-LexA (TM3)', 'GAL4' : '64B03AD-GAL4 (L5)', 'response' : 'pos', 'color' : (1, 0.5, 0.5), 'rois' : ['Body', 'M1', 'M3', 'Background']}
	
    
#colors2 = [(0, 0.5, 0), (0.1, 0.2, 0.4), (0.3, 0.4, 0.6),  (0.5, 0.8, 1), (0.7, 1, 1)]
rois = {'Body': [0.3, 0.15, 0.5] , 'M1': [0.6, 0.2, 0.5], 'M3':[1, 0.25, 0.15], 'M4' : [0.9, 0.5, 0], 'M8-10' : [0.9, 0.9, 0.5], 'Background': [1, 0, 0] } #dictionary lists rois and color associated


