'''
Created on Dec 13, 2016

@author: Surf32
used to load data from the A08_Ramp_data
'''
import platform
import pandas as pd
import os
from pathlib import Path
path1 = Path.cwd()
path1 =str([path1.parents[len(path1.parts) - i-1] for i, cfolder in enumerate(path1.parts) if cfolder=='CircuitCatcher'][0])
print('path1', path1)
import numpy as np
projectname = 'A74'
date = '20181101'

saveFig = str(Path(path1) / str('Analysis' + date))
if not os.path.isdir(saveFig):
    os.mkdir(saveFig)


ExcelFile = str(Path(path1) / str(projectname + '_Summary1.xlsx'))
output = str(Path(path1) / saveFig / "SummaryOutput.xlsx")
save_data = str(Path(path1) / saveFig / projectname) + 'Compiled_Data'+ '_' + date + '.hdf5'


transgene = {} 
transgene ['A74-1'] ={'sensor' : '20XUASGCaMP6f 5X9317B', 'position': 1, 'trigger': '10xUAS-Chrimson-tdT', 'LexA' : 'VT45561-LexA (attp80)', 'GAL4' : 'Vt38111-GAL4 (Jli22) ', 'response' : 'pos', 'color' : (0.4, 0.6, 0.9), 'rois' : ['ganna1-L', 'gamma1-R',  'Background']}

    
#colors2 = [(0, 0.5, 0), (0.1, 0.2, 0.4), (0.3, 0.4, 0.6),  (0.5, 0.8, 1), (0.7, 1, 1)]
rois = {'gamma1-R': [0.3, 0.15, 0.5] , 'gamma1-L': [0.6, 0.2, 0.5], 'Background' : [0.9, 0.9, 0.5] } #dictionary lists rois and color associated


