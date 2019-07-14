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
projectname = 'A78'
date = '20181030'

saveFig = str(Path(path1) / str('Analysis' + date))
if not os.path.isdir(saveFig):
    os.mkdir(saveFig)


ExcelFile = str(Path(path1) / str(projectname + '_Summary1.xlsx'))
output = str(Path(path1) / saveFig / "SummaryOutput.xlsx")
save_data = str(Path(path1) / saveFig / projectname) + 'Data'+ '_' + date + '.hdf5'


transgene = {} 
transgene ['1'] ={'sensor' : 'GCaMP', 'position': 6, 'trigger': '10xUAS-Chrimson-tdT', 'LexA' : 'unknown', 'GAL4' : 'unknown', 'response' : 'pos', 'color' : (0.4, 0.6, 0.9), 'rois' : ['Body', 'M1', 'M4', 'M8-10', 'Background'], 'kd' : 33}

    
#colors2 = [(0, 0.5, 0), (0.1, 0.2, 0.4), (0.3, 0.4, 0.6),  (0.5, 0.8, 1), (0.7, 1, 1)]
rois = {'Cell': [0.3, 0.15, 0.5] ,  'Dendrite' : [0.9, 0.5, 0], 'Axon' : [0.9, 0.9, 0.5], 'Background': [1, 0, 0] } #dictionary lists rois and color associated


