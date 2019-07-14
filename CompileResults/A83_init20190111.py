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
projectname = 'A83'
date = '20190111'

saveFig = str(Path(path1) / str('Analysis' + date))
if not os.path.isdir(saveFig):
    os.mkdir(saveFig)


ExcelFile = str(Path(path1) / str(projectname + '_Summary1.xlsx'))
output = str(Path(path1) / saveFig / "SummaryOutput.xlsx")
save_data = str(Path(path1) / saveFig / projectname) + 'Data'+ '_' + date + '.hdf5'

commonitems = {'sensor' : 'LexAop2-Syn21-opGCaMP6s in JK16F, LexAop2-Syn21-opGCaMP6s in su(Hw)attP8', 'trigger': '10XUAS-Syn21-Chrimson88-tdT-3.1 in attP18', 'LexA' : '64A11-LexAp65 in JK73A'}

crosses = {} 
crosses ['A83-20'] ={'GAL4' : 'SS33917' }
crosses ['A83-21'] ={'GAL4' : 'MB310C' }
crosses ['A83-22'] ={'GAL4' : 'MB082C' }
crosses ['A83-23'] ={'GAL4' : 'SS45234' }
crosses ['A83-24'] ={'GAL4' : 'SS32259' }
crosses ['A83-25'] ={'GAL4' : 'MB011B' }
crosses ['A83-26'] ={'GAL4' : 'MB434B' }
crosses ['A83-27'] ={'GAL4' : 'MB112C' }
crosses ['A83-28'] ={'GAL4' : 'MB077C' }
crosses ['A83-29'] ={'GAL4' : 'SS01308' }
crosses ['A83-30'] ={'GAL4' : 'MB433B' }

#add common items to crosses
for ccross in crosses.keys():
    for ccom in commonitems.keys():
        if ccom not in crosses[ccross].keys():
            crosses[ccross][ccom] = commonitems[ccom]
    
#colors2 = [(0, 0.5, 0), (0.1, 0.2, 0.4), (0.3, 0.4, 0.6),  (0.5, 0.8, 1), (0.7, 1, 1)]
rois = {'Region1': [0.3, 0.4, 0.7] ,  'Region2': [0.3, 0.8, 0.5], 'Background': [1, 0, 0] } #dictionary lists rois and color associated


