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
path1 =str([path1.parents[len(path1.parts) - i-1] for i, cfolder in enumerate(path1.parts) if cfolder=='CircuitCatcher2'][0])
print('path1', path1)
import numpy as np
projectname = 'A99'
date = '20190523'

saveFig = str(Path(path1) / str('Analysis' + date))
if not os.path.isdir(saveFig):
    os.mkdir(saveFig)


ExcelFile = str(Path(path1) / str(projectname + '_Summary1.xlsx'))
output = str(Path(path1) / saveFig / "SummaryOutput.xlsx")
save_data = str(Path(path1) / saveFig / projectname) + 'Data'+ '_' + date + '.hdf5'


#upstream crosses
crosses = pd.read_excel(Path(path1) / str(projectname + '_Cross_Summary.xlsx'), index_col=0)
crosses =crosses.transpose().to_dict()

rois = {'g5L': [0.2, 0.15, 0.5] ,
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
    
                 "b'p2aL": [0.8, 0.5, 0],
                 "b'p2aR": [0.8, 0.5, 0],
                 "b'p2mpL": [0.8, 0.5, 0],
                 "b'p2mpR": [0.8, 0.5, 0],
  
                 'Background' : [0.9, 0.9, 0.5]}




