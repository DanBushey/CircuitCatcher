'''
Created on Dec 13, 2016

@author: Surf32
used to load data from the A08_Ramp_data
'''
import platform
import pandas as pd
import os
import Path from pathlib

if 'Windows' in platform.platform():
    path1 = r'C:\Users\Surf32\Desktop\ResearchDSKTOP\DataJ\A\A08_19F01-LexA_CaMPARI\A08_Data\2017-40-02'
    #path1 = os.path.normpath(path1)

else:
    #path1 = '/data/JData/A/A08_19F01-LexA_CaMPARI/A08_Data/2017-04-02'
    path1 = ''
    
projectname = 'A08_'
ReadFile = os.path.join(path1, projectname + "20170402.xlsx")
WriteFile = os.path.join(path1, "SummaryOutput_B_Compile.xlsx")
save_data = os.path.join(path1, 'A08-20170402.pckl')



#genotypes = ['A08-5', 'A08-2',  'A08-6',  'A08-4']
#genotypes = ['A08-5', 'A08-2',  'A08-4']
#transgene = ['13XLexAOP2-IVS-Syn21-opGCaMP6s-p10', 'p13XLexAOP2-IVS-Syn21-coCaMPARI-p10', 'p13XLexAOP2-IVS-Syn21-coCaMPARI.V398L-p10',  'p13XLexAOP2-IVS-Syn21-coCaMPARI.T394A-p10']
#transgene = ['coCaMPARI', 'CaMPARI-V398L',  'CaMPARI-G395A', 'CaMPARI-T394A']
genotypes = ['A08-1', 'A08-5', 'A08-2',  'A08-6',  'A08-4']
#transgene = ['GCaMP6s' , 'coCaMPARI', 'CaMPARI-V398L',  'CaMPARI-G395A', 'CaMPARI-T394A']
transgene = {'A08-1':'GCaMP6s' , 'A08-5':'coCaMPARI', 'A08-2':'CaMPARI-V398L',  'A08-6':'CaMPARI-G395A',  'A08-4':'CaMPARI-T394A'}


inclusion = {'Lamina Present': 'yes', 'Responds': 'yes', 'Random' : 'no', 'Grade': 1, 'Saline': 'regular', 'Stim Protocol' : 'campari expts'}
#inclusion = {'Lamina Present': 'yes', 'Responds': 'yes', 'Random' : 'no', 'Grade': 1, 'Saline': 'regular', 'Stim Protocol' : 'ramp'}
#inclusion = {'Lamina Present': 'yes',  'Random' : 'no', 'Grade': 1, 'Saline': 'regular', 'Stim Protocol' : 'ramp'}
saveFig = os.path.join(path1, 'FiguresStim-' + inclusion['Stim Protocol']  + '-Lamina-' + inclusion['Lamina Present'])

#where to save figures
controlGenotype = ['A08-1']
control1name =['GCaMP6s']

chrimson = ['L1 (27G06)']
target   = ['Mi1 (19F01)']

signsDict = {'A08-1': 'neg', 'A08-5': 'pos', 'A08-2': 'pos', 'A08-3': 'pos', 'A08-4': 'pos', 'A08-6': 'pos'}
rois = ['M8-10', 'M4', 'M1', 'Body']

if not os.path.isdir(saveFig):
    os.mkdir(saveFig)
    
colors1 = [(0, 1, 0), (0,0,1)]
#colors2 = [(0, 0.5, 0), (0.1, 0.2, 0.4), (0.3, 0.4, 0.6),  (0.5, 0.8, 1), (0.7, 1, 1)]
colors2 = {'A08-1':(0, 0.5, 0), 'A08-5': (0.1, 0.2, 0.4), 'A08-2':(0.3, 0.4, 0.6),   'A08-6':(0.5, 0.8, 1), 'A08-4':(0.7, 1, 1)}
