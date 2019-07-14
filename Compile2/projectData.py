import platform
import pandas as pd
import os

if 'Windows' in platform.platform():
    path1 = r'/data/JData/A/A18/nSyb/Ca'
    #path1 = os.path.normpath(path1)

else:
    #path1 = '/HD4A/HD4A-JData/A32_Saline/Mg2'
    path1 = '/media/daniel/Seagate Backup Plus Drive1/HD4A-JData/A32_Saline/Mg-Compiled'


projectname = 'A32_'
ReadFile = os.path.join(path1, 'A32_Mg_Summary2.xlsx')
WriteFile = os.path.join(path1, "SummaryOutput2.xlsx")
save_data = os.path.join(path1, projectname + 'DATA2.hdf5')


transgene = {} 
transgene ['A12-1'] ={'sensor' : 'GCaMP6s', 'trigger': 'lexAOP2-CsChrimson', 'LexA' : '27G06-LexA (L1)', 'GAL4' : '19F01-GAL4 (Mi1)', 'response' : 'neg', 'color' : (0, 0.5, 0), 'rois' : ['Body', 'M1', 'M4', 'M8-10', 'Background']}
transgene ['A12-3']  ={'sensor' : 'coCaMPARI', 'trigger': 'lexAOP2-CsChrimson', 'LexA' : '27G06-LexA (L1)', 'GAL4' : '19F01-GAL4 (Mi1)', 'response' : 'pos', 'color' : (0.3, 0.4, 0.6), 'rois' : ['Body', 'M1', 'M4', 'M8-10', 'Background']}
transgene ['A12-6']  ={'sensor' : 'coCaMPARI-T394A', 'trigger': 'lexAOP2-CsChrimson', 'LexA' : '27G06-LexA (L1)', 'GAL4' : '19F01-GAL4 (Mi1)', 'response' : 'pos', 'color' : (0.7, 1, 1), 'rois' : ['Body', 'M1', 'M4', 'M8-10', 'Background']}
transgene ['A12-9']  ={'sensor' : 'ASAP2s', 'trigger': 'lexAOP2-CsChrimson', 'LexA' : '27G06-LexA (L1)', 'GAL4' : '19F01-GAL4 (Mi1)', 'response' : 'pos', 'color' : (0.9, 1, 1), 'rois' : ['Body', 'M1', 'M4', 'M8-10', 'Background']}
transgene ['A32-2']  ={'sensor' : 'UAS-GCaMP6s', 'trigger': 'UAS-Chrimson', 'LexA' : '59C10-LexA (TM3)', 'GAL4' : '64B03AD-GAL4 (L5)', 'response' : 'pos', 'color' : (1, 0.5, 0.5), 'rois' : ['Body', 'M1', 'M3', 'M8-10', 'Background']}
	
#where to save figures
#inclusion = {'Lamina Present': 'yes', 'Responds': 'yes', 'Random' : 'no', 'Grade': 1, 'Saline': 'regular', 'Stim Protocol' : 'campari expts'}
#inclusion = {'Lamina Present': 'yes', 'Responds': 'yes', 'Random' : 'no', 'Grade': 1, 'Saline': 'regular', 'Stim Protocol' : 'ramp'}
#inclusion = {'Lamina Present': 'yes',  'Random' : 'no', 'Grade': 1, 'Saline': 'regular', 'Stim Protocol' : 'ramp'}
#saveFig = os.path.join(path1, 'FiguresStim-' + inclusion['Stim Protocol']  + '-Lamina-' + inclusion['Lamina Present'])
saveFig = os.path.join(path1, 'Analysis')
if not os.path.isdir(saveFig):
    os.mkdir(saveFig)


    
#colors2 = [(0, 0.5, 0), (0.1, 0.2, 0.4), (0.3, 0.4, 0.6),  (0.5, 0.8, 1), (0.7, 1, 1)]
rois = {'Body': [0.3, 0.15, 0.5] , 'M1': [0.6, 0.2, 0.5], 'M3':[1, 0.25, 0.15], 'M4' : [0.9, 0.5, 0], 'M8-10' : [0.9, 0.9, 0.5], 'Background': [1, 0, 0] } #dictionary lists rois and color associated
Solute_Colors ={1:[1, 0, 0], 4:[0, 1, 0], 10:[0, 0, 1]}

#specific type of saline tested
Saline = 'Saline MgCl (mM)'