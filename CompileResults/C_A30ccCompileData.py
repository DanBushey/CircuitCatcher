#generates a Analysis/Compiled_data.hdf5 file that contains all the roi intensity data
#generates excel files each containing intensity data from each roi

import pandas as pd
import os
import sys
import itertools
import numpy as np
#folderModules = '/data/JData/A/A30FastROI/CircuitCatcher'
#sys.path.append(folderModules)
import ccModules
#import parallelDB
import ipyparallel as ipp
#parallelDB.startLocalWait(15, 20) #(=number of engines, #number of to attempts connect to engines)
rc = ipp.Client()
dview = rc[:]

from A83_init20190411 import *
exceldata = pd.read_excel(output, sheetname='Summary')
    
#exceldata.to_csv(outputcsv, encoding='utf-8', sep = ",")
#exceldata.to_excel(output)
#change the index so there are no repeatd numbers
newindex = range(0, len(exceldata))
exceldata.index = newindex


#exceldata.to_csv(outputcsv, encoding='utf-8', sep=",")


#generate output folder
targetfolder = 'IndividualFigures'
if not os.path.isdir(os.path.join(saveFig, targetfolder)):
    os.mkdir(os.path.join(saveFig, targetfolder))
outputfolder = os.path.join(saveFig, targetfolder)

listexceldata=[exceldata.loc[row] for row in exceldata.index ]

'''
test if intensity data files exist
from pathlib import Path
output1 =[]
for dseries in listexceldata:
    output1.append(Path(dseries['Intensity_Data_File']).is_file())
exceldata['Intensity_Data_File_Exists'] = output1
exceldata.to_excel
'''

output1 = dview.map(ccModules.compileTimeSeriesData, listexceldata)
output1.wait_interactive()
output1.get()
#convert list of pandas series to dataframe
exceldata = pd.DataFrame(index=range(len(output1.get())), columns = output1.get()[0].index)
for row in exceldata.index:
    exceldata.loc[row] = output1.get()[row]

#add genotype to genotype column
for cgene in crosses.keys():
    for row in exceldata.index:
        if cgene in exceldata['Sample Name'].loc[row]:
            exceldata['Genotype'].loc[row] = crosses[cgene]['LexA']
#remove ones with empty genotype 
exceldata['Genotype'].replace('', np.nan, inplace=True)
exceldata.dropna(subset = ['Genotype'], inplace = True)

#add cross to cross column
exceldata['Cross'] = ''
for cgene in crosses.keys():
    for row in exceldata.index:
        if cgene in exceldata['Sample Name'].loc[row]:
            exceldata['Cross'].loc[row] = cgene

    
#save pandas data frame
exceldata.to_hdf(os.path.join(saveFig, 'Compiled_data.hdf5'), 'data')

#saved data into an excel file
#pull out intensity data so that rois have separate rows rathater than being embbedded in the dicitonary
dataframe = pd.DataFrame(columns = exceldata.columns)
for row, dseries in exceldata.iterrows():
    intensity_data=pd.DataFrame(dseries['intensity_data'])
    for rowI, dseriesIntensity in intensity_data.iterrows():
        for ccolumn in intensity_data.columns:
            dseries[ccolumn] = dseriesIntensity[ccolumn]
        #dseries.drop('intensity_data', inplace = True)
        dataframe = dataframe.append(dseries)
dataframe = dataframe.rename(columns = {'Name': 'roi'})
dataframe.drop(['intensity_data'], axis = 1, inplace=True)
exceldata = dataframe
#output so each sheet has separate genotype



for roi in rois.keys():
    excelfile= os.path.join(saveFig, 'Compiled_data_roi-' + roi + '.xlsx') 
    writer = pd.ExcelWriter(excelfile, engine = 'xlsxwriter')
    for tg in transgene.keys():
        croiFrame = exceldata[exceldata['roi'] == roi]
        ctgFrame = croiFrame[croiFrame['Genotype'] == tg]
        data = pd.DataFrame(index = croiFrame.index, columns = ['{0:.2f}'.format(float(i)) for i in croiFrame['timestamp'].iloc[0]], data=croiFrame['intensity'].tolist() )
        alldata = croiFrame.join(data)
        dropcolumns = ['intensity_data', 'voltage', 'timestamp']
        for dc in dropcolumns:
            alldata = alldata.drop(dc, 1)
        alldata.transpose().to_excel(writer, tg)
    #add a sheet showing voltage
    voltage = exceldata['voltage'].iloc[0]
    timestamp =  exceldata['timestamp'].iloc[0]
    binaryvolt = voltage.copy()
    binaryvolt[binaryvolt > 0] =1
    starttime = np.where(np.diff(binaryvolt.flatten()) == 1)[0]+1
    stoptime = np.where(np.diff(binaryvolt.flatten()) == -1)[0]
    voltageattime = np.zeros(len(timestamp))
    for i, cstart in enumerate(starttime):
        voltageattime[int(round(cstart/100)):int(round(stoptime[i]/100))] = voltage[cstart]
    voltageFrame = pd.DataFrame(index = ['{0:.2f}'.format(float(i)) for i in croiFrame['timestamp'].iloc[0]], columns = ["Voltage"], data = voltageattime)
    voltageFrame.to_excel(writer, 'voltage')
    writer.save()



