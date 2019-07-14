#designed to run through all rows in excel sheet and generate figures summarizing results
#there are high memory requirements when running this script so keep number of workers low

import pandas as pd
import os
import sys
import itertools
from pathlib import Path
#folderModules = '/data/JData/A/A30FastROI/CircuitCatcher'
#sys.path.append(folderModules)
import ccModules
#import parallelDB
import ipyparallel as ipp
#parallelDB.startLocalWait(7, 20) #(=number of engines, #number of attempts to connect to engines)
rc = ipp.Client()
dview = rc[:]
from A83_init20190411 import *

#excel file to write summary data
writer = pd.ExcelWriter(output, engine='xlsxwriter')

xl = pd.ExcelFile(ExcelFile)
print(xl.sheet_names)
exceldata=xl.parse(xl.sheet_names[0])
for sheet in xl.sheet_names[1:]:
    if not 'Protocol' in sheet:
        exceldata = exceldata.append(xl.parse(sheet, encoding='utf-8'))
#change the index so there are no repeatd numbers
newindex = range(0, len(exceldata))
exceldata.index = newindex
exceldata.to_excel(writer, sheet_name = 'Summary', encoding='utf-8')
writer.save()

#get list of files
import pathlibDB as pbDB
files = pbDB.getDirContents(path1)
print(files)

#create a new pandas dataframe matching excel data to file data
## search for directories that match a directory and create a new row
columns = list(exceldata.columns)
columns.extend(['Paths', 'Directory?'])
newframe = [] 
files = files[files['Directory']]
files = files[~files['File_Name'].str.contains('_2color_')]
files = files[~files['File_Name'].str.contains('_1040nm_')]
#remove rows with nan in 'Sample Name'
exceldata.dropna(subset = ['Sample Name'], inplace =True)
stims = {'stim30s06V' : 1, 'stimDur300at15' : 2}
for row, dseries in exceldata.iterrows():
    print(dseries)
    cfiles = files[files['File_Name'].str.contains(dseries['Sample Name'])]
    for row2, dseries2 in cfiles.iterrows():
        cdict = dseries.to_dict()
        cdict['Paths'] = dseries2['Full_Path']
        cdict['Directory?'] = dseries2['Directory']
        #make stim protocol specific
        for cstim in stims:
            print(cstim)
            if cstim in dseries2['File_Name']:
                cdict['Stim Protocol'] = cstim
                cdict['No.'] = stims[cstim]
        newframe.append(cdict)
exceldata = pd.DataFrame(newframe)

#find the file with the intensity data and image data
intensitydatafile =[]
for row in exceldata.index:
    if os.path.isdir(exceldata['Paths'].loc[row]):
        files, indx = ccModules.getFileContString(exceldata['Paths'].loc[row], 'IntensityData.hdf5')
        if len(indx[0]) == 1:
            intensitydatafile.append(os.path.join(exceldata['Paths'].loc[row], files.values[0]))
        else:
            intensitydatafile.append('No intensity data file')
        #search for the file containing image data
        
    else:
         intensitydatafile.append('Directory does not exist')
exceldata['Intensity_Data_File'] = intensitydatafile
writer = pd.ExcelWriter(output, engine='xlsxwriter')
exceldata.to_excel(writer, sheet_name = 'Summary')
writer.save()

#remove any rows that have Intensity_Data_File
exceldata = exceldata[exceldata['Intensity_Data_File'].str.contains('No intensity data file') == False]
#exceldata = exceldata[exceldata['Saline'].str.contains('regular') == True]

#test for duplicated sample names
#bool1 = exceldata['Sample Name'].duplicated()
#exceldata['Duplicated'] = bool1
#exceldata.to_excel(WriteFile)
#confirm all rois are present
roicount = dview.map(ccModules.confirmROIs, exceldata['Intensity_Data_File'].values, itertools.repeat(list(rois.keys()), len(exceldata['Intensity_Data_File'].values)))
roicount.wait_interactive()
'''
path = exceldata['Intensity_Data_File'].values[0]
roilist = rois.keys()
'''

roicombine = pd.concat([exceldata[['No.', 'Sample Name', 'Intensity_Data_File']].reset_index(), pd.DataFrame(roicount.get()).reset_index()], axis=1)
writer = pd.ExcelWriter(output, engine='xlsxwriter')
roicombine.to_excel(writer, sheet_name = 'ROI_count')
#create a spreadsheet for each roi with animals that have no instances of that roi
for croi in rois.keys():
    missingroi = roicombine[roicombine[croi] ==0]
    missingroi.to_excel(writer, sheet_name = 'Missing_' + croi)


#find image files
##find timeseries image files
exceldata['Path'] = ''
exceldata['TimeSeries_Image_Files'] = ''
for row, dseries in exceldata.iterrows():
    exceldata['Path'].loc[row] = str(Path(dseries['Intensity_Data_File']).parent)
    tif, index = ccModules.getFileContString(exceldata['Path'].loc[row], '.tif')
    exceldata['TimeSeries_Image_Files'].loc[row] = tif.values
## find color images for each
files = pbDB.getDirContents(path1)
files = files[files['File_Name'].str.contains('_2color_')]
files = files[files['Directory'] == False]
files = files[~files['File_Name'].str.contains('Mask.hdf5')]
files = files[files['File_Name'].str.contains('.hdf5')]
exceldata['Two_Colored_Image'] = ''
for row, dseries in exceldata.iterrows():
    targets = files[files['File_Name'].str.contains(dseries['Sample Name'])]
    exceldata['Two_Colored_Image'].loc[row] = targets['Full_Path'].values[0]
exceldata.to_excel(writer, sheet_name= 'Summary')
writer.save()    

#generate summary figure for all tests
#generate output folder
targetfolder = 'IndividualFigures'
if not os.path.isdir(os.path.join(saveFig, targetfolder)):
    os.mkdir(os.path.join(saveFig, targetfolder))
outputfolder = os.path.join(saveFig, targetfolder)


#add genotype to genotype column
for cgene in crosses.keys():
    for row in exceldata.index:
        if cgene in exceldata['Sample Name'].loc[row]:
            exceldata['Genotype'].loc[row] = crosses[cgene]['GAL4']
#remove ones with empty genotype 

listexceldata=[exceldata.loc[row] for row in exceldata.index ]


'''
#Delete intensitydata file if does not contain specified roi
missing = pd.read_excel(output, sheet_name = 'Missing_Region1')
for row, dseries in missing.iterrows():
    Path(dseries['Intensity_Data_File']).unlink()


'''
output1 = dview.map(ccModules.makeTimeSeriesFig, exceldata.index, listexceldata, itertools.repeat(outputfolder, len(listexceldata)), itertools.repeat(rois, len(listexceldata)))
output1.wait_interactive()
output1.get()
'''
#debugging lines
i=60
seriesdata = listexceldata[i]
dataframeposition = exceldata.index[i]

for dataframeposition, seriesdata in exceldata.iterrows():
    print(dataframeposition)
    ccModules.makeTimeSeriesFig(dataframeposition,
    seriesdata, outputfolder,  rois)
'''
'''

#parallelDB.stopLocal()
# single computation methods
'''
