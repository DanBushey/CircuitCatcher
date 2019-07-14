#designed to run through all rows in excel sheet and generate figures summarizing results
#there are high memory requirements when running this script so keep number of workers low

import pandas as pd
import os
import sys
import itertools
#folderModules = '/data/JData/A/A30FastROI/CircuitCatcher'
#sys.path.append(folderModules)
import ccModules
import parallelDB
import ipyparallel as ipp
#parallelDB.startLocalWait(7, 20) #(=number of engines, #number of attempts to connect to engines)
#rc = ipp.Client()
#dview = rc[:]
import sys
sys.path.append(r'/media/daniel/Seagate Backup Plus Drive2/JData/A/A12_19F01-Gal4_CaMPARI/A12 Data')
#sys.path.append(r'/media/daniel/Windows/Users/dnabu/Desktop/ResearchYogaWindows/DataJ/Programming/Python/Modules')


ReadFile = '/media/daniel/Seagate Backup Plus Drive2/JData/A/A12_19F01-Gal4_CaMPARI/A12 Data/A12_Summary.xlsx'
WriteFile = '/media/daniel/Seagate Backup Plus Drive2/JData/A/A12_19F01-Gal4_CaMPARI/A12 Data/scoreROIlist.xlsx'
path1 = '/media/daniel/Seagate Backup Plus Drive2/JData/A/A12_19F01-Gal4_CaMPARI/A12 Data'

selectioncriteria = {
    'Beam strength' : 7,
    'Lamina Present' : 'yes',
    'Saline' : 'regular',
    }

#excel file to write summary data
writer = pd.ExcelWriter(WriteFile, engine='xlsxwriter')

xl = pd.ExcelFile(ReadFile)
print(xl.sheet_names)
exceldata=xl.parse(xl.sheet_names[0])
for sheet in xl.sheet_names[1:]:
    if not 'Protocol' in sheet:
        exceldata = exceldata.append(xl.parse(sheet, encoding='utf-8'))
exceldata.to_excel(writer, sheet_name = 'Summary', encoding='utf-8')
writer.save()
#exceldata.to_csv(outputcsv, encoding='utf-8', sep = ",")
#change the index so there are no repeatd numbers
newindex = range(0, len(exceldata))
exceldata.index = newindex

#remove rows that do not meat selection criteria
for cselect in selectioncriteria.keys():
    exceldata = exceldata[exceldata[cselect] == selectioncriteria[cselect]]

#generate paths to all acquisitions
paths = []
notdir = []
for row in exceldata.index:
    filename1 = exceldata['Sample Name'].loc[row] + '_' + '%05d' % exceldata['No.'].loc[row]
    #path2 = os.path.join(os.path.split(ExcelFile)[0], str(exceldata['Date'].loc[row])[:4] + '-' + str(exceldata['Date'].loc[row])[4:6] + '-' + str(exceldata['Date'].loc[row])[6:8])
    path2 = os.path.join(path1, str(exceldata['Date'].loc[row] ))
    paths.append(os.path.join(path2, filename1))
    notdir.append(os.path.isdir(os.path.join(path2, filename1)))

exceldata['Paths']  = paths
exceldata['Directory?'] = notdir
#exceldata.to_csv(outputcsv, encoding='utf-8', sep=",")

#find the image
intensitydatafile =[]
for row in exceldata.index:
    if os.path.isdir(exceldata['Paths'].loc[row]):
        files, indx = ccModules.getFileContString(exceldata['Paths'].loc[row], 'STDEV.hdf5')
        if len(indx[0]) == 1:
            intensitydatafile.append(os.path.join(exceldata['Paths'].loc[row], files.values[0]))
        else:
            intensitydatafile.append('No intensity data file')
        #search for the file containing image data
        
    else:
         intensitydatafile.append('Directory does not exist')
exceldata['Image_File'] = intensitydatafile
writer = pd.ExcelWriter(WriteFile, engine='xlsxwriter')
exceldata = exceldata[exceldata['No.'] == 1]
exceldata.to_excel(writer, sheet_name = 'Summary')
writer.save()
#remove images done in succession that can use the same mask

#remove any rows that have Intensity_Data_File
exceldata = exceldata[exceldata['Image_File'].str.contains('No intensity data file') == False]

