import tables
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import pathlibDB
import pathlib
import os
import tifffile
import matplotlib

#find 'intensityData files'
targetdir = '/home/daniel/Desktop/ResearchUbuntuYoga720/A58_MeanGirl/nSyb'
dataframe = pathlibDB.getDirContents(targetdir)
dataframe.head()
dataframe = dataframe[dataframe['File_Name'].str.contains('IntensityData.hdf5')]
dataframe




#change dataframe index to match groupFrame columns
newindex = []
for row, sampleFrame in dataframe.iterrows():
    newindex.append(sampleFrame['File_Name'][:-19])
print(newindex)
newindex = dict(zip(dataframe.index, newindex))
dataframe = dataframe.rename(newindex)

counts = pd.DataFrame(index = dataframe.index, columns=['ROI_Count'])
for rowD, sampleFrame in dataframe.iterrows():
    print(rowD)
    intensity_data = pd.read_hdf(sampleFrame['Full_Path'],'intensity_data')
    counts['ROI_Count'].loc[rowD] = len(intensity_data['Name'].unique())
print(counts)    
outputfile = '/home/daniel/Desktop/ResearchUbuntuYoga720/A58_MeanGirl/nSyb/counts.xlsx'
counts.to_excel(outputfile)
    
    
    
    