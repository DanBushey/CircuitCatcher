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

#need to find the registered mask file
dataframe['Registered Mask File'] =''
dataframe2 = pathlibDB.getDirContents(targetdir)
searchstrings = ['Registration5_', 'baselineMask.hdf5']
for row, sampleFrame in dataframe.iterrows():
    #find corresponding files 
    dataframe3 = dataframe2[dataframe2['File_Name'].str.contains(sampleFrame['File_Name'][:-20])]
    #search files for specific string
    for cs in searchstrings:
        dataframe3 = dataframe3[dataframe3['Full_Path'].str.contains(cs)]
    dataframe['Registered Mask File'].loc[row] = dataframe3['Full_Path'].values[0]
print(dataframe['Registered Mask File'])


#rename the roi names and give them distinct colors

def getColors(num):
    #get num of distinct colors
    cm = plt.get_cmap('gist_rainbow')
    cNorm = matplotlib.colors.Normalize(vmin = 0, vmax=num-1)
    scalarMap = matplotlib.cm.ScalarMappable(norm = cNorm, cmap = cm)
    return [scalarMap.to_rgba(i) for i in range(num)]

# make outputdirectories
dataframe['Output'] = ''
for row, sampleFrame in dataframe.iterrows():   
    outputfolder = pathlib.Path(sampleFrame['Registered Mask File'])
    outputfolder = outputfolder.parent / 'output'
    if not outputfolder.is_dir():
        outputfolder.mkdir()
    dataframe['Output'].loc[row] = str(outputfolder)

#get the image file
dataframe['Image_File'] =''
dataframe2 = pathlibDB.getDirContents(targetdir)
searchstrings = ['Registration5_', 'baseline.tif']
for row, sampleFrame in dataframe.iterrows():
    #find corresponding files 
    dataframe3 = dataframe2[dataframe2['File_Name'].str.contains(sampleFrame['File_Name'][:-20])]
    #search files for specific string
    for cs in searchstrings:
        dataframe3 = dataframe3[dataframe3['Full_Path'].str.contains(cs)]
    dataframe['Image_File'].loc[row] = dataframe3['Full_Path'].values[0]
print(dataframe['Image_File'].values)

#find the corresponding numbers for each region
regions  = ['Dorsal', 'Lateral', 'Medial']
for r in regions:
    dataframe[r] = ''
for row, sampleFrame in dataframe.iterrows():
    #find corresponding files 
    basedir = pathlib.Path(sampleFrame['Full_Path']).parent
    basedir = basedir / 'output'
    for r in regions:
        cbasdir = basedir / r
        numbers  = []
        for i in cbasdir.iterdir():
            numbers.append(i.parts[-1][:-5])
        dataframe[r].loc[row] = numbers
print(dataframe[r].values)
    


#write figures for each region into the output folder
import numpy as np
def getSummaryImage(img_file):
    #img_files = pandas dataframe with row = individual images and columns name = where images can be found
    #need to determine whether the files are hdf5 or tif
        
    #sometimes path file name change so need to search again for #####.hdf5 file
    timeseries = tifffile.imread(img_file)
    MIP = timeseries.max(axis=0)
    return MIP #STD.compute()

def concatenateIndex(index1):
    #contenate the index from multiple intensity_data files
    index2 = []
    for i in index1:
        #print(i[0].shape)
        index2.append(i[0])
    return np.concatenate(index2)
        
        
import ccModules
import skimage
row = dataframe.index[2]
sampleFrame = dataframe.loc[row]
for rowSample, sampleFrame in dataframe.iterrows():
    print('start', sampleFrame['Full_Path'])
    intensity_data = pd.read_hdf(sampleFrame['Full_Path'],'intensity_data')
    rois = pd.read_hdf(sampleFrame['Registered Mask File'],'roi')
    img_file = sampleFrame['Image_File']
    MIP = tifffile.imread(sampleFrame['Image_File']).max(axis = 0).squeeze()
    hdf5 = tables.open_file(sampleFrame['Full_Path'])
    voltage = hdf5.root.voltage[:]
    timeStamp = hdf5.root.timeStamp[:]
    hdf5.close()
    #creating a figure for each region
    for cr in regions:
        ## add rois from the same region within the axis
        names = sampleFrame[cr]
        if len(names) > 0:
            colors = getColors(len(names))
            colors = dict(zip(names, colors))
            fig1=plt.figure(figsize=(10,8))
            ax1 = fig1.add_axes([0.01, 0.95, 0.9, 0.5])
            title2 = cr + '-' + sampleFrame['File_Name'][:-19]
            ax1.text(0,0, title2)
            ax1.axis('off')
            ax2=fig1.add_axes([0.1, 0.1, 0.8, 0.4])
            for name in names:
                grouped = intensity_data['Name'] == name
                cintensitydata = intensity_data['intensity'].loc[grouped].values
                cintensitydata = np.concatenate(cintensitydata)
                ax2.plot(timeStamp, np.mean(cintensitydata, axis=0), color = colors[name], label = str(int(name)))
            ax2.set_ylabel('Raw Intensity ')
            ax21 = ax2.twinx()
            ax21.set_ylabel('LED Power (V)')
            stimTimesRange = np.array(range(len(voltage)))/100.0
            ax21.plot(stimTimesRange,  voltage, color = 'r', linestyle = '--', alpha=0.4)
            ax2.set_xlabel('Time (s)')
            #ax2.set_title('Raw Intensity Traces')
    
            # show maximum intensity projection 
            ax3 = fig1.add_axes([0.1, 0.55, 0.4, 0.4])
            ax3.imshow(MIP, cmap='Greys_r')
            ax3.set_aspect('equal')
            ax3.axis('off') 
            ax3.set_title('MIP over time')
        
            
            for name in names:
                grouped = rois['Name']==name
                mask1 = np.zeros(rois['image_shape'].loc[grouped].values[0][1:-1])
                if len(rois['mask_index'].loc[grouped].values[0] ) >0 : #roi must contain more than one pixel
                    index = rois['mask_index'].loc[grouped].values
                    index = concatenateIndex(index)
                    mask1.flat[index]=1
                    mask1 = np.sum(mask1, axis = 0)
                    mask1[mask1 > 0 ] = 1
                    contours = skimage.measure.find_contours(mask1, 0.8)
                    for n, contour in enumerate(contours):
                        ax3.plot(contour[:, 1], contour[:, 0], linewidth = 2, color = colors[name], label = str(int(name)))
                #ax4.plot(contour[:, 1], contour[:, 0], linewidth = 2, color = color1)
            #lbl = scipy.ndimage.label(mask1)
            #indexC = scipy.ndimage.center_of_mass(mask1)
            #ax3.text(indexC[1], indexC[0], intensity_data['Name'].loc[roi], color=intensity_data['Color'].loc[grouped[cg]].values[0])
            #ax4.text(indexC[1], indexC[0], intensity_data['Name'].loc[roi], color=intensity_data['Color'].loc[grouped[cg]].values[0])
    
            #plt.show()
            ax2.legend()
            name = cr + '.jpeg'
            print(sampleFrame['Output'])
            fig1.savefig(pathlib.Path(sampleFrame['Output']) / name , dpi =300)
            plt.close(fig1)
    
    

