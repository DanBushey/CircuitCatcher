'''
write file IntensityData.hdf5 containing the intensity data for rois
Project A57

'''
import os
import numpy as np
import pandas as pd
import ccModules
import ccModules2 as cc2
import ipyparallel as ipp
import pathlibDB as pbDB
import pathlib as pb
import matplotlib.pyplot as plt
import skimage
import scipy
from dask.array.image import imread


#search directory tree for all files containing numpy
#targetfolder = r'/media/daniel/Windows1/Users/dnabu/Desktop/ResearchYogaWindows/DataJ/A/A57_GtACR/A57_Data'
targetfolder = r'/data/JData/A/A57_GtACR_Mi1/A57_Data'

#get full directory
files = pbDB.getDirContents(targetfolder)
maskfiles = files[files['File_Name'].str.contains('Mask.hdf5')]
print(maskfiles)
maskfiles = maskfiles[maskfiles['Full_Path'].str.contains('Scope1')]

#search for corresponding Image files
maskfiles['Image_files'] = ''
for row, dseries in maskfiles.iterrows():
    HDF5files, index = ccModules.getFileContString(dseries['Parent'], '.tif')
    if len(HDF5files) > 0:
        maskfiles['Image_files'].loc[row] = str(pb.Path(dseries['Parent']) / HDF5files.values[0])
print(maskfiles['Image_files'].values)

#generate paths to outputfiles
maskfiles['Output_files'] = ''
for row, dseries in maskfiles.iterrows():
    name = str(pb.Path(dseries['Parent']).parts[-1]) + '_IntensityData.hdf5'
    pathname = str(pb.Path(dseries['Parent']) / name)
    maskfiles['Output_files'].loc[row] = pathname

maskfiles.to_csv(str(pb.Path(targetfolder) / 'generateIntensityDataFiles.csv'))

#delete rows with empty cells
maskfiles.replace('', np.nan, inplace = True)
maskfiles.dropna(inplace = True)

#generate figures
rois1 = {'M8-10': [0.3, 0.15, 0.5] , 'M4': [0.6, 0.2, 0.5],'M1' : [0.9, 0.5, 0], 'Background' : [0.9, 0.9, 0.5], 'Body': [1, 0, 0]}
rois = {'M8-10': [0.3, 0.15, 0.5] , 'M4': [0.6, 0.2, 0.5],'M1' : [0.9, 0.5, 0], 'Body': [1, 0, 0]}
summaryfolder = '/data/JData/A/A57_GtACR_Mi1/A57_Data/Scope1Summary'
#cut off the first 5 seconds because there is an artifact at the beginning of each run
cutoff = 10

for row, dseries in maskfiles.iterrows():
    print(row)
    
    fig1=plt.figure(figsize=(10,8))
    ## generate title for page
    ax1 = fig1.add_axes([0.01, 0.95, 0.9, 0.5])
    title2 =  dseries['File_Name']
    ax1.text(0,0, title2)
    ax1.axis('off') 
    #get voltage and timestamp
    hdf5 = cc2.readIntensityData(dseries['Output_files'])
    voltage = hdf5.getVoltage()[:]
    timeStamp = hdf5.getTimeStamp()[:]
    intensity_data = hdf5.getROIdata()
    #img_files = pd.read_hdf(seriesdata['Intensity_Data_File'], 'image_files')
    #img_files = pd.read_hdf(seriesdata['Intensity_Data_File'], 'image_files')
    
    ## generate a series of axes with the signal from each roi
    grouped = intensity_data.groupby(['Name'], axis=0).groups 
    #determine how many and size of each axis
    numRois = len(grouped)
    ysize = 0.6 / len(rois.keys())
    #get background intensity
    back_index = intensity_data.index[intensity_data['Name'] == 'Background']
    back_intensity = np.mean(np.vstack(intensity_data['intensity'].loc[back_index].values), axis =0)
    #plot the background
    ax2=fig1.add_axes([0.1, 0.1, 0.5, 0.2])
    croi = 'Background'
    ax2.plot(timeStamp[cutoff:], back_intensity[cutoff:], color = rois1[croi])
    ax2.set_ylabel('Raw Intensity ' + croi)
    ax21 = ax2.twinx()
    ax21.set_ylabel('LED Power (V)')
    stimTimesRange = np.array(range(len(voltage)))/100.0
    ax21.plot(stimTimesRange,  voltage, color = 'r', linestyle = '--', alpha=0.4)
    ax2.set_xlabel('Time (s)')
    
    #add axis and plot
    for i, croi in enumerate(rois.keys()):
        index = intensity_data.index[intensity_data['Name'] == croi]
        ax2=fig1.add_axes([0.1, 0.3+(i)*ysize, 0.5, ysize])
        if len(index) > 0:
            cintensity = np.mean(np.vstack(intensity_data['intensity'].loc[index].values), axis =0)
            cintensity =cintensity - back_intensity
            ax2.plot(timeStamp[cutoff:], cintensity[cutoff:], color = rois[croi])
            #remove axis if no the first
            if i !=0:
                ax2.axes.get_xaxis().set_visible(False)
            ax2.set_ylabel('Raw Intensity ' + croi)
            ax21 = ax2.twinx()
            ax21.set_ylabel('LED Power (V)')
            stimTimesRange = np.array(range(len(voltage)))/100.0
            ax21.plot(stimTimesRange,  voltage, color = 'r', linestyle = '--', alpha=0.4)
            #ax2.set_ylim(np.min(seriesdata['RawTimeSeries'][seriesdata['RawTimeSeries'] != 0]), np.max(seriesdata['RawTimeSeries']))
            #ax2.set_xlim(0, 205)
            #if i == 0:
            #    ax2.set_xlabel('Time (s)')
            #ax2.set_title('Raw Intensity Traces')
    
    # show standard deviation image 
    ## get image standard deviation and MIP
    ## need to load either 
    image = imread(dseries['Image_files']).squeeze()
    stdImg = np.std(image, axis =0).squeeze()
    #stdImg = stdImg.T
    ax3 = fig1.add_axes([0.63, 0.5, 0.4, 0.4])
    ax3.imshow(stdImg, cmap='Greys_r')
    ax3.set_aspect('equal')
    ax3.axis('off') 
    ax3.set_title('Standard Deviation over time')
    
    #Generate a MIP image
    MIP= np.max(image, axis =0).squeeze()
    #MIP= MIP.T
    ax4 = fig1.add_axes([0.63, 0.05, 0.4, 0.4])
    ax4.imshow(MIP, cmap='Greys_r')
    ax4.set_aspect('equal')
    ax4.axis('off') 
    ax4.set_title('MIP over time')
 
    #outline rois
    for roi in intensity_data.index:
        #roi=intensity_data.index[7]
        mask1 = np.zeros(intensity_data['image_shape'].loc[roi][1:-1]).flatten()
        if len(intensity_data['mask_index'].loc[roi][0] ) >0 : #roi must contain more than one pixel
            mask1[intensity_data['mask_index'].loc[roi] ]=1
            #mask1 = np.flipud(mask1)
            mask1 = mask1.reshape(intensity_data['image_shape'].loc[roi][1:-1])
            mask1 = np.sum(mask1, axis = 0)
            mask1[mask1 > 0 ] = 1
            contours = skimage.measure.find_contours(mask1, 0.8)
            if intensity_data['Name'].loc[roi] in rois.keys():
                color1 = rois[intensity_data['Name'].loc[roi]]
            else:
                color1 = [1, 0, 1]
            for n, contour in enumerate(contours):
                ax3.plot(contour[:, 1], contour[:, 0], linewidth = 2, color = color1)
                ax4.plot(contour[:, 1], contour[:, 0], linewidth = 2, color = color1)
            lbl = scipy.ndimage.label(mask1)
            indexC = scipy.ndimage.center_of_mass(mask1)
            ax3.text(indexC[1], indexC[0], intensity_data['Name'].loc[roi], color=color1)
            ax4.text(indexC[1], indexC[0], intensity_data['Name'].loc[roi], color=color1)
 
    #save figure
    fig1.savefig(os.path.join(dseries['Parent'], dseries['File_Name'][:-9] + '.jpeg'))
    fig1.savefig(os.path.join(summaryfolder, dseries['File_Name'][:-9] + '.jpeg'))
    plt.close(fig1)
