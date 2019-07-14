'''
Project A66
Yoshi provided binary tif files representing masks covering compartments in the mushroom body.
Script combines rois from the same compartment.
It outputs binary tif images from the combined masks.
Script also produces a mask.hdf5 that can be viewed in circuit catcher.

'''
import pandas as pd
import numpy as np
from dask.array.image import imread 
import pathlibDB as pldb
import xarray as xr
import skimage
import matplotlib.pyplot as plt
from pathlib import Path
import skimage.measure
import scipy
import tifffile
import ccModules as cc
import ccModules2 as cc2

targetfolder = '/media/daniel/Windows1/Users/dnabu/Desktop/ResearchYogaWindows/DataJ/A/A66_DAN_Response/A66_Data2/MB_compartment_mask/Original'
outputfolder = '/media/daniel/Windows1/Users/dnabu/Desktop/ResearchYogaWindows/DataJ/A/A66_DAN_Response/A66_Data2/MB_compartment_mask/Combined'

files = pldb.getDirContents(targetfolder)
#files = files[files['File_Name'].str.contains('.tif')]
files        

#divide dataframe by compartment (into a dictionary)
compartments = ["a'1", "a'2", "a1", "a2", "a'3", "a3", "b1", "b'1", "b'2", "b2", "g1", "g2", "g3", "g4", "g5"]
#colors = cc.getColors(len(compartments))
#colors =[list(ccol) for ccol in colors]
#colors = dict(zip(compartments, colors))
colors = {"a'1": [0.3, 0.0, 0.16, 1.0],
          "a'2": [0.5, 0.21939586645469003, 0.0, 1.0],
          'a1': [0.8, 0.6009538950715422, 0.5, 1.0],
          'a2': [1.0, 0.9825119236883944, 0., 1.0],
          "a'3": [0.6147323794382618, 1.0, 0.0, 1.0],
          'a3': [0.23317435082140958, 1.0, 0.5, 1.0],
          'b1': [1.0, 1.0, 0.14758591608686492, 1.0],
          "b'1": [0.0, 0.5, 0.5481762597512125, 1.0],
          "b'2": [0.0, 0.0, 0.9276829011174367, 1.0],
          'b2': [0.2, 0.6894714407502134, 1.0, 1.0],
          'g1': [0.8, 0.3058397271952257, 1.0, 1.0],
          'g2': [0.09910485933503851, 0.3, 1.0, 1.0],
          'g3': [0.4827365728900258, 0.6, 1.0, 1.0],
          'g4': [0.8663682864450132, 0.8, 0.2, 1.0],
          'g5': [1.0, 0.3, 0.75, 1.0]}
compart_dict = {}
for cct in compartments:
    cfiles = files[files['File_Name'].str.contains(cct)]
    #compart.loc['comp': cc] = cfiles
    compart_dict[cct] = cfiles.to_dict()
#print(compart_dict)
    
#create a mask file that can be imported by circuit catcher into basic mask.hdf5 file layout
fileaddon = 'Mask.hdf5'
columns = ['Name', 'Color', 'Type', 'Z:XY', 'mask_index', 'image_shape','image_file']
cmaskdata = pd.DataFrame(columns = columns)

def getXYZ(index, image_shape):
    #return dictionary containing perimeter for each z layer
    #return = XYZ = {z: x,y}
    #index = ([z],[x],[y])
    #image_shape = [z x y] of original image
    
    mask = np.zeros(image_shape, dtype=np.bool)
    mask[index] = 1
    ZXY={}
    mask = np.rollaxis(mask, 2, 1)
    for z in range(0, mask.shape[0]):
        contours = skimage.measure.find_contours(mask[z], 0.8)
        if len(contours) > 0:
            index = [[i, len(con)] for i, con in enumerate(contours)] #there may be more than 2 regions so take the largest region. Circuit catcher assumes only one
            index2 = np.argmax(np.array(index)[:,1])
            ZXY[z] = contours[index2][0::10, :].tolist() #need to slice 0::4  to reduce the number of indices. Circuit catcher runs very slow with too many.
    return ZXY

for ckey in list(compart_dict.keys()):
    cfiles = pd.DataFrame(compart_dict[ckey])
    img1 = imread(cfiles['Full_Path'].iloc[0]).compute()
    for row, dseries in cfiles.iloc[1:].iterrows():
        img1 = img1 + imread(dseries['Full_Path']).compute()
    img1[img1>0] = 1
    #find how many regions (connected areas exist) because each has to be separate rows in the mask.hdf5 file
    regions =skimage.measure.label(np.squeeze(img1), neighbors=8)
    tifffile.imsave(str(Path(outputfolder) / str(ckey + '.tif')), img1)
    #each connected region is saved as a different row
    for i in np.unique(regions):
        if i != 0:
            index = np.where(regions == i)
            print(index[0].shape)
            if index[0].shape[0] >40:
                ZXY= getXYZ(index, np.squeeze(img1).shape)
                crow = pd.Series({'Name': ckey, 'Color': colors[ckey], 'Type': 'polyArea', 'mask_index': np.where(regions.flatten() == i), 'image_shape': np.squeeze(img1).shape, 'image_file': cfiles['Full_Path'].tolist(), 'Z:XY' : ZXY })
                cmaskdata = cmaskdata.append(crow, ignore_index =True)

cmaskdata.to_hdf(str(Path(outputfolder) / fileaddon), 'roi')

#read former mask file to see format
testmask = pd.read_hdf('/media/daniel/Windows1/Users/dnabu/Desktop/ResearchYogaWindows/DataJ/A/A66_DAN_Response/A66_Data2/MB_compartment_mask/Original/JFRC2013Mask.hdf5')
