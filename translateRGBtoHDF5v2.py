'''
Creates RGB images where
red = mean from channel 1 from 2 color image taken using fastz
green = mean from channel 2 from 2 color image taken using fastz
blue = MIP from fast time series

'''

import pandas as pd
import pathlibDB as pldb
from pathlib import Path
import ipyparallel as ipp
import ccModules as cc
import pathlib
rc = ipp.Client()
dview = rc[:]

outputfolder = pathlib.Path.cwd().parents[0]
#targetfolder = '/media/daniel/Seagate Backup Plus Drive/A83'
#outputfolder = '/home/daniel/Desktop/ResearchUbuntuYoga720/A83_A6411/A83_Data2'
targetfolder = '/media/daniel/Seagate Backup Plus Drive/A83/A83_data3'

print('targetfolder: ', targetfolder)
print('outputfolder: ', outputfolder)
#outputfolder = None

#if not outputfolder is desginated targetfolder and outputfolder are the same
if outputfolder == None:
    outputfolder = targetfolder

#get all files in targetfolder
files = pldb.getDirContents(targetfolder)
print(files)
files = files[files['Directory'] == True]
files = files[files['File_Name'].str.contains('_2color_')]
print(files['File_Name'].values)

files['output_folder'] = ''
files['output'] = '' 
files['output_exists'] = ''
for row, dseries in files.iterrows():
    #row = 11867
    #dseries = files.loc[row]
    files['output_folder'].loc[row] = str(Path(outputfolder) / dseries['Parent'][len(targetfolder)+1:] /  dseries['File_Name'])
    pldb.makeTree(files['output_folder'].loc[row])  
    files['output'].loc[row] = str(Path(files['output_folder'].loc[row]) /  str(dseries['File_Name'] + '.hdf5'))
    #test if RGB file already exists
    files['output_exists'].loc[row] = Path(files['output'].loc[row]).is_file()

# get location of the timeseriesd data to include in one channel of the rgb image
files2 = pldb.getDirContents(targetfolder)
files2 = files2[files2['Directory'] == True] # searching through directories
files2 = files2[files2['File_Name'].str.contains('_stim30s06V')]
files['TimeSeries_Path'] = ''
for row, dseries in files.iterrows():
    files['TimeSeries_Path'].loc[row] = files2[files2['File_Name'].str.contains(dseries['File_Name'][:-13])]['Full_Path'].values[0]

files=files[files['output_exists'] == False]
print(files)
output1 = dview.map(cc.generateHDF_from_2colorTif2, files['output'].tolist(),  files['Full_Path'].tolist(), files['TimeSeries_Path'].tolist())
output1.wait_interactive()
print(output1.get())
'''
for row in range(len(files)):
    output1 = cc.generateHDF_from_2colorTif(files['output'].tolist()[row], files['Full_Path'].tolist()[row],  files['TimeSeries_Path'].tolist()[row], )

row=28
index = np.where(files.index == row)
row = index[0][0]
output = files['output'].tolist()[row]
path2color = files['Full_Path'].tolist()[row]
TimeSeries_Path = files['TimeSeries_Path'].tolist()[row]

def getMeanInTime_from_TIF(path, tif_files):
    #path = full file path to tif image including image
    #tif_files = [list of .tif files to be included]
    tif_files.sort()
    sample = np.squeeze(imread(str(Path(path) / tif_files[0])).compute())
    sample = sample.astype(np.float64)
    for cfile in tif_files[1:]:
        sample = sample + np.squeeze(imread(str(Path(path) / cfile)).compute())
    sample = sample / len(tif_files)
    shape = sample.shape
    red_range = np.arange(0, shape[0], 2)
    green_range = np.arange(1, shape[0], 2)
    green = normalize(sample[green_range, :, :])
    red = normalize(sample[red_range, :, :])
    rgb = np.zeros([int(shape[0] /2), shape[1], shape[2], 3], dtype = np.float64)
    rgb[:, :, :, 0] = green
    rgb[:, :, :, 1] = red
    return np.squeeze(rgb)


def generateHDF_from_2colorTif2(output, path2color, TimeSeries_Path = None):
    #output = output path for .hdf5 file
    #path2color =  path to directory containing multiple images with 2 channels at high laser power
    #TimeSeries_path = path to directorying containing times series data with single channel
    #get list of tif files
    tif_files, index  = getFileContString(path2color, '.tif')
    rgb = getMeanInTime_from_TIF(path2color, tif_files.tolist())
    trgb = np.zeros([1, rgb.shape[0], rgb.shape[1], rgb.shape[2], 3], dtype= rgb.dtype)
    trgb[0] =rgb
    #plt.imshow(skimage.img_as_ubyte(np.max(rgb, axis=0)))
    #plt.imshow(skimage.img_as_ubyte(np.max(rgb[:, :, :, 0], axis=0)))
    #plt.imshow(skimage.img_as_ubyte(np.max(rgb[:, :, :, 1], axis=0)))
    #get timeseries data
    if os.path.isdir(TimeSeries_Path):
        tif_files, index = getFileContString(TimeSeries_Path, '.tif')
        blue = getMaxInTime_from_TIF(TimeSeries_Path, tif_files.tolist())
        blue = normalize(blue)
    #for layer in green_range
    if 'blue' in locals():
        trgb[:, :, :, :, 2] = blue[:, :, :]
    
    trgb[trgb > 1] = 1
    trgb = np.rollaxis(trgb, 3, 2)
    #plt.imshow(skimage.img_as_ubyte(np.squeeze(np.max(trgb, axis=1))))
    trgb = skimage.img_as_int(trgb)
    hdf5_file = tables.open_file(output, mode='w')
    #filters = tables.Filters(complevel=5, complib='blosc') #faster but poor compression
    filters = tables.Filters(complevel=3, complib='zlib') #slower but better compression
    data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                  tables.Atom.from_dtype(rgb.dtype),
                                  shape = trgb.shape,
                                  filters=filters)
    data_storage[:] = trgb
    data_storage.flush()
    hdf5_file.close()
'''