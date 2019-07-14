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

files = files[files['File_Name'].str.contains('_2colstack_') & files['File_Name'].str.contains('.tif')]
print(files['File_Name'].values)

files['output_folder'] = ''
files['output'] = '' 
files['output_exists'] = ''
for row, dseries in files.iterrows():
    #row = 0
    #dseries = files.iloc[row]
    files['output_folder'].loc[row] = str(Path(outputfolder) / dseries['Parent'][len(targetfolder)+1:] /  dseries['File_Name'][:-4])
    pldb.makeTree(files['output_folder'].loc[row])  
    files['output'].loc[row] = str(Path(files['output_folder'].loc[row]) /  str(dseries['File_Name'][:-3] + 'hdf5'))
    #test if RGB file already exists
    files['output_exists'].loc[row] = Path(files['output'].loc[row]).is_dir()

# get location of the timeseriesd data to include in one channel of the rgb image
files2 = pldb.getDirContents(targetfolder)
files2 = files2[files2['Directory'] == True] # searching through directories
files2 = files2[files2['File_Name'].str.contains('_stim30s06V_')]
files['TimeSeries_Path'] = ''
for row, dseries in files.iterrows():
    files['TimeSeries_Path'].loc[row] = files2[files2['File_Name'].str.contains(dseries['File_Name'][:-22])]['Full_Path'].values[0]

files=files[files['output_exists'] == False]
output1 = dview.map(cc.generateHDF_from_2colorTif, files['output'].tolist(),  files['Full_Path'].tolist(), files['TimeSeries_Path'].tolist())
output1.wait_interactive()
print(output1.get())
'''
for row in range(len(files)):
    output1 = cc.generateHDF_from_2colorTif(files['output'].tolist()[row], files['Full_Path'].tolist()[row],  files['TimeSeries_Path'].tolist()[row], )

row =2
output = files['output'].tolist()[row]
RGB_image =  files['Full_Path'].tolist()[row]
TimeSeries_Path = files['TimeSeries_Path'].tolist()[row]
    
def getMaxInTime_from_TIF(path, tif_files):
    #path = full file path to tif image including image
    #tif_files = [list of .tif files to be included]
    tif_files.sort()
    sample = np.squeeze(imread(str(Path(path) / tif_files[0])).compute())
    img = np.zeros([2, sample.shape[0], sample.shape[1],sample.shape[2]], dtype=sample.dtype)
    img[0] = sample
    for cfile in tif_files[1:]:
        sample = np.squeeze(imread(str(Path(path) / cfile)).compute())
        img[1] = sample
        img[0] = np.max(img, axis =0)
    return np.squeeze(img[0])    
    
    
    
def generateHDF_from_2colorTif(output, RGB_image, TimeSeries_Path = None):
    #RGB_image = full file path to tif image including image
    #output = output path for .hdf5 file
    #if os.path.isfile(RGB_image):
    #get image data from colored image - high laser power taken after timeseries
    img = imread(RGB_image).compute()
    #create an rgb image
    shape = img.shape
    rgb = np.zeros([1, int(shape[1] /2), shape[2], shape[3], 3], dtype = np.float64)
    red_range = np.arange(0, shape[1], 2)
    green_range = np.arange(1, shape[1], 2)
    green = normalize(img[:, green_range, :, :])
    red = normalize(img[:, red_range, :, :])
    #get timeseries data
    if os.path.isdir(TimeSeries_Path):
        tif_files, index = getFileContString(TimeSeries_Path, '.tif')
        blue = getMaxInTime_from_TIF(TimeSeries_Path, tif_files.tolist())
        blue = normalize(blue)
    
    for layer in range(blue.shape[0]):
        print(np.max(blue[layer,:, :])
        
    #for layer in green_range
    if 'green' in locals():
        rgb[:, :, :, :, 0] = green
        rgb[:, :, :, :, 1] = red
    if 'blue' in locals():
        rgb[:, :, :, :, 2] = blue[:42, :, :]
    
    rgb[rgb > 1] = 1
    rgb = np.rollaxis(rgb, 3, 2)
    rgb = skimage.img_as_int(rgb)
    hdf5_file = tables.open_file(output, mode='w')
    #filters = tables.Filters(complevel=5, complib='blosc') #faster but poor compression
    filters = tables.Filters(complevel=3, complib='zlib') #slower but better compression
    data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                  tables.Atom.from_dtype(rgb.dtype),
                                  shape = rgb.shape,
                                  filters=filters)
    data_storage[:] = rgb
    data_storage.flush()
    hdf5_file.close()
