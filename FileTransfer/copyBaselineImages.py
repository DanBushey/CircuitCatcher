#transfering files whose name contains baseline to laptop
import pandas as pd
import pathlib as pl
import pathlibDB as pDB
import shutil
import osDB
import numpy as np

#get full directory
targetdir = '/run/user/1000/gvfs/smb-share:server=busheyd-ws1,share=dr1tba/JData/A/A55_NewCaMPARI/A55_Data'
outputdir = pl.Path('/media/daniel/Windows1/Users/dnabu/Desktop/ResearchYogaWindows/DataJ/A/A55_NewCampari/A55_data/ForAnalysis')


files = pDB.getDirContents(targetdir)

#get files with baseline in name
files1 = files[files['File_Name'].str.contains('STDEV.hdf5')]

#remove files that already have a mask file_to_laptop
files1['Mask_file'] = ''
for row, drow in files1.iterrows():
    cfile, index = osDB.getFileContString(drow['Parent'], 'Mask.hdf5')
    if len(cfile) > 0:
        files1['Mask_file'].loc[row] = str(pl.Path(drow['Parent']) / cfile.values[0])
    else:
        files1['Mask_file'].loc[row] = np.nan

files1.to_excel(str(outputdir / 'file_to_laptop.xlsx'))
#location on laptop


#remove rows that have a mask file_to_laptop
files1 = files1[files1['Mask_file'].isnull()]

#generate copy paths
files1['outputdir'] = ''
files1['outputfile'] = ''

for row, drow in files1.iterrows():
    files1['outputdir'].loc[row] = str(outputdir / drow['File_Name'][:-4])
    files1['outputfile'].loc[row] = str(outputdir / drow['File_Name'][:-4] / drow['File_Name'])
files1.to_excel(str(outputdir / 'file_to_laptop.xlsx'))
print(files1)
#copy files
for row, drow in files1.iterrows():
    #make directory
    cdir = pl.Path(drow['outputdir'])
    if not cdir.is_dir():
        cdir.mkdir()

    shutil.copy2(drow['Full_Path'], drow['outputfile'])
        

#output an excelfile containing a list of images for circuitcatcher
imageexcel = pd.DataFrame(data = files1['outputfile'].tolist(), columns=['Image_File'])
imageexcel.to_excel(str(outputdir / 'Image_file_list.xlsx'))
