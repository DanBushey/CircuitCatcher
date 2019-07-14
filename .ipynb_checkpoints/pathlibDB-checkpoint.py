import pathlib as pl
import pandas as pd
import pathlib as pl
import pathlibDB as pldb
import os
import numpy as np

def getDirContents(path, columns = ['File_Name', 'Parent', 'Full_Path', 'Modified', 'File_Size', 'File', 'Directory']):
    if isinstance(path, str):
        path = pl.Path(path)
    all_files = []
    #for i in path.glob('**/*'): #only includes files not directories
    for i in path.glob('**/*'):
        if not i.name.startswith('.'): #exclude hidden files/folder
            try:
                all_files.append((i.name, str(i.parent), str(i.absolute()), i.lstat().st_mtime, i.stat().st_size, i.is_file(), i.is_dir()))
            except:
                all_files.append((i.name, str(i.parent), str(i.absolute()), np.nan, np.nan, np.nan, np.nan))
    df = pd.DataFrame(all_files, columns = columns)
    return df


def mkdir(directory):
    #directory = can either be a string or pathlib.PosixPath indicating full path of folder to be made
    if type(directory) == str:
        directory = pl.Path(directory)
    if not directory.is_dir():
        directory.mkdir()

def changeName(directory, string1, string2, execute = True):
    #directory = directory where names have to be changed
    #string1 = string to replace
    #string2 = string replacement
    #execute = if False then trial run
    files = pldb.getDirContents(directory)
    files['New_Name'] = ''
    for row, dseries in files.iterrows():
        files['New_Name'].loc[row] = str(pl.Path(dseries['Parent']) / dseries['File_Name'].replace(string1, string2))
    
    if execute == True:
        #need to rename files first
        filesonly = files[files['Directory'] == False]
        for row, dseries in filesonly.iterrows():
            os.rename(dseries['Full_Path'], dseries['New_Name'])
        #now change directory names
        ## directory names must be changed furthest on the file tree first
        directories =  files[files['Directory'] == True]
        #get distance in tree
        directories['Rank'] = np.zeros(len(directories))
        for row, dseries in directories.iterrows():
            directories['Rank'].loc[row] = len(pl.Path(dseries['Full_Path']).parts)
        directories = directories.sort_values(by=['Rank'], ascending = False)
        for row, dseries in directories.iterrows():
            os.rename(dseries['Full_Path'], dseries['New_Name'])
            

def makeTree(path):
    #path = folder path to be created. The top level has to exist but makes all subsequent folder if they do not already exist
    cpath = pl.Path('/')
    for cpart in pl.Path(path).parts[1:]:
        cpath = cpath / cpart
        if not cpath.is_dir():
            cpath.mkdir()
