# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 10:29:44 2016
Script generates an excel file listing the folders in a given directory

@author: Surf32
"""

import os
import pandas as pd
import numpy as np
import shutil
import code
from datetime  import datetime

def convertList2Dict(headers, list1, dict1=None):
    #converts lists of list into a dictionary  - then can be added to dataframe
    #headers for each item in list
    #list1 is a list of lists
    if dict1 == None:
        dict1 = {}
    for i, head in enumerate(headers):
        ct = []
        for clist in list1:
            ct.append(clist[i])
        dict1[head] = ct
    return dict1

def getFolders(targetdir, outputdir=False):
#get list of folders in targetdir and returns them as a pandas dataframe and writes them to csv file
    #targetdir = 'C:\Users\Surf32\Desktop\ResearchDSKTOP\ResearchJ\A02 Align Images\Images\FruNeurons'
    #outputdir = 'C:\Users\Surf32\Desktop\ResearchDSKTOP\ResearchJ\A02 Align Images\Images\FruNeurons'
    list1 = os.listdir(targetdir)

    folderlist=[]
    dirlist=[]
    for int in list1:
        fulldir = targetdir + "/" + int
        #print fulldir
        if os.path.isdir(fulldir):
            dirlist.append(fulldir)
            folderlist.append(int)
        
    full_list=pd.DataFrame({'folderlist': folderlist, 'path': dirlist})  
    if outputdir:      
        full_list.to_csv(outputdir + '/Folderlist.csv')
        
    return full_list
    
    
def getFileList(targetdir):
    #get the list of files in the targetdir 
    list1 = os.listdir(targetdir)
    filelist=[]
    dirlist=[]
    for int in list1:
        fulldir = targetdir + "/" + int
        #print fulldir
        if os.path.isfile(fulldir):
            dirlist.append(fulldir)
            filelist.append(int)
            
    full_list=pd.DataFrame({'file': filelist, 'path': dirlist})        
    #full_list.to_csv(outputdir + '\\Folderlist.csv')
    return full_list
    
def getFileContString(targetdir, string1):
    #code.interact(local=locals())
    filelist=getFileList(targetdir)
    if len(filelist) > 0:
        indx=filelist['file'].str.contains(string1)
        filenames=filelist['file'][indx]
        indx2=np.where(indx)
    else:
        filenames = filelist['file']
        indx2 = ([],)
            
    
    return filenames, indx2
      
def moveFileContString(targetdir, string, targetfolder):
    filelist, indx = getFileContString(targetdir, string)
    if indx > 0:
        if not os.path.isdir(targetfolder):
            os.makedirs(targetfolder)
        for file1 in filelist:
            os.rename( targetdir + '/' + file1, targetfolder + '/' + file1)
            
def renameFileContString(targetdir, oldname, newname):
    #changes name in target and subdirectories
    #import pdb; pdb.set_trace()
    renamefiles=[]
    for cdir, csub, cfilelist in os.walk(targetdir): 
        filelist, indx = getFileContString(cdir, oldname)
        if len(filelist) == 1:
            os.rename(cdir + '/' +filelist.values[0], cdir + '/' +newname)
            renamefiles.append([cdir + '/' +filelist.values[0], cdir + '/' +newname])

    return renamefiles
          
def copyFileContString(targetdir, string, targetfolder):
    filelist, indx = getFileContString(targetdir, string)
    if indx > 0:
        if not os.path.isdir(targetfolder):
            os.makedirs(targetfolder)
        for file1 in filelist:
            shutil.copy( targetdir + '/' + file1, targetfolder + '/' + file1)
            
def getFolderSize(targetdir):
    return sum([os.path.getsize(os.path.join(targetdir, file1)) for file1 in os.listdir(targetdir) if os.path.isfile(os.path.join(targetdir, file1))])
        
          

def dirData(targetdir):
    #gets directory and subdirectory size, date in dataframe
    dir1 =[] 
    for cdir, csub, cfileList in os.walk(targetdir): 
        #cdir = directory
        #csub = all subdirectories in croot
        #cfile = all files in croot
        #dir1.append([cdir, format(os.path.getsize(cdir), '.2e'), datetime.fromtimestamp(os.path.getctime(cdir)).strftime('%Y-%m-%d %H: %M: %S')])
        dir1.append([cdir, getFolderSize(cdir), datetime.fromtimestamp(os.path.getctime(cdir)).strftime('%Y-%m-%d %H: %M: %S')])
    
    directoryP = pd.DataFrame(convertList2Dict(['Directory', 'Size', 'Last_Modified'], dir1))
    return directoryP
        
def fileData(targetdir):
    #gets all files (+subdirectories including size and modification date from targetdir
    file1 = []
    for cdir, csub, cfileList in os.walk(targetdir): 
        #cdir = directory
        #csub = all subdirectories in croot
        #cfile = all files in croot
            for cfile in cfileList:
                fullpath = os.path.join(cdir, cfile)
                if os.path.isfile(fullpath):
                    #file1.append([cfile, fullpath, format(os.path.getsize(fullpath), '.2e'), datetime.fromtimestamp(os.path.getctime(fullpath)).strftime('%Y-%m-%d %H: %M: %S')])
                    file1.append([cfile, fullpath, os.path.getsize(fullpath), datetime.fromtimestamp(os.path.getctime(fullpath)).strftime('%Y-%m-%d %H: %M: %S')])
        

    filesP = pd.DataFrame(convertList2Dict(['Filename', 'Directory', 'Size', 'Last_Modified'], file1))
    return filesP

def deleteFolders(targetdir, string1):
    deletedir=[]
    for cdir, csub, cfilelist in os.walk(targetdir): 
        if string1 in cdir:
            shutil.rmtree(cdir)
            deletedir.append(cdir)

    return deletedir


          
        
    