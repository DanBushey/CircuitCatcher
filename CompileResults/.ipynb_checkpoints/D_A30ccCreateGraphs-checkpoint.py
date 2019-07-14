#designed to run through all rows in excel sheet and generate figures summarizing results
#there are high memory requirements when running this script so keep number of workers low

import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#folderModules = '/data/JData/A/A30FastROI/CircuitCatcher'
#sys.path.append(folderModules)
import ccModules
import scipy
import graphDB

from A55_init20180703 import *

#load pandas data frame
exceldata = pd.read_hdf(os.path.join(saveFig, 'Compiled_data.hdf5'), 'data')

#pull out intensity data so that rois have separate rows rathater than being embbedded in the dicitonary
dataframe = pd.DataFrame(columns = exceldata.columns)
for row, dseries in exceldata.iterrows():
    intensity_data=pd.DataFrame(dseries['intensity_data'])
    for rowI, dseriesIntensity in intensity_data.iterrows():
        for ccolumn in intensity_data.columns:
            dseries[ccolumn] = dseriesIntensity[ccolumn]
        #dseries.drop('intensity_data', inplace = True)
        dataframe = dataframe.append(dseries)
dataframe = dataframe.rename(columns = {'Name': 'roi'})
dataframe.drop(['intensity_data'], axis = 1, inplace=True)
exceldata = dataframe

# check columns with selection criteria to make sure there are only the expected categories
#checking that stim types are only two types
stims = exceldata['Stim Protocol'].unique()
print(stims)
#select only the timestimLong protocols
exceldata = exceldata[exceldata['Stim Protocol'] == 'timestimLong']

#divide by genotype and Ca solution
groups = exceldata.groupby(['Genotype', 'roi'])
#remove tests that graded 0
groups['Sample Name'].count().to_excel(os.path.join(saveFig, 'CountsAll.xlsx'))

# basic reiteration through groups    
for name, genotype in exceldata.groupby(['Genotype']):
    print('name-G', name)

    




#function get intensity data from pandas group
def getIntensityData(groupFrame):
    raw_intensity = np.vstack(groupFrame['intensity'].values)
    voltage = groupFrame['voltage'].iloc[0]
    timeStamp = groupFrame['timestamp'].iloc[0]
    name = []
    for row, dSeries in groupFrame.iterrows():
        name.append(str(dSeries['No.']) +'-'+ dSeries['Sample Name'])
    return raw_intensity, voltage, timeStamp, name

#used to match two dataframes for sample name and no
def matchingFrame(matchFrame, allFrame, roi):
    #matchframe = dataframe to match to
    #allFrame = entire data frame
    output = pd.DataFrame(columns=allFrame.columns)
    for row, dseries in matchFrame.iterrows():
        No = allFrame['No.'] == dseries['No.']
        Name =  allFrame['Sample Name'] == dseries['Sample Name']
        roilist = allFrame['roi'] == roi
        output = output.append( allFrame[No.values & Name.values & roilist.values])
    return output

def getSpacingSize(number, margin1, margin2, spacing):
    #get spacing for graphs
    return (1- margin1 - margin2 - spacing*(number))/(number)


def getXYSScoordinates(xnum, ynum, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, bottommargin, spacingL, spacingT):
    #returns the [x, y, xdistance, ydistance] for current axis position
    return [ leftmargin +(xnum)*(graphsizeX + spacingL), 1-((ynum+1)*(graphsizeY+spacingT))-topmargin, graphsizeX, graphsizeY]
####################################################
#relabeling by transgene
newdict = {}
for ckey in transgene.keys():
    newdict[transgene[ckey]['sensor']] = transgene[ckey]
transgene = newdict

####################################################################
#graph comparing normalized plots among all genotypes for each roi
plt.close('all')
spacingL = 0.05
spacingT =0.015
leftmargin = 0.08
topmargin = 0.02
rightmargin = 0.01
figsize = [22, 16]


#each figure will have one transgene, one stim type,  and each axis will display a different roi
for page_group_names, pageFrame in exceldata.groupby(['Genotype']):
    fig1=plt.figure(figsize=figsize) #each figures will be specific for the roi
    #determining graph placement
    n_rows =len(pageFrame['roi'].unique())  # determine the number of rows
    n_cols = 1
    graphsizeX = getSpacingSize(n_cols, leftmargin, rightmargin, spacingL)
    graphsizeY = getSpacingSize(n_rows, topmargin, topmargin, spacingT)
    #each axes will contain one type of roi
    rowG = 0
    for fig_group, figFrame in pageFrame.groupby(['roi']):
        ax1 = fig1.add_axes(getXYSScoordinates(0, rowG, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, topmargin, spacingL, spacingT))
        raw_intensity, voltage, timeStamp, name = getIntensityData(figFrame)
        norm_intensity = raw_intensity.T / np.mean(raw_intensity, axis =1)
        for row in range(norm_intensity.shape[1]):
            ax1.plot(timeStamp, norm_intensity[:,row], alpha = 0.4, linewidth = 1, color =  transgene[page_group_names]['color']) #label = name[row], 
        ax1.plot(timeStamp, np.median(norm_intensity, axis=1), color = transgene[page_group_names]['color'], linewidth = 1.5, alpha=0.8)
        yaxis = ax1.get_ylim()
            #ax1.text(0, yaxis[1], cgen + ':' + transgene[cgen], va = 'top')
        ax1.set_title(transgene[page_group_names]['sensor'] + ' ' + fig_group)
        ax11 = ax1.twinx()
        ax11.set_ylabel('LED Power (V)')
        stimTimesRange = np.array(range(len(voltage)))/100.0
        ax11.plot(stimTimesRange,  voltage, color = [0.5,0.5,0.5], linestyle = '--', alpha=0.4)
        ax1.set_xlim([0, timeStamp[-1]])
        lg = ax1.legend(bbox_to_anchor = (1.07, 1))
        ax1.set_ylabel('Normalized Intensity')
        if rowG != len(transgene[page_group_names]['rois'])-1:
            ax1.xaxis.set_visible(False)
        
        #norm_intensity = raw_intensity.T / np.mean(raw_intensity, axis =1)
        rowG = rowG +1
    if not os.path.isdir(os.path.join(saveFig, 'ComparingFullTimeLine')):
        os.mkdir(os.path.join(saveFig, 'ComparingFullTimeLine'))
    fig1.savefig(os.path.join(os.path.join(saveFig, 'ComparingFullTimeLine'), page_group_names +  '.jpeg'))
    plt.close(fig1)
            
############################################################################################################################################
#Generate graphs showing deltaF/F for each response period per transgene
#each figure will contain a single geneotype
#columns will contain stim period
#rois will contain roi
plt.close('all')
spacingL = 0.05
spacingT =0.015
leftmargin = 0.08
topmargin = 0.04
rightmargin = 0.01
figsize = [22, 16]
prestim = 5 # time in s to include before stimulation
poststim = 20 # time after first stimulation to include


rois = exceldata['roi'].unique().tolist()
rois = [roi for roi in rois if roi != 'Background']
#each figure will have one transgene, one stim type,  and each axis will display a different roi containing multiple Ca levels
for page_group_names, pageFrame in exceldata.groupby(['Genotype']):
    print('starting graph')
    fig1=plt.figure(figsize=figsize) #each figures will be specific for the roi
    #determining graph placement
    #remove background from roi list
    
    n_rows =len(rois)  # determine the number of rows
    #need to determine the number of stim periods??
    #find an example and periods from it
    voltage = exceldata['voltage'].iloc[0]
    voltageDiff = np.diff(voltage, axis = 0)
    stimPeriods = (np.where(voltageDiff > 0 )[0] +1) / 100
    n_cols =  len(stimPeriods)#number of columns
    graphsizeX = getSpacingSize(n_cols, leftmargin, rightmargin, spacingL)
    graphsizeY = getSpacingSize(n_rows, topmargin, topmargin, spacingT)
    rowG = 0 # rows are rois
    for  fig_group in rois:
        colG = 0 # columns are stim periods
        for cstim in stimPeriods:
            ax1 = fig1.add_axes(getXYSScoordinates(colG, rowG, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, topmargin, spacingL, spacingT))
            figFrame = pageFrame[pageFrame['roi'] == fig_group]
            matchBackground = matchingFrame(figFrame, exceldata, 'Background')
            background_intensity, voltage, stimesta, nameB = getIntensityData(matchBackground)
            raw_intensity, voltage, timeStamp, name = getIntensityData(figFrame)
            #remove missing data
            indx = [i for i, s in enumerate(name) if 'NaN' != s]
            background_intensity = background_intensity[indx, :]
            
            prestart = np.argmax(timeStamp > cstim-prestim)
            stop = np.argmax(timeStamp > cstim+poststim)
            start = np.argmax(timeStamp > cstim)
            xrange1 = timeStamp[prestart : stop]
            dFF = ccModules.fluorescent(raw_intensity[:, prestart : stop], background_intensity[:, prestart : stop], start-prestart, transgene[page_group_names]['response'],  xrange1)
            '''
            variables to debug ccModules.fluorescent
            data =raw_intensity[:, prestart : stop]
            background = background_intensity[:, prestart : stop]
            start = start - prestart
            response = transgene[Gen_name]['response']
            timeStamp1 = timeStamp[:, prestart : stop]
            '''
            ax1.plot(xrange1, dFF.deltaFF().T, color = transgene[page_group_names]['color'], linewidth =1, alpha = 0.3 )
            ax1.plot(xrange1, np.mean(dFF.deltaFF(), 0), color = transgene[page_group_names]['color'], linewidth =1.5, alpha = 0.8)
        
                
            #plot changes in voltage to led
            ax11 = ax1.twinx()
            
            stimTimesRange = np.array(range(len(voltage)))/100.0
            ax11.plot(stimTimesRange[int((cstim-prestim)*100) : int((cstim+poststim)*100)],  voltage[int((cstim-prestim)*100) : int((cstim+poststim)*100)], color = [0.5,0.5,0.5], linestyle = '--', alpha=0.4)
            ax11.set_ylim([0, 0.4])
            #ax1.set_xlim([0, timeStamp[-1]])
            if rowG == 0:
                ax1.set_title('Stim Time ' + str(cstim.round().astype(np.int)))
            if colG == len(stimPeriods)-1:
                #lg = ax1.legend(bbox_to_anchor = (1.0, 1))
                ax11.set_ylabel('LED Power (V)')
            if colG == 0:
                ax1.set_ylabel('Normalized Intensity' + '-' + fig_group)
            if rowG != len(rois)-1: 
                ax1.xaxis.set_visible(False)
            #if colG != len(stimPeriods)-1:
                #ax11.yaxis.set_visible(False)

            #ax1.set_ylim([0, 1])
            colG +=1
        rowG +=1
    if not os.path.isdir(os.path.join(saveFig, 'ComparingFulldeltaFF')):
        os.mkdir(os.path.join(saveFig, 'ComparingFulldeltaFF'))
    fig1.savefig(os.path.join(os.path.join(saveFig, 'ComparingFulldeltaFF'), page_group_names  + '.jpeg'))
    plt.close(fig1)

############################################################################################################################################
#comparing deltaF/F between transgenes within each axis at each stim period
#rows are specific to roi
#columns are specific stim period
#each figure will contain a single geneotype
#columns will contain stim period
#rois will contain roi
plt.close('all')
spacingL = 0.03
spacingT =0.01
leftmargin = 0.08
topmargin = 0.02
rightmargin = 0.09
figsize = [22, 16]
prestim = 5 # time in s to include before stimulation
poststim = 20 # time after first stimulation to include


rois = exceldata['roi'].unique().tolist()
rois = [roi for roi in rois if roi != 'Background']
#each figure will have one transgene, one stim type,  and each axis will display a different roi containing multiple Ca levels

print('starting graph')
fig1=plt.figure(figsize=figsize) #each figures will be specific for the roi
#determining graph placement
#remove background from roi list

n_rows =len(rois)  # determine the number of rows
#need to determine the number of stim periods??
#find an example and periods from it
voltage = exceldata['voltage'].iloc[0]
voltageDiff = np.diff(voltage, axis = 0)
stimPeriods = (np.where(voltageDiff > 0 )[0] +1) / 100
n_cols =  len(stimPeriods)#number of columns
graphsizeX = getSpacingSize(n_cols, leftmargin, rightmargin, spacingL)
graphsizeY = getSpacingSize(n_rows, topmargin, topmargin, spacingT)
rowG = 0 # rows are rois
for  row_group in rois:
    colG = 0 # columns are stim periods
    for cstim in stimPeriods:
        ax1 = fig1.add_axes(getXYSScoordinates(colG, rowG, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, topmargin, spacingL, spacingT))
        for ax_group, axFrame in exceldata.groupby(['Genotype']):
            #ax_group = 'A12-4'
            #axFrame = exceldata[exceldata['Genotype'].str.contains(ax_group)]
            rowFrame = axFrame[axFrame['roi'] == row_group]
            matchBackground = matchingFrame(rowFrame, exceldata, 'Background')
            background_intensity, voltage, stimesta, nameB = getIntensityData(matchBackground)
            raw_intensity, voltage, timeStamp, name = getIntensityData(rowFrame)
            #remove missing data
            indx = [i for i, s in enumerate(name) if 'NaN' != s]
            background_intensity = background_intensity[indx, :]
        
            prestart = np.argmax(timeStamp > cstim-prestim)
            stop = np.argmax(timeStamp > cstim+poststim)
            start = np.argmax(timeStamp > cstim)
            xrange1 = timeStamp[prestart : stop]
            dFF = ccModules.fluorescent(raw_intensity[:, prestart : stop], background_intensity[:, prestart : stop], start-prestart, transgene[ax_group]['response'],  xrange1)
            '''
            variables to debug ccModules.fluorescent
            data =raw_intensity[:, prestart : stop]
            background = background_intensity[:, prestart : stop]
            start = start - prestart
            response = transgene[Gen_name]['response']
            timeStamp1 = timeStamp[:, prestart : stop]
            '''
            #ax1.plot(xrange1, dFF.deltaFF().T, color = transgene[page_group_names]['color'], linewidth =1, alpha = 0.3 )
            ax1.plot(xrange1, np.nanmean(dFF.deltaFF(), 0), color = transgene[ax_group]['color'], linewidth =1.5, alpha = 0.8, label = transgene[ax_group]['sensor'])
        
                
        #plot changes in voltage to led
        ax11 = ax1.twinx()
        
        stimTimesRange = np.array(range(len(voltage)))/100.0
        ax11.plot(stimTimesRange[int((cstim-prestim)*100) : int((cstim+poststim)*100)],  voltage[int((cstim-prestim)*100) : int((cstim+poststim)*100)], color = [0.5,0.5,0.5], linestyle = '--', alpha=0.4)
        ax11.set_ylim([0, 0.4])
        #ax1.set_xlim([0, timeStamp[-1]])
        if rowG == 0:
            ax1.set_title('Stim Time ' + str(cstim.round().astype(np.int)))
        if colG == len(stimPeriods)-1:
            lg = ax1.legend(bbox_to_anchor = (1.7, 1))
            ax11.set_ylabel('LED Power (V)')
        if colG == 0:
            ax1.set_ylabel('Normalized Intensity' + '-' + row_group)
        if rowG != len(rois)-1:
            ax1.xaxis.set_visible(False)
        if colG != len(stimPeriods)-1:
            ax11.yaxis.set_visible(False)
        #ax1.set_ylim([0, 1])
        colG +=1
    rowG +=1
if not os.path.isdir(os.path.join(saveFig, 'ComparingFulldeltaFF')):
    os.mkdir(os.path.join(saveFig, 'ComparingFulldeltaFF'))
fig1.savefig(os.path.join(os.path.join(saveFig, 'ComparingFulldeltaFF'), 'ComparingAllTransgenes'+ '.jpeg'))
plt.close(fig1)

##################################################################################################################################################
#Generating statistics - bar graphs
#comparing each transgene  at each roi for each stim period
#columns will contain stim period
#rows will contain roi

#functionName = ['Max', 'Mean', 'Median', 'SNR', 'delay2Max']
#columns1 = ['Max_Absolute_dFF', 'Mean_dFF', 'Median_dFF', 'Signal2Noise', 'Delay_Peak_Amplitude', 'Min_Intensity_After', 'Decay_After'] # columngs for the dataframe holding data
statsDict = {'Max' : 'Max_Absolute_dFF', 
             'Mean': 'Mean_dFF',
             'Median': 'Median_dFF', 
             'SNR': 'Signal2Noise', 
             'delay2Max':'Delay_Peak_Amplitude'
             }


plt.close('all')
spacingL = 0.05
spacingT =0.015
leftmargin = 0.08
topmargin = 0.02
bottommargin = 0.1
rightmargin = 0.01
figsize = [22, 16]
prestim = 5 # time in s to include before stimulation
poststim = 20 # time after first stimulation to include
outputfolder = 'Statistics'
if not os.path.isdir(os.path.join(saveFig, outputfolder)):
    os.mkdir(os.path.join(saveFig, outputfolder))


rois = exceldata['roi'].unique().tolist()
rois = [roi for roi in rois if roi != 'Background']
#each figure will have one transgene, one stim type,  and each axis will display a different roi containing multiple Ca levels

#determining graph placement
#remove background from roi list

n_rows =len(rois)  # determine the number of rows
#need to determine the number of stim periods??
#find an example and periods from it
voltage = exceldata['voltage'].iloc[0]
voltageDiff = np.diff(voltage, axis = 0)
stimPeriods = (np.where(voltageDiff > 0 )[0] +1) / 100
n_cols =  len(stimPeriods)#number of columns
graphsizeX = getSpacingSize(n_cols, leftmargin, rightmargin, spacingL)
graphsizeY = getSpacingSize(n_rows, topmargin, bottommargin, spacingT)
    
#each figure will have different stat, rows = roi, columns = stim period

outputexcel= os.path.join(os.path.join(saveFig, outputfolder), 'Statistics_'+ '.xlsx') 
writer = pd.ExcelWriter(outputexcel , engine = 'xlsxwriter')
for cstat in statsDict.keys():
    fig1=plt.figure(figsize=figsize) #each figures will be specific for the roi
#determining graph placement
    n_rows =len(rois)  # determine the number of rows
    #need to determine the number of stim periods??
    #find an example and periods from it
    voltage = exceldata['voltage'].iloc[0]
    voltageDiff = np.diff(voltage, axis = 0)
    stimPeriods = (np.where(voltageDiff > 0 )[0] +1) / 100
    n_cols =  len(stimPeriods)#number of columns
    graphsizeX = getSpacingSize(n_cols, leftmargin, rightmargin, spacingL)
    graphsizeY = getSpacingSize(n_rows, topmargin, topmargin, spacingT)
    
    #create a pandas dataframe/multiindex to hold data
    iterables = [rois, stimPeriods.round().astype(np.int).tolist(), list(exceldata.groupby(['Genotype']).groups.keys())]
    statsFrameIndex = pd.MultiIndex.from_product(iterables, names = ['roi', 'Stim_Period', 'Genotype'])
    statsFrame = pd.DataFrame(index=statsFrameIndex, columns=[statsDict[cstat], 'Name'])
    columns2 = ['Number', 'Median', 'Mean', 'Standard Error', 'Kruskal_Wallis']
    columns2.extend(list(exceldata.groupby(['Genotype']).groups.keys()))
    outputFrame = pd.DataFrame(index=statsFrameIndex, columns=columns2)
    outputFrame['Name'] = ''
    
    rowG = 0 # rows are rois
    for row_group in rois:
        colG = 0 # columns are stim periods
        for cstim in stimPeriods:
            ax1 = fig1.add_axes(getXYSScoordinates(colG, rowG, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, bottommargin, spacingL, spacingT))

            for ax_group, axFrame in exceldata.groupby(['Genotype']):
                rowFrame = axFrame[axFrame['roi'] == row_group]
                matchBackground = matchingFrame(rowFrame, exceldata, 'Background')
                background_intensity, voltage, stimesta, name = getIntensityData(matchBackground)
                raw_intensity, voltage, timeStamp, name = getIntensityData(rowFrame)
                if len(raw_intensity.shape) != 0: #control for groups that have no observations
                    prestart = np.argmax(timeStamp > cstim-prestim)
                    stop = np.argmax(timeStamp > cstim+poststim)
                    start = np.argmax(timeStamp > cstim)
                    xrange1 = timeStamp[prestart : stop]
                    dFF = ccModules.fluorescent(raw_intensity[:, prestart : stop], background_intensity[:, prestart : stop], start-prestart, transgene[ax_group]['response'],  xrange1)
                    '''
                    variables to debug ccModules.fluorescent
                    data =raw_intensity[:, prestart : stop]
                    background = background_intensity[:, prestart : stop]
                    start = start - prestart
                    response = transgene[Gen_name]['response']
                    timeStamp1 = timeStamp[:, prestart : stop]
                    '''
                    cfun = 'dFF.' + cstat +'()'
                    data = (eval(cfun))
                    statsFrame[statsDict[cstat]].loc[row_group, cstim.round().astype(np.int), ax_group] = data
                    data = data[~np.isnan(data)]
                    outputFrame['Mean'].loc[row_group, cstim.round().astype(np.int), ax_group] = np.mean(data)
                    outputFrame['Median'].loc[row_group, cstim.round().astype(np.int), ax_group] = np.median(data)
                    outputFrame['Standard Error'].loc[row_group, cstim.round().astype(np.int), ax_group] = np.std(data) / np.sqrt(np.sum(~np.isnan(data)))
                    outputFrame['Number'].loc[row_group, cstim.round().astype(np.int), ax_group] = len(data)
                    statsFrame['Name'].loc[row_group, cstim.round().astype(np.int), ax_group] = name
                    graphDB.scatterBarPlot(transgene[ax_group]['position'], data)



         
                if colG == 0:
                    ax1.set_ylabel(statsDict[cstat] + ' ' + row_group)
                if rowG ==0:
                    ax1.set_title('Stim Time ' + str(cstim.round().astype(np.int)))
            #add kruskal wallis and signrank to dataframe
            outputFrame['Kruskal_Wallis'].loc[row_group, cstim.round().astype(np.int), ax_group] =scipy.stats.mstats.kruskalwallis(*statsFrame[statsDict[cstat]].loc[row_group, cstim.round().astype(np.int)].values).pvalue   
            for ax_group, axFrame in exceldata.groupby(['Genotype']):
                for ax_group2, axFrame2 in exceldata.groupby(['Genotype']):
                    data1 = statsFrame[statsDict[cstat]].loc[row_group, cstim.round().astype(np.int), ax_group]
                    data1 = data1[~np.isnan(data1)]
                    data2 = statsFrame[statsDict[cstat]].loc[row_group, cstim.round().astype(np.int), ax_group2]
                    data2 = data2[~np.isnan(data2)]
                    outputFrame[ax_group].loc[row_group, cstim.round().astype(np.int), ax_group2] = scipy.stats.ranksums(data1,data2).pvalue
            if rowG < len(rois)-1:
                ax1.xaxis.set_visible(False)
            if rowG == len(rois)-1:
                xlabel = np.empty(len(transgene.keys()), dtype=np.object)
                for cgene in transgene.keys():
                    xlabel[transgene[cgene]['position']-1] = transgene[cgene]['sensor']
                ax1.xaxis.set_ticklabels(xlabel, rotation = 'vertical')           
            colG +=1
        rowG +=1

    


    fig1.savefig(os.path.join(os.path.join(saveFig, outputfolder),  cstat + '.jpeg'))
    plt.close(fig1)
    outputFrame.reset_index().to_excel(writer, sheet_name = statsDict[cstat])
writer.save()

'''
Looking at a specific time point to check traces
row_group  =  'M8-10'  #roiw
cstim      =  120
ax_group   =  'A08-4'
stat       =  'Max_Absolute_dFF'
stats = pd.DataFrame(index = statsFrame['Name'].loc[row_group, cstim, ax_group], columns = [stat], data=statsFrame[stat].loc[row_group, cstim, ax_group])

'''

#############################################################################################################



