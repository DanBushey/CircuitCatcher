#designed to run through all rows in excel sheet and generate figures summarizing results
#there are high memory requirements when running this script so keep number of workers low

import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#folderModules = '/data/JData/A/A30FastROI/CircuitCatcher'
#sys.path.append(folderModules)
import ccModules
import ccModules2 as cc2
import scipy
import graphDB
import pathlib


mystyle = '/media/daniel/Windows1/Users/dnabu/Desktop/ResearchYogaWindows/DataJ/Programming/Python/Modules/matplotlib_style_template'
plt.style.use(mystyle)


from A55_init20180827 import *

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
#use only one stim type
exceldata = exceldata[exceldata['Stim Protocol'] == 'voltstim']
print(exceldata['Stim Protocol'].unique())
#exceldata['Stim Protocol'].replace('timestime', 'timestim', inplace = True)

#divide by genotype and Ca solution
groups = exceldata.groupby(['Genotype', 'roi'])
#remove tests that graded 0
groups[['Sample Name', 'Cross']].count().to_excel(os.path.join(saveFig, 'CountsAll.xlsx'))

# basic reiteration through groups    
for name, genotype in exceldata.groupby(['Genotype']):
    print('name-G', name)
exceldata['Grade'].unique()
exceldata = exceldata[exceldata['Grade'] ==1]
print(exceldata['Grade'].unique())
groups = exceldata.groupby(['Genotype', 'Cross', 'roi'])
groups[['Sample Name']].count().to_excel(os.path.join(saveFig, 'CountsAll.xlsx'))



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
###################################################################
#get plot lines for voltage
def getVoltageStopStarts(voltage):
    voltageDiff = np.diff(voltage, axis = 0)
    stimStarts = (np.where(voltageDiff > 0 )[0] +1) / 100
    stimStops = (np.where(voltageDiff < 0 )[0] ) / 100
    return stimStarts, stimStops

#relabeling by transgene
newdict = {}
for ckey in transgene.keys():
    newdict[transgene[ckey]['sensor']] = transgene[ckey]
transgene = newdict

#drop transgenes that are not found in the data
for ckey in transgene.keys():
    if ckey not in exceldata['Genotype'].unique():
        newdict.pop(ckey, None)
print(len(newdict.keys()))
print(len(exceldata['Genotype'].unique()))
transgene = newdict      

####################################################################
#graph comparing normalized plots among all genotypes for each roi
#page contains all genotypes for a single roi
#
plt.close('all')
spacingL = 0.05
spacingT =0.015
leftmargin = 0.08
topmargin = 0.02
rightmargin = 0.01
figsize = [10,8]

targetfolder = os.path.join(saveFig, 'ComparingFullTimeLine')
if not os.path.isdir(targetfolder):
    os.mkdir(targetfolder)

page_group_names = 'VS5_Trunk' #target roilist
for cgen in transgene.keys():
    pageFrame = exceldata[exceldata['roi'] == page_group_names]
    fig1=plt.figure(figsize=figsize) #each figures will be specific for the roi
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    fig_group = cgen

    figFrame = pageFrame[pageFrame['Genotype'] == fig_group]
    if len(figFrame) > 0:
        raw_intensity, voltage, timeStamp, name = getIntensityData(figFrame)
        norm_intensity = raw_intensity.T / np.mean(raw_intensity, axis =1)

        #draw in stim periods
        #this method draws full lines
        stimStarts, stimStops = getVoltageStopStarts(voltage)
        ax2 = ax1.twinx()
        for int1 in zip(stimStarts, stimStops):
            #ax1.axvspan(int1[0], int1[1], color = 'r', alpha=0.2)
            #ax1.plot(int1, [1, 1,], color = 'r', linewidth = 4, alpha =0.5)
            #ax1.plot(int1, [1, 1], color = 'r', linewidth = 4, alpha = 0.5)
            #this method draws in lines corresponding
            ax2.axvspan(int1[0], int1[1], 0, voltage[int(int1[0]*100)], color ='r', alpha=0.2)

            
        for row in range(norm_intensity.shape[1]):
            ax1.plot(timeStamp, norm_intensity[:, row], color = (0.5, 0.5, 0.5, 0.5))     
        #graphDB.lineSEM(timeStamp, norm_intensity.tolist(), transgene[fig_group]['color'], ax1)
        ax1.plot(timeStamp, np.mean(norm_intensity, axis=1), color = transgene[fig_group]['color'], linewidth=2)
    

        yaxis = ax1.get_ylim()
        #ax1.text(0, yaxis[1], cgen + ':' + transgene[cgen], va = 'top')
        #ax1.set_title(transgene[fig_group]['sensor'])
        #ax11 = ax1.twinx()
        #ax11.set_ylabel('LED Power (V)')
        #stimTimesRange = np.array(range(len(voltage)))/100.0
        ylim = ax1.get_ylim()
        ax1.set_ylim(ylim)

        ax1.set_xlim([0, timeStamp[-1]])
        #ylim = ax1.get_ylim()
        #ax1.set_ylim([0 , ylim[1]])
        #lg = ax1.legend(bbox_to_anchor = (1.07, 1))
        #ax1.set_ylabel('Normalized Intensity')
        #if rowG != len(rois)-1:
        #ax1.xaxis.set_visible(False)
        #ax1.set_xlim([0, 120])

        #norm_intensity = raw_intensity.T / np.mean(raw_intensity, axis =1)
        #rowG = rowG +1
        #fig1.show()
        if not os.path.isdir(os.path.join(saveFig, targetfolder)):
            os.mkdir(os.path.join(saveFig, targetfolder))
        fig1.savefig(os.path.join(os.path.join(saveFig, targetfolder), cgen + page_group_names + '_rawtrace.jpeg'))
        plt.close(fig1)

############################################################################################################################################
#comparing deltaF/F between transgenes within each axis at each stim period
#each figure contains on axis
#each figure will contain a multiple geneotype
#output is multiple graphs specific to roi, stim time
plt.close('all')
spacingL = 0.03
spacingT =0.01
leftmargin = 0.08
topmargin = 0.02
rightmargin = 0.09
figsize = [11, 8]
prestim = 5 # time in s to include before stimulation
poststim = 20 # time after first stimulation to include


crois = exceldata['roi'].unique().tolist()
crois = [roi for roi in crois if roi != 'Background']
#each figure will have one transgene, one stim type,  and each axis will display a different roi containing multiple Ca levels

print('starting graph')
#each figures will be specific for the roi
#determining graph placement
#remove background from roi list

#need to determine the number of stim periods??
#find an example and periods from it
voltage = exceldata['voltage'].iloc[0]
voltageDiff = np.diff(voltage, axis = 0)
stimPeriods = (np.where(voltageDiff > 0 )[0] +1) / 100
stimoffPeriods = (np.where(voltageDiff < 0 )[0]) / 100 
rowG = 0 # rows are rois
targetfolder1 = ['ComparingFulldeltaFF', 'individual_Plots']
targetdirectory = cc2.createBranch(targetfolder1, saveFig)



genotypes = list(transgene.keys())
for  row_group in crois:
    for cstimindex, cstim in enumerate(stimPeriods):
        fig1=plt.figure(figsize=figsize) 
        ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        stimStarts, stimStops = getVoltageStopStarts(voltage)
        ax2 = ax1.twinx()
        ax2.axvspan(cstim, stimoffPeriods[cstimindex], 0, voltage[int(cstim*100)], color ='r', alpha=0.2)
        
        
        for ax_group in genotypes:
            axFrame = exceldata[exceldata['Genotype'] == ax_group]
            #ax_group = 'A12-4'
            #axFrame = exceldata[exceldata['Genotype'].str.contains(ax_group)]
            rowFrame = axFrame[axFrame['roi'] == row_group]
            if len(rowFrame) > 0:
                matchBackground = matchingFrame(rowFrame, exceldata, 'Background')
                background_intensity, voltage, stimesta, nameB = getIntensityData(matchBackground)
                raw_intensity, voltage, timeStamp, name = getIntensityData(rowFrame)
                #remove missing data
                indx = [i for i, s in enumerate(name) if 'NaN' != s]
                background_intensity = background_intensity[indx, :]
            
                prestart = np.argmax(timeStamp > cstim-prestim)
                stop = np.argmax(timeStamp > stimoffPeriods[cstimindex]+poststim)
                if stop ==0: # raw_intensity[:, prestart : stop] > len(raw_intensity)
                    stop = raw_intensity.shape[1]
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
                ax1.plot(xrange1, np.nanmean(dFF.deltaFF(), 0), color = transgene[ax_group]['color'], linewidth =3, alpha = 0.8, label = transgene[ax_group]['sensor'])
            #ax1.set_ylim([-0.1, 1])
            ax1.set_xlim([cstim-prestim, stimoffPeriods[cstimindex]+poststim])
            
            fig1.savefig(os.path.join(targetdirectory, row_group + "{:.0f}".format(cstim)  + '.jpeg'))
            plt.close(fig1)

##################################################################################################################################################
#Generating statistics - bar graphs
#comparing each transgene  at each roi for each stim period
#each figure contains one axis

#functionName = ['Max', 'Mean', 'Median', 'SNR', 'delay2Max']
#columns1 = ['Max_Absolute_dFF', 'Mean_dFF', 'Median_dFF', 'Signal2Noise', 'Delay_Peak_Amplitude', 'Min_Intensity_After', 'Decay_After'] # columngs for the dataframe holding data
statsDict = {'Max' : 'Max_Absolute_dFF', 
             'Mean': 'Mean_dFF',
             'Median': 'Median_dFF', 
             'SNR': 'Signal2Noise', 
             'delay2Max':'Delay_Peak_Amplitude'
             }


plt.close('all')

figsize = [22, 16]
prestim = 5 # time in s to include before stimulation
poststim = 3 # time after off period to be included
#create figure storage location
targetfolder = ['Statistics', 'IndividualPlots']
targetdirectory = cc2.createBranch(targetfolder, saveFig)



crois = exceldata['roi'].unique().tolist()
crois = [roi for roi in crois if roi != 'Background']
#each figure will have multiple transgene, one stim type,  and each axis will display a different roi containing multiple Ca levels


outputexcel= os.path.join(targetdirectory, 'Statistics_'+ '.xlsx') 
writer = pd.ExcelWriter(outputexcel , engine = 'xlsxwriter')
for cstat in statsDict.keys():
    

    
    #create a pandas dataframe/multiindex to hold data
    iterables = [crois, stimPeriods.round().astype(np.int).tolist(), list(exceldata.groupby(['Genotype']).groups.keys())]
    statsFrameIndex = pd.MultiIndex.from_product(iterables, names = ['roi', 'Stim_Period', 'Genotype'])
    statsFrame = pd.DataFrame(index=statsFrameIndex, columns=[statsDict[cstat], 'Name'])
    columns2 = ['Number', 'Median', 'Mean', 'Standard Error', 'Kruskal_Wallis']
    columns2.extend(list(exceldata.groupby(['Genotype']).groups.keys()))
    outputFrame = pd.DataFrame(index=statsFrameIndex, columns=columns2)
    outputFrame['Name'] = ''
    
    rowG = 0 # rows are rois
    for row_group in crois:
        colG = 0 # columns are stim periods
        
        for stim_int, cstim in enumerate(stimPeriods):
            fig1=plt.figure(figsize=figsize) #each figures will be specific for the roi
            ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])

            for ax_group, axFrame in exceldata.groupby(['Genotype']):
                rowFrame = axFrame[axFrame['roi'] == row_group]
                matchBackground = matchingFrame(rowFrame, exceldata, 'Background')
                background_intensity, voltage, stimesta, name = getIntensityData(matchBackground)
                raw_intensity, voltage, timeStamp, name = getIntensityData(rowFrame)
                if len(raw_intensity.shape) != 0: #control for groups that have no observations
                    prestart = np.argmax(timeStamp > cstim-prestim)
                    stop = np.argmax(timeStamp > stimoffPeriods[stim_int]+poststim)
                    if stop ==0: # raw_intensity[:, prestart : stop] > len(raw_intensity)
                        stop = raw_intensity.shape[1]
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

            xlabel = np.empty(len(transgene.keys()), dtype=np.object)
            for cgene in transgene.keys():
                xlabel[transgene[cgene]['position']-1] = transgene[cgene]['sensor']
            ax1.xaxis.set_ticklabels(xlabel, rotation = 'vertical')           

            fig1.savefig(os.path.join(targetdirectory,  cstat+  row_group + "{:.0f}".format(cstim)  + '.jpeg'))
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


##################################################################################################################################################
#Generating statistics - line plots
#each figure contains one axis
#x = stim duration
#y= stat
#each line corresponds to a different transgene
#multiple figures corresponding to a different statistic

#functionName = ['Max', 'Mean', 'Median', 'SNR', 'delay2Max']
#columns1 = ['Max_Absolute_dFF', 'Mean_dFF', 'Median_dFF', 'Signal2Noise', 'Delay_Peak_Amplitude', 'Min_Intensity_After', 'Decay_After'] # columngs for the dataframe holding data
statsDict = {'Max' : 'Max_Absolute_dFF', 
             'Mean': 'Mean_dFF',
             'Median': 'Median_dFF', 
             'SNR': 'Signal2Noise', 
             'delay2Max':'Delay_Peak_Amplitude'
             }


plt.close('all')

figsize = [10, 8]
prestim = 5 # time in s to include before stimulation
poststim =5 # time after first stimulation to include

outputfolder1 = ['Statistics', 'IndividualPlots2']
targetdirectory = cc2.createBranch(outputfolder1, saveFig)

#get the duration for each stim period
voltage = exceldata['voltage'].iloc[0]
stimPeriods, stopPeriods = getVoltageStopStarts(voltage)
#stimPeriods = stimPeriods[:-1]
#stopPeriods = stopPeriods[:-1]
stimDur = []
for i in zip(stimPeriods, stopPeriods):
    stimDur.append(i[1] - i[0])
stimVolt = []
for i in stimPeriods:
    stimVolt.append(round(float(voltage[int(i *100)][0]), 2))
print(stimVolt)


crois = exceldata['roi'].unique().tolist()
crois = [roi for roi in crois if roi != 'Background']
#each figure will have multiple transgene, one stim type,  and each axis will display a different roi containing multiple Ca levels


null_genotype = list(transgene.keys())[0]
#stimPeriods = stimPeriods[:3]
for cstat in statsDict.keys():
    #create a pandas dataframe/multiindex to hold data
    iterables = [crois, stimPeriods.round().astype(np.int).tolist(), genotypes]
    statsFrameIndex = pd.MultiIndex.from_product(iterables, names = ['roi', 'Stim_Period', 'Genotype'])
    statsFrame = pd.DataFrame(index=statsFrameIndex, columns=[statsDict[cstat], 'Name'])
    columns2 = ['Number', 'Median', 'Mean', 'Standard Error', 'Kruskal_Wallis']
    columns2.extend(list(exceldata.groupby(['Genotype']).groups.keys()))
    outputFrame = pd.DataFrame(index=statsFrameIndex, columns=columns2)
    outputFrame['Name'] = ''
    
    rowG = 0 # rows are rois
    for row_group in crois:
        colG = 0 # columns are stim periods
        for stim_int, cstim in enumerate(stimPeriods):
            for ax_group in genotypes: #axFrame in exceldata.groupby(['Genotype']):
                axFrame = exceldata[exceldata['Genotype'] == ax_group]
                rowFrame = axFrame[axFrame['roi'] == row_group]
                matchBackground = matchingFrame(rowFrame, exceldata, 'Background')
                background_intensity, voltage, stimesta, name = getIntensityData(matchBackground)
                raw_intensity, voltage, timeStamp, name = getIntensityData(rowFrame)
                if len(raw_intensity.shape) != 0: #control for groups that have no observations
                    prestart = np.argmax(timeStamp > cstim-prestim)
                    #stop = np.argmax(timeStamp > stimoffPeriods[stim_int]+poststim)
                    stop = np.argmax(timeStamp > stimoffPeriods[stim_int] + poststim)
                    if stop ==0: # raw_intensity[:, prestart : stop] > len(raw_intensity)
                        stop = raw_intensity.shape[1]
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
                                                    
            #add kruskal wallis and signrank to dataframe
            #if all the results do not have the same value one can do a kruskal wallis test otherwise  make the kruskalwallis ==1
            if np.any(np.concatenate(statsFrame[statsDict[cstat]].loc[row_group, cstim.round().astype(np.int)].values) != np.concatenate(statsFrame[statsDict[cstat]].loc[row_group, cstim.round().astype(np.int)].values)[0]):    
                outputFrame['Kruskal_Wallis'].loc[row_group, cstim.round().astype(np.int), ax_group] =scipy.stats.mstats.kruskalwallis(*statsFrame[statsDict[cstat]].loc[row_group, cstim.round().astype(np.int)].values).pvalue
            else:
                outputFrame['Kruskal_Wallis'].loc[row_group, cstim.round().astype(np.int), ax_group] = 1
            for ax_group in genotypes:#axFrame in exceldata.groupby(['Genotype']):
                axFrame = exceldata[exceldata['Genotype'] == ax_group]
                for ax_group2 in genotypes: #, axFrame2 in exceldata.groupby(['Genotype']):
                    axFrame2 = exceldata[exceldata['Genotype'] == ax_group2]
                    data1 = statsFrame[statsDict[cstat]].loc[row_group, cstim.round().astype(np.int), ax_group]
                    data1 = data1[~np.isnan(data1)]
                    data2 = statsFrame[statsDict[cstat]].loc[row_group, cstim.round().astype(np.int), ax_group2]
                    data2 = data2[~np.isnan(data2)]
                    outputFrame[ax_group].loc[row_group, cstim.round().astype(np.int), ax_group2] = scipy.stats.ranksums(data1,data2).pvalue

            xlabel = []
            for cgene in transgene.keys():
                xlabel.append(transgene[cgene]['sensor'])
        #graph mean results in line plots
        fig1=plt.figure(figsize=figsize) #each figures will be specific for the roi
        ax1 = fig1.add_axes([0.1, 0.1, 0.5, 0.8])
        statsFrame = statsFrame.sort_index() # sorting is necessary to select the dseries otherwise produces an error
        for ax_group in genotypes: #, axFrame in exceldata.groupby(['Genotype']):
            axFrame = exceldata[exceldata['Genotype'] == ax_group]
            dseries  = statsFrame[statsDict[cstat]].loc[row_group, :, ax_group].copy()
            dseries = dseries.reset_index()
            dseries = dseries[statsDict[cstat]]
            dseries = dseries.rename(dict(zip(dseries.index, stimVolt)))
            graphDB.SElinePlotFrame(dseries, transgene[ax_group]['color'], ax1, transgene[ax_group]['sensor'])
            #add statistics
            t=[]
            for i in dseries:
                t.append(np.median(i))
            u = []
            for i2, cstim2 in enumerate(stimPeriods):
                if outputFrame['Kruskal_Wallis'].loc[row_group, cstim2.round().astype(np.int), genotypes[-1]] < 0.05:
                    #print(outputFrame['Kruskal_Wallis'].loc[row_group, cstim2.round().astype(np.int), ax_group2])
                    if outputFrame[null_genotype].loc[row_group, cstim2.round().astype(np.int), ax_group] < 0.05:
                            ax1.plot(stimVolt[i2], outputFrame['Mean'].loc[row_group, cstim2.round().astype(np.int), ax_group], 'x', markersize = 10, color = 'k')
                            u.append(outputFrame['Median'].loc[row_group, cstim2.round().astype(np.int), ax_group])
        ax1.set_ylabel(statsDict[cstat])
        #ax1.set_xlim([-0.2, 7])
        #ax1.set_xlim([0, 8.2])
        ax1.legend(bbox_to_anchor=(1.05,1), loc= 2)
        #fig1.show()
        fig1.savefig(os.path.join(targetdirectory, row_group + cstat + '.jpeg'))   
        plt.close(fig1)
    

############################################################################################################################################
#comparing deltaF/F between rois each axis at each stim period
#each figure contains on axis
#each figure will contain a multiple rois
#output is multiple graphs specific transgene and stim time
plt.close('all')
spacingL = 0.03
spacingT =0.01
leftmargin = 0.08
topmargin = 0.02
rightmargin = 0.09
figsize = [22, 16]
prestim = 5 # time in s to include before stimulation
poststim = 3 # time after first stimulation to include


crois = exceldata['roi'].unique().tolist()
crois = [r for r in crois if r != 'Background']
#each figure will have one transgene, one stim type,  and each axis will display a different roi containing multiple Ca levels

print('starting graph')
#each figures will be specific for the roi
#determining graph placement
#remove background from roi list

#need to determine the number of stim periods??
#find an example and periods from it
voltage = exceldata['voltage'].iloc[0]
voltageDiff = np.diff(voltage, axis = 0)
stimPeriods = (np.where(voltageDiff > 0 )[0] +1) / 100
rowG = 0 # rows are rois
targetfolder1 = 'Compare_deltaFF_Compartments'
targetfolder2 = 'Individual_Plots'
targetdirectory = str(pathlib.Path(saveFig) / targetfolder1 / targetfolder2)
for  row_group in transgene.keys():
    for cstim in stimPeriods:
        fig1=plt.figure(figsize=figsize) 
        ax1 = fig1.add_axes([0.1, 0.1, 0.6, 0.8])
        
        for ax_group in crois:
            #ax_group = 'A12-4'
            #axFrame = exceldata[exceldata['Genotype'].str.contains(ax_group)]
            axFrame = exceldata[exceldata['roi'] == ax_group]
            rowFrame = axFrame[axFrame['Genotype'] == row_group]
            matchBackground = matchingFrame(rowFrame, exceldata, 'Background')
            background_intensity, voltage, stimesta, nameB = getIntensityData(matchBackground)
            raw_intensity, voltage, timeStamp, name = getIntensityData(rowFrame)
            #remove missing data
            indx = [i for i, s in enumerate(name) if 'NaN' != s]
            background_intensity = background_intensity[indx, :]
        
            prestart = np.argmax(timeStamp > cstim-prestim)
            stop = np.argmax(timeStamp > cstim+poststim)
            if stop ==0: # raw_intensity[:, prestart : stop] > len(raw_intensity)
                stop = raw_intensity.shape[1]
            start = np.argmax(timeStamp > cstim)
            xrange1 = timeStamp[prestart : stop]
            dFF = ccModules.fluorescent(raw_intensity[:, prestart : stop], background_intensity[:, prestart : stop], start-prestart, transgene[row_group]['response'],  xrange1)
            '''
            variables to debug ccModules.fluorescent
            data =raw_intensity[:, prestart : stop]
            background = background_intensity[:, prestart : stop]
            start = start - prestart
            response = transgene[Gen_name]['response']
            timeStamp1 = timeStamp[:, prestart : stop]
            '''
            #ax1.plot(xrange1, dFF.deltaFF().T, color = transgene[page_group_names]['color'], linewidth =1, alpha = 0.3 )
            ax1.plot(xrange1, np.nanmean(dFF.deltaFF(), 0), color = rois[ax_group], linewidth =3, alpha = 0.8, label = ax_group)
        
                
        #plot changes in voltage to led
        ylim = ax1.get_ylim()
        xlim = ax1.get_xlim()
        ax1.set_ylim(ylim)
        stimStarts, stimStops = getVoltageStopStarts(voltage)
        for int1 in zip(stimStarts, stimStops):
            ax1.plot(int1, [ylim[0], ylim[0]], color = 'r', linewidth = 15)
            ax1.plot(int1, [ylim[1], ylim[1]], color = 'r', linewidth = 15)
        ax1.set_xlim(xlim)
        ax1.legend(bbox_to_anchor=(1.05,1), loc= 2)
        if not os.path.isdir(targetdirectory):
            os.mkdir(targetdirectory)
        fig1.savefig(os.path.join(targetdirectory, row_group + "{:.0f}".format(cstim)  + '.jpeg'))
        plt.close(fig1)

##################################################################################################################################################
############################################################################################################################################
#comparing deltaF between transgenes within each axis at each stim period
#each figure contains on axis
#each figure will contain a multiple geneotype
#output is multiple graphs specific to roi, stim time
plt.close('all')
spacingL = 0.03
spacingT =0.01
leftmargin = 0.08
topmargin = 0.02
rightmargin = 0.09
figsize = [11, 8]
prestim = 5 # time in s to include before stimulation
poststim = 20 # time after first stimulation to include


crois = exceldata['roi'].unique().tolist()
crois = [roi for roi in crois if roi != 'Background']
#each figure will have one transgene, one stim type,  and each axis will display a different roi containing multiple Ca levels

print('starting graph')
#each figures will be specific for the roi
#determining graph placement
#remove background from roi list

#need to determine the number of stim periods??
#find an example and periods from it
voltage = exceldata['voltage'].iloc[0]
voltageDiff = np.diff(voltage, axis = 0)
stimPeriods = (np.where(voltageDiff > 0 )[0] +1) / 100
stimoffPeriods = (np.where(voltageDiff < 0 )[0]) / 100 
rowG = 0 # rows are rois
targetfolder1 = ['ComparingFulldeltaF', 'Individual_Plots']
targetdirectory = cc2.createBranch(targetfolder1, saveFig)



genotypes = list(transgene.keys())
for  row_group in crois:
    for cstimindex, cstim in enumerate(stimPeriods):
        fig1=plt.figure(figsize=figsize) 
        ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        stimStarts, stimStops = getVoltageStopStarts(voltage)
        ax2 = ax1.twinx()
        ax2.axvspan(cstim, stimoffPeriods[cstimindex], 0, voltage[int(cstim*100)], color ='r', alpha=0.2)
        
        
        for ax_group in genotypes:
            axFrame = exceldata[exceldata['Genotype'] == ax_group]
            #ax_group = 'A12-4'
            #axFrame = exceldata[exceldata['Genotype'].str.contains(ax_group)]
            rowFrame = axFrame[axFrame['roi'] == row_group]
            if len(rowFrame) > 0:
                matchBackground = matchingFrame(rowFrame, exceldata, 'Background')
                background_intensity, voltage, stimesta, nameB = getIntensityData(matchBackground)
                raw_intensity, voltage, timeStamp, name = getIntensityData(rowFrame)
                #remove missing data
                indx = [i for i, s in enumerate(name) if 'NaN' != s]
                background_intensity = background_intensity[indx, :]
            
                prestart = np.argmax(timeStamp > cstim-prestim)
                stop = np.argmax(timeStamp > stimoffPeriods[cstimindex]+poststim)
                if stop ==0: # raw_intensity[:, prestart : stop] > len(raw_intensity)
                    stop = raw_intensity.shape[1]
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
                ax1.plot(xrange1, np.nanmean(dFF.deltaF(), 0), color = transgene[ax_group]['color'], linewidth =3, alpha = 0.8, label = transgene[ax_group]['sensor'])
            #ax1.set_ylim([-0.1, 1])
            ax1.set_xlim([cstim-prestim, stimoffPeriods[cstimindex]+poststim])
            
            fig1.savefig(os.path.join(targetdirectory, row_group + "{:.0f}".format(cstim)  + '.jpeg'))
            plt.close(fig1)





#########################################################################################
#plotting changes in voltage over timestim
plt.close('all')
voltage = exceldata['voltage'].iloc[0]
fig1 = plt.figure(figsize = [10,5])
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
time1 = np.arange(len(voltage)) / 100
ax1.plot(time1, voltage, '--', color = (1,0,0))
fig1.show()
fig1.savefig(os.path.join(saveFig, 'voltage.jpeg'))

