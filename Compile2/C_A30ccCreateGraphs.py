#designed to run through all rows in excel sheet and generate figures summarizing results
#there are high memory requirements when running this script so keep number of workers low

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#folderModules = '/data/JData/A/A30FastROI/CircuitCatcher'
#sys.path.append(folderModules)
import ccModules
import scipy
import graphDB

from projectData import *
dropforexcel = ['intensity_data', 'timestamp', 'voltage'] # columns to remove if writing to excel sheet

#save pandas data frame
exceldata = pd.read_hdf(os.path.join(saveFig, 'Compiled_data_All.hdf5'), 'data')
#exceldata = pd.read_hdf('/home/daniel/Desktop/ResearchUbuntuYoga720/A32_Saline/Ca-Compile/Compiled_data2.hdf5', 'data')

#generate a genotype column from name
for ckey in transgene.keys():
    for i, dseries in exceldata.iterrows():
        if ckey+ '-' in dseries['Sample Name']:
            exceldata['Genotype'].loc[i] = ckey
            
#remove rows in exceldata with genotypes that are not be studied
exceldata = exceldata.dropna(subset=['Genotype'])

# check columns with selection criteria to make sure there are only the expected categories
#checking Ca categories
Ca = exceldata[Saline].unique()
print(Ca.sort() == list(Solute_Colors.keys()).sort())

#checking that stim types are only two types
stims = exceldata['Stim Protocol'].unique()
print(stims)
exceldata['Stim Protocol'].replace('timestime', 'timestim', inplace = True)

#remove double trials labeled 3 and 4 in the 'No.' column
exceldata = exceldata[exceldata['No.'] < 3]

#divided data into groups 
#divide by genotype and Ca solution
groups = exceldata.groupby(['Stim Protocol', 'Genotype', 'No.', Saline])
groups.count().to_excel(os.path.join(saveFig, 'CountsAll.xlsx'))
exceldata[['No.', 'Sample Name', 'Stim Protocol', 'Genotype', 'No.', Saline]].to_excel(os.path.join(saveFig, 'GroupsList.xlsx'))


# basic reiteration through groups    
for name, genotype in exceldata.groupby(['Genotype']):
    print('name-G', name)
    for name, Ca in genotype.groupby([Saline]):
        print('name-Saline', name)
    


#function to get data using indexFrame
def getIntensityData(dataframe1, roi):
    #dataframe of all experiments to be included
    #roi current roi to collect data from
    data = []
    name = []
    for row, dSeries in dataframe1.iterrows():
        cdata = [] # collect all intensity data here for roi from all animals
        cintensityFrame = pd.DataFrame(dSeries['intensity_data'])
        index = cintensityFrame['Name'].str.contains(roi)
        cintensityFrame=cintensityFrame[index]
        for row2, dSeries2 in cintensityFrame.iterrows():
            cdata.append(dSeries2['intensity'])
        if len(cdata) >0:
            data.append(np.mean(cdata, axis=0))
            name.append(str(dSeries['No.']) +'-'+ dSeries['Sample Name'])
        else:
            #data.append(np.nan)
            name.append('NaN')
    voltage = dSeries['voltage']
    timeStamp = dSeries['timestamp']
    return np.vstack(data), voltage, timeStamp, name



def getSpacingSize(number, margin1, margin2, spacing):
    #get spacing for graphs
    return (1- margin1 - margin2 - spacing*(number))/(number)


def getXYSScoordinates(xnum, ynum, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, bottommargin, spacingL, spacingT):
    #returns the [x, y, xdistance, ydistance] for current axis position
    return [ leftmargin +(xnum)*(graphsizeX + spacingL), 1-((ynum+1)*(graphsizeY+spacingT))-topmargin, graphsizeX, graphsizeY]

####################################################################
#graph comparing normalized plots among all genotypes for each roi
plt.close('all')
spacingL = 0.05
spacingT =0.015
leftmargin = 0.08
topmargin = 0.02
rightmargin = 0.01
figsize = [22, 16]


#each figure will have one transgene, one stim type,  and each axis will display a different roi containing multiple Ca levels
for Gen_name, genFrame in exceldata.groupby(['Genotype']):
    for stimName, stimFrame in genFrame.groupby(['Stim Protocol']):
        fig1=plt.figure(figsize=figsize) #each figures will be specific for the roi
        #determining graph placement
        n_rows =len(transgene[Gen_name]['rois'])  # determine the number of rows
        n_cols = 1
        graphsizeX = getSpacingSize(n_cols, leftmargin, rightmargin, spacingL)
        graphsizeY = getSpacingSize(n_rows, topmargin, topmargin, spacingT)
        #each axes will contain one type of roi
        rowG = 0
        for roi in transgene[Gen_name]['rois']:
            print(Gen_name)
            ax1 = fig1.add_axes(getXYSScoordinates(0, rowG, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, topmargin, spacingL, spacingT))
            for Ca_name, Ca in stimFrame.groupby([Saline]):
                raw_intensity, voltage, timeStamp, name = getIntensityData(Ca, roi)
                norm_intensity = raw_intensity.T / np.mean(raw_intensity, axis =1)
                for row in range(norm_intensity.shape[1]):
                    ax1.plot(timeStamp, norm_intensity[:,row], alpha = 0.4, linewidth = 1, color = Solute_Colors[Ca_name]) #label = name[row], 
                ax1.plot(timeStamp, np.median(norm_intensity, axis=1), color = Solute_Colors[Ca_name], linewidth = 1.5, label = Ca_name, alpha=0.8)
                yaxis = ax1.get_ylim()
                #ax1.text(0, yaxis[1], cgen + ':' + transgene[cgen], va = 'top')
            ax1.set_title(roi)
            ax11 = ax1.twinx()
            ax11.set_ylabel('LED Power (V)')
            stimTimesRange = np.array(range(len(voltage)))/100.0
            ax11.plot(stimTimesRange,  voltage, color = [0.5,0.5,0.5], linestyle = '--', alpha=0.4)
            ax1.set_xlim([0, timeStamp[-1]])
            lg = ax1.legend(bbox_to_anchor = (1.07, 1))
            ax1.set_ylabel('Normalized Intensity')
            if rowG != len(transgene[Gen_name]['rois'])-1:
                ax1.xaxis.set_visible(False)
            
            #norm_intensity = raw_intensity.T / np.mean(raw_intensity, axis =1)
            rowG = rowG +1
        if not os.path.isdir(os.path.join(saveFig, 'ComparingFullTimeLine')):
            os.mkdir(os.path.join(saveFig, 'ComparingFullTimeLine'))
        fig1.savefig(os.path.join(os.path.join(saveFig, 'ComparingFullTimeLine'), Gen_name + '-' + stimName + '.jpeg'))
        plt.close(fig1)
            
############################################################################################################################################
#Generate graphs showing deltaF/F for each response period
#each figure will contain a single genetype
#columns will contain stim period
#rois will contain roi
plt.close('all')
spacingL = 0.05
spacingT =0.015
leftmargin = 0.08
topmargin = 0.02
rightmargin = 0.01
figsize = [22, 16]
prestim = 5 # time in s to include before stimulation
poststim = 20 # time after first stimulation to include

#each figure will have one transgene, one stim type,  and each axis will display a different roi containing multiple Ca levels
for Gen_name, genFrame in exceldata.groupby(['Genotype']):
    for stimName, stimFrame in genFrame.groupby(['Stim Protocol']):
        fig1=plt.figure(figsize=figsize) #each figures will be specific for the roi
        #determining graph placement
        #remove background from roi list
        crois = transgene[Gen_name]['rois']
        if 'Background' in crois: crois.remove('Background') 
        n_rows =len(crois)  # determine the number of rows
        #need to determine the number of stim periods??
        #find an example and periods from it
        voltage = stimFrame['voltage'].iloc[0]
        voltageDiff = np.diff(voltage, axis = 0)
        stimPeriods = (np.where(voltageDiff > 0 )[0] +1) / 100
        n_cols =  len(stimPeriods)#number of columns
        graphsizeX = getSpacingSize(n_cols, leftmargin, rightmargin, spacingL)
        graphsizeY = getSpacingSize(n_rows, topmargin, topmargin, spacingT)
        
        
        rowG = 0 # rows are rois

        for roi in crois:
            colG = 0 # columns are stim periods
            for cstim in stimPeriods:
                ax1 = fig1.add_axes(getXYSScoordinates(colG, rowG, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, topmargin, spacingL, spacingT))
                for Ca_name, Ca in stimFrame.groupby([Saline]):
                    background_intensity, voltage, stimesta, nameB = getIntensityData(Ca, 'Background')
                    raw_intensity, voltage, timeStamp, name = getIntensityData(Ca, roi)
                    #remove missing data
                    indx = [i for i, s in enumerate(name) if 'NaN' != s]
                    background_intensity = background_intensity[indx, :]
                    
                    prestart = np.argmax(timeStamp > cstim-prestim)
                    stop = np.argmax(timeStamp > cstim+poststim)
                    start = np.argmax(timeStamp > cstim)
                    xrange1 = timeStamp[prestart : stop]
                    dFF = ccModules.fluorescent(raw_intensity[:, prestart : stop], background_intensity[:, prestart : stop], start-prestart, transgene[Gen_name]['response'],  xrange1)
                    '''
                    variables to debug ccModules.fluorescent
                    data =raw_intensity[:, prestart : stop]
                    background = background_intensity[:, prestart : stop]
                    start = start - prestart
                    response = transgene[Gen_name]['response']
                    timeStamp1 = timeStamp[:, prestart : stop]
                    '''
                    ax1.plot(xrange1, dFF.deltaFF().T, color = Solute_Colors[Ca_name], linewidth =1, alpha = 0.3 )
                    ax1.plot(xrange1, np.median(dFF.deltaFF(), 0), color = Solute_Colors[Ca_name], linewidth =1.5, alpha = 0.8, label = Ca_name )
                
                    
                #plot changes in voltage to led
                ax11 = ax1.twinx()
                
                stimTimesRange = np.array(range(len(voltage)))/100.0
                ax11.plot(stimTimesRange[int((cstim-prestim)*100) : int((cstim+poststim)*100)],  voltage[int((cstim-prestim)*100) : int((cstim+poststim)*100)], color = [0.5,0.5,0.5], linestyle = '--', alpha=0.4)
                ax11.set_ylim([0, 0.4])
                #ax1.set_xlim([0, timeStamp[-1]])
                if rowG == 0:
                    ax1.set_title('Stim Time ' + str(cstim.round().astype(np.int)))
                if colG == len(stimPeriods)-1:
                    lg = ax1.legend(bbox_to_anchor = (1.0, 1))
                    ax11.set_ylabel('LED Power (V)')
                if colG == 0:
                    ax1.set_ylabel('Normalized Intensity' + '-' + roi)
                if rowG != len(transgene[Gen_name]['rois'])-1:
                    ax1.xaxis.set_visible(False)
                    
                colG +=1
            rowG +=1
        if not os.path.isdir(os.path.join(saveFig, 'ComparingFulldeltaFF')):
            os.mkdir(os.path.join(saveFig, 'ComparingFulldeltaFF'))
        fig1.savefig(os.path.join(os.path.join(saveFig, 'ComparingFulldeltaFF'), Gen_name + '-' + stimName + '.jpeg'))
        plt.close(fig1)

##################################################################################################################################################
#Generating statistics -
#each figure will contain a single genetype
#columns will contain stim period
#rois will contain roi

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
rightmargin = 0.01
figsize = [22, 16]
prestim = 5 # time in s to include before stimulation
poststim = 20 # time after first stimulation to include
outputfolder = 'Statistics'
if not os.path.isdir(os.path.join(saveFig, outputfolder)):
    os.mkdir(os.path.join(saveFig, outputfolder))
#each figure will have one transgene, one stim type,  and each axis will display a different roi containing multiple Ca levels
for Gen_name, genFrame in exceldata.groupby(['Genotype']):
    #create a new xlsx file for each transgene

    for stimName, stimFrame in genFrame.groupby(['Stim Protocol']):
        outputexcel= os.path.join(os.path.join(saveFig, outputfolder), 'Statistics_'+ Gen_name + '_' + stimName +'.xlsx') 
        writer = pd.ExcelWriter(outputexcel , engine = 'xlsxwriter')
        for cstat in statsDict:
            fig1=plt.figure(figsize=figsize) #each figures will be specific for the roi
            #determining graph placement
            #remove background from roi list
            crois = transgene[Gen_name]['rois']
            if 'Background' in crois: crois.remove('Background') 
            n_rows =len(crois)  # determine the number of rows
            #need to determine the number of stim periods??
            #find an example and periods from it
            voltage = stimFrame['voltage'].iloc[0]
            voltageDiff = np.diff(voltage, axis = 0)
            stimPeriods = (np.where(voltageDiff > 0 )[0] +1) / 100
            n_cols =  len(stimPeriods)#number of columns
            graphsizeX = getSpacingSize(n_cols, leftmargin, rightmargin, spacingL)
            graphsizeY = getSpacingSize(n_rows, topmargin, topmargin, spacingT)
            
            #create a pandas dataframe/multiindex to hold data
            iterables = [crois, stimPeriods.round().astype(np.int).tolist(), list(stimFrame.groupby([Saline]).groups.keys())]
            statsFrameIndex = pd.MultiIndex.from_product(iterables, names = ['roi', 'Stim_Period', 'Ca_level'])
            statsFrame = pd.DataFrame(index=statsFrameIndex, columns=[statsDict[cstat]])
            columns2 = ['Number', 'Median', 'Mean', 'Standard Error', 'Kruskal_Wallis']
            columns2.extend(list(stimFrame.groupby([Saline]).groups.keys()))
            outputFrame = pd.DataFrame(index=statsFrameIndex, columns=columns2)
                    
            rowG = 0 # rows are rois
            for roi in crois:
                colG = 0 # columns are stim periods
                for cstim in stimPeriods:
                    ax1 = fig1.add_axes(getXYSScoordinates(colG, rowG, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, topmargin, spacingL, spacingT))
                    for Ca_name, Ca in stimFrame.groupby([Saline]):
                        background_intensity, voltage, stimesta, name = getIntensityData(Ca, 'Background')
                        raw_intensity, voltage, timeStamp, name = getIntensityData(Ca, roi)
                        if len(raw_intensity.shape) != 0: #control for groups that have no observations
                            prestart = np.argmax(timeStamp > cstim-prestim)
                            stop = np.argmax(timeStamp > cstim+poststim)
                            start = np.argmax(timeStamp > cstim)
                            xrange1 = timeStamp[prestart : stop]
                            dFF = ccModules.fluorescent(raw_intensity[:, prestart : stop], background_intensity[:, prestart : stop], start-prestart, transgene[Gen_name]['response'],  xrange1)
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
                            statsFrame[statsDict[cstat]].loc[roi, cstim.round().astype(np.int), Ca_name] = data
                            data = data[~np.isnan(data)]
                            outputFrame['Mean'].loc[roi, cstim.round().astype(np.int), Ca_name] = np.mean(data)
                            outputFrame['Median'].loc[roi, cstim.round().astype(np.int), Ca_name] = np.median(data)
                            outputFrame['Standard Error'].loc[roi, cstim.round().astype(np.int), Ca_name] = np.std(data) / np.sqrt(np.sum(~np.isnan(data)))
                            outputFrame['Number'].loc[roi, cstim.round().astype(np.int), Ca_name] = len(data)
                            graphDB.scatterBarPlot(Ca_name, data)

        
                 
                    if colG == 0:
                        ax1.set_ylabel(statsDict[cstat] + ' ' + roi)
                    if rowG ==0:
                        ax1.set_title('Stim Time ' + str(cstim.round().astype(np.int)))
                    #add kruskal wallis and signrank to dataframe
                    outputFrame['Kruskal_Wallis'].loc[roi, cstim.round().astype(np.int), Ca_name] =scipy.stats.mstats.kruskalwallis(*statsFrame[statsDict[cstat]].loc[roi, cstim.round().astype(np.int)].values).pvalue   
                    for Ca_name, Ca in stimFrame.groupby([Saline]):
                        for Ca_name2, Ca2 in stimFrame.groupby([Saline]):
                            data1 = statsFrame[statsDict[cstat]].loc[roi, cstim.round().astype(np.int), Ca_name]
                            data1 = data1[~np.isnan(data1)]
                            data2 = statsFrame[statsDict[cstat]].loc[roi, cstim.round().astype(np.int), Ca_name2]
                            data2 = data2[~np.isnan(data2)]
                            outputFrame[Ca_name].loc[roi, cstim.round().astype(np.int), Ca_name2] = scipy.stats.ranksums(data1,data2).pvalue
                                
                    colG +=1
                rowG +=1
                
            
        

            fig1.savefig(os.path.join(os.path.join(saveFig, outputfolder), Gen_name + '-' + stimName + cstat + '.jpeg'))
            plt.close(fig1)
            outputFrame.reset_index().to_excel(writer, sheet_name = statsDict[cstat])
        writer.save()

    

##################################################################################################################################################
#Generating statistics - line plot
#each figure will contain a single genotype
#columns = stat
#row = roi
#x = stim period
#y = statistic see stats Dict
#multiple lines for each saline concentration

#functionName = ['Max', 'Mean', 'Median', 'SNR', 'delay2Max']
#columns1 = ['Max_Absolute_dFF', 'Mean_dFF', 'Median_dFF', 'Signal2Noise', 'Delay_Peak_Amplitude', 'Min_Intensity_After', 'Decay_After'] # columngs for the dataframe holding data
statsDict = {'Max' : 'Max_Absolute_dFF', 
             'Mean': 'Mean_dFF',
             'Median': 'Median_dFF', 
             'SNR': 'Signal2Noise', 
             'delay2Max':'Delay_Peak_Amplitude'
             }

def map_level(df, dct, level=0):
    index = df.index
    index.set_levels([[dct.get(item, item) for item in names] if i==level else names for i, names in enumerate(index.levels)], inplace=True)              



def SElinePlotFrame(dseries, color, ax1, label1):
    #dseries = pandas series
    #x = dseries.index
    #y = data
    median = []
    SE = []
    for cindex in dseries.index:
        data = dseries[cindex]
        data = data[~np.isnan(data)]
        median.append(np.median(data))
        SE.append (scipy.stats.sem(data))
    #plot median values
    x = dseries.index
    ax1.plot(x, median, '-o', color = color, label=label1)
    #plot standard error bars
    for i, cSE in enumerate(SE):
        ax1.plot([x[i], x[i]], [median[i] - SE[i], median[i]+SE[i]], color=color, alpha = 0.5)


plt.close('all')
spacingL = 0.05
spacingT =0.015
leftmargin = 0.08
topmargin = 0.02
rightmargin = 0.01
figsize = [22, 16]
prestim = 5 # time in s to include before stimulation
poststim = 20 # time after first stimulation to include
outputfolder = 'Statistics2-LinePlots'


if not os.path.isdir(os.path.join(saveFig, outputfolder)):
    os.mkdir(os.path.join(saveFig, outputfolder))
#each figure will have one transgene, one stim type,  and each axis will display a different roi containing multiple Ca levels
for Gen_name, genFrame in exceldata.groupby(['Genotype']):
    #create a new xlsx file for each transgene

    for stimName, stimFrame in genFrame.groupby(['Stim Protocol']):
        #for cstat in statsDict:
        fig1=plt.figure(figsize=figsize) #each figures will be specific for the roi
        #determining graph placement
        #remove background from roi list
        crois = transgene[Gen_name]['rois']
        if 'Background' in crois: crois.remove('Background') 
        n_rows =len(crois)  # determine the number of rows
        #need to determine the number of stim periods??
        #find an example and periods from it
        voltage = stimFrame['voltage'].iloc[0]
        voltageDiff = np.diff(voltage, axis = 0)
        stimPeriods = (np.where(voltageDiff > 0 )[0] +1) / 100
        stimPeriods = stimPeriods.round().astype(np.int).tolist()
        #need to change x axis depending on stim protocol
        stimDict ={}
        if stimName == 'voltstim':
            for i in np.where(voltageDiff > 0 )[0] +1:
                int1 = i/100
                stimDict[int1.round().astype(np.int)] = float('{0:0.3f}'.format(voltage[i][0]))
        elif stimName == 'timestim':
            stimPeriods = stimPeriods[:-1]
            starts = np.where(voltageDiff > 0 )[0] +1
            stops = np.where(voltageDiff < 0 )[0]
            for i, cstart in enumerate(starts):
                time1 = (stops[i] - cstart)/100
                int1 = cstart/100
                stimDict[int1.round().astype(np.int)] = float('{0:0.3f}'.format(time1))
            
        
        
        # remove last stim period if stim protocol timestim
        #setting number of columns in figure
        n_cols =  len(statsDict)#number of columns
        graphsizeX = getSpacingSize(n_cols, leftmargin, rightmargin, spacingL)
        graphsizeY = getSpacingSize(n_rows, topmargin, topmargin, spacingT)
        
        #create a pandas dataframe/multiindex to hold data
        iterables = [crois, list(stimFrame.groupby([Saline]).groups.keys()), stimPeriods]
        statsFrameIndex = pd.MultiIndex.from_product(iterables, names = ['roi', 'Ca_level', 'Stim_Period'])
        statsFrame = pd.DataFrame(index=statsFrameIndex, columns=list(statsDict.keys()))

                
        #rowG = 0 # rows are rois
        for cstat in statsDict.keys():
            for roi in crois:
                #colG = 0 # columns are stats
                for cstim in stimPeriods:
                    #ax1 = fig1.add_axes(getXYSScoordinates(colG, rowG, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, topmargin, spacingL, spacingT))
                    for Ca_name, Ca in stimFrame.groupby([Saline]):
                        background_intensity, voltage, stimesta, name = getIntensityData(Ca, 'Background')
                        raw_intensity, voltage, timeStamp, name = getIntensityData(Ca, roi)
                        if len(raw_intensity.shape) != 0: #control for groups that have no observations
                            prestart = np.argmax(timeStamp > cstim-prestim)
                            stop = np.argmax(timeStamp > cstim+poststim)
                            start = np.argmax(timeStamp > cstim)
                            xrange1 = timeStamp[prestart : stop]
                            dFF = ccModules.fluorescent(raw_intensity[:, prestart : stop], background_intensity[:, prestart : stop], start-prestart, transgene[Gen_name]['response'],  xrange1)
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
                            statsFrame[cstat].loc[roi, Ca_name, cstim] = data
        #need to replace index stim times with either voltage or time stimulated
        #map_level(statsFrame, stimDict, level=1) # for some reason when converting to series it uses the stime interval 
              
        rowG = 0 # rows are rois
        for roi in crois:
            colG = 0 # columns are stats
            for cstat in statsDict.keys():
                ax1 = fig1.add_axes(getXYSScoordinates(colG, rowG, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, topmargin, spacingL, spacingT))

    
             
                if colG == 0:
                    ax1.set_ylabel( roi )
                if rowG ==0:
                    ax1.set_title(statsDict[cstat])
                for Ca_name, Ca in stimFrame.groupby([Saline]):
                    dseries = statsFrame[cstat].loc[roi,Ca_name]
                    dseries.rename(stimDict, inplace = True)
                    SElinePlotFrame(dseries, Solute_Colors[Ca_name], ax1, Ca_name)
                    
                #add p-value statistics
                Salinelist = list(stimFrame.groupby([Saline]).groups.keys())
                for cstim in stimPeriods:
                    dseries = statsFrame[cstat].loc[roi, Salinelist, cstim]
                    #perform kruskal wallis test for given stimulation period
                    krusk = scipy.stats.mstats.kruskalwallis(*dseries.values).pvalue   
                    if krusk < 0.05:
                        for csal in np.array(Salinelist)[np.array([0,2])]:
                            data1 = dseries.loc[roi, Salinelist[1], cstim]
                            data1 = data1[~np.isnan(data1)]
                            data2 = dseries.loc[roi,csal, cstim]
                            data2 = data2[~np.isnan(data2)]
                            ranksum = scipy.stats.ranksums(data1,data2).pvalue
                            if ranksum < 0.05:
                                ax1.plot(stimDict[cstim], np.median(data2), 'x', color ='k', markersize=12)
                if colG == len(statsDict.keys())-1:
                    lg = ax1.legend(bbox_to_anchor = (1.0, 1))
                colG +=1
            rowG +=1
            

        fig1.savefig(os.path.join(os.path.join(saveFig, outputfolder), Gen_name + '-' + stimName +  '.jpeg'))
        plt.close(fig1)


##################################################################################################################################################
NOT FINISHED
#Generating statistics - comparing baseline fluorescence levels for Campari variants only 
#each figure will contain a single statistic
#columns will contain baseline + stim period
# x = roi compartment
# y = statistics
#using the soma as baseline and normalize all other rois to this compartment

statsDict = {'Mean': 'Mean_Relative_Soma',
             'Std': 'Standard_Deviation_Relative_Soma',
             }

def map_level(df, dct, level=0):
    index = df.index
    index.set_levels([[dct.get(item, item) for item in names] if i==level else names for i, names in enumerate(index.levels)], inplace=True)              



def SElinePlotFrame(dseries, color, ax1, label1):
    #dseries = pandas series
    #x = dseries.index
    #y = data
    median = []
    SE = []
    for cindex in dseries.index:
        data = dseries[cindex]
        data = data[~np.isnan(data)]
        median.append(np.median(data))
        SE.append (scipy.stats.sem(data))
    #plot median values
    x = dseries.index
    ax1.plot(x, median, '-o', color = color, label=label1)
    #plot standard error bars
    for i, cSE in enumerate(SE):
        ax1.plot([x[i], x[i]], [median[i] - SE[i], median[i]+SE[i]], color=color, alpha = 0.5)


plt.close('all')
spacingL = 0.05
spacingT =0.015
leftmargin = 0.08
topmargin = 0.02
rightmargin = 0.01
figsize = [22, 16]
prestim = 5 # time in s to include before stimulation
poststim = 20 # time after first stimulation to include
outputfolder = 'Statistics3-BaelineByROI'


if not os.path.isdir(os.path.join(saveFig, outputfolder)):
    os.mkdir(os.path.join(saveFig, outputfolder))
    
Gen_name = 'A12
# each figure will have a different statistic
for cstat in statsDict.keys():
    fig1=plt.figure(figsize=figsize) #each figures will be specific for the statistic
    crois = transgene['A12-3']['rois']
    if 'Background' in crois: crois.remove('Background') 
    ax1 = fig1.add_axes([0.1, 0.1, 0.9, 0.9])
    #collect data into the following dataframe
    iterables = [crois, list(stimFrame.groupby([Saline]).groups.keys()), stimPeriods]
    statsFrameIndex = pd.MultiIndex.from_product(iterables, names = ['roi', 'Ca_level', 'Stim_Period'])
    statsFrame = pd.DataFrame(index=statsFrameIndex, columns=list(statsDict.keys()))
    
    for Ca_name, Ca in stimFrame.groupby([Saline]):
        background_intensity, voltage, stimesta, name = getIntensityData(Ca, 'Background')
        raw_intensity, voltage, timeStamp, name = getIntensityData(Ca, roi)
    
    
    
for Gen_name, genFrame in exceldata.groupby(['Genotype']):
    #create a new xlsx file for each transgene

    for stimName, stimFrame in genFrame.groupby(['Stim Protocol']):
        #for cstat in statsDict:
        fig1=plt.figure(figsize=figsize) #each figures will be specific for the roi
        #determining graph placement
        #remove background from roi list
        crois = transgene[Gen_name]['rois']
        if 'Background' in crois: crois.remove('Background') 
        n_rows =len(crois)  # determine the number of rows
        #need to determine the number of stim periods??
        #find an example and periods from it
        voltage = stimFrame['voltage'].iloc[0]
        voltageDiff = np.diff(voltage, axis = 0)
        stimPeriods = (np.where(voltageDiff > 0 )[0] +1) / 100
        stimPeriods = stimPeriods.round().astype(np.int).tolist()
        #need to change x axis depending on stim protocol
        stimDict ={}
        if stimName == 'voltstim':
            for i in np.where(voltageDiff > 0 )[0] +1:
                int1 = i/100
                stimDict[int1.round().astype(np.int)] = float('{0:0.3f}'.format(voltage[i][0]))
        elif stimName == 'timestim':
            stimPeriods = stimPeriods[:-1]
            starts = np.where(voltageDiff > 0 )[0] +1
            stops = np.where(voltageDiff < 0 )[0]
            for i, cstart in enumerate(starts):
                time1 = (stops[i] - cstart)/100
                int1 = cstart/100
                stimDict[int1.round().astype(np.int)] = float('{0:0.3f}'.format(time1))
            
        
        
        # remove last stim period if stim protocol timestim
        #setting number of columns in figure
        n_cols =  len(statsDict)#number of columns
        graphsizeX = getSpacingSize(n_cols, leftmargin, rightmargin, spacingL)
        graphsizeY = getSpacingSize(n_rows, topmargin, topmargin, spacingT)
        
        #create a pandas dataframe/multiindex to hold data
        iterables = [crois, list(stimFrame.groupby([Saline]).groups.keys()), stimPeriods]
        statsFrameIndex = pd.MultiIndex.from_product(iterables, names = ['roi', 'Ca_level', 'Stim_Period'])
        statsFrame = pd.DataFrame(index=statsFrameIndex, columns=list(statsDict.keys()))

                
        #rowG = 0 # rows are rois
        for cstat in statsDict.keys():
            for roi in crois:
                #colG = 0 # columns are stats
                for cstim in stimPeriods:
                    #ax1 = fig1.add_axes(getXYSScoordinates(colG, rowG, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, topmargin, spacingL, spacingT))
                    for Ca_name, Ca in stimFrame.groupby([Saline]):
                        background_intensity, voltage, stimesta, name = getIntensityData(Ca, 'Background')
                        raw_intensity, voltage, timeStamp, name = getIntensityData(Ca, roi)
                        if len(raw_intensity.shape) != 0: #control for groups that have no observations
                            prestart = np.argmax(timeStamp > cstim-prestim)
                            stop = np.argmax(timeStamp > cstim+poststim)
                            start = np.argmax(timeStamp > cstim)
                            xrange1 = timeStamp[prestart : stop]
                            dFF = ccModules.fluorescent(raw_intensity[:, prestart : stop], background_intensity[:, prestart : stop], start-prestart, transgene[Gen_name]['response'],  xrange1)
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
                            statsFrame[cstat].loc[roi, Ca_name, cstim] = data
        #need to replace index stim times with either voltage or time stimulated
        #map_level(statsFrame, stimDict, level=1) # for some reason when converting to series it uses the stime interval 
              
        rowG = 0 # rows are rois
        for roi in crois:
            colG = 0 # columns are stats
            for cstat in statsDict.keys():
                ax1 = fig1.add_axes(getXYSScoordinates(colG, rowG, graphsizeX, graphsizeY, leftmargin, rightmargin, topmargin, topmargin, spacingL, spacingT))

    
             
                if colG == 0:
                    ax1.set_ylabel( roi )
                if rowG ==0:
                    ax1.set_title(statsDict[cstat])
                for Ca_name, Ca in stimFrame.groupby([Saline]):
                    dseries = statsFrame[cstat].loc[roi,Ca_name]
                    dseries.rename(stimDict, inplace = True)
                    SElinePlotFrame(dseries, Solute_Colors[Ca_name], ax1, Ca_name)
                    
                #add p-value statistics
                Salinelist = list(stimFrame.groupby([Saline]).groups.keys())
                for cstim in stimPeriods:
                    dseries = statsFrame[cstat].loc[roi, Salinelist, cstim]
                    #perform kruskal wallis test for given stimulation period
                    krusk = scipy.stats.mstats.kruskalwallis(*dseries.values).pvalue   
                    if krusk < 0.05:
                        for csal in np.array(Salinelist)[np.array([0,2])]:
                            data1 = dseries.loc[roi, Salinelist[1], cstim]
                            data1 = data1[~np.isnan(data1)]
                            data2 = dseries.loc[roi,csal, cstim]
                            data2 = data2[~np.isnan(data2)]
                            ranksum = scipy.stats.ranksums(data1,data2).pvalue
                            if ranksum < 0.05:
                                ax1.plot(stimDict[cstim], np.median(data2), 'x', color ='k', markersize=12)
                if colG == len(statsDict.keys())-1:
                    lg = ax1.legend(bbox_to_anchor = (1.0, 1))
                colG +=1
            rowG +=1
            

        fig1.savefig(os.path.join(os.path.join(saveFig, outputfolder), Gen_name + '-' + stimName +  '.jpeg'))
        plt.close(fig1)



'''
####################################################################################################################
#


##poster: graphing line plot with increasing stim times
matplotlib.rcParams.update({'font.size':6})
matplotlib.rcParams.update({'font.family': 'sans-serif'})
matplotlib.rcParams.update({'font.sans-serif': 'Helvetica'})
functionName = ['Max', 'delay2Max', 'SNR']
columns1 = ['Maximum Absolute dFF', 'Delay to Peak Signal', 'Signal to Noise'] # columngs for the dataframe holding data
stimTimesShort = startStim[:-1]
plt.close('all')
spacingL = 0.01
spacingT =0.08
leftmargin = 0.09
topmargin = 0.02
numofplotsX =4
graphsizeX = 0.9/(numofplotsX+ spacingL*numofplotsX )
graphsizeY = 0.7/(len(functionName) + spacingT*len(functionName))
figsize = [10, 10]
plt.close('All')
matplotlib.rcParams.update({'font.size':12})
roi = rois[0]
fig1=plt.figure(figsize=figsize)
poststim1=2
for if1, cfun in enumerate(functionName):
    
    ax1 = fig1.add_axes([ leftmargin , 1-((if1+1)*(graphsizeY+spacingT))-topmargin, 0.9, graphsizeY])
    
    min1=[] #get the y min values
    max1 = [] #get the y max values
    for ig1, cgen in enumerate(genotypes):
        index = cIndex['index'].loc[cgen]
        if len(index) > 0:
            data1 = np.vstack(exceldata[roi+'-RawTimeSeries'].loc[index].values)
            #plot line for each stim time
            dataAllTimes = []
            for is1, cstim in enumerate(stimTimesShort):
                data1=np.vstack(data1) 
                prestart = np.argmax(FrameTimeRange > cstim-prestim)
                stop = np.argmax(FrameTimeRange > stopStim[is1] +poststim1)
                #stop = np.argmax(FrameTimeRange > stopStim[is1])
                start = np.argmax(FrameTimeRange > cstim)
                dFF = imageDB.fluorescent(data1[:, prestart : stop], np.nanmin(exceldata['Offset1'].values), start-prestart, signsDict[cgen],  framerate=1/8.9)
                dataAllTimes.append(eval('dFF.' + cfun + '()'))
                
            #graphing data
            #get stim times
            #find the time from when the
            xrange = []
            if inclusion['Stim Protocol'] == 'campari expts':
                for ci, cstim in  enumerate(stimTimesShort):
                    xrange.append(stopStim[ci]-cstim )
            elif inclusion['Stim Protocol'] == 'ramp':
                index11 = stimTimesShort*100
                xrange = voltage[index11.astype(int)]
                    
            gdata=np.vstack(dataAllTimes)
            gdataMed= np.nanmedian(gdata, axis=1)
        
            #plot line
            ax1.plot(xrange, gdataMed, linewidth = 2, color=colors2[cgen])
            #plot markers
            ax1.plot(xrange, gdataMed, 'o', color=colors2[cgen], markersize =10)
            min1.append(np.nanmin(gdataMed))
            max1.append(np.nanmax(gdataMed))
            #write in SE bars
            for xi, x1 in enumerate(xrange):
                SE = np.nanstd(gdata[xi, :]) / np.sqrt(sum(~np.isnan(gdata[xi, :] )))
                plt.plot([x1, x1], [gdataMed[xi]-SE, gdataMed[xi]+SE], color=colors2[cgen], alpha = 0.35, linewidth = 2.5)
                
    plt.title(columns1[if1])
    plt.xlim([xrange[0]-xrange[0]*0.1, xrange[-1]+xrange[-1]*0.1])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ybuffer =0.1*(np.nanmax(max1)-np.nanmin(min1))
    ax1.set_ylim([np.nanmin(min1)-ybuffer, np.nanmax(max1)+ybuffer])
    ax1.locator_params(axis='y',nbins=4)
    #range = np.arange(np.min(min1), np.max(max1), (np.max(max1)-np.min(min1))/4.0)
    if inclusion['Stim Protocol'] == 'ramp':
        ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1e'))
    yticks1 = ax1.get_yticks()
    ax1.set_yticks(yticks1[1:])
    ax1.set_xticks(xrange)
    
plt.savefig(saveFig + '/'  + 'StatisticsByStimDuration'  + '.svg')
            
 
##paper: graphing line plot with increasing stim times
matplotlib.rcParams.update({'font.size':6})
matplotlib.rcParams.update({'font.family': 'sans-serif'})
matplotlib.rcParams.update({'font.sans-serif': 'Arial'})
functionName = ['Max', 'delay2Max', 'SNR']
columns1 = ['Maximum Absolute dFF', 'Delay to Peak Signal', 'Signal to Noise'] # columngs for the dataframe holding data
stimTimesShort = startStim[:-1]
plt.close('all')
spacingL = 0.01
spacingT =0.08
leftmargin = 0.09
topmargin = 0.02
numofplotsX =4
graphsizeX = 0.9/(numofplotsX+ spacingL*numofplotsX )
graphsizeY = 0.7/(len(functionName) + spacingT*len(functionName))
figsize = [2.5, 1.]
plt.close('All')
roi = rois[0]

poststim1=2
for if1, cfun in enumerate(functionName):
    fig1=plt.figure(figsize=figsize)
    ax1 = fig1.add_axes([0.05, 0.05, 0.9, 0.9])
    
    min1=[] #get the y min values
    max1 = [] #get the y max values
    for ig1, cgen in enumerate(genotypes):
        index = cIndex['index'].loc[cgen]
        if len(index) > 0:
            data1 = np.vstack(exceldata[roi+'-RawTimeSeries'].loc[index].values)
            #plot line for each stim time
            dataAllTimes = []
            for is1, cstim in enumerate(stimTimesShort):
                data1=np.vstack(data1) 
                prestart = np.argmax(FrameTimeRange > cstim-prestim)
                stop = np.argmax(FrameTimeRange > stopStim[is1] +poststim1)
                #stop = np.argmax(FrameTimeRange > stopStim[is1])
                start = np.argmax(FrameTimeRange > cstim)
                dFF = imageDB.fluorescent(data1[:, prestart : stop], np.nanmin(exceldata['Offset1'].values), start-prestart, signsDict[cgen],  framerate=1/8.9)
                dataAllTimes.append(eval('dFF.' + cfun + '()'))
                
            #graphing data
            #get stim times
            #find the time from when the
            xrange = []
            if inclusion['Stim Protocol'] == 'campari expts':
                for ci, cstim in  enumerate(stimTimesShort):
                    xrange.append(stopStim[ci]-cstim )
            elif inclusion['Stim Protocol'] == 'ramp':
                index11 = stimTimesShort*100
                xrange = voltage[index11.astype(int)]
                    
            gdata=np.vstack(dataAllTimes)
            gdataMed= np.nanmedian(gdata, axis=1)
        
            #plot line
            ax1.plot(xrange, gdataMed, linewidth = 1, color=colors2[cgen])
            #plot markers
            ax1.plot(xrange, gdataMed, 'o', color=colors2[cgen], markersize =2)
            min1.append(np.nanmin(gdataMed))
            max1.append(np.nanmax(gdataMed))
            #write in SE bars
            for xi, x1 in enumerate(xrange):
                SE = np.nanstd(gdata[xi, :]) / np.sqrt(sum(~np.isnan(gdata[xi, :] )))
                plt.plot([x1, x1], [gdataMed[xi]-SE, gdataMed[xi]+SE], color=colors2[cgen], alpha = 0.35, linewidth = 1.5)
                
    plt.title(columns1[if1])
    plt.xlim([xrange[0]-xrange[0]*0.1, xrange[-1]+xrange[-1]*0.1])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ybuffer =0.1*(np.nanmax(max1)-np.nanmin(min1))
    ax1.set_ylim([np.nanmin(min1)-ybuffer, np.nanmax(max1)+ybuffer])
    ax1.locator_params(axis='y',nbins=4)
    #range = np.arange(np.min(min1), np.max(max1), (np.max(max1)-np.min(min1))/4.0)
    if inclusion['Stim Protocol'] == 'ramp':
        ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1e'))
    yticks1 = ax1.get_yticks()
    ax1.set_yticks(yticks1[1:])
    ax1.set_xticks(xrange)
    
    plt.savefig(saveFig + '/'  + 'StatisticsByStimDuration' + cfun + '.svg')


'''



















  
        
    
