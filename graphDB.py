import matplotlib.pyplot as plt 
import numpy as np
import scipy

def scatterBarPlot(xcoord, ydata, dist1=0.3, markersize = 20):
    #xcoord = x coord to place data
    #data = ydata at xcoord
    #dist1 = width for data
    #markersize = size for idividual points

    lim1=[np.min(ydata), np.max(ydata)]
    bins=10
    yrange1 = np.linspace(lim1[0], lim1[1], bins) #
    binned, range1 = np.histogram(ydata,yrange1)

    xlist=[]
    ylist=[]

    #plot individual data points
    #through the y-range plot points for each region
    for c1, cx in enumerate(yrange1[1:]):
        #import pdb; pdb.set_trace()
       indx=np.where(np.logical_and(ydata >= yrange1[c1],  ydata <= cx)) #looking for points within the given y region
       
       if indx[0].size > 0: #has to have at least one point
           # get first point
           xlist.append(xcoord)
           ylist.append(ydata[indx[0][0]])
           alt1=1
           distA=0
           distB=0
           dist2 = dist1 / len(indx[0])
           for i in indx[0][1:]: #running through each point in y region to determine x-coord
               if alt1 == 1:
                   distA = distA + dist2
                   xlist.append(xcoord + distA)
                   alt1 = -1
               else:
                   distB = distB - dist2
                   xlist.append(xcoord + distB)
                   alt1 = 1
               ylist.append(ydata[i])
    
    plt.scatter(xlist, ylist, s = 20, color = 'b', facecolor = 'none')
    plt.plot([xcoord-dist1, xcoord+dist1],[np.mean(ydata), np.mean(ydata)], color = 'k', linewidth = 2 )
    plt.plot([xcoord-dist1, xcoord+dist1],[np.median(ydata), np.median(ydata)], color = 'g', linewidth = 1.5 )
    plt.plot([xcoord, xcoord], [np.mean(ydata) - np.std(ydata), np.mean(ydata) + np.std(ydata)], color = '0.7', linewidth =3, alpha = 0.5)
    plt.plot([xcoord, xcoord], [np.mean(ydata) - scipy.stats.sem(ydata), np.mean(ydata) + scipy.stats.sem(ydata)], color = 'k', linewidth =1.5, alpha = 0.8)

#plot the data with SEM and mean
def lineSEM(x, data, color, ax, label = None):
    #if getting values from data use np.transpose before passing
    mean1 = []
    sem1 = []
    for i, col in enumerate(data):
        col = np.array(col)
        col = col[~np.isnan(col)]
        mean1.append(np.mean(col))
        sem1.append(scipy.stats.sem(col))
    ax.plot(x, mean1, color = color, label=label)

    for i, m, in enumerate(mean1):
        ax.plot([x[i], x[i]], [m+sem1[i], m-sem1[i]], color =color, alpha =0.3, linewidth = 1)
        
def simplePlot(data, colors = ['r', 'y', 'g', 'b', 'k']):
    for i, line in enumerate(data):
        plt.plot(line, color = colors[i])

def barSEM(x, data, color = 'b'):
    plt.bar(x, np.mean(data), facecolor = color, align = 'center')
    plt.plot([x,x], [np.mean(data), np.mean(data) + np.std(data)/np.sqrt(len(data))], color ='k', linewidth = 2)
    
def timeBar(start, stop, length, width, figsize, savefile):
    fig1 = plt.figure(figsize =figsize)
    ax1 = fig1.add_axes([0.05, 0.05, 0.9, 0.9])
    ax1.plot([0, length], [0,0], color = 'k', linewidth =2)
    for i, cstart in enumerate(startStim):
        ax1.plot([cstart, stop[i] ], [0, 0], color = 'r', linewidth=2)
    ax1.axis('off')
    plt.savefig(savefile)
    
def SEbarsToLinePlot(data, color, axes):
    #data = np array ploted along axis=1
    for i in range(data.shape[1]):
        se = np.std(data[:, i]) / np.sqrt(data.shape[0])
        me = np.mean(data[:, i], axis=0)
        axes.plot([i, i], [me-se, me+se], alpha = 0.4, color = 'r', linewidth=0.2)


def SElinePlotFrame(dseries, color, ax1, label1):
    #dseries = pandas series
    #x = dseries.index
    #y = data
    median = []
    SE = []
    for cindex in dseries.index:
        data = dseries[cindex]
        data = data[~np.isnan(data)]
        #median.append(np.median(data))
        median.append(np.mean(data))
        SE.append (scipy.stats.sem(data))
    #plot median values
    x = dseries.index
    ax1.plot(x, median, '-o', color = color, label=label1)
    #plot standard error bars
    for i, cSE in enumerate(SE):
        ax1.plot([x[i], x[i]], [median[i] - SE[i], median[i]+SE[i]], color=color, alpha = 0.5)

    
def getColors(num):
    #get num of distinct colors
    cm = plt.get_cmap('gist_rainbow')
    cNorm = matplotlib.colors.Normalize(vmin = 0, vmax=num-1)
    scalarMap = matplotlib.cm.ScalarMappable(norm = cNorm, cmap = cm)
    return [scalarMap.to_rgba(i) for i in range(num)]
        
