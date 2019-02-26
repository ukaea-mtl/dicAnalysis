### Import Packages

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

from scipy import stats
# from scipy.stats import binned_statistic

import matplotlib.cm as cmx
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.tri as mtri
from matplotlib.tri.triangulation import Triangulation


### Data class

class dic3D:

    def __init__(self, dataFile):
        
        # load data
        self.data = pd.read_csv(dataFile, encoding = "latin1")
        self.x = self.data['x [mm]']
        self.y = self.data['y [mm]']
        self.z = self.data['z [mm]']

        self.xDim = np.max(self.x)-np.min(self.x)
        self.yDim = np.max(self.y)-np.min(self.y)
        
        self.meanMaxSS = np.nanmean(self.data['maximum shear strain [S]'])
    
    def info(self):
        print(self.data.info())
        
    def scatter2d(self, x=None, y=None, cs=None, colorsMap='viridis',vmin=0,vmax=0.005, s=1, label=None, axisVis="off"):
        
        # define terms
        if x==None:
            x=self.data['x [mm]']
            y=self.data['y [mm]']
        if cs == None:
            cs=self.data['maximum shear strain [S]']
            label='$\\gamma$ $_{shear}^{max}$ [  ]'
        xDim = np.max(x)-np.min(x)
        yDim = np.max(y)-np.min(y)
        
        # plot data
        fig = plt.figure(figsize=(xDim+1,yDim+1))
        ax = fig.add_subplot(111)
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        ax.scatter(x,y,c=scalarMap.to_rgba(cs),s=s)
        
        ax.axis(axisVis)
        scalarMap.set_array(cs)
        cb = fig.colorbar(scalarMap,shrink=0.8)
        cb.set_label(label=label,size=12)
        
        
    def scatter3d(self, x=None, y=None, z=None, cs=None, colorsMap='viridis', vmin=0, vmax=0.005, label=None):
        
        # define terms
        if x==None:
            x=self.data['x [mm]']
            y=self.data['y [mm]']
            z=self.data['z [mm]']
        if cs==None:
            label = '$\\gamma$ $_{shear}^{max}$ [  ]'
            cs=self.data['maximum shear strain [S]']
        
        # plot data
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
        scalarMap.set_array(cs)
        
        # scalebar
        fig.colorbar(scalarMap,shrink=0.6,label=label)
        
        plt.show()


    def lineariseValues(self, dists=None,plotVals=None,nbins=100,plotData2D=False,plotData3D=False,newFig=True,timeStep=1,
                       xLabel=None,yLabel=None,zLabel=None):

        # define data
        if dists is None:
            dists = self.data['y [mm]']
        if plotVals is None:
            plotVals = self.data['maximum shear strain [S]']

        # initialise bins
        bins = np.linspace(np.min(dists), np.max(dists), nbins)
        plotVals_binned = []
        for i in range(nbins):
            plotVals_binned.append([])

        # bin the plot data according to the distance bins
        for dist,plotVal in zip(dists,plotVals):
            for i,binVal in enumerate(bins[:-1]):
                if bins[i] <= dist < bins[i+1]:
                    plotVals_binned[i].append(plotVal)

        # get stats on data
        plotVals_binned_means=[]
        plotVals_binned_stds=[]
        for plotVal in plotVals_binned:

            if len(plotVal) > 0:
                meanVal = np.nanmean(plotVal)
                stdVal = np.nanstd(plotVal)
            else:
                meanVal = np.nan
                stdVal = np.nan

            plotVals_binned_means.append(meanVal)
            plotVals_binned_stds.append(stdVal)

        self.linearised_bins=bins
        self.linearised_plotValsMeans=plotVals_binned_means
        self.linearised_plotValsStds=plotVals_binned_stds
        
        
        # plot data in 3D (with timeStep as third dimension)
        if plotData3D:
            
            if newFig:
                fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

            y = self.linearised_bins
            y = np.max(y) - y
            x = np.ones(len(y))*timeStep
            z = self.linearised_plotValsMeans

#             ax.plot(x,y,z)

            for i in range((len(x))-1):
                ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.viridis(z[i]/np.nanmax(z)))
            
#             plt.set_xlabel('x')
#             plt.set_ylabel('y')
#             plt.set_zlabel('z')
            
        # plot data in 2D (distance vs. values)
        if plotData2D:
            
            if newFig:
                fig, ax = plt.subplots()

            x = np.max(self.linearised_bins) - self.linearised_bins
            plt.errorbar(x, self.linearised_plotValsMeans, yerr=self.linearised_plotValsStds, fmt='.')
            
    def plotHistogram(self, data=None,xScale='linear',yScale='linear',xLabel=None,yLabel=None,plot=True,bins=30):

        if data is None:
            data = self.data['maximum shear strain [S]']

        hist = np.histogram(data,bins=bins)
        x = 0.5*(hist[1][1:]+hist[1][:-1])
        y = hist[0]/np.nansum(hist[0])
        
        self.hist_x = x
        self.hist_y = y

        if plot:
            
            fig, ax = plt.subplots()
            
            ax.plot(x,y)

            ax.set_xscale(xScale)
            ax.set_yscale(yScale)
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)

	## Functions that allows user to click on a linescan image multiple times and store the coordinates such that a plot in another dimrension can be obtained

    def clickLineScan(self,x=None,y=None):
        if x is None:
            x = self.linearised_bins
        if y is None:
            y = self.linearised_plotValsMeans

        # plot the figure
        global fig
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.plot(x,y)

        self.clickCoords = []

        # Call click func
        global cid
        cid = fig.canvas.mpl_connect('button_press_event', self.onClickLineScan)

        plt.show(1)


    def onClickLineScan(self, event):

        global ix, iy
        ix, iy = event.xdata, event.ydata

        # print 'x = %d, y = %d'%(
        #     ix, iy)

        # assign global variable to access outside of function
        #         global self.coords
        self.clickCoords.append((ix, iy))

    #     # Disconnect after 3 clicks
    #     if len(self.clickCoords) == 3:
    #         fig.canvas.mpl_disconnect(cid)
    #         plt.close(1)
        return
 
    def plotColourbar(self, vMin=None, vMax=None, colorsMap=None, label=None,shrink=0.8):

        if colorsMap is None:
            colorsMap="rainbow"
        if vMin is None:
            vMin = 0
        if vMax is None:
            vMax = np.nanmax(self.data['maximum shear strain [S]'])

        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=float(vMin), vmax=float(vMax))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

        scalarMap.set_array(colorsMap)
        plt.colorbar(scalarMap,shrink=shrink,label=label)



    # TRI
    
    def onclick(self,event):
        global ix, iy
        ix, iy = event.xdata, event.ydata

        # print 'x = %d, y = %d'%(
        #     ix, iy)

        # assign global variable to access outside of function
#         global self.coords
        self.clickCoords.append((ix, iy))

        # Disconnect after 3 clicks
        if len(self.clickCoords) == 3:
            fig.canvas.mpl_disconnect(cid)
            plt.close(1)
        return

    def find_nearest(self,array,value):
        idx = (np.abs(array-value)).argmin()
        return array[idx]
    
    def define_circle(self,p1, p2, p3):
        """
        Returns the center and radius of the circle passing the given 3 points.
        In case the 3 points form a line, returns (None, infinity).
        """
        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

        if abs(det) < 1.0e-6:
            return (None, np.inf)

        # Center of circle
        cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

        radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
        return ((cx, cy), radius)
    
    def clickDefineNotch(self,x=None,y=None):
        
        if x is None:
            x = self.linearised_bins
        if y is None:
            y = self.linearised_plotValsMeans
            
        # plot the figure
        global fig
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.plot(x,y)

        self.clickCoords = []

        # Call click func
        global cid
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)

        plt.show(1)
        
        
    def findBridgman(self,r_core=None,r_notch=None):
        eta = (1/3) + np.log((r_core/(2*r_notch))+1)
        return eta
        
    def findBridgmanTriaxiality(self,x=None,y=None,coords=None,plotData=False,thickness_initial=2):
        
        if coords is None:
            coords = self.clickCoords
        if x is None:
            x = self.linearised_bins
        if y is None:
            y = self.linearised_plotValsMeans
            
        # find and the data points that will define the notch
        x0 = self.find_nearest(x,coords[0][0])
        x1 = self.find_nearest(x,coords[1][0])
        x2 = self.find_nearest(x,coords[2][0])

        x0_idx = x.tolist().index(x0)
        x1_idx = x.tolist().index(x1)
        x2_idx = x.tolist().index(x2)

        y0 = y[x0_idx]
        y1 = y[x1_idx]
        y2 = y[x2_idx]

        xVals=[x0,x1,x2]
        yVals=[y0,y1,y2]

        # plt.plot(x,y)
        # plt.scatter(xVals,yVals)

        xVals, yVals = zip(*sorted(zip(xVals, yVals)))

        x0_ = xVals[0]
        y0_ = (yVals[0]+yVals[2])/2

        x1_ = (xVals[0]+xVals[2])/2
        y1_ = yVals[1]

        x2_ = xVals[2]
        y2_ = (yVals[0]+yVals[2])/2

        xVals=[x0_,x1_,x2_]
        yVals=[y0_,y1_,y2_]

        # plt.scatter(xVals,yVals)
        
        # find the circle centre and radius
        centre,radius = self.define_circle((x0_,y0_),(x1_,y1_),(x2_,y2_))
        
        self.notchRadius = radius
        self.NotchCentre = centre
        
        if plotData:
            fig, ax = plt.subplots()
            circle1 = plt.Circle(centre,radius, edgecolor='C1',linestyle='--',fill=False)
            ax.add_artist(circle1)
            ax.scatter(xVals,yVals)
        
        thickness_initial = thickness_initial-abs(y0) #mm
        thickness_final = thickness_initial - abs(y0_-y1)
        r_final = thickness_final/2
        
        eta = self.findBridgman(r_core=r_final,r_notch=radius)
        
        self.bridgmanTriaxiality = eta

### End