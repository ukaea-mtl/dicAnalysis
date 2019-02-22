######
# Module for display and analysis of data from the Phoenix rigs
#####

class rigData:

    def __init__(self, dataFile):

        # load data
        loadData = np.loadtxt(dataFile, skiprows=1, delimiter=',', usecols=[0,1,3])

        self.time = loadData[:,0]
        self.stroke = loadData[:,1]
        self.load = loadData[:,2]
        
    def info(self):
        print("time   : "+ "% .2f" % np.nanmin(self.time) + ' - ' + "% .2f" % np.nanmax(self.time))
        print("stroke : "+ "% .2f" % np.nanmin(self.stroke) + ' - ' + "% .2f" % np.nanmax(self.stroke))
        print("load   : "+ "% .2f" % np.nanmin(self.load) + ' - ' + "% .2f" % np.nanmax(self.load))
        
    def find_nearest(self,array,value):
        idx = (np.abs(array-value)).argmin()
        return array[idx]
    
    def findNearestIndex(self,array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
    
    def trimData(self,minVal,maxVal):
        m = minVal
        n = maxVal

        self.time = self.time[m:n]
        self.stroke = self.stroke[m:n]
        self.load = self.load[m:n]

