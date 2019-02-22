#####
# Module for display and analysis of extensometry data
#####

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import pandas as pd

class extensometryData:

    def __init__(self, dataFile):

        # load data
#         self.data = pd.read_csv(dataFile, encoding = "latin1",delimiter=";",skiprows=range(1, 2))
        
        self.time = []
        data = pd.read_csv(dataFile, encoding = "latin1",delimiter=";",skiprows=range(1, 2),usecols=[0])
        data = data.values
        for i in data:
            for i_ in i:
                self.time.append(i_)
        self.time = np.asarray(self.time)
            
        self.l0 = []
        data = pd.read_csv(dataFile, encoding = "latin1",delimiter=";",skiprows=range(1, 2),usecols=[1])
        data = data.values
        for i in data:
            for i_ in i:
                self.l0.append(i_)
        self.l0 = np.asarray(self.l0)
            
        self.lt = []
        data = pd.read_csv(dataFile, encoding = "latin1",delimiter=";",skiprows=range(1, 2),usecols=[2])
        data = data.values
        for i in data:
            for i_ in i:
                self.lt.append(i_)
        self.lt = np.asarray(self.lt)
        
        # remove NANs
        timeTemp,l0Temp,ltTemp = [],[],[]
        for t,l0,lt in zip(self.time,self.l0,self.lt):
            if not np.isnan(t) and not np.isnan(l0) and not np.isnan(lt):
                timeTemp.append(t)
                l0Temp.append(l0)
                ltTemp.append(lt)
        self.time = np.asarray(timeTemp)
        self.l0 = np.asarray(l0Temp)
        self.lt = np.asarray(ltTemp)
                
        self.extension = self.lt - self.l0
        self.strain = self.extension/self.l0
        
    def trimData(self,minVal,maxVal):
        m = minVal
        n = maxVal
        
        self.time = self.time[m:n]
        self.l0 = self.l0[m:n]
        self.lt = self.lt[m:n]
        self.extension = self.extension[m:n]
        self.strain = self.strain[m:n]