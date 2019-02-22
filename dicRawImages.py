#####
# Module for hacking LaVision images and masks in .im7 and .vc7 format, respectively.
#####

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import ReadIM


# Data class

class dicRawImages:

    def __init__(self, dataFile):
        
        # load data
        vbuff, vatts = ReadIM.extra.get_Buffer_andAttributeList(dataFile)
        v_array, vbuff = ReadIM.extra.buffer_as_array(vbuff)
        del(vbuff)

        self.data = v_array
        
    def plotData(self,cmap=cm.Greys_r):
        my_dpi=96
        fig = plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
        plt.imshow(self.data[0], cmap = cmap)
#         return fig
    
    def saveData(self,filePath):
        # define the pixels in a list of lists
        pixels = self.data[0].tolist()

        # Convert the pixels into an array using numpy
        array = np.array(pixels)
        array_max = array.max()
        image_max = 2**16
        fac = image_max/array_max

        array_ = (array*fac).astype(int)

        # Use PIL to create an image from the new array of pixels
        new_image = Image.fromarray(array_)
        new_image.save(filePath)
        

