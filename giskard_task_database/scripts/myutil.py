"""Contains various helpful methods that don't belong to a specific class"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class DataUtil:
    """Contains all utility functions for working with data (text, MongoDB and otherwise)"""
    
    @staticmethod
    def readdata(filename):
        """Read constraint data
        Returns: [columnnames, data], where columnnames is a list and data is a 2d numpy array"""
        # Read column headers (to be variable naames)
        with open(filename) as f:
            firstline = f.readline()                    # Read first line of csv
            firstline = firstline.replace("\n","")      # Remove new line characters
            firstline = firstline.replace(" ","")       # Remove spaces
            column_headers = firstline.split(",")        # Get array of column headers
        # Read in the data (omitting the first row containing column headers)
        rawdata=np.genfromtxt(filename,skip_header=1,delimiter=",",dtype=None)
    
        # Assign the data to arrays
        data=list()
        for i in xrange(len(column_headers)):
            #vars()[Var]=[x[Ind] for x in rawdata]         # Assign the columns of the data to variables names after the column headers - this creates the columns as separate variables
            data.append([x[i] for x in rawdata])   # Creates 2d array with columns of data stored per row, for example, all data under 'episode' heading can be easily accessed as data[0] now
        
        tdata = np.array(data) #convert to numpy array for easier access over columns/rows
        rdata = np.transpose(tdata) #transpose data so that one row is a single observation (historically it used to be useful to be able to access one column of data as data[0], but this is because we didn't use np.array)
        #return column names and the datamatrix
        return [column_headers, rdata]
    
def removeObs(columnnames, data, column_oi, val):
    """ Remove rows (observations) from data in which column_oi(string of columnname) has value val
    Returns: data, a 2d numpy array
    """
    #Get the index of the column of interest
    if not column_oi in columnnames:
        print "Tried to remove " + column_oi + "=" + str(val) + " from data, but " + column_oi + " does not exist"
        return data
    else: 
        idx_coi = columnnames.index(column_oi)
        res = [data[irow,:] for irow in range(0,data.shape[0]) if not data[irow,idx_coi]==val]
        res = np.array(res)
        return res

def plotBoxplot(data_to_plot, xlabel, ylabel, columnnames=None):
    plt.boxplot(data_to_plot)
    xtickvals = [x for x in xrange(data_to_plot.shape[1])]
    if not columnnames == None:
        plt.xticks(xtickvals, columnnames)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def adjustedRS(r2, N, P):
    """Adjusts R^2 to take into account the number of predictor variables (degrees of freedom) and number of observations, given that R^2 is already given
    R2: R^2
    
    Returns: Adjusted R^2
    """
    #r2 = r2_score(y, y_pred)
    if N<=(P+1):
        print "WARNING: number of observations N is equal or smaller than the number of predictors +1. Adjusted R squared cannot be computed reliably"
        return None
    r2ad = 1-(((1-r2)*(N-1))/(N-P-1))
    return r2ad

def convertableToFloat(str_var):
    """Returns true if the given string can be converted to a float, false otherwise"""
    try:
        float(str_var)
        return True
    except ValueError:
        return False 