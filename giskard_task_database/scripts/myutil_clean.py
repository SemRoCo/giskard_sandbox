"""Contains various helpful methods that don't belong to a specific class

Shortened version for giskard
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class DataUtil:
    """Contains all utility functions for working with data (text, MongoDB and otherwise)"""

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