# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:03:02 2016

@author: Alan
"""

import numpy as np

def laplacian(tmp):
    return -6*tmp+np.roll(tmp,-1,axis=0)+np.roll(tmp,1,axis=0)+np.roll(tmp,-1,axis=1)+np.roll(tmp,1,axis=1)+np.roll(tmp,-1,axis=2)+np.roll(tmp,1,axis=2)
