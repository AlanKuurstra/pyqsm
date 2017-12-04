# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:48:10 2016

@author: Alan
"""
import numpy as np

def gradientWeights(iMag,Mask,voxel_size,thresh):
    WGradx,WGrady,WGradz=np.gradient(iMag,*voxel_size)
    WGradx=(Mask>0)*WGradx
    #imshow3d(W1x)    
    WGradx=WGradx<thresh
    #imshow3d(W1x)
    WGrady=(Mask>0)*WGrady
    WGrady=WGrady<thresh
    WGradz=(Mask>0)*WGradz
    WGradz=WGradz<thresh
    return WGradx,WGrady,WGradz