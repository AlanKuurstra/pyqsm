# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:42:21 2016

@author: Alan
"""
import numpy as np

def dipole_kernel(pixelSize,FOV):
    nx,ny,nz=pixelSize
    FOVx,FOVy,FOVz=FOV
    
    kx=np.arange(-np.ceil((nx-1)/2.0),np.floor((nx-1)/2.0)+1)*1.0/FOVx
    ky=np.arange(-np.ceil((ny-1)/2.0),np.floor((ny-1)/2.0)+1)*1.0/FOVy
    kz=np.arange(-np.ceil((nz-1)/2.0),np.floor((nz-1)/2.0)+1)*1.0/FOVz
    
    KX,KY,KZ=np.meshgrid(kx,ky,kz)
    KX=KX.transpose(1,0,2)
    KY=KY.transpose(1,0,2)
    KZ=KZ.transpose(1,0,2)
    
    K2=KX**2+KY**2+KZ**2
    
    dipole_f=1.0/3-KZ**2/K2
    dipole_f=np.fft.ifftshift(dipole_f) #note ifftshift([-2,-1,0,1,2])=[0,1,2,-2,-1]
    dipole_f[0,0,0]=0
    dipole_f=dipole_f.astype('complex')
    return dipole_f

