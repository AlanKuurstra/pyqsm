# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:42:39 2016

@author: Alan
"""

from dipole_kernel import dipole_kernel
from dipole_convolve import dipole_convolve
from CG_engine import CG_engine
from CG_A_matrices import Apply_A_dipole_projection as Apply_A

def dipole_project(fieldmap,WData,DataMask,BackgroundMask,FOV,maxiter):
    print "Performing bg removal using dipole projection"
    dipole_f=dipole_kernel(fieldmap.shape,FOV)
    WData_sqrd=DataMask*WData**2
    Apply_A_variables={}
    Apply_A_variables['dipole_f']=dipole_f
    Apply_A_variables['WData_sqrd']=WData_sqrd #which data should be used for fidelity (field inside the ROI)    
    Apply_A_variables['WDipoles']= BackgroundMask #dipole locations for projection (dipoles outside the ROI)    

    b=BackgroundMask*dipole_convolve(WData_sqrd*fieldmap,dipole_f)
    
    bg_susc=CG_engine(Apply_A, Apply_A_variables,b, 1, alpha=1, maxiter=maxiter, x_init=None)
    bg_bfield=DataMask*dipole_convolve(BackgroundMask*bg_susc,dipole_f)
    return bg_bfield,bg_susc

