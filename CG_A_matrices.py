# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:16:24 2016

@author: Alan
"""

import numpy as np
from dipole_convolve import dipole_convolve as dipole_convolve
from laplacian import laplacian as laplacian

def Apply_A_dp_invert_w_l2_grad(p,Apply_A_variables, lamda, alpha=1, preAllocatedAp=None):
    
    dipole_f=Apply_A_variables['dipole_f']
    WData_sqrd=Apply_A_variables['WData_sqrd']
    WGradx_sqrd=Apply_A_variables['WGradx_sqrd']
    WGrady_sqrd=Apply_A_variables['WGrady_sqrd']
    WGradz_sqrd=Apply_A_variables['WGradz_sqrd']
        
    p_x=np.roll(p,-1,axis=0)-p
    p_y=np.roll(p,-1,axis=1)-p
    p_z=np.roll(p,-1,axis=2)-p

    A=WGradx_sqrd*p_x 
    B=WGrady_sqrd*p_y 
    C=WGradz_sqrd*p_z     

    A=np.roll(A,1,axis=0)-A
    B=np.roll(B,1,axis=1)-B
    C=np.roll(C,1,axis=2)-C
    if preAllocatedAp is not None:
        preAllocatedAp[:,:,:]=alpha*dipole_convolve(WData_sqrd*dipole_convolve(p,dipole_f),dipole_f) + lamda*(A+B+C)
    else:
        return alpha*dipole_convolve(WData_sqrd*dipole_convolve(p,dipole_f),dipole_f) + lamda*(A+B+C)  
        
#this is the only original idea that works well
def Apply_A_dp_invert_w_laplacian_fidelity_l2_grad(p,Apply_A_variables, lamda, alpha=1, preAllocatedAp=None):    
    dipole_f=Apply_A_variables['dipole_f']    
    WData_sqrd=Apply_A_variables['WData_sqrd']
    WGradx_sqrd=Apply_A_variables['WGradx_sqrd']
    WGrady_sqrd=Apply_A_variables['WGrady_sqrd']
    WGradz_sqrd=Apply_A_variables['WGradz_sqrd']
        
    p_x=np.roll(p,-1,axis=0)-p
    p_y=np.roll(p,-1,axis=1)-p
    p_z=np.roll(p,-1,axis=2)-p

    A=WGradx_sqrd*p_x 
    B=WGrady_sqrd*p_y 
    C=WGradz_sqrd*p_z     

    A=np.roll(A,1,axis=0)-A
    B=np.roll(B,1,axis=1)-B
    C=np.roll(C,1,axis=2)-C
    
    
    tmp=dipole_convolve(p,dipole_f)
    tmp=-6*tmp+np.roll(tmp,-1,axis=0)+np.roll(tmp,1,axis=0)+np.roll(tmp,-1,axis=1)+np.roll(tmp,1,axis=1)+np.roll(tmp,-1,axis=2)+np.roll(tmp,1,axis=2)
    tmp*=WData_sqrd
    tmp=-6*tmp+np.roll(tmp,-1,axis=0)+np.roll(tmp,1,axis=0)+np.roll(tmp,-1,axis=1)+np.roll(tmp,1,axis=1)+np.roll(tmp,-1,axis=2)+np.roll(tmp,1,axis=2)
    tmp=dipole_convolve(tmp,dipole_f)
    
    
    if preAllocatedAp is not None:
        preAllocatedAp[:,:,:]=alpha*tmp + lamda*(A+B+C)
    else:
        return alpha*tmp + lamda*(A+B+C)

def Apply_A_dipole_projection(p,Apply_A_variables,lamda,alpha=1,preAllocatedAp=None):
    #note that lamda and alpha are dummy variables not used
    #use dipoles outside the ROI to fit the field inside of the ROI
    dipole_f=Apply_A_variables['dipole_f']
    WData_sqrd=Apply_A_variables['WData_sqrd'] #which data should be used for fidelity (field inside the ROI)
    WDipoles=Apply_A_variables['WDipoles'] #dipole locations for projection (dipoles outside the ROI)
    if preAllocatedAp is not None:
        preAllocatedAp[:,:,:]=WDipoles*dipole_convolve(WData_sqrd*dipole_convolve(WDipoles*p,dipole_f),dipole_f)
    else:
        return WDipoles*dipole_convolve(WData_sqrd*dipole_convolve(WDipoles*p,dipole_f),dipole_f)
        
def Apply_A_bg_removal_l2_laplacian(p,Apply_A_variables,lamda,alpha=1,preAllocatedAp=None):
    WData_sqrd=Apply_A_variables['WData_sqrd']
    WLaplacian_sqrd=Apply_A_variables['WLaplacian_sqrd']
    if preAllocatedAp is not None:
        preAllocatedAp[:,:,:]=alpha*WData_sqrd*p+lamda*laplacian(WLaplacian_sqrd*laplacian(p))  
    else:
        return alpha*WData_sqrd*p+lamda*laplacian(WLaplacian_sqrd*laplacian(p))

def Apply_A_bg_removal_l2_laplacian_mask_laplacian(p,Mask_sqrd,lamda,alpha=1,preAllocatedAp=None):
    if preAllocatedAp is not None:
        preAllocatedAp[:,:,:]=alpha*p+lamda*laplacian(Mask_sqrd*laplacian(p))  
    else:
        return alpha*p+lamda*laplacian(Mask_sqrd*laplacian(p))   

def Apply_A_bg_removal_l2_laplacian_mask_fidelity(p,Mask_sqrd,lamda,alpha=1,preAllocatedAp=None):
    if preAllocatedAp is not None:
        preAllocatedAp[:,:,:]=alpha*Mask_sqrd*p+lamda*laplacian(laplacian(p))  
    else:
        return alpha*Mask_sqrd*p+lamda*laplacian(laplacian(p))     
        
def Apply_A_dp_invert_l2_laplacian(p,Apply_A_variables,lamda,alpha=1,preAllocatedAp=None):
    WData_sqrd=Apply_A_variables['WData_sqrd']
    WBext_sqrd=Apply_A_variables['WBext_sqrd']
    if preAllocatedAp is not None:
        preAllocatedAp[:,:,:]=WData_sqrd*p+lamda*laplacian(WBext_sqrd*laplacian(p))  
    else:
        return WData_sqrd*p+lamda*laplacian(WBext_sqrd*laplacian(p))  
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    