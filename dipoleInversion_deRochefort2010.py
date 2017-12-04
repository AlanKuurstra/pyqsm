# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:40:20 2016

@author: Alan

||DX-B_loc||_2 + lamda ||Grad X||_2

TODO: take voxel size into account in the TV...gradients should take spacing into account
"""

import numpy as np
from dipole_kernel import dipole_kernel
from dipole_convolve import dipole_convolve
from fidelity_weighting import dataFidelityWeight as dataFidelityWeight
from gradient_weighting import gradientWeights as gradientWeights
from CG_engine import CG_engine
    
def getXmap(iMag,Mask,voxel_size,B_loc,lamda=1,maxiter=100):
    #create gradient weights that preserve edges
    WData_sqrd=dataFidelityWeight(iMag)    
    WGradx,WGrady,WGradz=gradientWeights(iMag,Mask,voxel_size,2) #the user might want to make these masks manually...because you gotta twiddle with the threshold
    WGradx_sqrd=WGradx**2
    WGrady_sqrd=WGrady**2
    WGradz_sqrd=WGradz**2
    
    #dipole kernel
    dipole_f=dipole_kernel(B_loc.shape,list(voxel_size*B_loc.shape))
    
    #create dictionary for Apply_A
    Apply_A_variables={\
    'dipole_f':dipole_f,\
    'WData_sqrd':WData_sqrd,\
    'WGradx_sqrd':WGradx_sqrd,\
    'WGrady_sqrd':WGrady_sqrd,\
    'WGradz_sqrd':WGradz_sqrd,\
    }
    from CG_A_matrices import Apply_A_dp_invert_w_l2_grad as Apply_A #note that this is missing one of the regularizers
    b=dipole_convolve(WData_sqrd*B_loc,dipole_f)
    Xmap=CG_engine(Apply_A,Apply_A_variables,b,lamda,maxiter=maxiter)
    return Xmap

if __name__=="__main__":
    import scipy.io as sio
    from vidi3d import imshow3d, compare3d
        
    #import some data
    fileloc='/cfmm/data/akuurstr/code/matlab/SFMM/RDF.mat'
    imgs=sio.loadmat(fileloc)
    CF=imgs['CF'][0][0]
    Mask=np.ascontiguousarray(imgs['Mask'])
    iMag=np.ascontiguousarray(imgs['iMag'])
    RDF_orig=np.ascontiguousarray(imgs['iFreq'].astype('complex'))
    RDF_filtered=np.ascontiguousarray(imgs['RDF'].astype('complex'))    
    delta_TE=imgs['delta_TE'][0][0]
    voxel_size=imgs['voxel_size'].T[0]   
    
    #filter to remove bg....replace with better algorithm after we have the code working
    import scipy.ndimage
    B_loc=RDF_filtered-scipy.ndimage.filters.gaussian_filter(np.real(RDF_filtered),np.around(1.0/voxel_size))
        
    result=getXmap(iMag,Mask,voxel_size,B_loc,maxiter=20)    
    savefileloc='/Users/Alan/python/QSM/initial_tests'
    tmp=np.load(savefileloc+'/alpha_1.0_lam_1.0_filter_1.0'+'.npy')
    compare3d((tmp/CF/delta_TE*1e6,result/CF/delta_TE*1e6))    
