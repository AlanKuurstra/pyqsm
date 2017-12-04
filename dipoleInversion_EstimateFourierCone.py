#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:12:37 2017

@author: akuurstr

There are some methods that suggest that solving the inverse problem of 
dipole inversion could be done in Fourier domain.  The deconvolution is 
performed by dividing the dipole kernel in Fourier domain.  The only problem
is a cone of undefined points. It has been noted that this cone introduces
streaking artefacts into the susceptibility solution. Some methods, such as 
iswim, try to reduce streaking by finding better values in the fourier cone of
the susceptibility solution.

This script investigates a way to find better values inside the fourier cone.
The idea is that since the bad cone values are causing streaking, we should find
cone values that remove streaking outside the brain. We make the assumption
is that there is no susceptibility outside the brain, and erroneous cone values
cause streaking susceptibility values outside the brain.  If we find cone values
that remove streaking outside the brain, then these values will likely be values
that remove streaking inside the brain too.

min X_f_insideCone:
|| M_insideBrain * ( IFT{ D*(X_f_insideCone+X_f_outsideCone) } - Fieldmap ) || 
+ lambda || M_outsideBrain * X_img ||

Here we assume X_f_outsideCone is known.  It is found by doing a deconvolution
in Fourier domain using the fieldmap.  Then we just try to find better values
inside the cone by finding a solution which forces the image domain susceptibility
to be zero outside the brain.

First we investigate the idea using a full fieldmap.  Then try again with a 
fieldmap measured only inside the brain.
"""

import scipy.ndimage
import scipy.io
import numpy as np
from dipole_kernel import dipole_kernel
from dipole_convolve import dipole_convolve, getFourierDomain, getImgDomain, viewF
from fidelity_weighting import dataFidelityWeight as dataFidelityWeight
from gradient_weighting import gradientWeights as gradientWeights
from CG_engine import CG_engine
from vidi3d import compare3d
from scipy.ndimage import binary_erosion, binary_dilation

def Apply_A(p,Apply_A_variables, lamda, alpha=1, preAllocatedAp=None):
    
    dipole_f=Apply_A_variables['dipole_f']
    mask_inBrain=Apply_A_variables['mask_inBrain']   
    mask_outBrain=Apply_A_variables['mask_outBrain']   
    cone=Apply_A_variables['cone']    
    x_cone_img=getImgDomain(p*cone)    
    part1=dipole_convolve(mask_inBrain*dipole_convolve(x_cone_img,dipole_f),dipole_f)    
    part2=mask_outBrain*x_cone_img
    result=cone*getFourierDomain(part1+lamda*part2)
    
    if preAllocatedAp is not None:
        preAllocatedAp[:,:,:]=result
    else:
        return result

def estimateFourierConeByForcingSusceptibilityOutsideBrainToZero(LFS,FOV,mask_inBrain,mask_inCone_threshold):    
    mask_outBrain=np.invert(mask_inBrain)
    shape=LFS.shape
    print "preparing dipole_f"
    dipole_f=dipole_kernel(shape,FOV)
        
    thr=mask_inCone_threshold
    coneIndx=np.abs(dipole_f)<thr      
    
    print "done"
    
    print "prepare inverse dipole_f"
    one_over_dipole_f=1.0/dipole_f
    one_over_dipole_f[coneIndx]=0
    one_over_dipole_f=one_over_dipole_f+coneIndx*(np.sign(dipole_f)*(dipole_f/thr)**2/thr)
    one_over_dipole_f[dipole_f==0]=0
    print "done"
      
    orig_qsm_result=dipole_convolve(LFS,one_over_dipole_f)
    
    #parameters for optimization
    lamda=10 #how stronlgy we force outside brain mask to zero
    maxiter=200 #how many iterations of conjugate gradient we perform
    
    x_f_initial=getFourierDomain(orig_qsm_result)
    x_noCone_f=np.invert(coneIndx)*x_f_initial
    x_noCone_img=getImgDomain(x_noCone_f)
    
    bpart1=getFourierDomain(dipole_convolve(mask_inBrain*(LFS-dipole_convolve(x_noCone_img,dipole_f)),dipole_f))
    
    bpart2=-getFourierDomain(mask_outBrain*x_noCone_img)
    b=coneIndx*(bpart1+lamda*bpart2)    
    Apply_A_variables={\
        'dipole_f':dipole_f,\
        'mask_inBrain':mask_inBrain,\
        'mask_outBrain':mask_outBrain,\
        'cone':coneIndx\
        }        
    print "Finding optimal Fourier coefficients for cone:"        
    x_cone_est_f=CG_engine(Apply_A,Apply_A_variables,b,lamda,maxiter=maxiter)#,x_init=xlastrun)
    x_cone_est_f=coneIndx*x_cone_est_f
    x_est_f=x_noCone_f+x_cone_est_f
    x_est=getImgDomain(x_est_f)
    
    return x_est

if __name__=="__main__":
        
    print "loading data"    
    fileloc="/cfmm/data/akuurstr/tmp/sagar/3DBrain_only.mat"
    suscData=scipy.io.loadmat(fileloc)['model']
    #suscData=scipy.ndimage.zoom(suscData,0.1,order=0)
    suscData=scipy.ndimage.zoom(suscData,0.2,order=0)
    mask_inBrain=np.abs(suscData)>0
    mask_outBrain=np.invert(mask_inBrain)
    shape=suscData.shape
    FOV=shape
    x_groundTruth_f=getFourierDomain(suscData)
    print "done"
        
    print "preparing dipole_f"
    dipole_f=dipole_kernel(shape,FOV)
        
    thr=0.05
    coneIndx=np.abs(dipole_f)<thr
    print "done"
    
    print "prepare inverse dipole_f"
    one_over_dipole_f=1.0/dipole_f
    one_over_dipole_f[coneIndx]=0
    one_over_dipole_f=one_over_dipole_f+coneIndx*(np.sign(dipole_f)*(dipole_f/thr)**2/thr)
    one_over_dipole_f[dipole_f==0]=0
    print "done"
    
    print "convolving"
    LFS_data=dipole_convolve(suscData,dipole_f)
    print "done"
    
    fieldmap_full=LFS_data
    fieldmap_brain_masked=LFS_data*mask_inBrain
    orig_qsm_result=dipole_convolve(fieldmap_brain_masked,one_over_dipole_f)
    
    #to give the idea a chance, let's see what happens when we use the full fieldmap 
    x_est_fullFieldmap=estimateFourierConeByForcingSusceptibilityOutsideBrainToZero(fieldmap_full,FOV,mask_inBrain,thr)
    #now let's give more realistic data - we only measure the fieldmap inside the brain where there's signal    
    x_est_maskedFieldmap=estimateFourierConeByForcingSusceptibilityOutsideBrainToZero(fieldmap_brain_masked,FOV,mask_inBrain,thr)
    
    
    dimensionTransposeArgument=(0,2,1)
    compare3d((\
               #suscData.transpose(0,2,1),\
               orig_qsm_result.transpose(dimensionTransposeArgument),\
               x_est_fullFieldmap.transpose(dimensionTransposeArgument),\
               x_est_maskedFieldmap.transpose(dimensionTransposeArgument),\
               np.abs(suscData-orig_qsm_result).transpose(dimensionTransposeArgument),\
               np.abs(suscData-x_est_fullFieldmap).transpose(dimensionTransposeArgument),\
               np.abs(suscData-x_est_maskedFieldmap).transpose(dimensionTransposeArgument)),\
        subplotTitles=("X_est_orig","X_est_full","X_est_masked","X_est_orig error","X_est_full error","X_est_masked error"))
    
    """
    The problem is that we don't have the full fieldmap. By only measuring the 
    fieldmap inside the brain, we've masked the fieldmap in the image
    domain... which corresponds to a convolution in the Fourier domain.
    Thus the assumption that it's only the values inside the Fourier cone that 
    are incorrect is wrong. In fact, the values outside the Fourier cone are also
    a problem since they've been corrupted by a convolution.
    
    To make this more clear.  Let's use the true values for fourier cone values 
    from our model susceptibility. Let's combine the true cone values with the
    outs outside cone values obtained from fourier deconvolution of the masked 
    fieldmap. What's the best soultion we can get?
    """
    
    coneIndx=np.abs(dipole_f)<thr      
    x_f_initial=getFourierDomain(orig_qsm_result)
    x_noCone_f=np.invert(coneIndx)*x_f_initial
    x_cone_groundTruth=coneIndx*x_groundTruth_f
    x_bestHope=getImgDomain(x_noCone_f+x_cone_groundTruth)
    
    compare3d((\
               #suscData.transpose(0,2,1),\
               orig_qsm_result.transpose(dimensionTransposeArgument),\
               x_bestHope.transpose(dimensionTransposeArgument),\
               np.abs(suscData-orig_qsm_result).transpose(dimensionTransposeArgument),\
               np.abs(suscData-x_bestHope).transpose(dimensionTransposeArgument)),\
        subplotTitles=("X_est_orig","X_bestHope","X_est_orig error","X_bestHope error"))
    
    """
    The natural question is if we can do better at guessing the entire fieldmap?
    
    It won't make sense to make the value of the fieldmap outside the brain
    another unknown in the optimization problem. By admitting that X_f_outsideCone
    is corrupted, we are basically saying we don't know X_f anywhere...which
    means we might as well stick with MEDI type solutions to the inverse problem.
    
    If we want to continue using cone methods, we need some way of modeling the
    fieldmap outside the brain so that we can trust the values X_f_outsideCone.
    """
    
    