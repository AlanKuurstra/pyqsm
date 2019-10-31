# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:08:32 2016

@author: Alan
"""

import nibabel as nib
import numpy as np
import scipy.io as sio
import Unwrap3d as unwrap3d
from scipy import ndimage
from Unwrap3d import calculate_reliability

def estimateNonlinearPhase(data,te,W2=None,mask=None):
    #estimate linear and an x**2 part of phase (polynomial of degree 2)
    orig_shape=data.shape
    n=np.prod(orig_shape[:-1])    
    data=data.reshape((n,orig_shape[-1]))
    W2=W2.reshape((n,orig_shape[-1]))        
    
    a11=np.dot(W2*te,te)    
    a12=np.dot(W2*te,te**2)
    a21=a12
    a22=a11
    det=a11*a22-a12*a21
    
    data_te=np.dot(W2*data,te)
    data_te2=np.dot(W2*data,te**2)
    
    f=1.0/det*(a22*data_te-a12*data_te2)
    f_nl=1.0/det*(-a21*data_te+a11*data_te2)
    
    
    f=f.reshape(orig_shape[:-1])
    f_nl=f_nl.reshape(orig_shape[:-1])
    if mask is not None:        
        f=f*mask
        f_nl=f_nl*mask
    return f,f_nl

def estimateFrequency(data,te,W2=None,mask=None):
    #different weights for every voxel
    #estimate = inverse(T' W^2 T) T' W^2 phase
    orig_shape=data.shape
    n=np.prod(orig_shape[:-1])    
    data=data.reshape((n,orig_shape[-1]))
    if W2 is None:        
        freqest=1.0/np.dot(te,te) * np.dot(data,te)
    else:
        W2=W2.reshape((n,orig_shape[-1]))        
        freqest=1.0/np.dot(W2*te,te) * np.dot(W2*data,te)   

    freqest=freqest.reshape(orig_shape[:-1])
    if mask is not None:        
        freqest=freqest*mask
    return freqest

def quickEstimateFrequency(data,W2_avg,te):
    #weights only differ in temporal direction, W2_avg should be same size as te
    orig_shape=data.shape
    n=np.prod(orig_shape[:-1])    
    freqest=np.empty(n)
    data=data.reshape((n,orig_shape[-1])).T
    freqest=1.0/np.dot(te.T,W2_avg*te)*np.dot(te.T,W2_avg[:,np.newaxis]*data)
    freqest=freqest.reshape(orig_shape[:-1])
    return freqest  

def phaseUnwrap(phImg,te,mask,weight,removePhase1=True):
    print("unwrapping phase")
    mask=mask.astype('bool')
    unwrapped=phImg.copy()    
    f=np.empty(unwrapped.shape[:-1])        
    #f_nl=np.empty(unwrapped.shape[:-1])    
    unreliabilityWeights=np.ones_like(phImg)
    phase_estimates=np.empty(unwrapped.shape[:-1])    
    
    if weight is None:
        W2=None
    else:
        W2=weight**2
        #W2_avg=np.average(W2.reshape(np.prod(W2.shape[:-1]),W2.shape[-1]),axis=0)/float(np.prod(W2.shape[:-1]))    
    if removePhase1:        
        unwrapped=np.angle(np.exp(1j*(unwrapped-unwrapped[...,0,np.newaxis])))
        te=te-te[0]       
    
    if not removePhase1:
        unreliabilityWeights[...,0]=calculate_reliability(unwrapped[...,0].astype('float32'),mask.astype('bool'))
        unwrapfloat32=unwrap3d.unwrap3d(unwrapped[...,0].astype('float32'),mask.astype('bool'))
        unwrapped[...,0]=(unwrapped[...,0]+np.round((unwrapfloat32-unwrapped[...,0])/(2*np.pi))*2*np.pi)*mask        
        nextEchoToUnwrap=1
    else:
        echo=1
        unreliabilityWeights[...,echo]=calculate_reliability(unwrapped[...,echo].astype('float32'),mask.astype('bool'))
        unwrapfloat32=unwrap3d.unwrap3d(unwrapped[...,echo].astype('float32'),mask.astype('bool'))                
        unwrapped[...,echo]=(unwrapped[...,echo]+np.round((unwrapfloat32-unwrapped[...,echo])/(2*np.pi))*2*np.pi)*mask                  
        nextEchoToUnwrap=2
     
    for echo in range(nextEchoToUnwrap,phImg.shape[-1]):       
        unreliabilityWeights[...,echo]=calculate_reliability(unwrapped[...,echo].astype('float32'),mask.astype('bool'))
        #f,f_nl=estimateNonlinearPhase(unwrapped[...,:echo],te[...,:echo],W2=W2[...,:echo],mask=mask)
        #phase_estimates=f*te[echo]+f_nl*te[echo]**2        
               
        f=estimateFrequency(unwrapped[...,:echo],te[...,:echo],W2=W2[...,:echo],mask=mask)
        phase_estimates=f*te[echo]
        unwrapped[...,echo]=np.angle(np.exp(1j*(unwrapped[...,echo] - phase_estimates)))
        #after subtraction of phase estimate, all values should be close to zero if frequency doesn't change
        #but maybe there's a susceptibility change due to breathing and a linear change in freq happens across the brain
        #let's just unwrap to make sure
        unwrapfloat32=unwrap3d.unwrap3d(unwrapped[...,echo].astype('float32'),mask.astype('bool'))
        unwrapped[...,echo]=(unwrapped[...,echo]+np.round((unwrapfloat32-unwrapped[...,echo])/(2*np.pi))*2*np.pi)*mask        
        #each echo could have a multiple of 2pi offset depending on where unwrapping
        # seeds from.  The algorithm seeds from highest snr area and starts unwrapping.
        #If the algorithm picks a different starting spot each echo, there could be 
        #global wpi jumps between echoes. This looks for and corrects that.        
        avg_difference_from_zero=unwrapped[...,echo][mask].mean()        
        two_pi_offset_integer=int(avg_difference_from_zero/(2*np.pi))        
        unwrapped[...,echo]=unwrapped[...,echo]+phase_estimates+two_pi_offset_integer*2*np.pi
    
    return unwrapped,unreliabilityWeights

def estimateFrequencyFromWrappedPhase(phImg,voxelSize,te,mask,weight,truncateEcho=None,removePhase1=True):
    phImg=phImg[...,:truncateEcho]
    te=te[:truncateEcho]
    weight=weight[...,:truncateEcho]    
    phaseEstimates,unreliability=phaseUnwrap(phImg,te,mask,weight,removePhase1=removePhase1)
    
    reliable=(unreliability<30)
    rtmp=reliable.copy()
    kernelSz_mm=1.5 #mm
    KernelSz_vxl=np.ceil(kernelSz_mm/np.array(voxelSize)).astype('int')
    for i in range(10):
        rtmp=ndimage.gaussian_filter(rtmp.astype('float'),np.concatenate((KernelSz_vxl,(0,))))
        rtmp=rtmp*reliable          
    rtmp=ndimage.gaussian_filter(rtmp.astype('float'),(0,0,0,.75))    
    W2=(weight*rtmp)**2
    W2=W2+1e-30
    
    if removePhase1:
        #f,f_nl=estimateNonlinearPhase(phaseEstimates,te-te[0],W2=W2,mask=mask)
        f=estimateFrequency(phaseEstimates,te-te[0],W2=W2,mask=mask)
    else:
        #f,f_nl=estimateNonlinearPhase(phaseEstimates,te,W2=W2,mask=mask)
        f=estimateFrequency(phaseEstimates,te,W2=W2,mask=mask)    
    return f #rad/s
if __name__=="__main__":
    pass
        
