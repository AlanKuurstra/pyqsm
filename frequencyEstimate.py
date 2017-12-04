# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:08:32 2016

@author: Alan
"""

import nibabel as nib
import numpy as np
import scipy.io as sio
import unwrap3d
from vidi3d import compare3d
from scipy import ndimage
import calculateReliability as cr   


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
    print "unwrapping phase"
    unwrapped=phImg.copy()    
    f=np.empty(unwrapped.shape[:-1])        
    #f_nl=np.empty(unwrapped.shape[:-1])    
    unreliabilityWeights=np.ones_like(phImg)
    phase_estimates=np.empty_like(unwrapped) 
    
    if weight is None:
        W2=None
    else:
        W2=weight**2
        #W2_avg=np.average(W2.reshape(np.prod(W2.shape[:-1]),W2.shape[-1]),axis=0)/float(np.prod(W2.shape[:-1]))    
    if removePhase1:        
        unwrapped=np.angle(np.exp(1j*(unwrapped-unwrapped[...,0,np.newaxis])))
        te=te-te[0]       
    
    if not removePhase1:
        unreliabilityWeights[...,0]=cr.calculateReliability(unwrapped[...,0].astype('float32'),mask.astype('bool'))
        unwrapfloat32=unwrap3d.unwrap3d(unwrapped[...,0].astype('float32'),mask.astype('bool'))
        unwrapped[...,0]=(unwrapped[...,0]+np.round((unwrapfloat32-unwrapped[...,0])/(2*np.pi))*2*np.pi)*mask        
        nextEchoToUnwrap=1
    else:
        echo=1
        unreliabilityWeights[...,echo]=cr.calculateReliability(unwrapped[...,echo].astype('float32'),mask.astype('bool'))
        unwrapfloat32=unwrap3d.unwrap3d(unwrapped[...,echo].astype('float32'),mask.astype('bool'))                
        unwrapped[...,echo]=(unwrapped[...,echo]+np.round((unwrapfloat32-unwrapped[...,echo])/(2*np.pi))*2*np.pi)*mask                  
        nextEchoToUnwrap=2
     
    for echo in range(nextEchoToUnwrap,phImg.shape[-1]):       
        unreliabilityWeights[...,echo]=cr.calculateReliability(unwrapped[...,echo].astype('float32'),mask.astype('bool'))        
        #f,f_nl=estimateNonlinearPhase(unwrapped[...,:echo],te[...,:echo],W2=W2[...,:echo],mask=mask)
        #phase_estimates[...,echo]=f*te[echo]+f_nl*te[echo]**2        
        #unwrapped[...,echo]=np.angle(np.exp(1j*(unwrapped[...,echo] - phase_estimates[...,echo])))
        #unwrapped[...,echo]=unwrapped[...,echo]+phase_estimates[...,echo]      
        
        f=estimateFrequency(unwrapped[...,:echo],te[...,:echo],W2=W2[...,:echo],mask=mask)
        phase_estimates[...,echo]=f*te[echo]
        unwrapped[...,echo]=np.angle(np.exp(1j*(unwrapped[...,echo] - phase_estimates[...,echo])))        
        unwrapped[...,echo]=unwrapped[...,echo]+phase_estimates[...,echo]                
        
    return unwrapped,unreliabilityWeights

def estimateFrequencyFromWrappedPhase(phImg,voxelSize,te,mask,weight,truncateEcho=-1,removePhase1=True):
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        