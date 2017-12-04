#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:12:37 2017

@author: akuurstr

Only solve for the fourier coefficients in the cone of zeros.
Look for places that we know should be zero but exhibit the streaking artifact 
which solve for cone of zero coefficients which cause those places to be zero.
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



n=1000
x_orig=np.random.randn(n)+1j*np.random.randn(n)
A_orig=np.random.randn(n,n)+1j*np.random.randn(n,n)
b_orig=np.dot(A_orig,x_orig)

#Ax=b
#A'Ax=A'b
A=np.dot(A_orig.conj().T,A_orig) #force it to be positive definite
b=np.dot(A_orig.conj().T,b_orig) 


maxiter=20
Apply_A_variables={'A':A}
def Apply_A(p,Apply_A_variables,lamda=1,alpha=1,preAllocatedAp=None):
    A=Apply_A_variables['A']
    if preAllocatedAp is not None:
        preAllocatedAp[:]=np.dot(A,p)
        return
    else:
        return np.dot(A,p)
        
x_est=CG_engine(Apply_A,Apply_A_variables,b,1,maxiter=maxiter)

import matplotlib.pyplot as plt
plt.plot(x_orig)
plt.plot(x_est)

print np.linalg.norm(x_est-x_orig)


stop

#does it work for identity?
Apply_A_variables={}
def Apply_A(p,Apply_A_variables,lamda=1,alpha=1,preAllocatedAp=None):    
    if preAllocatedAp is not None:
        preAllocatedAp[:]=p
        return
    else:
        return p
b_est=CG_engine(Apply_A,Apply_A_variables,b,1,maxiter=20)

import matplotlib.pyplot as plt
plt.plot(b)
plt.plot(b_est)

print np.linalg.norm(b-b_est)