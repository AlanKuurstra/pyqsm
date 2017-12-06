# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:28:26 2016

@author: Alan
"""
import numpy as np
try:
	import pyfftw

	nthreads=16 #this shouldn't be hard coded!
	np.fft.fftn=pyfftw.interfaces.numpy_fft.fftn
	np.fft.ifftn=pyfftw.interfaces.numpy_fft.ifftn

	def getFourierDomain(x):    
	    return np.fft.fftn(np.fft.ifftshift(x),threads=nthreads)
	def getImgDomain(x_f):    
	    return np.fft.fftshift(np.fft.ifftn(x_f,threads=nthreads))
	def viewF(x_f):
	    return np.fft.fftshift(x_f) #fourier has k=0 in the 0 index.  fftshift will put k=0 in the middle of matrix.
except:
	def getFourierDomain(x):    
	    return np.fft.fftn(np.fft.ifftshift(x))
	def getImgDomain(x_f):    
	    return np.fft.fftshift(np.fft.ifftn(x_f))
	def viewF(x_f):
	    return np.fft.fftshift(x_f) #fourier has k=0 in the 0 index.  fftshift will put k=0 in the middle of matrix.
def dipole_convolve(img,dipole_f,returnFourier=False,imgInFourier=False):
    #return np.real(np.fft.fftshift(np.fft.ifftn(dipole_f*np.fft.fftn(np.fft.ifftshift(img),threads=nthreads),threads=nthreads)))    
    if imgInFourier:
        if returnFourier:
            return dipole_f*img
        else:
            return np.real(getImgDomain(dipole_f*img))
    else:
        if returnFourier:
            return dipole_f*getFourierDomain(img)
        else:
            return np.real(getImgDomain(dipole_f*getFourierDomain(img)))

    
