# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:08:32 2016

@author: Alan
"""
import nibabel as nib
import numpy as np
import scipy.io as sio

import unwrap3d
from pyview.Viewers import imshow3d, compare3d


imgs=sio.loadmat('/mnt/Data/code/matlab/RatQSM/unwrappedEcho2')

Echo4ph=imgs['Echo2ph']
Echo4uph=imgs['Echo2uph']


fileloc='/mnt/Data/code/matlab/RatQSM/SUSC_rats_Cage239152_noEN_HDRHB40_W0_01_combinedPH.nii'
img=nib.load(fileloc).get_data()
img1=img[:,:,:,6]

#the unwrapping algorithm needs fortran style arrays
img1=np.asfortranarray(img1,'float32')
maskloc='/mnt/Data/code/matlab/RatQSM/Cage239152_noEN_W0_mask.nii'
mask=nib.load(maskloc).get_data()
mask=np.asfortranarray(mask,'bool')


"""
maskc=np.ascontiguousarray(mask) #row major: c and python
maskf=np.asfortranarray(mask,'bool') #column major: fortran and matlab

print "orig is f:", np.isfortran(mask)
#print "c is f:",np.isfortran(maskc)
#print "f is f:",np.isfortran(maskf)

print "orig same mem f:",np.may_share_memory(mask,maskf)
print "orig same mem c:",np.may_share_memory(mask,maskc)
#print "c same f:",np.may_share_memory(maskc,maskf)
"""


result = unwrap3d.unwrap3d(img1,mask)
compare3d((img1,result))