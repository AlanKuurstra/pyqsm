# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:08:32 2016

@author: Alan
"""
import sys,os
print os.path.abspath(os.getcwd()+"/../")
sys.path.append(os.path.abspath(os.getcwd()+"/../"))
import calculateReliability as cr

import nibabel as nib
#data_loc='/cfmm/data/akuurstr/data/MS_STUDY_VARIAN/Patients/P001/V1/SUSC/sub-701_ses-V1_rec-svd_part-phase_echo-02_GRE.nii'
data_loc='/softdev/akuurstr/python/modules/pyQSM_nipype/qsm_wf/freqest/freq.nii'
imgObj=nib.load(data_loc)
img=imgObj.get_data()    
mask_loc='/softdev/akuurstr/python/modules/pyQSM_nipype/qsm_wf/brainExtract/avg_restore_brain_mask.nii.gz'
mask=nib.load(mask_loc).get_data()
mask=mask.astype('bool')

test=cr.calculateReliability(img.astype('float32'),mask)
test[test>1e5]=0
import vidi3d as v
v.compare3d((img,test*mask))
v.compare3d((img,test>1e4))