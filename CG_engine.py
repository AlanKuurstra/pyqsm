# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:11:09 2016

@author: Alan
"""
import numpy as np
import time

from vidi3d import imshow3d, compare3d

#tmp=np.load('/Users/Alan/python/QSM/QSMapp/bg_removal/LFS_bg_removed_jacobi_30000iter.npy')  
#import nibabel as nib
#erodedmaskloc='/mnt/Data/code/matlab/P714_Siemens/MagEchoAvg_brain_eroded5.nii'
#WLaplacian=nib.load(erodedmaskloc).get_data()
  
def CG_engine(Apply_A, Apply_A_variables,b, lamda, alpha=1, maxiter=100, x_init=None,tmp=None,blockOutput=False):    
    """make sure Apply_A_variables dictionary matches the one from the Apply_A function you use"""
    
    if x_init is None:
        x=np.zeros_like(b)
    else:
        x=x_init
        
    #intialize residual r=A x0 - b
    r=Apply_A(x,Apply_A_variables, lamda, alpha=alpha)-b      
    #intialize y and p (not doing preconditioned CG so y=r)
    y=r
    p=-y    
    ry_kplus1=np.dot(r.ravel(),y.conj().ravel())
    
    Ap=np.empty_like(b)    
    for i in range(maxiter):
        t=time.time()        
        Apply_A(p,Apply_A_variables,lamda,alpha=alpha,preAllocatedAp=Ap) #Ap=Apply_A(p,Apply_A_variables,lamda,alpha=alpha) #
        ry=ry_kplus1    
        stepsz=ry/(np.dot(p.conj().ravel(),Ap.ravel())+np.finfo('float').eps)  
        x+=stepsz*p #x=x+stepsz*p#           
        r+=stepsz*Ap #r=r+stepsz*Ap   
        y=r #not doing preconditioned CG
        ry_kplus1=np.dot(y.conj().ravel(),r.ravel())
        enforce_conjugate=ry_kplus1/ry    
        
        #p=-y+enforce_conjugate*p
        p*=enforce_conjugate
        p-=y
        if not blockOutput:        
            print i,"time",str(time.time()-t), "cost",str(np.linalg.norm(r))
        if np.linalg.norm(r)<np.finfo('float').eps:
            return x
        #if np.linalg.norm(r)<rmin:
        #    rmin=np.linalg.norm(r)                
        #    xmin=x        
        #if (i+10)%20==0:
            #imshow3d(b-x)
        #    from laplacian import laplacian            
        #    compare3d((tmp,x,laplacian(x)*WLaplacian))
            #compare3d(tmp-x)
    return x