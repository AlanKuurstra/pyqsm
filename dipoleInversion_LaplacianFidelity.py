# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:31:15 2016

@author: Alan

Sharp erodes edges of brain.

Put the laplacian into the data fidelity term to avoid bg removal.

alpha ||laplacian(DX)-laplacian(fieldmap)||_2 + TV

This might not work well because the true solution, X_true, is equally as good
as DX_true + DX_harmonic.  Question: is there an X_harmonic susceptibility
field that could produce a harmonic field when convolved with the dipole kernel?
"""
import numpy as np
import time
from CG_engine import CG_engine
from CG_A_matrices import Apply_A_dp_invert_w_laplacian_fidelity_l2_grad as Apply_A #this is missing a regularizer
from dipole_kernel import dipole_kernel
from dipole_convolve import dipole_convolve
from fidelity_weighting import dataFidelityWeight as dataFidelityWeight
from gradient_weighting import gradientWeights as gradientWeights

def ShrinkAndUpdate(Xmap,dx,dy,dz,bregx,bregy,bregz,WGradx,WGrady,WGradz,lamda):
    
    WGradx_Gx_Xmap=WGradx*(np.roll(Xmap,-1,axis=0)-Xmap)
    WGradx_Gx_Xmap_Bx=WGradx_Gx_Xmap+bregx
    WGrady_Gy_Xmap=WGrady*(np.roll(Xmap,-1,axis=1)-Xmap)
    WGrady_Gy_Xmap_By=WGrady_Gy_Xmap+bregy
    WGradz_Gz_Xmap=WGradz*(np.roll(Xmap,-1,axis=2)-Xmap)
    WGradz_Gz_Xmap_Bz=WGradz_Gz_Xmap+bregz
    
    s=np.sqrt(WGradx_Gx_Xmap_Bx**2+WGrady_Gy_Xmap_By**2+WGradz_Gz_Xmap_Bz**2)
    s_max=np.maximum(s-1.0/lamda,0)
    
    #soft thresholding    
    dx=s_max*WGradx_Gx_Xmap_Bx/s
    dy=s_max*WGrady_Gy_Xmap_By/s
    dz=s_max*WGradz_Gz_Xmap_Bz/s    
    
    #hard thresholding 
    """
    s_max=s_max>0
    dx=s_max*MxGxX_Bx
    dy=s_max*MyGyX_By
    dz=s_max*MzGzX_Bz
    #"""
    bregx=bregx+WGradx_Gx_Xmap-dx
    bregy=bregy+WGrady_Gy_Xmap-dy
    bregz=bregz+WGradz_Gz_Xmap-dz
    
    return dx,dy,dz,bregx,bregy,bregz

def SplitBregmanL1Grad(B_loc,FOV,WData_sqrd, WGradx,WGrady,WGradz,alpha,lamda,Xmap_init=None,maxiter=200):
    if Xmap_init is None:
        Xmap=np.zeros_like(B_loc) 
        dx=dy=dz=bregx=bregy=bregz=0
    else:
        Xmap=Xmap_init
        """iniitialize bregman params...it's all the small edges (noise?)""" 
        dx,dy,dz,bregx,bregy,bregz=ShrinkAndUpdate(Xmap,0,0,0,0,0,0,WGradx,WGrady,WGradz,lamda)        
    
    #create dipole kernel
    dipole_f=dipole_kernel(B_loc.shape,FOV)
    
    #store Apply_A_variables in dictionary
    WGradx_sqrd=WGradx**2
    WGrady_sqrd=WGrady**2
    WGradz_sqrd=WGradz**2
    
    Apply_A_variables={\
    'dipole_f':dipole_f,\
    'WData_sqrd':WData_sqrd,\
    'WGradx_sqrd':WGradx_sqrd,\
    'WGrady_sqrd':WGrady_sqrd,\
    'WGradz_sqrd':WGradz_sqrd,\
    }    
    
    tmp=B_loc.copy()
    tmp=-6*tmp+np.roll(tmp,-1,axis=0)+np.roll(tmp,1,axis=0)+np.roll(tmp,-1,axis=1)+np.roll(tmp,1,axis=1)+np.roll(tmp,-1,axis=2)+np.roll(tmp,1,axis=2)
    tmp*=WData_sqrd
    tmp=-6*tmp+np.roll(tmp,-1,axis=0)+np.roll(tmp,1,axis=0)+np.roll(tmp,-1,axis=1)+np.roll(tmp,1,axis=1)+np.roll(tmp,-1,axis=2)+np.roll(tmp,1,axis=2)
    tmp=dipole_convolve(tmp,dipole_f)
    b_part=alpha*tmp   
    
    for bregit in range(maxiter):
        t=time.time() 
        
        #compute b
        A=WGradx*(dx-bregx)        
        A=np.roll(A,1,axis=0)-A
        B=WGrady*(dy-bregy)
        B=np.roll(B,1,axis=1)-B
        C=WGradz*(dz-bregz)
        C=np.roll(C,1,axis=2)-C        
        b=b_part + lamda*(A+B+C)        
        
        """note: no coordinate descent!"""
        #might want to play with how many iterations of conjugate gradient are done
        Xmap=CG_engine(Apply_A, Apply_A_variables, b, lamda, alpha=alpha, maxiter=1,x_init=Xmap)
        dx,dy,dz,bregx,bregy,bregz=ShrinkAndUpdate(Xmap,dx,dy,dz,bregx,bregy,bregz,WGradx,WGrady,WGradz,lamda)
                
        print "step: "+str(bregit)+" time: "+str(time.time()-t)+"s"
        
        if (bregit+15)%20==0:    
            #from vidi3d import compare3d
            #compare3d((tmp2/CF/delta_TE*1e6,Xmap/CF/delta_TE*1e6,filtered_LFS))
            #compare3d((suscData,(WData_sqrd>0)*Xmap))
            pass
    return Xmap  
    
def getXmap(iMag,Mask,voxel_size,B_loc,alpha,maxiter=100):    
    lamda=2*alpha
    
    WData_sqrd=Mask*dataFidelityWeight(iMag)    
    #WGradx,WGrady,WGradz=gradientWeights(iMag,Mask,voxel_size,2)
    WGradx=WGrady=WGradz=np.ones_like(iMag) 
    FOV=list(voxel_size*B_loc.shape)
    
    Xmap=SplitBregmanL1Grad(B_loc,FOV, WData_sqrd, WGradx,WGrady,WGradz,alpha,lamda,maxiter=maxiter)
    return Xmap
    

if __name__=="__main__":
    
    from dipole_convolve import dipole_convolve, getFourierDomain, getImgDomain, viewF
    import scipy.ndimage
    from vidi3d import compare3d
    
    print "loading data"    
    fileloc="/cfmm/data/akuurstr/tmp/sagar/3DBrain_only.mat"
    suscData=scipy.io.loadmat(fileloc)['model']    
    suscData=scipy.ndimage.zoom(suscData,0.2,order=0)
    mask_inBrain=np.abs(suscData)>0    
    shape=suscData.shape
    FOV=shape    
    print "done"
        
    print "preparing dipole_f"
    dipole_f=dipole_kernel(shape,FOV)
    print "done"   
   
    print "convolving"
    LFS_data=dipole_convolve(suscData,dipole_f)
    print "done"
    
    fieldmap_full=LFS_data
    fieldmap_brain_masked=LFS_data*mask_inBrain
    
    alpha=50.0
    
    iMag=suscData
    Mask=mask_inBrain
    RDF_orig=fieldmap_brain_masked
    voxel_size=np.array((1,1,1))
    result=getXmap(iMag,Mask,voxel_size,RDF_orig,alpha,maxiter=300) 
    
    compare3d((suscData,result))
    