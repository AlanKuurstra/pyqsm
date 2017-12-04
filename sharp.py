# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:28:26 2016

@author: Alan

Rough code for doing sharp in fourier domain.
"""

from scipy.special import gamma,jn
import numpy as np
from multiprocessing import Pool
import pyfftw
import calculateReliability as cr   
from scipy.ndimage import binary_dilation,binary_fill_holes,gaussian_filter

nthreads=16
np.fft.fftn=pyfftw.interfaces.numpy_fft.fftn
np.fft.ifftn=pyfftw.interfaces.numpy_fft.ifftn

def getFourierDomain(x):    
    return np.fft.fftn(np.fft.ifftshift(x),threads=nthreads)
def getImgDomain(x_f):    
    return np.fft.fftshift(np.fft.ifftn(x_f,threads=nthreads))
def viewF(x_f):
    return np.fft.fftshift(x_f) #fourier has k=0 in the 0 index.  fftshift will put k=0 in the middle of matrix.
def convolve(img,kernel_f,returnFourier=False,imgInFourier=False):
    #return np.real(np.fft.fftshift(np.fft.ifftn(dipole_f*np.fft.fftn(np.fft.ifftshift(img),threads=nthreads),threads=nthreads)))    
    if imgInFourier:
        if returnFourier:
            return kernel_f*img
        else:
            return np.real(getImgDomain(kernel_f*img))
    else:
        if returnFourier:
            return kernel_f*getFourierDomain(img)
        else:
            return np.real(getImgDomain(kernel_f*getFourierDomain(img)))
        
def inverseTransform(kImg,FOV):
    """
    Compute the inverse Fourier transform.
    
    Appropriately scale numpy's ifft to match a continuous transform
    with impulse sampling.  This function also takes care of swapping 
    image and fourier space using ifftshift and fftshift.

    Parameters
    ----------
    kimg : array_like
        Input array, can be complex.
    FOV : list 
        A list containing the field of view for each dimension of kImg.
    
    Returns
    -------
    out : complex ndarray
        The continuous inverse Fourier transform of kImg
    """
    if kImg.ndim!=len(FOV):
        "Error: number of kImg dimensions must match number of dimensions in FOV!"
        raise
    return float(kImg.size)/np.array(FOV).prod()*np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kImg)))
def forwardTransform(img,FOV):
    """
    Compute the Fourier transform.
    
    Appropriately scale numpy's fft to match a continuous transform
    with impulse sampling.  This function also takes care of swapping 
    image and fourier space using ifftshift and fftshift.     

    Parameters
    ----------
    img : array_like
        Input array, can be complex.
    FOV : list
        A list containing the field of view for each dimension of img.
    
    Returns
    -------
    out : complex ndarray
        The continuous Fourier transform of img
    """
    if img.ndim!=len(FOV):
        "Error: number of img dimensions must match number of dimensions in FOV!"
        raise
    return np.array(FOV).prod()/float(img.size)*np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img))) 

def phantomGrid3D(numXSamples,FOVx,numYSamples,FOVy,numZSamples,FOVz):
    FOVx=float(FOVx)
    FOVy=float(FOVy)
    FOVz=float(FOVz)
    kx,ky,kz=np.mgrid[-np.floor(numXSamples/2.0):-np.floor(numXSamples/2.0)+numXSamples,
            -np.floor(numYSamples/2.0):-np.floor(numYSamples/2.0)+numYSamples,
            -np.floor(numZSamples/2.0):-np.floor(numZSamples/2.0)+numZSamples]
    return kx/FOVx,ky/FOVy,kz/FOVz
    
  

def kSpaceShift(kimg,k, shift):
    #shifts in positive direction along the axis
    #apply phase shift to k space data to shift corresponding image domain image
    if np.sum(np.abs(np.array(shift)))!=0:
        if len(k)==1 and len(shift)==1:
            kimg*=np.exp(-(1j*2*np.pi*(k[0]*shift[0])))
        elif len(k)==2 and len(shift)==2:
            kimg*=np.exp(-(1j*2*np.pi*(k[0]*shift[0]+k[1]*shift[1])))
        elif len(k)==3 and len(shift)==3:
            kimg*=np.exp(-(1j*2*np.pi*(k[0]*shift[0]+k[1]*shift[1]+k[2]*shift[2])))
        return kimg
    else:
        return kimg
    
    #code for making spheres
def sincn(n,r):
    if r==0:
        return .5
    else:
        return .5*gamma(1+n/2.0)*jn(n/2.0,r)/(r/2.0)**(n/2.0)
sincn=np.vectorize(sincn)

def sincnForImgSpaceUnitCircle(n,radMap):
    #return sincn(n,radMap*2*np.pi)*2*np.pi #makes it the same as jinc...but doesn't work for 3d
    return sincn(n,radMap*np.pi)*np.pi/n #this might be wrong...done completely empirically 
    
    
def scSincn(n,radMap,scale):
    scale=float(scale)
    return scale**n*sincnForImgSpaceUnitCircle(n,scale*radMap)
    
def _kSphereHelper(i):      
    radMapPart=radMap[:,:,i*chunksize:min((i+1)*chunksize,radMap.shape[2])]
    return scSincn(3,radMapPart,2*ksphererad)
#"""
def kSphere(kx,ky,kz,radius,amplitude=1,xShift=0,yShift=0,zShift=0):
    poolsz=8
    
    global ksphererad,chunksize,radMap
    ksphererad=radius
    chunksize=int(np.ceil(float(kz.shape[2])/poolsz))
    radMap=np.sqrt(kx**2+ky**2+kz**2)
    p=Pool(poolsz)    
    vals=p.map(_kSphereHelper,np.arange(poolsz))    
    vals=np.concatenate(vals,axis=2)
    returnvalue= amplitude*kSpaceShift(vals,[kx,ky,kz],[xShift,yShift,zShift])
    p.close()
    return returnvalue

class SpherePhantom():
    def __init__(self,phantomDiam, phantomIntensity=1, shift=[0.0,0.0,0.0]):        
        self.phantomRad=phantomDiam/2.0        
        self.phantomIntensity=phantomIntensity        
        self.shift=shift
        
    def sample(self,kx,ky,kz):        
        phantom= kSphere(kx,ky,kz,radius=self.phantomRad,amplitude=self.phantomIntensity,xShift=self.shift[0],yShift=self.shift[1],zShift=self.shift[2])
        return phantom

from vidi3d import compare3d
def sharp(freq,FOV,mask,sphereDiam=5.0,threshold=0.05):
    sphereDiam=float(sphereDiam)
    threshold=float(threshold)
    FOVx,FOVy,FOVz=FOV
    acquisitionX,acquisitionY,acquisitionZ=img.shape
        
    print "start grid"
    kx,ky,kz = phantomGrid3D(acquisitionX,FOVx,acquisitionY,FOVy,acquisitionZ,FOVz) 
    print "done grid"   
    vol=4.0/3*np.pi*(sphereDiam/2)**3   
    signal=1.0/vol*float(FOVx*FOVy*FOVz)/float(acquisitionX*acquisitionY*acquisitionZ) #multiplying by the size of the voxels accounts for the difference between continuous and discrete fft scaling   
    #need to create new sphere data
    sphere = SpherePhantom(sphereDiam, phantomIntensity=signal, shift=[0.0,0.0,0.0])   
    sphereKData=sphere.sample(kx,ky,kz)
    sphereImgData=np.real(inverseTransform(sphereKData,(FOVx,FOVy,FOVz)))             
    sk=getFourierDomain(sphereImgData) 
    unreliability=cr.calculateReliability(img.astype('float32'),mask)
    
    #edges
    reliabilityMask=np.abs(unreliability)<5e3
    
    tmp=reliabilityMask.copy()
    tmp=binary_dilation(tmp)
    tmp=binary_fill_holes(tmp)
    tmp_smooth=gaussian_filter(np.invert(tmp).astype('float'),2.5)
    tmp_smooth=tmp_smooth<.5
    tmp_smooth=binary_fill_holes(tmp_smooth)
    reliabilityMask=tmp_smooth
        
    mask_reliable=mask*reliabilityMask   
    img_reliable=img*mask_reliable
        
    SMV=(1.0-sk)
    #if it's close to 0...it should be 0    
    SMV[np.abs(SMV)<threshold]=0
    one_over_SMV=1.0/SMV
    one_over_SMV[np.abs(SMV)<threshold]=0    
    Bint_corrupted=convolve(img_reliable,SMV)
    mask_erode=convolve(np.invert(mask_reliable),sk)<.005           
    Bint_corrupted=Bint_corrupted*mask_erode    
    Bint=convolve(Bint_corrupted,one_over_SMV)*mask_erode    
    return Bint
   
    
    
if __name__=="__main__":
    #"""
    import nibabel as nib
    data_loc='/softdev/akuurstr/python/modules/pyQSM_nipype/qsm_wf/freqest/freq.nii'    
    imgObj=nib.load(data_loc)    
    img=imgObj.get_data()  
    
    mask_loc='/softdev/akuurstr/python/modules/pyQSM_nipype/qsm_wf/brainExtract/avg_restore_brain_mask.nii.gz'
    mask=nib.load(mask_loc).get_data()
    mask=mask.astype('bool')   
    
    acquisitionX,acquisitionY,acquisitionZ=imgObj.header['dim'][1:4]
    szX,szY,szZ=imgObj.header['pixdim'][1:4]
    FOVx,FOVy,FOVz=szX*acquisitionX,szY*acquisitionY,szZ*acquisitionZ
    
    lfs=sharp(img,(FOVx,FOVy,FOVz),mask)
    
    import scipy.io as sio
    matlab_lfs=sio.loadmat('/softdev/akuurstr/python/modules/pyQSM_nipype/qsm_wf/freqest/matlab_lfs.mat')['lfs'].transpose(1,0,2)[::-1,:,:]
    from vidi3d import compare3d
    compare3d((lfs,matlab_lfs))
    
    
    STOP
    #"""
    
    #EXPLORE!
    
    if 'sphereImgData' not in locals():
        import nibabel as nib
        data_loc='/softdev/akuurstr/python/modules/pyQSM_nipype/qsm_wf/freqest/freq.nii'    
        imgObj=nib.load(data_loc)    
        img=imgObj.get_data()    
        
        data_loc='/softdev/akuurstr/python/modules/pyQSM_nipype/qsm_wf/freqest/freq_firstEchoOnly.nii'    
        img_firstEchoOnly=nib.load(data_loc).get_data()
        
        mask_loc='/softdev/akuurstr/python/modules/pyQSM_nipype/qsm_wf/brainExtract/avg_restore_brain_mask.nii.gz'
        mask=nib.load(mask_loc).get_data()
        mask=mask.astype('bool')   
        
        acquisitionX,acquisitionY,acquisitionZ=imgObj.header['dim'][1:4]
        szX,szY,szZ=imgObj.header['pixdim'][1:4]
        FOVx,FOVy,FOVz=szX*acquisitionX,szY*acquisitionY,szZ*acquisitionZ
        
        print "start grid"
        kx,ky,kz = phantomGrid3D(acquisitionX,FOVx,acquisitionY,FOVy,acquisitionZ,FOVz) 
        print "done grid"
        diam=5.0 #mm
        vol=4.0/3*np.pi*(diam/2)**3   
        signal=1.0/vol*szX*szY*szZ #multiplying by the size of the voxels accounts for the difference between continuous and discrete fft scaling
              
        #need to create new sphere data
        sphere = SpherePhantom(diam, phantomIntensity=signal, shift=[0.0,0.0,0.0])
        import time
        t1=time.time()
        sphereKData=sphere.sample(kx,ky,kz)
        sphereImgData=np.real(inverseTransform(sphereKData,(FOVx,FOVy,FOVz)))   
        
        t2=time.time()
        print t2-t1
        print "vol: ",np.sum(sphereImgData.ravel())
        
    
    from vidi3d import imshow3d,compare3d,compare2d  
    
    tmp=cr.calculateReliability(img.astype('float32'),mask)
    
    #edges
    reliabilityMask=np.abs(tmp)<5e3
    
    #n=9
    #tmp2=binary_dilation(binary_erosion(reliabilityMask,iterations=n),iterations=n)
    #tmp2=binary_fill_holes(tmp2)
    tmp2=reliabilityMask.copy()
    tmp2=binary_dilation(tmp2)
    tmp2=binary_fill_holes(tmp2)
    tmp2_smooth=gaussian_filter(np.invert(tmp2).astype('float'),3)
    tmp2_smooth=tmp2_smooth<.5
    tmp2_smooth=binary_fill_holes(tmp2_smooth)
    reliabilityMask=tmp2_smooth
    
    
    if 'img_inpaint' not in locals():
        #inpaint the veins...I don't like this. We're really messing with the data here.
        from skimage.restoration import inpaint
        vein_error=np.abs(tmp)>1e4
        vein_error[np.invert(reliabilityMask)]=False
        
        img_inpaint=(img*reliabilityMask*np.invert(vein_error))
        scale=np.abs(img_inpaint).max()
        
        img_inpaint=inpaint.inpaint_biharmonic(img_inpaint/scale, vein_error)
        img_inpaint=img_inpaint*scale
        
        compare3d((img,img_inpaint,vein_error))
        
    
    
    img_reliable=img*mask*reliabilityMask
    img_firstEchoOnly_reliable=img_firstEchoOnly*mask*reliabilityMask
    mask_reliable=mask*reliabilityMask
    
    sk=getFourierDomain(sphereImgData)
    SMV=(1.0-sk)
    #if it's close to 0...it should be 0
    thr=.05
    SMV[np.abs(SMV)<thr]=0
    one_over_SMV=1.0/SMV
    one_over_SMV[np.abs(SMV)<thr]=0
    #Bint_corrupted=convolve(img_firstEchoOnly_reliable,SMV)
    Bint_corrupted=convolve(img_reliable,SMV)
    mask_erode=convolve(np.invert(mask_reliable),sk)<.005
    Bint_corrupted=Bint_corrupted*mask_erode
    Bint=convolve(Bint_corrupted,one_over_SMV)*mask_erode
    #sharp_bg=img_firstEchoOnly_reliable-Bint
    #Bint=img_reliable-sharp_bg
    
    if 'dp_bg' not in locals():
        from pyQSM.dipole_project import dipole_project
        dp_bg,_=dipole_project(img_firstEchoOnly_reliable,mask_reliable,mask_reliable,np.invert(mask_reliable),(FOVx,FOVy,FOVz),30)
        #dp_bg,_=dipole_project(img_reliable,mask_reliable,mask_reliable,np.invert(mask_reliable),(FOVx,FOVy,FOVz),50)
    Bint2=(img_reliable-dp_bg)*mask_erode
    
    
    """**************************INPAINTING**************************"""
    
    
    img_inpaint_reliable=img_inpaint*reliabilityMask
    
    Bint_inpaint_corrupted=convolve(img_inpaint_reliable,SMV)
    Bint_inpaint_corrupted=Bint_inpaint_corrupted*mask_erode
    Bint_inpaint=convolve(Bint_inpaint_corrupted,one_over_SMV)*mask_erode
    Bint2_inpaint=(img_inpaint_reliable-dp_bg)*mask_erode
        
    
    """
    daveloc='/cfmm/data/drudko/vnmrsys/data/studies/MS/LongitudinalAnalysis/P711_67F/contrasts/QS/QS_Visit1_P11.nii'
    daveloc2='/cfmm/data/drudko/vnmrsys/data/studies/MS/LongitudinalAnalysis/P701_69M/contrasts/LFS/LFS_Visit1.nii'
    daveimg=nib.load(daveloc).get_data()
    daveimg2=nib.load(daveloc2).get_data()
    compare3d(daveimg)
    """
    compare3d((Bint,Bint2,Bint_inpaint,Bint2_inpaint),
              subplotTitles=('sharp','dp','sharp inpaint','dp inpaint'))




