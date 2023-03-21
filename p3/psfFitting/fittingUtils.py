#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 3 13:30:49 2021

@author: mdumont
"""

#%% IMPORTING LIBRAIRIES
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
from random import random
from configparser import ConfigParser
import os

from astropy.io import fits
from astropy.table import QTable

from p3.psfao21.psfao21 import psfao21
from p3.psfFitting.psfFitting import psfFitting
import p3.aoSystem.FourierUtils as FourierUtils
import p3.telemetry.keckUtils as keckUtils
from p3.psfFitting.psfFitting import displayResults
from p3.psfFitting.imageModel import imageModel

mpl.rcParams['font.size'] = 12
usetex = False
plt.rcParams.update({
    "text.usetex": usetex,
    "font.family": "serif",
    "font.serif": ["Palatino"],
}) 


#%% SPLIT AND JOINT FITTING
def split_fitting_psfao(im, path_ini, x0, fixed, weights,path_root='') : 
    '''
    '''
    # Instantiating the model
    psfao = psfao21(path_ini,path_root=path_root)
    
    # fitting - 7 PSD parameters + no static aberrations
    xo_atm   = x0 + list(np.zeros((psfao.ao.tel.nModes)))
    res_atm  = psfFitting(im,psfao,xo_atm,verbose=2,fixed=fixed,ftol=1e-5,gtol=1e-5,xtol=1e-5)
    
    # fitting - PSD parameters + static aberrations
    x0    = list(res_atm.x[0:7]) + [0,0,0,1.0,0,0,0] + list(np.zeros((psfao.ao.tel.nModes))) 
    fixed = (False, False, False, False, False, False, False) +(True,)*3 + (False,False,False,False) + (False,)*psfao.ao.tel.nModes
    # redefining bounds
    bounds  = psfao.updateBounds(res_atm.x,res_atm.xerr,sig=5)
    res_all = psfFitting(im,psfao,x0,verbose=2,fixed=fixed,bounds=bounds)
    
    return res_all

def joint_fit(im, psfao, x0, fixed, weights):
    x0    = [0.7,4e-2,0.5,1e-2,1,0,1.8,0,0,0,1.0,0,0,0] + list(np.zeros((psfao.ao.tel.nModes)))
    fixed = (False, False, False, False, False, False, False) +(True,)*3 + (False,False,False,False) + (False,)*psfao.ao.tel.nModes
    res_psfao21_joint  = psfFitting(im,psfao,x0,verbose=2,fixed=fixed,ftol=1e-5,gtol=1e-5,xtol=1e-5)
    return res_psfao21_joint

#%% BUILDING PSFS SPATIAL VARIATIONS MODELS
def generate_psfs_grid_from_image(im,mag,coo,path_ini,nSeg,nPx,
        nPSFMax=1,interpolation='psfs',crit='center',magMin=None,magMax=None,
        display=False,tol=1e-5,verbose=2,fact=2):
    '''
        Calibrate the psfao21 model from an astronomical image of stellar fields by : 
            1. Segmenting the image in nSeg x nSeg parts
            2. Extracting sub-images according to the criterion crit and the inputs magnitude/coordinates
            3. Detecting the number of stars in each sub-image
            4. Fitting the psfao21 model instantiated from the .ini file at path_ini
            5. Interpolating the PSFs at the image segments
       INPUTS:
           - im : the astronomical image
           - coo : the coordinates of the stars in pixel ((0,0) : top-left corner)
           - path_ini : the path to the .ini file to instantiate the psfao21 model
           - nSeg : the number of segments to split the image along one direction
           - nPx : the number of pixels to extract sub-images from the image
           - nPSFMAX : the number of sub-images to extract for each segment, default=1
           - interpolation : the interpolation method, either 'psfs' or 'parameters'
           - crit : the criterion for selecting PSF candidates for fitting, must be either 'center' or 'isolated'
           - magMin/magMax : min/max value for the stars magnitude that can be fitted
           - display : if True, the results fo the fit is diplayed, to be use when a few sub-images are fitted
           - tol : tolerances on the parameters, function and gradient increment for the fitting, default=1e-5
           - verbose : makes the fitting talkative (-1:no info up to 2: full info)
           - fact, multiplication factor to nPx to compute the PSF model, default=2
       OUTPUTS:
           - psfs_interp : a numpy array of size nOut x (fact x nPx) x (fact x nPx) containing the interpolated PSFs 
           - psfs_fit  : a numpy array of size nPSF x (fact x nPx) x (fact x nPx) contained the fitted PSFs
           - X :  a numpy array of size (2*nParam + 10) x nPSF containing the results of the fit
           - sub_im :  a numpy array of size nPSf x nPx x nPx containing the extracted sub-images
           - coo_interp :  the 2D numpy array of size nOut x 2, where  nOut=nSeg**2
           - coo_extr : the 2D numpy array of size nPSF x 2, where  nPSF the number of sub_im extracted from the image
    '''
    nY , nX = im.shape
    
    from configparser import ConfigParser
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(path_ini)
    parser.set('sensor_science','FieldOfView', str(int(2*(nPx*fact//2))))
    with open(path_ini, 'w') as configfile:
        parser.write(configfile)
            
    ################### PSF MODEL CALIBRATION
    # segmenting the images
    im_seg, centers = segment_image_in_tiles(im,nSeg)
    # getting coordinates of best PSF
    coo_extr = select_candidates_for_fitting(im_seg,centers,coo,mag,nPSFMax=nPSFMax,
        nBorder=nPx//2,magMin=magMin,magMax=magMax,xon=nX//2,yon=nY//2,crit=crit)
    # extract PSFs
    sub_im = extract_multiple_psf(im,coo_extr,nPx,xon=nX//2,yon=nY//2)
    # fit PSFs
    X , psfs_fit = fit_multiple_psfs(sub_im,coo_extr,path_ini,
                    tol=tol,verbose=verbose,display=False)

    ################### INTERPOLATION
    # defining the interpolation points
    tmp_x  = np.linspace(-nX//2 + nX/nSeg//2,nX//2 - nX/nSeg//2,num=nSeg)
    tmp_y  = np.linspace(-nY//2 + nY/nSeg//2,nY//2 - nY/nSeg//2,num=nSeg)
    x_interp,y_interp = np.meshgrid(tmp_x,tmp_y)
    coo_interp = np.array([y_interp.reshape(-1),x_interp.reshape(-1)]).T
    
    # interpolation
    if interpolation == 'psfs':
        psfs_interp = interpolate_psfs_in_fov(psfs_fit,coo_extr,coo_interp,X,nPSFMax=nPSFMax)
    elif interpolation == 'parameters':
        # spatial variation model
        M        = create_spatial_variations_psf_model(X,order=2)
        psfs_interp = compute_psfs_grid(path_ini,M,coo_interp,nPx)
    else:
        raise ValueError('The field interpolation is not valid: must be psfs or parameters')
        
    # display
    if display:
        psf_cube_to_image(psfs_interp,display=display)
    
        plt.figure()
        plt.plot(coo_extr[:,1],coo_extr[:,0],'ro',label='Extraction fields')
        plt.plot(coo_interp[:,1],coo_interp[:,0],'bs',label='Interpolation points')
        xseg  = np.linspace(-nX//2 , nX//2, nSeg+1)
        yseg  = np.linspace(-nY//2 , nY//2, nSeg+1)
        
        for x in range(len(xseg)):
            plt.plot([xseg[x],xseg[x]],[-nY//2,nY//2],'k--')
            
        for y in range(len(yseg)):
            plt.plot([-nX//2,nX//2],[yseg[y],yseg[y]],'k--')
        
        plt.legend()
        plt.xlabel('X-axis [pixels]')
        plt.ylabel('Y-axis [pixels]')
        plt.gca().set_aspect('equal')
        
    return psfs_interp , psfs_fit , X , sub_im , coo_interp , coo_extr
    
def read_catalog(path_cat,nSkip=1):
    '''
    Read Starfinder=produced catalogs identifyied from the path_cat.
    INPUTS:
        - path_Cat : path to the .txt file
        - nSkip=1 : number of lines to skip
    OUTPUTS:
        - Tab : numpy array of shape nCols x nLines, where nCOls is the number of parameters and nLines the number of stars
        - labels : corresponding label for each column
    '''    
    
    # Identifying the number of columns
    tmp     = open(path_cat)
    labels  = tmp.readline().split()
    nCols   = len(labels)
    
    # Identifying the number of lines
    tmp = open(path_cat)
    nLines = sum(1 for line in tmp) - nSkip
    tmp.close()
    
    # reading each line
    tmp = open(path_cat)
    lines = [tmp.readline().split() for n in range(nLines+1)]
    tmp.close()
    
    # creating the table
    Tab = np.zeros((nCols,nLines))
    for k in range(nCols):
        Tab[k] = [lines[n][k] for n in range(nSkip,nLines+1)]
    
    return Tab , labels

def segment_image_in_tiles(im_in,n1D):
    '''
    Segment the image over a squared grid of n1D x n1D tiles and returns a cube of tiles.
    INPUTS : 
        - im_in : the input image
        - n1D the number of segment along one axis
    OUTPUTS :
        - im_seg, the cube of sub-images
        - centers, the 2D coordinates of the center of each segment, in pixels
    '''
    
    # allocating memory
    nY , nX = im_in.shape
    nPx_y   = nY//n1D
    nPx_x   = nX//n1D
    nTiles  = int(n1D**2)
    im_seg  = np.zeros((nTiles,nPx_y,nPx_x))
    centers = np.zeros((nTiles,2)).astype(int)
    
    # loop
    k=-1
    for y in range(n1D):
        # defining the y interval
        idY1 = y*nPx_y
        idY2 = min([(y+1)*nPx_y,nY]) 
        cy   = y*nPx_y + (idY2 - idY1)//2
        for x in range(n1D):
            #defining the x interval
            idX1 = x*nPx_x
            idX2 = min([(x+1)*nPx_x ,nX])
            cx   = x*nPx_x + (idX2 - idX1)//2
            # cropping
            k+=1
            im_seg[k] = im_in[idY1:idY2,idX1:idX2]
            # center of the segment
            centers[k] = [cy,cx]
            
    return im_seg, centers


def select_candidates_for_fitting(im_seg,centers,coo,mag,nPSFMax=1,nBorder=0,
                                  magMin=-np.inf,magMax=np.inf,yon=0,xon=0,
                                  crit='isolated'):
    '''
    Identify the coordinates of the most isolated stars for each segment.
    INPUTS:
        - im_seg : the cube of segmented images
        - centers :  the center of the segments given in pixels
        - coo : the coordinates of detected stars in [y,x] format
        - mag : the magnitude of detected stars
        - nPSFMax : the number of psfs to be fitted per segment
        - nBorder : the number of pixels to remove in the downselection of good psfs candidates
        - magMin : the minimal magnitude to be considered in thr downslection process
        - magMax : the max magnitude to be considered in thr downslection process
    OUTPUTS:
        - coo_seg : the 3D numpy array of size nSeg x nPSFMax x 2 containing the y/x coordinates of the nPSFMax stars to be fitted for each segment
    '''  
    
    from sklearn.neighbors import KDTree
    
    # getting coordinates
    nSeg , nY , nX = im_seg.shape
    y_all = coo[0]
    x_all = coo[1]
    coo_seg = np.zeros((nSeg,nPSFMax,2))
    
    # loop on image segments
    for k in range(nSeg):
        # get the center of the segment
        y0 = centers[k][0]
        x0 = centers[k][1]
        # remove borders
        idx = np.logical_and(x_all > x0 - nX//2 + nBorder, x_all < x0 + nX//2 - nBorder)
        idy = np.logical_and(y_all > y0 - nY//2 + nBorder, y_all < y0 + nY//2 - nBorder)
        idB = np.logical_and(idx,idy)
        if len(idB) < nPSFMax:
            raise ValueError('Error : the field does not contain enough PSFs within the borders')
        # remove too bright or too faint stars
        idM = np.logical_and(mag > magMin, mag < magMax)
        if len(idM) < nPSFMax:
            raise ValueError('Error : the field does not contain enough PSFs within the magnitude range')
        # downselect samples
        idJ   = np.logical_and(idB,idM)
        nJ    = len(idJ.nonzero()[0])
        if nJ < nPSFMax:
            raise ValueError('Error : there are not enough PSFs that meet the constrains')
                    
        y_k   = y_all[idJ]
        x_k   = x_all[idJ]
        coo_k = np.array([y_k,x_k]).T
        
        if crit == 'isolated':
            # grabbing most isolated stars with the k-D neighbors approach
            tree       = KDTree(coo_k)
            dist, ind  = tree.query(coo_k,k=nJ)
            dmean      = dist[:,1:].mean(axis=1)
            indmean    = np.argsort(dmean)[-nPSFMax:]         
        elif crit == 'center':
            # grabbing closest fields from the image center
            dist       = np.hypot(y_k-y0,x_k-x0)
            indmean    = np.argsort(dist)[:nPSFMax]
            
        coo_seg[k] = np.array([y_k[indmean]-yon,x_k[indmean]-xon]).T
        
    coo_seg = coo_seg.reshape((nSeg*nPSFMax,2))
            
    return coo_seg

def extract_multiple_psf(im,coo_seg,nPx,xon=0,yon=0):
    '''
    Extract psfs from an image given the coordinates.
    INPUTS:
        - im, the image as a 2D numpy array
        - coo_seg, the 2D numpy array of size nPSF x 2, where  nPSF the number of PSF extracted from the image
        - nPx, the size of the PSFs to be extracted in pixels
    OUTPUTS:
        - psfs, a 3D numpy array of size nPSF x nPx x nPx
    '''
    
    # allocating memory
    nPSF = coo_seg.shape[0]
    psfs = np.zeros((nPSF,nPx,nPx))
    
    # loop on segment
    for k in range(nPSF):
        y0 = coo_seg[k][0] + yon
        x0 = coo_seg[k][1] + xon
        psfs[k] = im[ int(y0 - nPx/2) : int(y0 + nPx/2) , int(x0 - nPx/2) : int(x0 + nPx/2)]
        
    return psfs
    
def fit_multiple_psfs(sub_im,coo_seg,path_ini,tol=1e-5,
                      threshold=3,fwhm=2,saturation = 2**15-1,verbose=2,display=False,nSeg=3):
    '''
    Fitting individually multiple PSFs by using the PSFAO21 model.
    INPUTS:
        - psfs, a 3D numpy array of size nPSF x nPx x nPx
        - coo_seg, a 2D numpy array of size nPSF x 2 that contains the PSFs coordinates in pixels
        - path_ini, the path to the .ini file that allows to instantiate the psfao21 model
        - xon/yon, are the coordinates of the center of the field in pixels
        - tol, tolerances for increments on gradient, parameters and model to stop the fitting
        - verbose, to make the fit talkative, fro -1 (no message) to 2 (detailled message)
    OUTPUTS:
        - X, a 2D numpy array of size (2*nParam + 10) x nPSF, where nParam=7 is the number of parameters for psfao21.
        X concatenates the fitted parameters, the associated 95% confidence interval, the PSFs coordinates (y,x,radius),
        as well as PSF-based metrics: Strehl-ratio (sky/fit), FWHM (x-sky, y-sky, x-fit, y-fit) and the Mean sqaure error.
        - psfs_out, a 3D numpy array of size nPSF x nPx x nPx
    '''
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    
    # Instantiating the PSF model
    psfao      = psfao21(path_ini)
    psInArcsec = psfao.ao.cam.psInMas/1e3
        
    # Allocating memory    
    nPSF     = sub_im.shape[0]
    nParam   = 7
    X        = np.zeros((nParam*2 + 10,nPSF))
    psfs_out = np.zeros((nPSF,psfao.ao.cam.fovInPix,psfao.ao.cam.fovInPix))
    
    # initial guess on PSF parameters
    x0    = [0.5,0,1,1e-1,0.5,0,1.8,0,0,0]
    fixed = (False,True,False,False,False,False,False) + (True,)*3 + (False,)*4
    fit_ok= np.zeros(nPSF).astype(bool)    
                  
    # ------- loop on images
    for k in range(nPSF):
        # default config
        y_s = [0.0]
        x_s = [0.0]
        xstars = [1,0,0,0]
        fixed = (False,True,False,False,False,False,False) + (True,)*3 + (False,)*4
            
        # get image statistics
        ymax , xmax    = sub_im[k].shape
        tmp            = np.copy(sub_im[k])
        _, median, std = sigma_clipped_stats(tmp, sigma=3.0)  
        
        # detect the peaks
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)  
        sources = daofind(tmp - median)  
           
        if sources:
            # initial guess on stellar parameters
            y_s = sources['ycentroid'] - ymax//2
            x_s = sources['xcentroid'] - xmax//2
            p_s = sources['flux']
            n_s = len(y_s)
            xstars  = list(p_s/p_s.sum()) + list(y_s) + list(x_s) + [0]
            fixed   = (False,True,False,False,False,False,False) + (True,)*3 + (False,)*(3*n_s+1)
                    
            # Instantiating the multiple-stars image model
            coo_stars = np.array([y_s , x_s])* psInArcsec
            psfao     = psfao21(path_ini , coo_stars = coo_stars)
        
            # performing the fit
            res = psfFitting(sub_im[k],psfao,x0+xstars,fixed=fixed,verbose=verbose,gtol=tol,ftol=tol,xtol=tol)
            
            # getting the PSFs coordinates
            y = coo_seg[k,0]
            x = coo_seg[k,1]
            r = np.hypot(y,x)
            
            # getting psfs-based metrics
            psf_metrics = [res.SR_sky,res.SR_fit,res.FWHMx_sky,res.FWHMy_sky,res.FWHMx_fit,res.FWHMy_fit,res.mse]
            
            # concatenating parameters
            X[:,k]      = list(res.x[0:nParam]) + list(res.xerr[0:nParam]) + [y,x,r] + psf_metrics
            fit_ok[k]   = True
            if display:
                displayResults(psfao,res)
            
    #  ----------- Regenerating PSFs
    psfao = psfao21(path_ini , coo_stars = np.array([0.0 , 0.0]))
    r0 = np.median(X[0,fit_ok])
    for k in range(nPSF):
        if fit_ok[k]:
            psfs_out[k] = imageModel(psfao([r0] + list(X[1:7,k]) + [0,0,0,1,0,0,0],nPix=psfao.ao.cam.fovInPix))
    
    return X , psfs_out
            
def interpolate_psfs_in_fov(psfs_in,coo_in,coo_out,X,nPSFMax=1,nNeighbors=4):
    '''
    Spatially interpolate the psfs at the coo_out coordinates from a input grid of PSFs obtained at coo_psfs.
    INPUTS:
        - psfs_in, a 3D numpy array of size nPSF x nPx x nPx
        - coo_in, a 2D numpy array of size nPSF x 2 containing the y/x coordinates
        - coo_out, a 2D numpy array of size nOut x 2 containing the targetted coordinates
        - X, a 2D numpy array of size (2*nParam + 10) x nPSF containing the retrieved parameters from the fit
        - nPSFMax, the number of fitted sub-images per segment, default=1
        - nNeighbors : the number of closest neighbors to be accounted for during the interpolation
    OUTPUTS:
        - psfs_out, the 3D numpy array of size nOut x nPx x nPx containing the interpolated PSFs.
    '''
    #allocating memory
    nPSF , nPx, _ = psfs_in.shape
    nOut = coo_out.shape[0]
    y_in = coo_in[:,0]
    x_in = coo_in[:,1]
    nSeg = int(nPSF/nPSFMax)
    psfs_out = np.zeros((nOut,nPx,nPx))
    
    # grab the mse value
    mse   = X[-1,:].reshape((nPSFMax,nSeg))
    w_mse = (1/mse/np.sum(1/mse,axis=0)).reshape(nPSF)
    # loop on target locations
    for k in range(nOut):
        # get position and distance
        y_out = coo_out[k,0]
        x_out = coo_out[k,1]
        d_out = w_mse * np.hypot(y_out - y_in, x_out - x_in)
        # finding closest input psfs
        id_out = np.argsort(d_out)[:nNeighbors]
        # interpolating
        w_out = 1/d_out[id_out] / (np.sum(1/d_out[id_out]))
        psfs_out[k] = np.sum(psfs_in[id_out] * w_out[:,np.newaxis,np.newaxis],axis=0)
    
    return psfs_out
    
def psf_cube_to_image(psfs_cube,display=False):
    '''
    Transform a 3D cube of psfs into a wide image concatenating all psfs.
    INPUTS:
        - psfs_cube, the 3D numpy array of size nPSF x nPx x nPx
    OUTPUTS:
        - psfs_im, the 2D numpy array of size (sqrt(nPSF)xnPx) x (sqrt(nPSF)xnPx)
    '''
    
    # allocating memory
    nSeg , nPx , _ = psfs_cube.shape
    nSeg           = int(np.sqrt(nSeg))
    tmp            = psfs_cube.reshape((nSeg,nSeg,nPx,nPx))
    psfs_im        = np.zeros((nSeg*nPx,nSeg*nPx))
    
    # loops
    for y in range(nSeg):
        idy0 = y * nPx
        idy1 = (y+1) * nPx
        for x in range(nSeg):
            idx0 = x * nPx
            idx1 = (x+1) * nPx
            psfs_im[idy0:idy1,idx0:idx1] = tmp[y,x]
        
    if display:
        plt.figure()
        plt.imshow(np.log10(psfs_im))
    return psfs_im

def create_spatial_variations_psf_model(X,order=2):
    '''
    Infer a model of spatial variations of psfao21 parameters from fits performed on multiple stars.
    INPUTS:
        - X, 2D numpy array of size nParam*2+10 x nPSF, where nParam is the number of retrieved parameters over the nPSF observations.
    '''
    
    # unpacking geometric parameters
    nParam = 7
    #y = X[nParam*2]
    #x = X[nParam*2+1]
    r = X[nParam*2+2]
    
    #1. r0 : does not vary across the field - weighted averaged
    r0  = X[0]
    dr0 = X[nParam]
    r0_mod     = np.zeros(order+1)
    r0_mod[-1] = np.sum(r0 * 1/dr0)/sum(1/dr0)
    
    #2. C : does not vary across the field - weighted averaged
    C  = X[1]
    dC = X[1+nParam] + 1e-20
    C_mod     = np.zeros(order+1)
    C_mod[-1] = np.sum(C * 1/dC)/sum(1/dC)
    
    #3. sig2 : must vary wrt the radial distance from the center
    A  = X[2]
    dA = X[2+nParam]
    A_mod = np.polyfit(r,A,order)
    
    #4. ax : assumed to not vary accross the field
    ax  = X[3]
    dax = X[3+nParam]
    ax_mod     = np.zeros(order+1)
    ax_mod[-1] = np.sum(ax * 1/dax)/sum(1/dax)
    
    #5. p : must vary wrt the radial distance from the center
    p  = X[4]
    p_mod = np.polyfit(r,p,order)
    
    #6.theta : ao-related elongation varies wrt np.arctan(y/x), but the residual jitter is uniform
    th  = X[5]
    dth = X[5+nParam]
    th_mod     = np.zeros(order+1)
    th_mod[-1] = np.sum(th * 1/dth)/sum(1/dth)
    
    #7.beta : assumed to not vary accross the field
    be  = X[6]
    dbe = X[6+nParam]
    be_mod     = np.zeros(order+1)
    be_mod[-1] = np.sum(be * 1/dbe)/sum(1/dbe)

    return np.array([r0_mod,C_mod,A_mod,ax_mod,p_mod,th_mod,be_mod])

def compute_psfs_grid(path_ini,M,coo_grid,nPx,path_save=''):
    '''
    Generating PSFAO21 PSFs from the .ini file and the model of spatial variations for parameters.
    INPUTS:
        - path_ini, the path to the .ini file that allows to instantiate the psfao21 model
        - M, the 2D numpy array of size 7x order, where order is the degree of the model
        - coo_grid, the 2D numpy array of size nPSF x 2 that gives the coordinates y/x of the nPSF to be computed
        - path_save, the path to save the grid as a .fits file
    OUTPUTS:
        - psfs, the grid of psf of size nPSF x nPx x nPx
    '''
    
    # instantiating the model
    psfao = psfao21(path_ini)    
    # allocating memory
    nPSF  = coo_grid.shape[0]
    nParam= M.shape[0]
    psfs  = np.zeros((nPSF,nPx,nPx))
        
    for k in range(nPSF):
        # coordinates
        y_k = coo_grid[k,0]
        x_k = coo_grid[k,1]
        r_k = np.hypot(y_k,x_k)
        # parameters
        p_psd = [np.polyval(M[n],r_k) for n in range(nParam)]
        # computing the PSF
        psfs[k] = np.squeeze(psfao(p_psd + [0,0,0,1,0,0,0],nPix=nPx))
    
    return psfs

#%% SPLIT FITTING
def choose_stat_modes(mode, path_ini):
    if mode == 'Piston':
        pathMode = 'aoSystem/data/KECK_CALIBRATION/KECK_STAT/keck_piston_modes_200px.fits'
    elif mode == 'Zernike':
        pathMode = 'aoSystem/data/KECK_CALIBRATION/KECK_STAT/keck_piston_modes_200px_all_lr.fits'
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(path_ini)
    parser.set('telescope', 'path_statModes', '\'' + pathMode + '\'')
    with open(path_ini, 'w') as configfile:
        parser.write(configfile)

def get_split_date(hdr): 
    date = hdr['DATE-OBS']
    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    return year, month, day

def get_PSF_number(psfName):
    return float(psfName[10:14])
    
def adjust_init_file(path_file, path_ini):
    path_filter = 'aoSystem/data/KECK_CALIBRATION/NIRC2_FILTERS/'
    im = fits.getdata(path_file)
    hdr = fits.getheader(path_file)

    PA = keckUtils.getPA(hdr)
    NCPAname = keckUtils.getNCPAstr(hdr)
    wvl,bw,tr,disp = keckUtils.samplingFilterProfile(path_filter,hdr)

    nPix = im.shape[0]

    wvl = float(wvl[0])
    
    year, month, day = get_split_date(hdr)

    pmsName = keckUtils.getPupilMask(hdr)
    if pmsName == 'OPEN':
        pathPup = 'aoSystem/data/KECK_CALIBRATION/NIRC2_MASK/keck_pupil_open_240px.fits'
    elif  pmsName == 'LARGEHEX':
        pathPup = 'aoSystem/data/KECK_CALIBRATION/NIRC2_MASK/keck_pupil_largeHex_272px.fits'

    parser = ConfigParser()
    parser.optionxform = str
    parser.read(path_ini)
    parser.set('telescope','path_pupil','\'' + pathPup + '\'')
    parser.set('telescope','Path_static','\'' + NCPAname + '\'')
    parser.set('telescope', 'pupilAngle', str(PA))
    
    parser.set('sources_HO','Wavelength',str(wvl))
    parser.set('sensor_science','FiedOfView',str(nPix))
    
    parser.set('sensor_HO','SpectralBandwidth',str(bw))
    parser.set('sensor_HO','Dispersion',str([disp[0], disp[1]]))
    parser.set('sensor_HO','Transmittance',str(tr))
    
    with open(path_ini, 'w') as configfile:
        parser.write(configfile)


def fit_all_PSFs(pathData, x0, fixed, weights, statMode, path_ini, pathFilter, outputName, NbPSF = None):
    
    files = os.listdir(pathData)
    names  = ['psfNumber', 'Day', 'Month', 'Year',  'Wvl', 'SRfit', 'SRsky', 'FWHMxsky', 'FWHMysky', 'FWHMxfit', 'FWHMyfit']
    TEMP   = ['Res_x'+str(s) for s in range(50)]
    TEMP  += ['Res_xerr'+str(s) for s in range(50)]
    names += TEMP
    Tab    = QTable(names=names)
    choose_stat_modes('Piston', path_ini)
    if NbPSF is None:
        NbPSF = len(files)

    for i in range (NbPSF):
        if NbPSF is not None:
            choosenFile = math.floor( random()*len(files) )
        else:
            choosenFile = i

        f        = files[choosenFile]
        pathFile = pathData + f
        im       = fits.getdata(pathFile)

        adjust_init_file(pathFile, path_ini)
        choose_stat_modes(statMode, path_ini)
        psfao = psfao21(path_ini)
        FWHM = FourierUtils.getFWHM(im,psfao.ao.cam.psInMas,nargout=1)
        SR = FourierUtils.getStrehl(im,psfao.ao.tel.pupil, psfao.freq.sampRef)
        
        if SR > 0.35 and FWHM > 35 : 

            res = split_fitting(im, psfao, x0, fixed, weights)

            hdr = fits.getheader(pathFile)
            year, month, day = get_split_date(hdr)
            wvl,bw,tr,disp = keckUtils.samplingFilterProfile(pathFilter,hdr)
            psfNumber = get_PSF_number(f)

            data  = (psfNumber ,day, month, year, wvl, res.SR_fit, res.SR_sky, res.FWHMx_sky, res.FWHMy_sky, res.FWHMx_fit, res.FWHMy_fit)
            data += tuple(res.x)
            data += tuple(res.xerr)
            Tab.add_row(data)
    Tab.write(outputName, overwrite = True)


def fits_2_dict(pathFile):
    data = fits.getdata(pathFile)
    nbPSF = len(data)
    names = ['psfNumber', 'Day', 'Month', 'Year',  'Wvl', 'SRfit', 'SRsky', 'FWHMxsky', 'FWHMysky', 'FWHMxfit', 'FWHMyfit']

    names += ['r0', 'C', 'A', 'ax', 'p', 'Theta', 'Beta']
    names += ['jitterx', 'jittery', 'jitterz']
    names += ['flux', 'posx', 'posy', 'bckgrnd']
    names += ['StatMode_x'+str(s) for s in range(36)]

    names += ['r0err', 'Cerr', 'Aerr', 'axerr', 'perr', 'Thetaerr', 'Betaerr']
    names += ['jitterxerr', 'jitteryerr', 'jitterzerr']
    names += ['fluxerr', 'posxerr', 'posyerr', 'bckgrnderr']
    names += ['StatMode_xerr'+str(s) for s in range(36)]
    
    res = dict()

    for num,featureName in enumerate(names):
        res[featureName] = [data[i][num] for i in range(nbPSF)]
    return res

def display_results(path_fits):
    dicti = fits_2_dict(path_fits)
    data  = fits.getdata(path_fits)
    nbPSF = len(data)
    nSeg  = 36
    plt.figure()
    plt.plot(dicti['SRsky'], dicti['SRfit'], '*k')
    plt.plot([ np.min(dicti['SRfit']),np.max(dicti['SRsky']) ], [ np.min(dicti['SRfit']),np.max(dicti['SRsky']) ], 'r')
    plt.xlabel('SRsky')
    plt.ylabel('SRfit')
    plt.title('Strehl')

    plt.figure()
    plt.plot(dicti['StatMode_x0'], dicti['StatMode_xerr0'], '*b')
    plt.xlabel('x0')
    plt.ylabel('xerr0')
    plt.title('Mode TIP-TILP xerr0(x0)')

    meanXList = []
    meanXerrList = []
    for i in range (nSeg):
        currentName = 'StatMode_x' + str(i)
        currentMean =  np.mean(dicti[currentName])
        meanXList.append(currentMean)

        currentName = 'StatMode_xerr' + str(i)
        currentMean =  np.mean(dicti[currentName])
        meanXerrList.append(currentMean)

    plt.figure()
    plt.plot(meanXList, meanXerrList, '*b')
    plt.xlabel('mean x0')
    plt.ylabel('mean xerr0')
    plt.title('Mean')

    paramsList = ['r0', 'C', 'A', 'ax', 'p', 'Theta', 'Beta']
    for num, paramName in enumerate(paramsList):
        ratio = np.array(dicti[paramName]) / np.array(dicti[paramName +'err'])
        plt.figure()
        plt.plot(ratio, 'b')
        plt.title(paramName + ' ratio')
        plt.xlabel('PSFs')
        plt.ylabel('ratio')

    #r0 Histogram
    plt.figure()
    plt.hist(dicti['r0'])
    plt.title('Histogram r0')
    plt.xlabel('r0')
    plt.ylabel('Nb PSFs')

    meanXList = []
    meanXerrList = []
    Tab = QTable(names = ['StatMode_x' + str(s) for s in range(36)])
    Tableauresult = []
    for i in range (nbPSF):
        data = []
        for j in range(nSeg):
            currentName = 'StatMode_x' + str(j)
            currentStatMode =  dicti[currentName][i]
            data.append(currentStatMode)
        Tab.add_row(data)
        Tableauresult.append(data)

    meanList = []
    stdList  = []
    for j in range (nSeg):
        statModeList = []
        for i in range (nbPSF):
            statModeList.append( np.abs(Tableauresult[i][j]) )
        currentMean = np.median(statModeList)
        meanList.append(currentMean)
        currentStd = np.std(statModeList)/np.sqrt(nbPSF)
        stdList.append(currentStd)

    #median values with error bars and weighted mean
    plt.figure()
    plt.errorbar(range(len(stdList)), np.abs(meanList), yerr = stdList, ecolor = 'red', capthick = 3, capsize = 3)
    plt.title('Amplitude fonction du nombre de modes')
    plt.xlabel('Modes')
    plt.ylabel('Amplitude')

    Wlist = []
    for i in range(nbPSF):
        W = dicti['SRsky'][i] / np.abs(dicti['SRfit'][i] - dicti['SRsky'][i])
        Wlist.append(W)
    Wlist = Wlist/np.sum(Wlist)

    sumList  = []
    for i in range(nSeg):
        currentSum = 0
        for num,row in enumerate(Tableauresult):
            currentSum += np.abs(row[i]) * Wlist[num]
        sumList.append(currentSum)
    meanWeightedList = [currentSum / 170 for currentSum in sumList]

    #Weighted mean
    plt.figure()
    plt.plot(np.sort(meanWeightedList), 'o')
    #plt.errorbar(range(len(stdList)), np.abs(meanWeightedList), yerr = stdList, ecolor = 'red', capthick = 3, capsize = 3)
    plt.title('Amplitude fonction du nombre de modes')
    plt.xlabel('Modes')
    plt.ylabel('Amplitude')
    plt.show()
