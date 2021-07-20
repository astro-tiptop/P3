#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:37:55 2021

@author: omartin
"""

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

def estimateLoopDelay(hdr):
            
    wssmprg  = int(hdr['WSSMPRG'])
    _,aoMode = getStagePositionWFS(hdr)
    
    if wssmprg == 0:
        t0 = 8000e-6
    if wssmprg == 1:
        t0 = 2600e-6
    if wssmprg == 2:
        t0 = 850e-6
    if wssmprg == 3:
        t0 = 700e-6
    if wssmprg == 4:
        t0 = 28000e-6
    if wssmprg == 5:
        t0 = 9500e-6
    if wssmprg == 6:
        t0 = 2000e-6
    if wssmprg == 7:
        t0 = 1350e-6
        
    # Total delays
    t8    = 40e-6
    delay = t0+t8

    if aoMode == 'NGS':
        t9        = 1e-6
        t10       = 2e-6
        textra_tt = 200e-6 # retrieved from rejection transfer function fit
        delay_tt  = t0 + t9 + t10 + textra_tt
    else:
        delay_tt  = 1e-3 #STRAP

    return delay, delay_tt
        
def getScale(hdr):
    """
    Returns the instrument pixel scale in mas
    """
    instName = getInstName(hdr)
    if instName == 'NIRC2':
        # Setup NIRC2 plate scales
        scales = {"NARROW": 9.942,
              "MEDIUM": 19.829,
              "WIDE": 39.686}
        return scales[hdr['CAMNAME'].upper()]  
    
    elif instName == 'OSIRIS':
        return 9.95
    else:
        return None

def getPA(hdr):
    #theta = float(hdr['ROTPOSN']) - float(hdr['EL']) - getInstAngle(hdr)
    theta = float(hdr['ROTPOSN']) - getInstAngle(hdr)
    return theta

def getPupilMask(hdr):    
    return hdr['PMSNAME']

def getTelescopeAirmass(hdr):
    el = float(hdr['EL'])
    zenithAngle = 90 - el
    airmass      = 1/np.cos(zenithAngle*np.pi/180)
    return zenithAngle,airmass

def getInstAngle(hdr):
    """
    Returns the instrument pupil position in degrees
    """
    instName = getInstName(hdr)
    if instName == 'NIRC2':
        return float(hdr['INSTANGL']) 
    elif instName == 'OSIRIS':
        return float(hdr['INSTANGL']) - 42.5
    else:
        return 0
    
def getInstName(hdr):
    return hdr['CURRINST']

def getCentralWavelength(hdr):
    """
    Returns the central wavelength in meters
    """
    return float(hdr['CENWAVE'])*1e-6

def getGain(hdr):
    return hdr['DETGAIN']
    
def getFilterName(hdr):
    instName = getInstName(hdr)
    if instName == 'NIRC2':
        filter1 = hdr['fwiname']
        filter2 = hdr['fwoname']
        filt = filter1
        if (filter1.startswith('PK')):
            filt = filter2

        return filt
    elif instName == 'OSIRIS':
        f = hdr['ifilter']
        return f.split('-')[0]
    else:
        return None
    
def getFilterProfile(path_filter,spectral_filter):
    """
    Returns the wavelength (in microns) and the transmission for 
    the specified NIRC2 filter.

    Example: 
    (wave, trans) = nirc2.photometry.get_filter_profile('Kp')
    py.clf()
    py.plot(wave, trans)
    py.xlabel('Wavelength (microns)')
    py.ylabel('Transmission')
    """
    spectral_filter = spectral_filter.replace('_','')
    
    filters = ['J', 'H', 'K', 'Kcont', 'Kp', 'Ks', 'Lp', 'Ms',
               'Hcont', 'Brgamma', 'FeII']
    
    filters_upper = [x.upper() for x in filters]
    
    if spectral_filter.upper() not in filters_upper:
        print( 'Could not find profile for filter %s.' % spectral_filter)
        print( 'Choices are: ', filters)
        return
    
    spectral_filter = filters[filters_upper.index(spectral_filter.upper())]
    table = Table.read(path_filter + spectral_filter + '.dat', format='ascii')

    wavelength = table[table.colnames[0]]
    transmission = table[table.colnames[1]]

    # Lets fix wavelength array for duplicate values
    diff = np.diff(wavelength)
    idx = np.where(diff <= 0)[0]
    wavelength[idx+1] += 1.0e-7

    # Get rid of all entries with negative transmission
    idx = np.where(transmission > 1)[0]
    wavelength = wavelength[idx]
    transmission = transmission[idx] / 100.0 # convert from % to ratio

    return (wavelength*1e-6, transmission)

def getDate(hdr):
    return hdr['DATE-OBS']

def getNCPAstr(hdr):
    """
    """
    date = getDate(hdr)
    numMonth = date[5:7]
    strMonth = ''
    strYear = date[0:4]
    if numMonth == '01': pass
    elif numMonth == '02':
        strMonth = 'Feb'
    elif numMonth == '03':
        strMonth = 'March'
        strDay = date [8:10]
        NCPAstr = strDay+ strMonth + strYear + '_PD1'
    elif numMonth == '04': pass
    elif numMonth == '05': pass
    elif numMonth == '06': pass
    elif numMonth == '07': pass
    elif numMonth == '08':
        strMonth = 'Aug'
    elif numMonth == '09':
        strMonth = 'Sept'
    elif numMonth == '10': pass
    elif numMonth == '11': pass
    elif numMonth == '12': pass    
    
    if numMonth != '03':
        NCPAstr = strMonth + '_' + strYear
        
    return 'NCPA_Keck_' + NCPAstr + '.fits'

def getExposureConfig(hdr):
    
    ittime   = hdr['ITIME']
    coadds   = hdr['COADDS']
    sampmode = hdr['SAMPMODE']
    numreads = hdr['MULTISAM']
    gain     = hdr['GAIN']
    if sampmode == 2:
        ron = 60
    else:
        ron = 15.0 * (16.0 / numreads)**0.5
        
    
    return ittime,coadds,sampmode,numreads,ron,gain
        
def samplingFilterProfile(path_filter,hdr,nWvl=3,thresold=0.9,bwMin = 100e-9,display=False):
    """
    """
    # reading the .dat file
    instName = getInstName(hdr)
    (wvl_all,tr_all)= getFilterProfile(path_filter,getFilterName(hdr))
    # fitting the transmission curve
    id_f    = tr_all > thresold*tr_all.max()
    tr_f    = tr_all[id_f]
    wvl_f   = wvl_all[id_f]
    # linear regression
    p       = np.polyfit(wvl_f,tr_f,1)
    tr_f    = np.polyval(p,wvl_f)
    # compressing the profile for PSF fitting; more bins means longer PSF model computation
    wvlMin = wvl_f.min()
    wvlMax = wvl_f.max()
    bw     = wvlMax - wvlMin
    # consider a monochromatic filter if the bandwith is smaller than the threshold
    if bw <= bwMin:
        nWvl = 1
    # perform the linear regression on the desired wavelengths
    if nWvl > 1:
        wvl_c  = np.linspace(wvlMin,wvlMax,nWvl)
        
    elif nWvl <=1:
         wvl_c  = np.array([getCentralWavelength(hdr)])
         bw     = 0.0
    tr_c   = np.polyval(p,wvl_c)
    
    # assumption : no chromatic dispersion
    disp = [list(np.zeros(nWvl)),list(np.zeros(nWvl))]
    
    if display:
        # plotting
        plt.figure(figsize=(7,7))
        plt.plot(wvl_all*1e6,tr_all,'b',label='True profile')
        plt.plot(wvl_f*1e6,tr_f,'r',label='Fitted profile')
        plt.plot(wvl_c*1e6,tr_c,'ks',label='Compressed profile')
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Transmission')
        plt.title(instName+' filter transmission')
        plt.legend()

    return wvl_c, bw, tr_c, disp

def getSaturationLevel(instName):
        """
        Set to the 95% saturation threshold in DN.
        """
        if instName == 'NIRC2':
            return 12000.0
        elif instName == 'OSIRIS':
            return 20000.0
        else:
            return np.inf

def getStagePositionWFS(hdr):
    """
    
    """
    OBWF   = float(hdr['OBWF'])
    LSPROP = hdr['LSPROP'].upper()
    if LSPROP == 'YES':
        aoMode = 'LGS'     
        AOFCLGFO    = float(hdr['AOFCLGFO'])*1e3 #mm
        zstage_defocus = abs(OBWF - AOFCLGFO)
    else:
        aoMode = 'NGS'
        AOFCNGFO    = float(hdr['AOFCNGFO'])*1e3
        zstage_defocus = abs(OBWF - AOFCNGFO)
        
    return zstage_defocus, aoMode

def getWFSwavelength(hdr):    
    return float(hdr['GUIDWAVE'])*1e-6