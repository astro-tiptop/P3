#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 22:30:39 2021

@author: omartin
"""

import numpy as np
import os
from astropy.io import fits
from astropy.table import Table, Column
from pandas import read_csv
import datetime
import pytz

class DIMM(object):
    """
    An object that can load a DIMM fil and find the nearest DIMM date
    given any time. 
    """
    def __init__(self, dimmfile):
        """
        Load up a DIMM file into this object. You can then access the following 
        object variables that contain numpy arrays:

        year 
        month
        day
        hour -- UT, 24 hour based (int)
        minute 
        second
        timeInHours -- UT, 24 hour based, float

        r0
        
        """
        table = read_csv(dimmfile, delim_whitespace=True, names= \
            ['year', 'month', 'day', 'hour', 'minute', 'second', 'seeing'])
        
        # Date and time are in HST
        self.year  = np.array(table['year'])
        self.month = np.array(table['month'])
        self.day   = np.array(table['day'])
    
        self.hour   = np.array(table['hour'])
        self.minute = np.array(table['minute'])
        self.second = np.array(table['second'])

        self.seeing = np.array(table['seeing'])

        # Convert from HST to UT
        self.hour += 10
        
        idx = np.where(self.hour >= 24)[0]
        self.day[idx] += 1
        self.hour[idx] -= 24
        

        self.r0 = 0.98 * 500e-7 * 206265.0 / self.seeing # in cm

        self.timeInHours = self.hour + (self.minute/60.0) + (self.second/3600.0)

    def indexTime(self, hhmmss):
        """
        Fetch the closest row of data for a specified time (UT).
        """
        closestVals = []
        closestDts  = []

        for item in hhmmss:
            timeDiff = abs(self.timeInHours - item)
            closestIndex = timeDiff.argmin()

            ## Make sure we aren't off by more than an hour
            if timeDiff[closestIndex] > 1.0:
                print('Could not find MASS data close to ', item) 
                closestVal = -1.
            else:
                closestVal = self.seeing[closestIndex]

            closestVals.append(closestVal)
            closestDts.append(timeDiff[closestIndex])

        closestVals = np.array(closestVals).T

        return closestVals, closestDts

class MASS(object):

    def __init__(self, massfile):
        """
        Load up a MASS file into this object. You can then access the following 
        object variables that contain numpy arrays:

        year 
        month
        day
        hour -- UT, 24 hour based (int)
        minute 
        second
        timeInHours -- UT, 24 hour based, float

        r0
        
        """
        self.file = massfile

        # redid this to not use asciidata module but pandas instead
        table = read_csv(massfile, delim_whitespace=True, names= \
            ['year', 'month', 'day', 'hour', 'minute', 'second', 'seeing'])
        
        # Date and time are in HST
        self.year  = np.array(table['year'])
        self.month = np.array(table['month'])
        self.day   = np.array(table['day'])
    
        self.hour   = np.array(table['hour'])
        self.minute = np.array(table['minute'])
        self.second = np.array(table['second'])

        self.free_seeing = np.array(table['seeing'])

        # Convert from HST to UT
        self.hour += 10
        
        idx = np.where(self.hour >= 24)[0]
        self.day[idx] += 1
        self.hour[idx] -= 24
        self.timeInHours = self.hour + (self.minute/60.0) + (self.second/3600.0)

    def indexTime(self, hhmmss):
        """
        Fetch the closest row of data for a specified time (UT).
        """

        closestVals = []
        closestDts  = []

        for item in hhmmss:
            timeDiff = abs(self.timeInHours - item)
            closestIndex = timeDiff.argmin()

            ## Make sure we aren't off by more than an hour
            if timeDiff[closestIndex] > 1.0:
                print('Could not find MASS data close to ', item) 
                closestVal = -1.
            else:
                closestVal = self.free_seeing[closestIndex]

            closestVals.append(closestVal)
            closestDts.append(timeDiff[closestIndex])

        closestVals = np.array(closestVals).T

        return closestVals, closestDts

class MASSPROF(object):

    def __init__(self, proffile):
        """
        Load up a MASS file into this object. You can then access the following 
        object variables that contain numpy arrays:

        year 
        month
        day
        hour -- UT, 24 hour based (int)
        minute 
        second
        timeInHours -- UT, 24 hour based, float

        r0
        
        """
        self.file = proffile

        # redid this to not use asciidata module but pandas instead
        table = read_csv(proffile, delim_whitespace=True, names= \
            ['year', 'month', 'day', 'hour', 'minute', 'second', \
            'cn2dh_05', 'cn2dh_1', 'cn2dh_2', 'cn2dh_4', 'cn2dh_8', \
            'cn2dh_16', 'seeing'])
        
        # Date and time are in HST
        self.year  = np.array(table['year'])
        self.month = np.array(table['month'])
        self.day   = np.array(table['day'])
    
        self.hour   = np.array(table['hour'])
        self.minute = np.array(table['minute'])
        self.second = np.array(table['second'])

        self.profs = np.array([table['cn2dh_05'], table['cn2dh_1'], table['cn2dh_2'], \
            table['cn2dh_4'], table['cn2dh_8'], table['cn2dh_16']]).T

        # Convert from HST to UT
        self.hour += 10
        
        idx = np.where(self.hour >= 24)[0]
        self.day[idx] += 1
        self.hour[idx] -= 24
        self.timeInHours = self.hour + (self.minute/60.0) + (self.second/3600.0)

    def indexTime(self, hhmmss):
        """
        Fetch the closest row of data for a specified time (UT).
        """

        closestVals = []
        closestDts  = []

        for item in hhmmss:
            timeDiff = abs(self.timeInHours - item)
            closestIndex = timeDiff.argmin()

            ## Make sure we aren't off by more than an hour
            if timeDiff[closestIndex] > 1.0:
                print('Could not find MASS data close to ', item) 
                closestVal = -1.
            else:
                closestVal = self.profs[closestIndex]

            closestVals.append(closestVal)
            closestDts.append(timeDiff[closestIndex])

        closestVals = np.array(closestVals).T

        return closestVals, closestDts

def fetch_data(utDate, saveTo):
    '''Saves massdimm files to directory specified. The output files
    will have the format:

    <utDate>.mass.dat
    <utDate>.dimm.dat
    <utDate>.massprof.dat

    Parameters
    ----------
    utDate : str
        The string UT date in the format such as 20170113 for 2017-01-13
    saveTo : str
        The directory where the retrieved MASS DIMM profiles will be stored.
    '''
    import urllib

    print('Saving MASS/DIMM data to directory:')
    print(saveTo)

    if not os.path.isdir(saveTo):
        os.mkdir(saveTo)

    urlRoot = 'http://mkwc.ifa.hawaii.edu/current/seeing/'
    # Save the DIMM file
    dimmFile = utDate + '.dimm.dat'
    url = urlRoot + 'dimm/' + dimmFile
    try:
        urllib.request.urlretrieve(url, saveTo + dimmFile)
    except:
        print('MASSDIMM not available during the acquisition')
        return 0
    
    # Save the MASS file
    massFile = utDate + '.mass.dat'
    url = urlRoot + 'mass/' + massFile
    urllib.request.urlretrieve(url, saveTo + massFile)

    # Save the MASS profile
    massproFile = utDate + '.masspro.dat'
    url = urlRoot + 'masspro/' + massproFile
    urllib.request.urlretrieve(url, saveTo + massproFile)
    
    return 1

def append_mass_dimm(stats_file, massdimm_dir):
    """Append new columns for the MASS seeing, the DIMM seeing, and the full MASS profile. 
    Save the resulting files to a new file called <stats_file_root>_mpd.<stats_file_extension>.

    Parameters
    ----------
    stats_file : str
        Full path to the stats file which will be read in to get
        dates and times of individual frames. 
    massdimm_dir : str
        The root directory of the MASS/DIMM data. Within this 
        directory, there should be sub-directories with <date>.
    """

    # Read in the stats file, which contains the times of the
    # individual exposures. 
    stats = Table.read(stats_file)

    # Fetch the UT dates and times from the table.

    # Loop through match up with the mass/dimm values.
    date_long_previous = None

    # Some output tables.
    all_dimm = np.zeros(len(stats), dtype=float)
    all_mass = np.zeros(len(stats), dtype=float)
    all_prof_005 = np.zeros(len(stats), dtype=float)
    all_prof_010 = np.zeros(len(stats), dtype=float)
    all_prof_020 = np.zeros(len(stats), dtype=float)
    all_prof_040 = np.zeros(len(stats), dtype=float)
    all_prof_080 = np.zeros(len(stats), dtype=float)
    all_prof_160 = np.zeros(len(stats), dtype=float)
    all_dimm_dt = np.zeros(len(stats), dtype=float)
    all_mass_dt = np.zeros(len(stats), dtype=float)
    all_prof_dt = np.zeros(len(stats), dtype=float)
    
    for ii in range(len(stats)):
        dt_str = stats['DATE_UTC'][ii] + ' ' + stats['TIME_UTC'][ii]
        dt_utc = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

        date_long = '{0:4d}{1:02d}{2:02d}'.format(dt_utc.year,
                                                  dt_utc.month,
                                                  dt_utc.day)
        time_hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.

        # Reload the MASS/DIMM tables if we update the day.
        if date_long != date_long_previous:
            t_dimm = DIMM(massdimm_dir + date_long + '.dimm.dat')
            t_mass = MASS(massdimm_dir + date_long + '.mass.dat')
            t_prof = MASSPROF(massdimm_dir + date_long + '.masspro.dat')

        dimm_val, dimm_dt = t_dimm.indexTime([time_hour])
        mass_val, mass_dt = t_mass.indexTime([time_hour])
        prof_val, prof_dt = t_prof.indexTime([time_hour])
        
        all_dimm[ii] = dimm_val[0]
        all_mass[ii] = mass_val[0]

        all_prof_005[ii] = prof_val[0][0]
        all_prof_010[ii] = prof_val[1][0]
        all_prof_020[ii] = prof_val[2][0]
        all_prof_040[ii] = prof_val[3][0]
        all_prof_080[ii] = prof_val[4][0]
        all_prof_160[ii] = prof_val[5][0]

        all_dimm_dt[ii] = dimm_dt[0]
        all_mass_dt[ii] = mass_dt[0]
        all_prof_dt[ii] = prof_dt[0]

        
    col_dimm = Column(name='DIMM', data=all_dimm)
    col_mass = Column(name='MASS', data=all_mass)
    col_prof_005 = Column(name='Cn2dh_005', data=all_prof_005)
    col_prof_010 = Column(name='Cn2dh_010', data=all_prof_010)
    col_prof_020 = Column(name='Cn2dh_020', data=all_prof_020)
    col_prof_040 = Column(name='Cn2dh_040', data=all_prof_040)
    col_prof_080 = Column(name='Cn2dh_080', data=all_prof_080)
    col_prof_160 = Column(name='Cn2dh_160', data=all_prof_160)
    col_dimm_dt = Column(name='DIMM_dt_hr', data=all_dimm_dt)
    col_mass_dt = Column(name='MASS_dt_hr', data=all_mass_dt)
    col_prof_dt = Column(name='Cn2dh_dt_hr', data=all_prof_dt)
    
    
    stats.add_columns([col_dimm, col_dimm_dt, col_mass, col_mass_dt,
                       col_prof_005, col_prof_010, col_prof_020,
                       col_prof_040, col_prof_080, col_prof_160, col_prof_dt])

    stats_file_root, stats_file_ext = os.path.splitext(stats_file)
    stats.write(stats_file_root + '_mdp' + stats_file_ext, overwrite=True)
    
    return


def get_mass_dimm_for_image(fits_file, massdimm_dir):
    """Print the MASS and DIMM seeing for a single FITS image. 

    Parameters
    ----------
    fits_file : str
        Full or relative path name to the FITS file.
    massdimm_dir : str
        The directory to look for the MASS/DIMM data. 
    """
    hdr = fits.getheader(fits_file)
    
    time_tmp = hdr['TIMEOBS']
    date_tmp = hdr['DATEOBS']
    hst_tz = pytz.timezone('US/Hawaii')
    
    if hdr['SHUTTER'] == True:
        dt_hst = datetime.datetime.strptime(date_tmp + ' ' + time_tmp, '%m/%d/%Y %H:%M:%S')
        dt_hst = hst_tz.localize(dt_hst)
    else:
        dt_hst = datetime.datetime.strptime(date_tmp + ' ' + time_tmp, '%d/%m/%Y %I:%M:%S %p')
        dt_hst = hst_tz.localize(dt_hst)

        # make hst date switch over at midnight 
        noon = datetime.time(12, 0, 0) # assuming you'll never be taking images at local noon...
        del_day = datetime.timedelta(days=1)
        if dt_hst.time() < noon:
            dt_hst += del_day
        
    dt_utc = dt_hst.astimezone(pytz.utc)

    date_long = '{0:4d}{1:02d}{2:02d}'.format(dt_utc.year,
                                              dt_utc.month,
                                              dt_utc.day)
    time_hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.

    # Reload the MASS/DIMM tables if we update the day.
    t_dimm = DIMM(massdimm_dir + date_long + '.dimm.dat')
    t_mass = MASS(massdimm_dir + date_long + '.mass.dat')
                            
    val_dimm = t_dimm.indexTime([time_hour])[0]
    val_mass = t_mass.indexTime([time_hour])[0]

    print('MASS:', val_mass, 'DIMM:', val_dimm)
    
    return


def CombineMASSandDIMM(seeing,seeingAlt,Cn2Alt,wvl=500e-9):
    """
    """
    radian2arcsec = 3600 * 180/np.pi
    seeing0       = (seeing**(5/3) - seeingAlt**(5/3))**(3/5)#/airmass^(3/5)
    mu0           = (0.976*radian2arcsec)**(5/3)*0.423*4*np.pi*np.pi/wvl**(1/3)
    Cn2Ground     = seeing0**(5/3)/mu0
    
    Cn2 = [Cn2Ground] +  list(Cn2Alt)
                    
    return np.array(Cn2)