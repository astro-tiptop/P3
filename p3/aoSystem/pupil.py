#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:26:14 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
from scipy.ndimage import rotate

import matplotlib as mpl
import matplotlib.pyplot as plt
from distutils.spawn import find_executable
import os

import p3.aoSystem.FourierUtils as FourierUtils
from p3.aoSystem.segment import segment
from p3.aoSystem.zernike import zernike
#%% DISPLAY FEATURES
mpl.rcParams['font.size'] = 16

if find_executable('tex'): 
    usetex = True
else:
    usetex = False

plt.rcParams.update({
    "text.usetex": usetex,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
    
# ISSUES : 
# case nPixels ~= 200 for the segment leads to overimposition of segments
    
#%% CLASS DEFINITION

class pupil:
    """ 
    Create a segmented telescope pupil 
    """
    
    def __init__(self,segClass=segment(6,1.4,100),segCoord=[(0,0)],
                 D=[], cobs =0, spiderClass=[], 
                 getPetal=False, fill_gap=True):
        
        # PARSING INPUTS
        self.segRef     = segClass
        self.D          = D
        self.cobs       = cobs
        self.spiRef     = spiderClass
        self.getPetal   = getPetal
        self.pupilAngle = 0
        
        # INITIALIZE MODES
        self.segModesType  = 'piston'
        self.segModes   = np.asarray([])
        self.nModesSeg  = 0
        self.petModesType  = 'piston'
        self.petalModes = np.asarray([])
        self.nModesPet  = 0
        
        # MANAGING THE SEGMENT COORDINATES
        if type(segCoord) == str:
            if os.path.exists(segCoord):
               A             =  np.loadtxt(segCoord)
               self.segCoord = np.asarray(list(zip(A[:,0],A[:,1])))
            else:
                print('Error : the file does not exist at your path location')
        elif type(segCoord) == list:
            self.segCoord   = np.asarray(segCoord)
        else:
            print('Error: the data format for the segCoord input is not understood, please use a list or provide a path to a .txt file')
        
        # CREATE LIST OF SEGMENTS
        self.nSegments  = self.segCoord.shape[0]
        self.pixelRatio = self.segRef.pixelRatio
        self.makeSegList()
        
        # DERIVE THE NUMBER OF PIXELS TO BUILD THE PUPIL
        self.SMC  = self.segCoord.max() - self.segCoord.min()  #size of area taken by the center
        maxOffset = 0
        for k in range(self.nSegments):
            maxOffset = max([maxOffset,max([self.segList[k].posXError,self.segList[k].posYError])])
                
        self.SMC    = self.SMC + 2*maxOffset
        self.nPixels= np.ceil(self.SMC * self.pixelRatio).astype('int') + self.segRef.nPixels
        self.radius = self.nPixels/self.pixelRatio/2
  
        # CREATE THE SPIDERS MASK
        if self.spiRef:
            self.spiRef.makeSpiders(self.nPixels,getPetal=self.getPetal)
            
        # DEFINING THE IMAGE CENTER IN PIXELS
        self.centerX = self.nPixels/2
        self.centerY = self.centerX
        
        # MAKE PUPIL
        self.makePupil()
        
        # REDEFINING THE SEGMENT SHAPE TO AVOID GAP ISSUES
        if fill_gap:
            self.gapMatrix = self.fillGap(self.findBestSegmentforFillGap())
        
        # INCLUDE SPIDERS
        self.matrix = self.makeCentObs(self.matrix)
        self.matrix = self.makeSpiders(self.matrix)
               
        # SAVE REFERENCES
        self.pupil = (np.abs(self.matrix)!=0) * 1
        self.phase = np.angle(self.matrix)
        self.reflexivity = self.pupil
        
#%% PUPIL MAKER    
    def makePupil(self):
        self.matrix = np.zeros((self.nPixels,self.nPixels),dtype=complex)
        for k in range(self.nSegments):
            posX, posY = self.getPosSeg(k)
            self.matrix[posX[0]:posX[1],posY[0]:posY[1]]+= self.segList[k].matrix
                
    def makeSpiders(self,matInt):
        if self.spiRef:
            matOut = np.real(matInt) * np.abs(self.spiRef.matrix)  + complex(0,1)* np.imag(matInt)*np.abs(self.spiRef.matrix)
            return matOut
        else:
            return matInt
        
    def makeCentObs(self,matInt):
        if self.cobs !=0:
            x1D         = np.linspace(-1,1,self.nPixels)
            x2D,y2D     = np.meshgrid(x1D,x1D)
            R           = np.hypot(x2D,y2D)
            matOut      = np.real(matInt) * (R>= self.cobs) + complex(0,1)*np.imag(matInt) * (R>= self.cobs)
            return matOut
        else:
            return matInt
        
    def reset(self):
        self.matrix = self.pupil.astype('complex')
        self.matrix = self.makeCentObs(self.matrix)
        self.matrix = self.makeSpiders(self.matrix)
        self.phase = 0*np.angle(self.matrix)
        self.reflexivity = abs(self.matrix)
    
    def rotatePupil(self,angInDegrees,reshape=False,order=3,mode='constant'):
        
        self.pupilAngle = angInDegrees
        if angInDegrees:
            # ROTATE REAL/IMAGINARY PARTS - NOTE THAT SCIPY 1.6 AUTOMATICALLY COVERS THIS FEATURE 
            modPart     = rotate(np.abs(self.matrix),angInDegrees,reshape=reshape,order=order,mode=mode)
            phPart      = rotate(self.phase,angInDegrees,reshape=reshape,order=order,mode=mode)
            # RECOMBINING
            modPart[modPart<0] = 0
            phPart[abs(phPart)<1e-3] = 0
            self.matrix = modPart*np.exp(1*complex(0,1)*phPart)
            self.pupil  = (modPart !=0)*1
            self.phase  = phPart
            self.reflexivity = modPart
            # SEGMENTS AND PETALS ARE NOT ROTATED
            
#%% SEGMENT POSITIONS 
    def makeSegList(self):
        nSides  = self.segRef.nSides
        radius  = self.segRef.radius
        nPx     = self.segRef.nPixels
        self.segList = [None]* self.nSegments
        for i in range(self.nSegments):
            self.segList[i]        = segment(nSides,radius,nPx)
            self.segList[i].matrix = self.segRef.matrix
            self.segList[i].posX   = self.segCoord[i,0] # in meters
            self.segList[i].posY   = self.segCoord[i,1] # in meters
            
    def getPosSeg(self,indSeg):
        '''
            Return the (x,y) indexes of self.matrix that contains the segments given by indSeg
        '''
        segI  = self.segList[indSeg]
        sx,sy = segI.matrix.shape
                        
        Ax = self.centerX + np.array([-sx/2,sx/2])
        Ay = self.centerY + np.array([-sy/2,sy/2])
        # pupil-centered domain
        posy = np.ceil(Ax + self.pixelRatio*(segI.posX + segI.posXError)).astype('int')
        posx = np.ceil(Ay + self.pixelRatio*(segI.posY + segI.posYError)).astype('int')
      
        return [posx,posy]
    
#%% GAPS MANAGEMENTS
    def fillGap(self,refSegNumber):
        '''
        hypothesis :
            all segments are regulary placed
            non overlaying pixels.
            all segments have the same geometry.
        '''
        # DEFINING THE GAP MATRIX
        sx         = np.argwhere(self.segRef.matrix)[:,0]
        posX,posY  = self.getPosSeg(refSegNumber)
        tmp        = self.matrix[posX[0]:posX[1],posY[0]:posY[1]]
        gapMatrix  = tmp.copy()
        gapMatrix[tmp==0] = 1
        gapMatrix[tmp>0]  = 0
            
            
        #Filling oblique sides (up left, up right)
        infX = sx.min()
        supX = np.round(self.segRef.matrix.shape[0]/2).astype('int')
        infY = 0
        supY = self.segRef.matrix.shape[1]
        self.segRef.matrix[infX:supX,infY:supY]+= gapMatrix[infX:supX, infY:supY]

        #Filling up side
        Y = np.argwhere(self.segRef.matrix[infX,:])[:,0]
        self.segRef.matrix[0:infX, min(Y):max(Y)]+= gapMatrix[0:infX, min(Y):max(Y)]
        
        # redo the pupil assembly
        self.makeSegList()
        self.makePupil()
        
        return gapMatrix
        
    def findBestSegmentforFillGap(self):
        
        segNumber= 0
        k        = 0
        flagFind = 0
        
        while flagFind ==0 and k<self.nSegments:
            posX,posY = self.getPosSeg(k)
            tmp       = np.real(self.matrix[posX[0]:posX[1],posY[0]:posY[1]])
            md        = np.round(tmp.shape[1]/2).astype('int')
           
            if tmp[0,0] and tmp[0,-1] and tmp[-1,0] and tmp[-1,-1]\
            and tmp[0,md] and tmp[md,0] and tmp[-1,md] and tmp[md,-1]:
                segNumber=k
                flagFind=1      
            k+=1
        return segNumber

#%% SEGMENTS MANAGEMENTS
    def removeSegment(self,indSeg):
        #indSeg  = np.asarray(indSeg)
        for k in range(len(indSeg)):
            posX,posY  = self.getPosSeg(indSeg[k])
            tmp        = self.matrix[posX[0]:posX[1],posY[0]:posY[1]].copy()
            msk        = (np.abs(self.segList[indSeg[k]].matrix)==1)
            if msk.any():
                tmp[msk] = 0
                self.matrix[posX[0]:posX[1],posY[0]:posY[1]] = tmp
            
    def shiftSegment(self,indSeg,x,y):
        '''
            Move a segment giving its number, X offset and Y offset in meters
        '''
        # convert in pixels
        x  = np.asarray(x)
        y  = np.asarray(y)
        nP = self.nPixels   
        
        # remove the segment
        self.removeSegment(indSeg)
        
        for k in range(len(indSeg)):
            dx        = x[k] # in meters
            dy        = y[k] # in meters
            # defining the maximal/minimal shifts
            posX,posY = self.getPosSeg(indSeg[k])
            dxMax     = (nP - posX.max()) / self.pixelRatio
            dxMin     = (-nP + posX.min()) / self.pixelRatio
            dyMax     = ( nP - posY.max()) / self.pixelRatio
            dyMin     = (-nP + posY.min()) / self.pixelRatio

            # shift the segment or remove it if the displacement exceeds the limits
            if  (dx > dxMax)  or (dx < dxMin) or  (dy > dyMax)  or (dy < dyMin):
                print('Caution : the shift value exceed the pupil dimension')
                self.removeSegment([indSeg[k]])
            else:
                self.segList[indSeg[k]].posXError += dx
                self.segList[indSeg[k]].posYError += dy
                posX,posY                         = self.getPosSeg(indSeg[k])
                self.matrix[posX[0]:posX[1],posY[0]:posY[1]]+= self.segList[indSeg[k]].matrix
            
        self.matrix = self.makeCentObs(self.matrix)
        self.matrix = self.makeSpiders(self.matrix)
           

    def rotateSegment(self,indSeg,segAngle):
        '''
            Rotate a segment giving its number and the angle in degrees
        '''        
        self.removeSegment(indSeg)
        for k in range(len(indSeg)):
            self.segList[indSeg[k]].angleError  = segAngle[k]
            tmp,_,_  = self.segList[indSeg[k]].makeSegment()
            # old reflexivity ans phase applied while re-making matrix
            posX,posY                           = self.getPosSeg(indSeg[k])
            self.matrix[posX[0]:posX[1],posY[0]:posY[1]]+= tmp

        self.matrix = self.makeCentObs(self.matrix)
        self.matrix = self.makeSpiders(self.matrix)
        
    def shrinkSegment(self,indSeg,shrinkFactor):
        '''
            Reduce size of a segment giving a value between 0 & 1
        '''
        self.removeSegment(indSeg);
        for k in range(len(indSeg)):
            self.segList[indSeg[k]].sizeError   = shrinkFactor[k]%1
            tmp,_,_  = self.segList[indSeg[k]].makeSegment()
            # old reflexivity ans phase applied while re-making matrix
            posX,posY                           = self.getPosSeg(indSeg[k])
            self.matrix[posX[0]:posX[1],posY[0]:posY[1]]+= tmp
            
        self.matrix = self.makeCentObs(self.matrix)
        self.matrix = self.makeSpiders(self.matrix)
        
#%% MANAGE REFLEXIVITY AND ABERRATIONS
        
    def computeModes(self, jIndex,area='segment'):
               
        if len(jIndex) == 0:
            print('Error : you must provide a valist list of Noll index')
            return []
        
        if area == 'segment':
            sX,sY = self.segRef.matrix.shape
            res   = max(sX,sY)
            zern  = zernike(jIndex,res,pupil= np.abs(self.segRef.matrix.copy()) == 1)
            modes = zern.modes
        elif area == 'petal':
            sX,sY = self.matrix.shape
            res   = max(sX,sY)
            modes = np.zeros((self.spiRef.nPetal,len(jIndex),res,res))
            S = np.abs(self.spiRef.matrixPetal.copy())
            for k in range(self.spiRef.nPetal):
                SS = S[k] * self.pupil
                SS = self.makeCentObs(SS)
                SS = self.makeSpiders(SS)
                zern  = zernike(jIndex,res,pupil= SS==1)
                modes[k] = zern.modes
                
        return modes
        
    def applyPhaseErrorSegment(self, indSeg, jIndex, modesCoeffs):
        
        # CHECKING INPUTS
        if type(indSeg) == int:
            indSeg = np.asarray([indSeg])
        elif type(indSeg) == list:
            indSeg = np.asarray(indSeg)
        if len(indSeg) == 0:
            print('Error : you must provide a valist list of Segment indexes')
            return []
        
        if len(jIndex) == 0:
            print('Error : you must provide a valist list of Noll index')
            return []
        else:
            if np.asarray(jIndex).ndim > 1:
                print('Warning : you must provide a single list of Noll index for each petal. Select the first dimension')
                jIndex = jIndex[0]
        
        if type(modesCoeffs) == int:
            modesCoeffs = np.asarray([modesCoeffs])
        elif type(modesCoeffs) == list:
            modesCoeffs = np.asarray(modesCoeffs)
        if modesCoeffs.ndim ==1: # case 1 petal
            modesCoeffs = np.asarray([modesCoeffs])
            
        if modesCoeffs.shape[1] != len(indSeg) or modesCoeffs.shape[0] != len(jIndex):
            print('Error : you must provide as many coefficients values as Noll indexes for each segment')
            return []
                    
        # COMPUTING MODES
        nSeg = len(indSeg)
        self.segModes = self.computeModes(jIndex,area = 'segment')
    
        # REMOVING SEGMENT
        self.removeSegment(indSeg)
        # APPLYING THE PHASE
        for k in range(nSeg):
            tmp = self.segList[indSeg[k]]
            # CHANGE SEGMENT ABERRATIONS
            tmp.matrix,tmp.phase = tmp.applyPhase(self.segModes,modesCoeffs[:,k])
            # REASSEMBLE
            posX,posY = self.getPosSeg(indSeg[k])
            self.matrix[posX[0]:posX[1],posY[0]:posY[1]] += tmp.matrix
      
        self.matrix = self.makeCentObs(self.matrix)
        self.matrix = self.makeSpiders(self.matrix)
        self.phase  = np.angle(self.matrix)
            
    def applyPhasePetal(self,indPetal,jIndex,modesCoeffs):
        
        # CHECKING INPUTS
        if not self.spiRef:
            print('Error : You must provide a spider class when instantiating the pupil class')
            return
                
        if type(indPetal) == int:
            indPetal = np.asarray([indPetal])
        elif type(indPetal) == list:
            indPetal = np.asarray(indPetal)
        if len(indPetal) == 0:
            print('Error : you must provide a valist list of petal indexes')
            return []
        
        if len(jIndex) == 0:
            print('Error : you must provide a valist list of Noll index')
            return []
        else:
            if np.asarray(jIndex).ndim > 1:
                print('Warning : you must provide a single list of Noll index for each petal. Select the first dimension')
                jIndex = jIndex[0]
                
        if type(modesCoeffs) == int:
            modesCoeffs = np.asarray([modesCoeffs])
        elif type(modesCoeffs) == list:
            modesCoeffs = np.asarray(modesCoeffs)
        if modesCoeffs.ndim ==1: # case 1 petal
            modesCoeffs = np.asarray([modesCoeffs])
            
            
        if np.any(np.asarray(indPetal) >= self.spiRef.nPetal):
            print('Warning : remove the non-existant indexes')
            idGood = np.asarray(indPetal) < self.spiRef.nPetal
            indPetal = indPetal[idGood]
            modesCoeffs = modesCoeffs[idGood,:]
            
        nPetal = len(indPetal)
        if modesCoeffs.shape[0] != nPetal or modesCoeffs.shape[1] != len(jIndex):
            print('Error : you must provide as many coefficients values as Noll indexes for each petal')
            return []
        
     
        # COMPUTING MODES
        self.petalModes = self.computeModes(jIndex,area = 'petal')
        nModes = len(jIndex)
        
        # APPLYING THE PHASE
        self.petalPhase = np.zeros((self.nPixels,self.nPixels),dtype=complex)
        for k in range(nPetal):
            for j in range(nModes):
                self.petalPhase +=  modesCoeffs[k,j] * self.petalModes[indPetal[k],j]
        ang = self.phase + self.petalPhase
        self.matrix = abs(self.matrix) * np.exp(1*complex(0,1)* ang)
           
    def applyReflexivitySegment(self,indSeg,coeffReflexion):
        
        # CHECKING INPUTS
        if type(coeffReflexion) == int:
            coeffReflexion = np.asarray([coeffReflexion])
        elif type(coeffReflexion) == list:
            coeffReflexion = np.asarray(coeffReflexion)
        if type(indSeg) == int:
            indSeg = np.asarray([indSeg])
        elif type(indSeg) == list:
            indSeg = np.asarray(indSeg)
        if np.any(np.asarray(indSeg) >= self.nSegments):
            print('Warning : remove the non-existant indexes')
            idGood = np.asarray(indSeg) < self.nSegments
            indSeg = indSeg[idGood]
            coeffReflexion = coeffReflexion[idGood]
        # CHECKING DATA FORMAT
        if len(indSeg) == 0:
            print('Error : you must provide a valist list of Segment indexes')
            return []
        if len(coeffReflexion) != len(indSeg):
            print('Error : you must provide as many coefficients values as segments')
            return []
        
        # REMOVE SEGMENT
        self.removeSegment(indSeg)
        
        # UPDATING THE REFLEXIVITY   
        for k in range(len(indSeg)):
            # CHANGE SEGMENT REFLEXIVITY
            tmp, _    = self.segList[indSeg[k]].applyReflexivity(coeffReflexion[k])
            # REASSEMBLE
            posX,posY = self.getPosSeg(indSeg[k])
            self.matrix[posX[0]:posX[1],posY[0]:posY[1]] += tmp
                                                          
        self.matrix = self.makeCentObs(self.matrix)
        self.matrix = self.makeSpiders(self.matrix)
        self.reflexivity = np.abs(self.matrix)        
#%% RESIZE
    def removeZeroBorder(self,flagReplace=1):
        
        print('removing border full of zeros')
         
        tmp = np.argwhere(self.matrix)
        Z1  = tmp.min()
        Z2  = tmp.max()
        self.matrix = self.matrix[Z1:Z2,Z1:Z2]
            
        if flagReplace==1: #to be tested
            self.nPixels= self.matrix.shape[0]
                
            oldCenterX = self.centerX
            oldCenterY = self.centerY
            self.centerX = self.nPixels/2
            self.centerY = self.centerX
            
            for k in  range(self.nSegments):
                self.segList[k].posX -= (oldCenterX-self.centerX) / self.pixelRatio
                self.segList[k].posY -= (oldCenterY-self.centerY) / self.pixelRatio    
                           
    def zeroPad(self,ratio,flagReplace=1):
        
        if ratio > 1:
            nX,nY       = self.matrix.shape
            nPadX       = np.round(nX*(ratio-1)/2).astype('int')
            nPadY       = np.round(nY*(ratio-1)/2).astype('int')
            self.matrix = np.pad(self.matrix,(nPadX,nPadY))
            
            if flagReplace: #to be tested
                self.nPixels= self.matrix.shape[0]
                oldCenterX  = self.centerX
                oldCenterY  = self.centerY
                self.centerX = self.nPixels/2
                
                self.centerY = self.centerX;
                for k in  range(self.nSegments):
                    self.segList[k].posX -= (oldCenterX-self.centerX)/self.pixelRatio
                    self.segList[k].posY -= (oldCenterY-self.centerY)/self.pixelRatio     
        
    def resize(self,newNPixels,flagReplace=0,kind='nearest'):
        
        if newNPixels != self.nPixels:
            self.matrix = FourierUtils.interpolateSupport(self.matrix,newNPixels,kind=kind)
            
            if flagReplace: #to be tested
                self.pixelRatio= self.pixelRatio*newNPixels/self.nPixels
                self.nPixels   = self.matrix.shape[0]
                self.centerX = self.nPixels/2    
                self.centerY = self.centerX
    
    def displayPupil(self):
        
        fig,axs = plt.subplots(1,2,figsize=(10,20), constrained_layout=True)
        
        pcm = axs[0].imshow(self.reflexivity,extent=[-self.radius,self.radius,-self.radius,self.radius])
        axs[0].set_title('Reflexivity')
        axs[0].set_xlabel('Position (m)')
        axs[0].set_ylabel('Position (m)')
        fig.colorbar(pcm,ax=axs[0],shrink=0.1)
        
        pcm = axs[1].imshow(self.phase,extent=[-self.radius,self.radius,-self.radius,self.radius])
        axs[1].set_title('Phase aberrations')
        axs[1].set_xlabel('Position (m)')
        axs[1].set_ylabel('Position (m)')
        fig.colorbar(pcm,ax=axs[1],shrink=0.1)
        plt.show()