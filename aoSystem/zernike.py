#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:21:10 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
import math
import scipy.special as ssp
from scipy import integrate

#%% CLASS DEFINITION
class zernike:
    ''' Zernike polynomials
       
    obj = zernike(j,nPixel) creates a Zernike polynomials
    object from the modes vector and the the number of pixel across the
    polynomials diameter
        
    Example:
    zern = zernike([1,2,3,4,5,6],32);
    Computes Zernike polynomials from piston to astigmastism on an
    pupil sampled on a 32x32 grid 
    '''


    def __init__(self,jIndex,resolution,D=None,pupil=[], unitNorm=False,radius=[],angle=[],cobs=0):
        
        # PARSING INPUTS
        if type(jIndex) != list:
            print('Error : you must provide a list for the Noll indexes')
            return
        
        if not jIndex or not resolution:
            print('Error : you must provide valid inputs arguments')
            
        self.jIndex     = jIndex
        self.resolution = resolution
        self.D          = D
        self.cobs       = cobs
        self.pupil      = pupil
        self.radius     = radius
        self.angle      = angle
        
        # DEFINE GEOMETRY
        if not self.radius:
            u           = self.resolution
            x1D         = 2*np.linspace(-(u-1)/2,(u-1)/2,u)/u
            x2D, y2D    = np.meshgrid(x1D,x1D)
            self.radius = np.hypot(x2D,y2D)
            self.angle  = np.arctan2(y2D,x2D)
        else:
            self.resolution = len(self.radius)
    
        # PUPIL
        if len(self.pupil) == 0:
            self.pupil = np.logical_and(self.radius >= self.cobs,self.radius<=1)
           
        # GET RADIAL/AZIMUTHAL ORDERS
        self.n,self.m = self.findNM()
        self.nollNorm = np.sqrt((2-(self.m==0))*(self.n+1))
        self.coeffs   = np.zeros(self.nModes)
        
        # DEFINING POLYNOMIALS
        self.modes = self.polynomials(unitNorm=unitNorm)
        
        
    def findNM(self,jIndex=[]):
        '''Get the Zernike index from a Noll index.
        Parameters
        -------
        jIndex : int, the list of Noll's indexes.
        Returns
        -------
        n : int, The radial Zernike order.
        m : int. The azimuthal Zernike order.
        '''
        
        
        if not jIndex:
            jIndex = self.jIndex
        
        if type(jIndex) != list:
            print('Error : you must provide a list for the Noll indexes')
            return [[],[]]
        
        jIndex = np.asarray(jIndex).astype('int')
        self.nModes   = len(self.jIndex)             
        
        n = np.zeros(self.nModes)
        m = np.zeros(self.nModes)
        
        for k in range(self.nModes):
            n[k] = int(np.sqrt(2 * jIndex[k] - 1) + 0.5) - 1
            if n[k]%2:
                m[k] = 2 * int((2 * (jIndex[k] + 1) - n[k] * (n[k] + 1)) // 4) - 1
            else:
                m[k] = 2 * int((2 * jIndex[k] + 1 - n[k] * (n[k] + 1)) // 4)
            m[k] = m[k]
            
        return n, m
    
    def nModeFromRadialOrder(n):
        '''
        NMODEFROMRADIALORDER Number of Zernike polynomials
        out = zernike.nModeFromRadialOrder(n) returns the number of
        Zernike polynomials (n+1)(n+2)/2 for a given radial order n
        '''
        return (n+1)*(n+2)/2
            
#%% MODES DEFINITION    
    def polynomials(self,unitNorm=False):
        '''
        POLYNOMIALS Zernike polynomials
        fun = polynomes(obj) Computes the Zernike polynomials for the
        Zernike object  sampled on the polar coordinates arrays radius and
        angle. The radius must be normalized to 1.
        '''
        
        def R_fun(r,n,m):
            R = np.zeros(r.shape)
            s1 = int( (n + m)/2 )
            s2 = int( (n - m)/2 )

            for s in range(s2+1):
                ff = math.factorial(s) * math.factorial(max(0,s1-s)) * math.factorial(max(0,s2-s))
                ff = math.factorial(int(n-s))/ff
                R += (-1)**s * ff * r**(n-2*s)
                
            return R
        
        if len(self.radius)==0 or len(self.angle)==0 :
            return []
        else:
            nv      = self.n
            mv      = self.m
            nf      = self.nModes
            pupLog  = self.pupil
            modes   = np.zeros((nf,self.resolution,self.resolution))
            r       = self.radius[pupLog]
            o       = self.angle[pupLog]
            
            # Null azimuthal order
            ind_m = np.argwhere(mv==0)
            if len(ind_m) > 0:
                for cpt in ind_m:
                    n = nv[cpt]
                    m = mv[cpt]
                    modes[int(cpt),pupLog] = np.sqrt(n+1)*R_fun(r,n,m)
                
            mod_mode = np.asarray(self.jIndex).astype('int')%2
            
            # Even polynomes
            ind_m = np.argwhere(np.logical_and(np.logical_not(mod_mode),mv)) 
            if len(ind_m) > 0:
                for cpt in ind_m:
                    n = int(nv[cpt])
                    m = int(mv[cpt])
                    modes[int(cpt),pupLog] = np.sqrt(n+1)*R_fun(r,n,m)*np.sqrt(2)*np.cos(m*o)
                
           # Odd polynomes
            ind_m = np.argwhere(np.logical_and(mod_mode,mv))
            if len(ind_m) > 0:
                for cpt in ind_m:
                    n = int(nv[cpt])
                    m = int(mv[cpt])
                    modes[int(cpt),pupLog] = np.sqrt(n+1)*R_fun(r,n,m)*np.sqrt(2)*np.sin(m*o)     
                
        if unitNorm:
            modes = modes * np.diag(1/self.nollNorm)
        
        return modes    
            
#%% 2ND ORDER MOMENTS
    def CoefficientsVariance(self,x):
        
        def nmOrder(jIndex):
            jj = np.copy(jIndex)
            n = np.floor((-1 + np.sqrt(8*(jj-1)+1))/2).astype(int)
            p = (jj -(n*(n+1))/2)
            k = np.mod(n,2)
            m = (np.floor((p+k)/2)*2 - k).astype(int)
            return n,m
        
        def newGamma(a,b):
            """
            NEWGAMMA Computes the function defined by Eq.(1.18) in R.J. Sasiela's book :
                 Electromagnetic Wave Propagation in Turbulence, Springer-Verlag.
            """
            return np.prod(ssp.gamma(a))/np.prod(ssp.gamma(b))
        
        def pochammerSeries(p,q,a,b,z,tol=1e-6,nmax=1e3):
            """
                POCHAMMERSERIES Computes power series in Pochammer notation
            """
            out = 0 
            if ( (p==(q+1)) & (abs(z)<1) ) | ( (abs(z)==1) & (np.real(sum(a)-sum(b))<0) ) | (p<(q+1)):
        
                if (p==len(a)) & (q==len(b)):
                    if type(z) != list:
                        z = [z]
                    out = np.zeros(np.size(z))                
                    indz = list(np.where(z==0)[0])
                    #import pdb
                    #pdb.set_trace()
                    if len(indz):
                        out[indz] = 1
                    
                    indnz = list(np.where(z!=0)[0])
                    if len(indnz):
                        z    = np.array([z[i] for i in indnz])
                        ck   = 1
                        step = np.inf
                        k    = 0
                        som  = ck
                        a    = np.array(a)
                        b    = np.array(b)
                        while (k<=nmax) & (step>tol):
                            ckp1 = np.prod(a+k) * z * ck/np.prod(b+k)
                            step = abs(abs(ck) - abs(ckp1))
                            som += ckp1
                            k   += 1
                            ck   = ckp1
                        if step>tol:
                            print('pochammerSeries','Maximum iteration reached before convergence')
                        out[indnz] = som                    
                else:
                    print('p and q must be the same length than vectors a and b, respectively')   
            else:
                print('This generalized hypergeometric function doesn''t converge')
            return out

        
        
        def UnParamEx4q2(mu,alpha,beta,p,a):
            """
             UNPARAMEX4Q2 Computes the integral given by the Eq.(2.33) of the thesis
            % of R. Conan (Modelisation des effets de l'echelle externe de coherence
            % spatiale du front d'onde pour l'Observation a Haute Resolution Angulaire
            % en Astronomie, University of Nice-Sophia Antipolis, October 2000)
            % http://www-astro.unice.fr/GSM/Bibliography.html#thesis
            """
            a1 = [(alpha+beta+1)/2,(2+mu+alpha+beta)/2,(mu+alpha+beta)/2]
            b1 = [1+alpha+beta,1+alpha,1+beta]
            a2 = [(1-mu)/2+p,1+p,p]
            b2 = [1+(alpha+beta-mu)/2+p,1+(alpha-beta-mu)/2+p,1+(beta-alpha-mu)/2+p]
            
            k  = (1/(2*np.sqrt(np.pi)*ssp.gamma(p)))
            f1 = newGamma(a1+[p-(mu+alpha+beta)/2],b1) * a**(mu+alpha+beta) * pochammerSeries(3,5,a1,[1-p+(mu+alpha+beta)/2]+b1+[1],a**2)
            f2 = newGamma([(mu+alpha+beta)/2-p] + a2,b2) * a**(2*p) * pochammerSeries(3,5,a2,[1-(mu+alpha+beta)/2+p]+ b2 +[1],a**2)
        
            return k * (f1 + f2)
       
        def zernCovCoef(dr0,dL0,i,j,ni,mi,nj,mj):
            if (mi==mj) & ( (np.remainder(abs(i-j),2)==0) | ((mi==0) & (mj==0))):
                if dL0==0:
                    if (i==1) & (j==1):
                        var=  np.inf
                    else:
                        var = (ssp.gamma(11./6)**2*ssp.gamma(14/3)/(2**(8./3)*np.pi))*(24*ssp.gamma(6./5)/5)**(5./6)*\
                            (dr0)**(5./3)*np.sqrt((ni+1)*(nj+1))*(-1)**((ni+nj-mi-mj)/2)*\
                            newGamma(-5/6+(ni+nj)/2, [23/6+(ni+nj)/2,17/6+(ni-nj)/2,17/6+(nj-ni)/2])
                else:   
                    cte =   (4*ssp.gamma(11./6)**2/np.pi**(14./3)) *\
                            (24*ssp.gamma(6./5)/5)**(5./6)*\
                            (dr0/dL0)**(5./3)/dL0**2*\
                            np.sqrt((ni+1)*(nj+1))*(-1)**((ni+nj-mi-mj)/2)
                    var = cte*UnParamEx4q2(0,ni+1,nj+1,11/6,np.pi*dL0)
            else:
                var = 0
                
            return var
        
        dr0 = x[0]
        dL0 = x[1]
        
        
        jv      = np.array(self.jIndex)
        nv,mv   = nmOrder(jv)
        nv0     = nv
        index   = np.diff(nv)!=0
        index   = [i for i, x in enumerate(index) if x]
        jv      = np.append(jv[index],jv[-1])
        mv      = np.append(mv[index],mv[-1])
        nv      = np.append(nv[index],nv[-1])
        nf      = len(nv)
        zern_var= np.zeros(self.nModes)

        for cpt in range(nf):
            j = jv[cpt]
            n = nv[cpt]
            m = mv[cpt]
            index   = [i for i, x in enumerate(nv0==n) if x]
            zern_var[index] = zernCovCoef(dr0,dL0,j,j,n,m,n,m)
            
        return zern_var

    def tiltsAngularCovariance(self,tel,atm,src,gs,tilt='Z',lag=0):
        
        def sumLayers(f,j,i, ax, ay):
            g = np.pi*( (-1)**i + (-1)**j - 2 )/4
            h = np.pi*( (-1)**i - (-1)**j )/4
            outSumLayers = 0;
            for k in range(atm.nL):
                srcVx = ax*atm.heights[k] + lag*atm.wSpeed[k]*np.exp(1*complex(0,1)*atm.wDir[k])
                srcVy = ay*atm.heights[k] + lag*atm.wSpeed[k]*np.exp(1*complex(0,1)*atm.wDir[k])
                srcV = srcVx + complex(0,1)*srcVy
                rho = abs(srcV)
                arg = np.angle(srcV)
                red = 2*np.pi*f*rho
                Itheta = -np.pi*( ssp.jn(2,red) * np.cos(2*arg+g) -   ssp.j0(red)* np.cos(h) )
                psd = atm.weights[k] *  psdCst*(f ** 2 + 1/atm.L0**2)**(-11./6)
                outSumLayers += psd * Itheta   
            return outSumLayers
        
        D  = tel.D
        R  = D/2
        psdCst = (24*ssp.gamma(6./5)/5)**(5./6) * (ssp.gamma(11./6)**2/(2*np.pi**(11./3))) * atm.r0 ** (-5./3)
        axs = src.direction[0] - gs.direction[0]
        ays = src.direction[1] - gs.direction[1]
        cov = np.zeros((len(axs),2,2))            

        if tilt == 'Z':
                tiltsFilter = lambda f: f*(2.*ssp.jn(2,np.pi*f*D)/(np.pi*f*R))**2
        elif tilt == 'G':
                tiltsFilter = lambda f: f*(ssp.j1(np.pi*f*D))**2
        elif tilt =='ZG':
                tiltsFilter = lambda f: f*2*ssp.jn(2,np.pi*f*D) * ssp.j1(np.pi*f*D)/(np.pi*f*R)
        else:
            print('tilts filters are either Z, G or ZG')
            
        # integration
        for s in range(len(axs)):
            ax=axs[s]
            ay=ays[s]
            cov[s,0,0] = integrate.quad( lambda f: sumLayers(f,2,2,ax,ay)*tiltsFilter(f) , 0 , np.inf)[0]
            cov[s,0,1] = integrate.quad( lambda f: sumLayers(f,2,3,ax,ay)*tiltsFilter(f) , 0 , np.inf)[0]
            cov[s,1,0] = integrate.quad( lambda f: sumLayers(f,3,2,ax,ay)*tiltsFilter(f) , 0 , np.inf)[0]
            cov[s,1,1] = integrate.quad( lambda f: sumLayers(f,3,3,ax,ay)*tiltsFilter(f) , 0 , np.inf)[0]

        return cov
    
    def anisokinetism(self,tel,atm,src,gs,tilt='Z'):
        
        C1 = self.tiltsAngularCovariance(tel,atm,src,src,tilt='Z',lag=0)
        C2 = self.tiltsAngularCovariance(tel,atm,gs,gs,tilt='Z',lag=0)
        C3 = self.tiltsAngularCovariance(tel,atm,src,gs,tilt='Z',lag=0)
        
        return C1 + C2 - 2*C3
            