#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:42:22 2021

@author: jonty
"""

import numpy as np


#Constants
h = 6.626e-34
c = 299792458.0 # m/s
k = 1.38e-23
sb = 5.67e-8 #
au     = 1.495978707e11 # m 
pc     = 3.0857e16 # m
lsol   = 3.828e26 # W
rsol   = 6.96342e8 # m
MEarth = 5.97237e24 # kg
G = 6.67e-11
Rsol = 6.96342e8 # m
Msol = 1.99e30 #kg
Lsol = 3.828e26 #W
pc = 3.086e16 #m
au = 1.496e11 #m
um = 1.e-6

__all__ = ['Exocomet']

class Exocomet:
    
    def __init__(self):
        #Define stellar parameters for modelling
        self.dstar = 10.0   # stellar distance in parsecs
        self.lstar = 1.0    # Luminosity of the star in solar luminosities
        self.rstar = 1.0    # Radius of the star in solar radii
        self.mstar = 1.0    # Stellar mass in solar masses
        self.tstar = 5770.0 # Stellar temperature in Kelvin
        self.npix  = 401    # Number of pixels in stellar image
        self.nstr  = 75     # Number of pixels in stellar radius
        #Define exocomet parameters for modelling
        self.a     = 5.0 # semi-major axis at point of transit in Rstar
        self.b     = 0.0 # Impact parameter in fraction of Rstar
        self.Rhead = 0.5 # Radius of the head in Rstar
        self.Rtail = 2.0 # Radius of the tail in Rstar
        self.tau_0 = 0.5 # Maximum opacity of the cometary cloud
    
    def planck_lam(wav, T):
        """
        Parameters
        ----------
        lam : float or array
            Wavelengths in metres.
        T : float
            Temperature in Kelvin.
    
        Returns
        -------
        intensity : float or array
            B(lam,T) W/m^2/m/sr
        """
        a = 2.0*h*c**2
        b = h*c/(wav*k*T)
        intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
        return intensity
    
    def get_ldc(self):

        """
        Function to read in the limb darkening coefficients for a 4 parameter
        model based on the Claret 2017 TESS LDCs.
        
        Parameters
        ----------
        tstar : float
            stellar temperature (K)
        
        Returns
        -------
        ldc : float array
            limb darkening coefficients
        
        """        
        
        from astropy.io import ascii
        from scipy import interpolate
    
        #Load limb darkening parameters
        ldc_file = 'Claret2017_TESS_LDC.dat'
        converters = {'a1LSM' : [ascii.convert_numpy(np.float64)],\
                      'a2LSM' : [ascii.convert_numpy(np.float64)],\
                      'a3LSM' : [ascii.convert_numpy(np.float64)],\
                      'a3LSM' : [ascii.convert_numpy(np.float64)],\
                      'Teff'  : [ascii.convert_numpy(np.float64)],\
                      'logg'  : [ascii.convert_numpy(np.float64)]}
        ldc_data = ascii.read(ldc_file,delimiter=';',comment='#',format='basic',fast_reader='False',guess='False',data_start=3)
        
        a1 = np.asarray(ldc_data['a1LSM'].data)
        a2 = np.asarray(ldc_data['a2LSM'].data)
        a3 = np.asarray(ldc_data['a3LSM'].data)
        a4 = np.asarray(ldc_data['a4LSM'].data)
        ts = np.asarray(ldc_data['Teff'].data)
        lg = np.asarray(ldc_data['logg'].data)
        #zz = ldc_data['Z'].data
        #vt = ldc_data['L/HP'].data
        
        if self.tstar <= np.min(ts):
            print("Stellar temperature below minimum in limb darkening grid (",ts[0],"), using lowest values in grid.")
            return a1[0],a2[0],a3[0],a4[0]
        elif self.tstar >= np.max(ts):
            print("Stellar temperature above maximum in limb darkening grid (",ts[-1],"), using highest values in grid.")
            return a1[-1],a2[-1],a3[-1],a4[-1]
        else:
            
            f = interpolate.interp1d(ts,a1)
            a1int = f(self.tstar)
            f = interpolate.interp1d(ts,a2)
            a2int = f(self.tstar)
            f = interpolate.interp1d(ts,a3)
            a3int = f(self.tstar)
            f = interpolate.interp1d(ts,a4)
            a4int = f(self.tstar)
    
            ldc = np.asarray([a1int,a2int,a3int,a4int])
    
            return ldc

    def make_spec(self,lmin=0.5,lmax=2.43,resolution=180,smodel='blackbody',filename=None):
        """
        Function to create a star using a blackbody to approximate the 
        photosphere.
        
        Parameters
        ----------
        
        smodel : string
            'blackbody' or 'spectrum'
        
        Returns
        -------
        wavelengths : float array
            Wavelengths in microns in ascending order.
        photosphere : float array
            Photospheric flux density in mJy in ascending order.
    
        """
        
        if lmax == lmin:
            self.lmin = lmin
            self.lmax = lmax
            self.resolution = 1
            self.nwav = 1
 
            if smodel == 'blackbody':          
                wavelengths = self.lmin #um
                photosphere = Exocomet.planck_lam(wavelengths*um,self.tstar) # W/m2/sr/m
                photosphere = np.pi * photosphere * ((self.rstar*Rsol)/(self.dstar*pc))**2 # W/m2/m            
                photosphere = photosphere*1e26*(wavelengths*um)**2 /c #convert to Jy
                
            elif smodel == 'spectrum':
                lambdas,photosphere = Exocomet.read_star(filename)
                
                wavelengths = self.lmin
                photosphere = np.interp(wavelengths,lambdas,photosphere)*1e26*(wavelengths*um)**2 /c #convert to Jy
    
        else: 
            nwav = int( (lmax - lmin) / ((lmin + 0.5*(lmax-lmin)) / resolution) ) # Resolution = lambda / dlambda
            
            self.lmin = lmin 
            self.lmax = lmax
            self.resolution = resolution
            self.nwav = nwav
            
            if smodel == 'blackbody':          
                wavelengths = np.logspace(np.log10(lmin),np.log10(lmax),num=nwav,base=10.0,endpoint=True) #um
                photosphere = Exocomet.planck_lam(wavelengths*um,self.tstar) # W/m2/sr/m
                photosphere = np.pi * photosphere * ((self.rstar*Rsol)/(self.dstar*pc))**2 # W/m2/m            
                photosphere = photosphere*1e26*(wavelengths*um)**2 /c #convert to Jy
                
            elif smodel == 'spectrum':
                lambdas,photosphere = Exocomet.read_star(filename)
                
                wavelengths = np.logspace(np.log10(lmin),np.log10(lmax),num=nwav,base=10.0,endpoint=True)
                photosphere = np.interp(wavelengths,lambdas,photosphere)*1e26*(wavelengths*um)**2 /c #convert to Jy
            
        self.wave = wavelengths
        self.flux = photosphere
        
        return wavelengths, photosphere
    
    def read_star(self,filename):
        """
        Function to read a VSO stellar atmopshere from an ascii file.
        
        Parameters
        ----------
        filename : string
            filename of the stellar atmosphere model
        star : object
            object containing stellar parameters
            
        Returns 
        -------
        wave : 1D float array
            wavelengths in angstroms
        
        flux : 1D float array
            stellar photosphere model, fluxes in ergs/cm2/Hz/s.
        """
        
        data = ascii.read(filename,comment='#',names=['Wave','Flux'])
                
        wavelengths =  data['Wave'].data #Angstroms
        model_spect =  data['Flux'].data #Ergs/cm**2/s/A -> 10^-7 * 10^4 * 10^10 W/m^2/Hz/m
        
        wave = wavelengths * 1e-4 # um
        flux = model_spect * (c/wavelengths**2) * 1e3 * ((self.rstar*Rsol)/ (self.dstar*pc))**2 #coversion to Flam units
        
        return wave, flux
    
    def make_imgs(self,ldc,npix=401,nstr=75):
        """
        Function to make stellar surface image for occulation calculations.
        
        Parameters
        ----------
        ldc : 1D float array 
            limb darkening coefficients
        
        npix : integer
            total size of the image for the stellar disc 
        
        nstr : integer
            radius of the star in pixels 
        
        Returns
        -------
        intensity : 2D float array
            npix by npix image with limb darkened stellar surface, radius nstr.

        """
        
        if npix != self.npix:
            self.npix = npix
        if nstr != self.nstr:
            self.nstr = nstr
        
        xg = np.arange(npix) - (npix/2)
        yg = np.arange(npix) - (npix/2)
        x,y = np.meshgrid(xg,yg)
        r = (x**2 + y**2)**0.5
        a = np.where(r < nstr)
        mu = np.zeros(r.shape)
        ld = np.zeros(r.shape)
        mu[a] = np.abs((r[a] - nstr) / nstr)
        ld[a] = 1. - ldc[0]*(1.- mu[a]**0.5) - ldc[1]*(1.- mu[a]) - ldc[2]*(1.- mu[a]**1.5) - ldc[3]*(1.- mu[a]**2) 

        intensity = ld / np.sum(ld)
        
        self.ldimage = intensity
        
        return intensity
                    
    def make_transit(self):
        """
        

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        image_width = ((self.npix/self.nstr)*self.rstar*Rsol)
        pixel_width = image_width/self.npix
        comet_moves = self.vorb*self.timestep
        dx = comet_moves/pixel_width
        nstep = int(self.npix/dx)
        nhead = int(self.Rhead*self.nstr)
        ntail = int(self.Rtail*self.nstr)
        
        if dx < 0.5 :
            print("Comet moves < 0.5 pixels in image per iteration. Consider using a smaller image or increasing the timestep.")
            
        if nstep < 10: 
            print("Fewer than 10 realizations across the eclipse event. Consider using a larger image or decreasing the timestep.")
    
        nimg = 0
        fig2 = plt.figure()
        ims = []
        overlap = np.where(self.ldimage > 0)
        
        xc = 0
        yc = self.b*self.nstr + self.npix/2
        
        lightcurve = []
        
        while (xc < self.npix) or (overlap[0].size > 0) :
            xt = np.arange(self.npix) - xc
            yt = np.arange(self.npix) - yc
            u,v = np.meshgrid(xt,yt)
            rc = (u**2 + v**2)**0.5
            cloud = np.zeros(rc.shape)
            
            #Front side
            a = np.where(u >= 0)
            cloud[a] = self.tau_0*np.e**(-0.5*((rc[a])/nhead)**2)
            #Tail side
            a = np.where(u < 0)
            cloud[a] = self.tau_0*np.e**(-0.5*( ( (u[a]/ntail)**2 + (v[a]/nhead)**2)**0.5 )**2)
            
            overlap = np.where((self.ldimage*self.flux >= 1e-9)&(cloud >= 1e-9))
            
            #If the comet is not covering the star, then no eclipse (yet)
        
            
            if overlap[0].size == 0:
                #print(xc,yc,overlap[0].size,np.min(cloud),np.max(cloud))
                transit = self.ldimage*self.flux
                
                lightcurve.append(1.0)
                
                ims.append((plt.imshow(transit,cmap='plasma',vmin=0.0,vmax=np.max(self.ldimage*self.flux),origin='lower'),))
                    
            if overlap[0].size != 0:
                #print(xc,yc,overlap[0].size,np.min(cloud),np.max(cloud))
                transit = self.ldimage*self.flux - (cloud*(self.flux*self.ldimage))
                
                lightcurve.append(np.sum(transit)/np.sum(self.ldimage*self.flux))
                
                ims.append((plt.imshow(transit,cmap='plasma',vmin=0.0,vmax=np.max(self.ldimage*self.flux),origin='lower'),))
            
            xc = xc + dx
            nimg = nimg+1
    
        im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
            blit=True)
        im_ani.save('exocomet_test_ani.mp4', metadata={'artist':'J.P.Marshall'})
        plt.close()
        plt.clf()
        #Plot eclipse
        nmin = np.argmin(lightcurve)
        plt.plot(180.*(np.arange(0,3*nmin) - nmin),lightcurve[0:3*nmin])
        plt.xlabel("Time (s)")
        plt.ylabel("Relative intensity (arb. units)")
        plt.savefig('exocomet_test_transit.png',dpi=200)
        plt.close()

    def read_optical_constants(self,filename):
        """
        Function to read in optical constants from a text file.
        
        Returns
        -------
        dl : float array
            Wavelength array of dust optical constants in microns.
        dn : float array
            Real part of refractive index of dust optical constants.
        dk : float array
            Imaginary part of refractive index of dust optical constants.
    
        
        """    
        from astropy.io import ascii
        
        data = ascii.read(filename,comment='#')
        
        dl = data["col1"].data
        dn = data["col2"].data
        dk = data["col3"].data
        
        dust_n = np.interp(self.wave,dl,dn)
        dust_k = np.interp(self.wave,dl,dk)        
        dust_nk = dust_n - 1j*np.abs(dust_k)
        
        self.oc_nk  = dust_nk

    def calculate_opacity(self):
        """
        Function to calculate the qabs,qsca values for the grains in the model.

        Returns
        -------
        None.

        """
        
        import miepython.miepython as mpy
        
        x = 2.*np.pi*self.agrain/self.wave
        qext, qsca, qback, g = mpy.mie(self.oc_nk,x)
        
        self.qabs = qext - qsca
        self.mgrain = (3.3*(4./3.)*np.pi*(self.agrain*1e-4)**3) 
        self.kappa = np.pi*(self.agrain*1e-4)**2*self.qabs / self.mgrain
        
    def calculate_mdust(self):
        """
        Function to create a 2D array of the comet, and calculate the mass
        required to produce the extinction based on the peak extinction.

        Returns
        -------
        None.

        """
        
        pix_wide = self.rstar*Rsol/self.nstr
        pix_area = pix_wide**2
        pix_volume = pix_area*1e4*self.dstar*pc*1e2 #in cm^2
        
        nhead = self.Rhead*self.nstr
        ntail = self.Rtail*self.nstr
        
        nx = int((3*nhead+3*ntail)+1)
        ny = int(6*nhead +1)
        
        comet_model = np.array((nx,ny))
        
        xc = 3*ntail
        yc = 3*nhead
        
        xt = np.arange(self.npix) - xc
        yt = np.arange(self.npix) - yc
        u,v = np.meshgrid(xt,yt)
        rc = (u**2 + v**2)**0.5
        comet_model = np.zeros(rc.shape)
        
        #Front side
        a = np.where(u >= 0)
        comet_model[a] = self.tau_0*np.e**(-0.5*((rc[a])/nhead)**2)
        #Tail side
        a = np.where(u < 0)
        comet_model[a] = self.tau_0*np.e**(-0.5*( ( (u[a]/ntail)**2 + (v[a]/nhead)**2)**0.5 )**2)
        
        #this bit is in cgs
        rho = np.log(comet_model)/(-1.*self.kappa)/(self.dstar*pc*1e2)
        
        mass = np.sum(rho*pix_volume)*1e-3 / MEarth #cloud mass is in grams, Earth mass in kg
        
        plt.imshow(comet_model)
        
        self.mdust = mass