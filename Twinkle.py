#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:42:22 2021

@author: jonty
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.models import Blackbody

#Constants
c = 3e8 #m/s
G = 6.67e-11
Rsol = 6.9634e8 #m
Msol = 1.99e30 #kg
Lsol = 6e24 #W
pc = 3.086e16 #m
au = 1.496e11 #m

class Exocomet:

    def __init__(self):
        print("Instantiated Twinkle exocomet model object.")
        self.parameters = {}
        self.sed_emit = 0.0
        self.sed_scat = 0.0
        self.sed_disc = 0.0 
        self.sed_star = 0.0
        self.sed_wave = 0.0

    def read_star(self):
        """
        Function to read in a stellar photosphere model from the SVO database.
        
        Stellar photosphere model is assumed to have wavelengths in Angstroms,
        and flux density in erg/s/cm2/A.
        
        The function will extrapolate to the longest wavelength required in the
        model, if necessary.
        
        Parameters
        ----------
        star_params : Dictionary
             Stellar parameters.
    
        Returns
        -------
        wav_um : float array
            Wavelengths in microns in ascending order.
        flx_mjy : float array
            Photospheric flux density in mJy in ascending order.
    
        
        """
        spectrum_file = self.parameters['model']
        
        data = ascii.read(spectrum_file,comment='#',names=['Wave','Flux'])
        
        rstar = self.parameters['rstar']
        dstar = self.parameters['dstar']
        
        wavelengths =  data['Wave'].data #Angstroms
        model_spect =  data['Flux'].data #Ergs/cm**2/s/A -> 10^-7 * 10^4 * 10^10 W/m^2/Hz/m
        
        wav_um = wavelengths * 1e-4
        flx_mjy =  model_spect * (c/wavelengths**2) * 1e-3 * ((rstar*rsol)/ (dstar*pc))**2 #coversion to Flam units
        
        return wav_um,flx_mjy

    def make_star(star_parameters,smodel='blackbody'):
        """
        Function to either create a star using a blackbody, or read in a 
        photosphere model.
        
        Returns
        -------
        wavelengths : float array
            Wavelengths in microns in ascending order.
        photosphere : float array
            Photospheric flux density in mJy in ascending order.
    
        """
        if smodel != 'blackbody' and smodel != 'spectrum' :
            print("Input 'stype' must be one of 'blackbody' or 'spectrum'.")
    
        if smodel == 'blackbody':
            lstar = self.parameters['lstar']
            rstar = self.parameters['rstar']
            tstar = self.parameters['tstar']
            dstar = self.parameters['dstar']
    
            lmin = self.parameters['lmin']
            lmax = self.parameters['lmax']
            nwav = int(self.parameters['nwav'])
            
            wavelengths = np.logspace(np.log10(lmin),np.log10(lmax),num=nwav,base=10.0,endpoint=True) #um
            photosphere = RTModel.planck_lam(wavelengths*um,tstar) # W/m2/sr/m
            photosphere = np.pi * photosphere * ((rstar*rsol)/(dstar*pc))**2 # W/m2/m
            
            lphot = RTModel.calc_luminosity(rstar,tstar)
            print("Stellar model has a luminosity of: ",lphot," L_sol")
            
            photosphere = (lstar/lphot)*photosphere
            
        elif smodel == 'spectrum':
            lambdas,photosphere = read_star(self)
    
            lmin = self.parameters['lmin']
            lmax = self.parameters['lmax']
            nwav = int(self.parameters['nwav'])
            
            wavelengths = np.logspace(np.log10(lmin),np.log10(lmax),num=nwav,base=10.0,endpoint=True)
    
            if np.max(wavelengths) > np.max(lambdas):
                interp_lam_arr = np.logspace(np.log10(lambdas[-1]),np.log10(1.1*wavelengths[-1]),num=nwav,base=10.0,endpoint=True)
                interp_pht_arr = photosphere[-1]*(lambdas[-1]/interp_lam_arr)**2
                photosphere = np.append(photosphere,interp_pht_arr)
                lambdas = np.append(lambdas,interp_lam_arr)
                
            photosphere = np.interp(wavelengths,lambdas,photosphere)
        
        elif smodel == 'function':
            print("starfish model not yet implemented.")
        
        self.sed_wave = wavelengths 
        self.sed_star = photosphere*1e3*1e26*(self.sed_wave*um)**2 /c
        
        return wavelengths, photosphere
