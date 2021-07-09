#Python script to create simulations of exocomet transits for Twinkle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Twinkle import Exocomet

direc = '/Users/jonty/mydata/exocomets/'

#Constants
c = 3e8 #m/s
G = 6.67e-11
Rsol = 6.9634e8 #m
Msol = 1.99e30 #kg
Lsol = 6e24 #W
pc = 3.086e16 #m
au = 1.496e11 #m

model = Exocomet()

#Make spectrum of the star 
wave_star, flux_star = Exocomet.make_spec(model,lmin=1.45,lmax=1.45,resolution=-1,smodel='blackbody')

#Get limb darkening coefficients
ldc = Exocomet.get_ldc(model)

#Create image of stellar surface (total intensity = 1, will be rescaled by flux_star for transits)
image_star = Exocomet.make_imgs(model,ldc,npix=401,nstr=75)

#Define comet orbit parameters for modelling
exocomet = Exocomet()

vorb = (2.*G*mstar*Msol /(rorb*rsol) )**0.5
timestep = 180.0 #seconds - sampling of time series

#Define comet physcial parameters
Rhead = 0.5 #Radial extent of dust cloud in fraction of stellar radius
Rtail = 2.0 #Length of dust tail in fraction of stellar radius
tau_0 = 0.1 #Maximum opacity of dust cloud

Exocomet.make_transit(model,intensity)