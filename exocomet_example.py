#Python script to create simulations of exocomet transits for Twinkle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Twinkle import Exocomet

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

model = Exocomet()

#Make spectrum of the star 
wave_star, flux_star = Exocomet.make_spec(model,lmin=1.45,lmax=1.45,resolution=-1,smodel='blackbody')

#Get limb darkening coefficients
ldc = Exocomet.get_ldc(model)

#Create image of stellar surface (total intensity = 1, will be rescaled by flux_star for transits)
image_star = Exocomet.make_imgs(model,ldc,npix=401,nstr=100)


#Define comet physcial parameters
model.Rhead = 0.5 #Radial extent of dust cloud in fraction of stellar radius
model.Rtail = 2.0 #Length of dust tail in fraction of stellar radius
model.tau_0 = 1.0 #Maximum opacity of dust cloud
#Define comet orbit parameters for modelling
model.vorb = (2.*G*model.mstar*Msol /(model.a*rsol) )**0.5 # relative velcoity to move cloud between timesteps
model.timestep = 180.0 #seconds - sampling of time series
model.agrain = 1.0 #grain size of dust in um


#Read in dust properties, caclulate opacity
Exocomet.read_optical_constants(model,'astrosil.lnk')
Exocomet.calculate_opacity(model)
Exocomet.make_transit(model)
Exocomet.calculate_mdust(model)