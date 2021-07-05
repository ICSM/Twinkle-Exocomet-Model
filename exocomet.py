#Python script to create simulations of exocomet transits for Twinkle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

direc = '/Users/jonty/mydata/exocomets/'

#Constants
c = 3e8 #m/s
G = 6.67e-11
Rsol = 6.9634e8 #m
Msol = 1.99e30 #kg
Lsol = 6e24 #W
pc = 3.086e16 #m
au = 1.496e11 #m
#Load limb darkening parameters
from astropy.io import ascii

#ldc_file = 'Claret2017_TESS_LDC.dat'
#ldc_data = ascii.read(direc+ldc_file,delimiter=';',comment='#',fast_read=False)

# a1 = ldc_data['a1'].data
# a2 = ldc_data['a2'].data
# a3 = ldc_data['a3'].data
# a4 = ldc_data['a4'].data
# ts = ldc_data['teff'].data
# lg = ldc_data['logg'].data
# zz = ldc_data['zfeh'].data

#Insert process to select nearest LDC based on stellar properties

a1,a2,a3,a4 = (0.5712,-0.1025,0.4745,-0.2657) #coefficients for 5750K, logg 4.5, FeH = 0.0, xi = 2 km/s

#Define stellar parameters for modelling
dstar = 10.0 #pc
tstar = 5770.0 #K
rstar = 1.0 #R_sol
mstar = 1.0 #M_sol
lstar = 1.0 #L_sol

#Define comet orbit parameters for modelling
a = 0.05 #au
e = 0.996
incl  = 0.0 #degrees
theta = 0.0 #degrees

rorb = a / (1 +e*np.cos((np.pi/180.)*theta))
vorb = (2.*G*mstar*Msol /(rorb*au) )**0.5

timestep = 180.0 #seconds

#Define comet physcial parameters
Rcom = 1e-3*rstar #Radial extent of planetesimal
Rcld = 0.5 #Radial extent of dust cloud in fraction of stellar radius
Rtal = 2.0 #Length of dust tail in fraction of stellar radius
Qcld = 0.1 #Opacity of dust cloud

#Define image space for transit calculation
npix = 401
nstr = 75
ncld = Rcld*nstr
ntal = Rtal*nstr

xg = np.arange(npix) - (npix/2)
yg = np.arange(npix) - (npix/2)
x,y = np.meshgrid(xg,yg)
r = (x**2 + y**2)**0.5
a = np.where(r < nstr)
mu = np.zeros(r.shape)
ld = np.zeros(r.shape)
mu[a] = np.abs((r[a] - nstr) / nstr)
ld[a] = 1. - a1*(1.- mu[a]**0.5) - a2*(1.- mu[a]) - a3*(1.- mu[a]**1.5) - a4*(1.- mu[a]**2) 

intensity = []


xc = 0

#work out if the size of the image is big enough to get decent sampling of the transit
image_width = ((npix/nstr)*rstar*Rsol)
pixel_width = image_width/npix
comet_moves = vorb*timestep
dx = comet_moves/pixel_width
nstep = int(npix/dx)

if dx < 0.5 :
    print("Comet moves < 0.5 pixels in image per iteration. Consider using a smaller image or increasing the timestep.")

if nstep < 10: 
    print("Fewer than 10 realizations across the eclipse event. Consider using a larger image or decreasing the timestep.")



nimg = 0
fig2 = plt.figure()
ims = []
overlap = np.where(ld > 0)
while (xc < npix) or (overlap[0].size > 0) :
    yc = npix/2
    
    xt = np.arange(npix) - xc
    yt = np.arange(npix) - yc
    u,v = np.meshgrid(xt,yt)
    rc = (u**2 + v**2)**0.5
    cloud = np.zeros(rc.shape)
    
    #Front side
    a = np.where(u >= 0)
    cloud[a] = Qcld*np.e**(-0.5*((rc[a])/ncld)**2)
    #Tail side
    a = np.where(u < 0)
    cloud[a] = Qcld*np.e**(-0.5*( ( (u[a]/ntal)**2 + (v[a]/ncld)**2)**0.5 )**2)
    
    overlap = np.where((ld != 0)&(cloud != 0))
    
    #print(xc, npix)
    #If the comet is not covering the star, then no eclipse (yet)

    
    if overlap[0].size == 0:
        transit = ld        
        
        intensity.append(1.0)
        
        ims.append((plt.imshow(ld,cmap='plasma',vmin=0.0,vmax=np.max(ld),origin='lower'),))
            
    if overlap[0].size != 0:
        transit = ld - (cloud*ld)
        
        intensity.append(np.sum(transit)/np.sum(ld))
        
        ims.append((plt.imshow(transit,cmap='plasma',vmin=0.0,vmax=np.max(ld),origin='lower'),))

        # plt.imshow(transit,origin='lower')
        # plt.savefig(direc+'exocomet_test_frame'+str(nimg)+'.png',dpi=200,overwrite=True)
        # plt.close()
        # plt.clf()
    xc = xc + dx
    nimg = nimg+1

im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
    blit=True)
im_ani.save(direc+'exocomet_test_ani.mp4', metadata={'artist':'J.P.Marshall'})
plt.close()
plt.clf()
#Plot eclipse
nmin = np.argmin(intensity)
plt.plot(180.*(np.arange(0,3*nmin) - nmin),intensity[0:3*nmin])
plt.xlabel("Time (s)")
plt.ylabel("Relative intensity (arb. units)")
plt.savefig(direc+'exocomet_test_transit.png',dpi=200,overwrite=True)
plt.close()