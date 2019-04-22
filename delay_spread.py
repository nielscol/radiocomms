import matplotlib.pyplot as plt
import numpy as np

L = 8.6
H = 2.5
W = 6.0

lin_divs = 101

DX = L/lin_divs
DY = W/lin_divs
DZ = H/lin_divs

p1 = {
    "x" : 1.0,
    "y" : 3.0,
    "z" : 1.0,
}


p2 = {
    "x" : 7.6,
    "y" : 3.0,
    "z" : 1.0,
}


pls = []

def p_dist(p1,p2):
    d = 0
    for k in p1:
        d += (p1[k] - p2[k])**2
    return np.sqrt(d)

""" x walls
"""
for y in np.linspace(0, W, lin_divs, endpoint=False):
    for z in np.linspace(0, H, lin_divs, endpoint=False):
        p0 = dict(x=0, y=y, z=z)
        pls.append(p_dist(p0, p1) + p_dist(p0,p2))
        p0 = dict(x=L, y=y, z=z)
        pls.append(p_dist(p0, p1) + p_dist(p0,p2))
""" y walls
"""
for x in np.linspace(0, L, lin_divs, endpoint=False):
    for z in np.linspace(0, H, lin_divs, endpoint=False):
        p0 = dict(x=x, y=0, z=z)
        pls.append(p_dist(p0, p1) + p_dist(p0,p2))
        p0 = dict(x=x, y=W, z=z)
        pls.append(p_dist(p0, p1) + p_dist(p0,p2))
""" z walls
"""
for x in np.linspace(0, L, lin_divs, endpoint=False):
    for y in np.linspace(0, W, lin_divs, endpoint=False):
        p0 = dict(x=x, y=y, z=0)
        pls.append(p_dist(p0, p1) + p_dist(p0,p2))
        p0 = dict(x=x, y=y, z=H)
        pls.append(p_dist(p0, p1) + p_dist(p0,p2))

_min = np.amin(pls)
_max = np.amax(pls)

n_bins = 101
_del = (_max-_min)/(n_bins-1)
full_n_bins = int(round((1.5*_max/_del)+1))
bins = np.round(np.linspace(0, 1.5*_max, full_n_bins),3)
hist = {_bin:0 for _bin in bins}

def phi(zenith, x, y, z):
    if zenith == "x":
        ph = np.arccos(x/np.sqrt(x**2+y**2+z**2))
    elif zenith == "y":
        ph = np.arccos(y/np.sqrt(x**2+y**2+z**2))
    elif zenith == "z":
        ph = np.arccos(z/np.sqrt(x**2+y**2+z**2))
    if np.isnan(ph):
        return 0.0
    else:
        return ph

def theta(zenith, x, y, z):
    if zenith == "x":
        th = np.arccos(y/np.sqrt(z**2+y**2))
    elif zenith == "y":
        th = np.arccos(z/np.sqrt(x**2+z**2))
    elif zenith == "z":
        th = np.arccos(x/np.sqrt(x**2+y**2))
    if np.isnan(th):
        return 0.0
    else:
        return th

def subtended_portion(p1, p2, dx, dy, dz):
    """ determine approx. angular solid subtended from source point p1 to sub area
        on box walls at p2 with dimensions {dx, dy, dz} (pick two)
    """
    thetas = []
    phis = []
    if dx and dy:
        zenith = "z"
        d = {k:p2[k]-p1[k] for k in p1}
        sin_phi = np.sqrt(d["x"]**2 + d["y"]**2)/np.sqrt(d["x"]**2 + d["y"]**2 + d["z"]**2)
        thetas.append(theta(zenith, **d))
        phis.append(phi(zenith, **d))
        offset = dict(x=dx, y=0, z=0)
        d = {k:p2[k]-p1[k]+offset[k] for k in p1}
        thetas.append(theta(zenith, **d))
        phis.append(phi(zenith, **d))
        offset = dict(x=0, y=dy, z=0)
        d = {k:p2[k]-p1[k]+offset[k] for k in p1}
        thetas.append(theta(zenith, **d))
        phis.append(phi(zenith, **d))
        offset = dict(x=dx, y=dy, z=0)
        d = {k:p2[k]-p1[k]+offset[k] for k in p1}
        thetas.append(theta(zenith, **d))
        phis.append(phi(zenith, **d))
    elif dy and dz:
        zenith = "x"
        d = {k:p2[k]-p1[k] for k in p1}
        sin_phi = np.sqrt(d["z"]**2 + d["y"]**2)/np.sqrt(d["x"]**2 + d["y"]**2 + d["z"]**2)
        thetas.append(theta(zenith, **d))
        phis.append(phi(zenith, **d))
        offset = dict(x=0, y=dy, z=0)
        d = {k:p2[k]-p1[k]+offset[k] for k in p1}
        thetas.append(theta(zenith, **d))
        phis.append(phi(zenith, **d))
        offset = dict(x=0, y=0, z=dz)
        d = {k:p2[k]-p1[k]+offset[k] for k in p1}
        thetas.append(theta(zenith, **d))
        phis.append(phi(zenith, **d))
        offset = dict(x=0, y=dy, z=dz)
        d = {k:p2[k]-p1[k]+offset[k] for k in p1}
        thetas.append(theta(zenith, **d))
        phis.append(phi(zenith, **d))
    elif dz and dx:
        zenith = "y"
        d = {k:p2[k]-p1[k] for k in p1}
        sin_phi = np.sqrt(d["x"]**2 + d["z"]**2)/np.sqrt(d["x"]**2 + d["y"]**2 + d["z"]**2)
        thetas.append(theta(zenith, **d))
        phis.append(phi(zenith, **d))
        offset = dict(x=dx, y=0, z=0)
        d = {k:p2[k]-p1[k]+offset[k] for k in p1}
        thetas.append(theta(zenith, **d))
        phis.append(phi(zenith, **d))
        offset = dict(x=0, y=0, z=dz)
        d = {k:p2[k]-p1[k]+offset[k] for k in p1}
        thetas.append(theta(zenith, **d))
        phis.append(phi(zenith, **d))
        offset = dict(x=dx, y=0, z=dz)
        d = {k:p2[k]-p1[k]+offset[k] for k in p1}
        thetas.append(theta(zenith, **d))
        phis.append(phi(zenith, **d))
    d_theta = np.abs(np.ptp(thetas))
    d_phi = np.abs(np.ptp(phis))
    return sin_phi*d_phi*d_theta/(4.0*np.pi)


""" x walls
"""
for y in np.linspace(0, W, lin_divs, endpoint=False):
    for z in np.linspace(0, H, lin_divs, endpoint=False):
        p0 = dict(x=0, y=y, z=z)
        pl = p_dist(p0, p1) + p_dist(p0,p2)
        density = subtended_portion(p1, p0, dx=0, dy=DY, dz=DZ)*subtended_portion(p2, p0, dx=0, dy=DY, dz=DZ)
        hist_bin = bins[np.argmin(np.abs(bins-pl))]
        hist[hist_bin] += density
        p0 = dict(x=L, y=y, z=z)
        pl = p_dist(p0, p1) + p_dist(p0,p2)
        density = subtended_portion(p1, p0, dx=0, dy=DY, dz=DZ)*subtended_portion(p2, p0, dx=0, dy=DY, dz=DZ)
        hist_bin = bins[np.argmin(np.abs(bins-pl))]
        hist[hist_bin] += density
""" y walls
"""
for x in np.linspace(0, L, lin_divs, endpoint=False):
    for z in np.linspace(0, H, lin_divs, endpoint=False):
        p0 = dict(x=x, y=0, z=z)
        pl = p_dist(p0, p1) + p_dist(p0,p2)
        density = subtended_portion(p1, p0, dx=DX, dy=0, dz=DZ)*subtended_portion(p2, p0, dx=DX, dy=0, dz=DZ)
        hist_bin = bins[np.argmin(np.abs(bins-pl))]
        hist[hist_bin] += density
        p0 = dict(x=x, y=W, z=z)
        pl = p_dist(p0, p1) + p_dist(p0,p2)
        density = subtended_portion(p1, p0, dx=DX, dy=0, dz=DZ)*subtended_portion(p2, p0, dx=DX, dy=0, dz=DZ)
        hist_bin = bins[np.argmin(np.abs(bins-pl))]
        hist[hist_bin] += density
""" z walls
"""
for x in np.linspace(0, L, lin_divs, endpoint=False):
    for y in np.linspace(0, W, lin_divs, endpoint=False):
        p0 = dict(x=x, y=y, z=0)
        pl = p_dist(p0, p1) + p_dist(p0,p2)
        density = subtended_portion(p1, p0, dx=DX, dy=DY, dz=0)*subtended_portion(p2, p0, dx=DX, dy=DY, dz=0)
        hist_bin = bins[np.argmin(np.abs(bins-pl))]
        hist[hist_bin] += density
        p0 = dict(x=x, y=y, z=H)
        pl = p_dist(p0, p1) + p_dist(p0,p2)
        density = subtended_portion(p1, p0, dx=DX, dy=DY, dz=0)*subtended_portion(p2, p0, dx=DX, dy=DY, dz=0)
        hist_bin = bins[np.argmin(np.abs(bins-pl))]
        hist[hist_bin] += density

density = np.array([v for k,v in hist.items()])
density /= np.amax(density)
#density += 0.1

normed_density = density/np.sum(density)
mean = np.sum(normed_density*bins)
variance = np.sum(normed_density*(bins-mean)**2)
stdev = np.sqrt(variance)
nz = np.where(density>0)
ptp = bins[nz[0][-1]] - bins[nz[0][0]]
print(ptp)
print("mean=%.2f ns, delay spread=%.2f ns, rms delay spread=%.2f ns"%(1e9*mean/3e8,1e9*ptp/3e8,1e9*stdev/3e8))

plt.plot(1e9*bins/3e8, density)
plt.xlabel("Time [ns]")
plt.ylabel("Power [relative]")
plt.title("Multipath propagation delay spread")
plt.show()
