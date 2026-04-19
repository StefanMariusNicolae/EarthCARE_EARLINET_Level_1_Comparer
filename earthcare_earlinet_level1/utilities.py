# utility functions to be used throughout the project
import numpy as np
from constants import EARTH_RADIUS_KM
from scipy import stats as _scipy_stats

def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

def moving_nanmean_1d(a, k):
    if k <= 1: return np.asarray(a, float).copy()
    w = np.ones(int(k))
    v = np.isfinite(a)
    num = np.convolve(np.where(v, a, 0.0), w, mode="same")
    den = np.convolve(v.astype(float), w, mode="same")
    out = num/den
    out[den == 0] = np.nan
    return out

def lin_ccc(x, y):
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    mx, my = x.mean(), y.mean()
    r = np.corrcoef(x, y)[0,1]
    return (2*r*np.sqrt(vx*vy)) / (vx + vy + (mx-my)**2)

def weighted_rmse(x, y, sx=None, sy=None):
    if sx is None or sy is None: return np.nan
    w = 1.0 / np.maximum(sx**2 + sy**2, 1e-12)
    d = x - y
    return np.sqrt(np.nansum(w*d**2) / np.nansum(w))

def reduced_chi2(x, y, sx=None, sy=None, remove_bias=True):
    if sx is None or sy is None: return np.nan
    d = x - y
    if remove_bias:
        d = d - np.nanmean(d)
        dof = max(np.isfinite(d).sum() - 1, 1)
    else:
        dof = max(np.isfinite(d).sum(), 1)
    var = np.maximum(sx**2 + sy**2, 1e-12)
    return np.nansum((d**2)/var) / dof

def tost_equivalence(diff, delta=0.05, alpha=0.05):

    d = diff[np.isfinite(diff)]
    if d.size < 3:
        return False, (np.nan, np.nan)
    se = d.std(ddof=1) / np.sqrt(d.size)
    tcrit = _scipy_stats.t.ppf(1-alpha, df=d.size-1)
    lo = d.mean() - tcrit*se
    hi = d.mean() + tcrit*se
    return (lo > -delta) and (hi < delta), (lo, hi)
