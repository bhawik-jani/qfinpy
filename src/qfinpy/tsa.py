import numpy as np
from .core import sliding_window
from numba import jit, prange


def ma(series, theta, mu=0.0, e0=None):
    series = np.atleast_1d(series)
    theta = np.atleast_1d(theta)
    if theta.ndim==2 and theta[0,:].size>1:
        theta = np.expand_dims(theta, 0)
    if theta.ndim==1:
        theta = np.expand_dims(theta, -1)
    len_theta = theta.shape[0]
    theta = theta[::-1]
    if e0 is None and series.shape[-1] < len_theta+1:
        raise RuntimeError("when e0=None, series.shape[-1] should be greater than theta.shape[0]")
    if e0 is not None:
        e0 = np.atleast_1d(e0)
        if e0.ndim<=series.ndim:
            e0 = np.broadcast_to(e0, series.shape[:-1] + (e0.shape[-1],))
        else:
            series = np.broadcast_to(series, e0.shape[:-1] + (series.shape[-1],))
        if e0.shape[-1]<len_theta:
            try:
                e0 = np.broadcast_to(e0, e0.shape[:-1] + (len_theta,))
            except:
                raise RuntimeError("e0.shape[-1] should be 1 or equal to theta.shape[0]")
        series = np.concatenate((e0[...,-len_theta:], series), axis=-1)
    if series.ndim<theta.ndim-1:
        raise RuntimeError("series.ndim<theta.ndim-1")
    for i in range(series.ndim-theta.ndim+1):
        theta = np.expand_dims(theta, 1) 
    mu = np.asarray(mu)
    mu = np.expand_dims(mu, -1)
    mu = np.broadcast_to(mu, series.shape[:-1] + (1,))
    tmp_series = sliding_window(series[...,:-1], len_theta)
    if theta.shape[-1]==1:
        return (series + mu)[...,len_theta:] + (theta * np.moveaxis(tmp_series, -1,0)).sum(axis=0)
    return (series + mu)[...,len_theta:] + (theta @ np.moveaxis(tmp_series, -1,0)).sum(axis=0)

@jit(nopython=True, parallel=True)
def _numba_ar_univariate(series, phi, out, out_x, mu, len_phi, n):
    for i in prange(len_phi,n):
        tmp = phi * out_x[i-len_phi:i,...]
        s = tmp[0,...,0]
        for j in prange(1,len_phi):
            s += tmp[j,...,0]
        out[...,i] = series[...,i-len_phi] + mu + s
    return out

@jit(nopython=True, parallel=True)
def _numba_ar_multivariate(series, phi, out, out_x, mu, len_phi, n, k):
    for i in prange(len_phi,n):
        tmp = np.zeros_like(out_x[i-len_phi:i,...])
        for l in prange(k):
            for m in prange(k):
                tmp[...,l,0] += phi[...,l,m] * out_x[i-len_phi:i,...,m,0] 
        s = tmp[0,...,0]
        for j in prange(1,len_phi):
            s += tmp[j,...,0]
        out[...,i] = series[...,i-len_phi] + mu + s
    return out

def ar(series, phi, mu=0.0, x0=None):
    series = np.atleast_1d(series)
    phi = np.atleast_1d(phi)
    if phi.ndim==2 and phi[...,-1].size>1:
        phi = np.expand_dims(phi, 0)
    if phi.ndim==1:
        len_phi = phi.size
    else:
        len_phi = phi.shape[0]
    phi = phi[::-1]
    if series.ndim<phi.ndim-1:
        raise RuntimeError("series.ndim<phi.ndim-1")
    for i in range(series.ndim-phi.ndim+1):
        phi = np.expand_dims(phi, 1) 
    mu = np.array(mu)
    mu = np.expand_dims(mu, -1)
    mu = np.broadcast_to(mu, series.shape[:-1]+(1,))
    out = np.empty(series.shape[:-1] + (series.shape[-1]+len_phi,))
    if x0 is None:
        out[...,:len_phi] = 0.0
    else:
        x0 = np.atleast_1d(x0)
        if x0.ndim<series.ndim:
            x0 = np.broadcast_to(x0, series.shape[:-1] + (x0.shape[-1],))
        elif series.ndim<x0.ndim:
            series = np.broadcast_to(series, x0.shape[:-1] + (series.shape[-1],))
        if x0.shape[-1]<len_phi:
            raise RuntimeError("x0.shape[-1] should be greater than or equal to phi.shape[0]")
        out[...,:len_phi] = x0[...,-len_phi:]-mu
    out_x = np.expand_dims(np.moveaxis(out, -1,0), -1)
    phi = np.broadcast_to(phi, (phi.shape[0],)+out_x.shape[1:-1]+(phi.shape[-1],))
    if phi.shape[-1]==1:
        return _numba_ar_univariate(series, phi, out, out_x , mu.squeeze(-1), len_phi, out.shape[-1])[...,len_phi:]
    return _numba_ar_multivariate(series, phi, out, out_x, mu.squeeze(-1), len_phi, out.shape[-1], phi.shape[-1])[...,len_phi:]

def ma_inverse(series, theta, mu=0.0):
    series = np.atleast_1d(series)
    theta = -np.atleast_1d(theta)
    mu = np.asarray(mu)
    mu = np.expand_dims(mu, -1)
    mu = np.broadcast_to(mu, series.shape[:-1] + (1,))
    series -= mu
    return ar(series, theta, mu=0.0)

def ar_inverse(series, phi, mu=0.0):
    series = np.atleast_1d(series)
    phi = -np.atleast_1d(phi)
    mu = np.asarray(mu)
    mu = np.expand_dims(mu, -1)
    mu = np.broadcast_to(mu, series.shape[:-1] + (1,))
    series -= mu
    return ma(series, phi, mu=0.0, e0=0.0)


