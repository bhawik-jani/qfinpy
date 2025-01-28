import numpy as np
from .core import sliding_window
from .core import chi2_cdf


def acovf(series, nlags):
    if not isinstance(nlags, int) or nlags<0:
        raise RuntimeError("nlags should be an integer >=0")
    series = np.atleast_1d(series)
    n = series.shape[-1]
    xs = np.atleast_1d(series.mean(axis=-1, keepdims=True))
    l = []
    for i in range(nlags+1):
        l.append( ((series[...,i:]-xs)*(series[...,:n-i]-xs)).sum(axis=-1, keepdims=True) / (n-i) )
    return np.concatenate(l, axis=-1)

def acf(series, nlags, qstat=False, dof_offset=0):
    if not isinstance(nlags, int) or nlags<0:
        raise RuntimeError("nlags should be an integer >=0")
    series = np.atleast_1d(series)
    temp = acovf(series, nlags)
    temp /= temp[...,0:1]
    if not qstat:
        return temp
    n = series.shape[-1]
    h = temp.shape[-1]
    q = n*(n+2)*np.cumsum(temp[...,1:]**2/np.arange(n-1,n-h,-1), axis=-1)
    p = 1 - chi2_cdf(q,np.arange(1-dof_offset,h-dof_offset))
    return temp, q, p

def ma(series, theta, mu=0.0, e0=None):
    series = np.atleast_1d(series)
    theta = np.atleast_1d(theta)
    if theta.ndim==2 and theta[0,:].size>1:
        theta = np.expand_dims(theta, 0)
    if theta.ndim==1:
        theta = np.expand_dims(theta, -1)
    len_theta = theta.shape[0]
    theta = theta[::-1]
    if e0 is None:
        e0 = 0.0
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

def ar(series, phi, mu=0.0, x0=None):
    series = np.atleast_1d(series)
    phi = np.atleast_1d(phi)
    if phi.ndim==2 and phi[...,-1].size>1:
        phi = np.expand_dims(phi, 0)
    if phi.ndim==1:
        len_phi = phi.size
        phi = np.concatenate((phi[::-1], np.array([1.0])))
    else:
        len_phi = phi.shape[0]
        phi = np.concatenate((phi[::-1], np.expand_dims(np.broadcast_to(np.eye(phi.shape[-1]), phi[0].shape), 0)), axis=0)
    if series.ndim<phi.ndim-1:
        raise RuntimeError("series.ndim<phi.ndim-1")
    for i in range(series.ndim-phi.ndim+1):
        phi = np.expand_dims(phi, 1) 
    mu = np.array(mu)
    mu = np.expand_dims(mu, -1)
    mu = np.broadcast_to(mu, series.shape[:-1]+(1,))
    out = np.empty(series.shape[:-1] + (series.shape[-1]+len_phi,))
    if x0 is None:
        out[...,:len_phi] = mu
    else:
        x0 = np.atleast_1d(x0)
        if x0.ndim<series.ndim:
            x0 = np.broadcast_to(x0, series.shape[:-1] + (x0.shape[-1],))
        elif series.ndim<x0.ndim:
            series = np.broadcast_to(series, x0.shape[:-1] + (series.shape[-1],))
        if x0.shape[-1]<len_phi:
            raise RuntimeError("x0.shape[-1] should be greater than or equal to phi.shape[0]")
        out[...,:len_phi] = x0[...,-len_phi:]
    psum = phi[-1]-phi[:-1].sum(axis=0)
    if phi.shape[-1]==1:
        out[...,len_phi:] = psum*mu + series
        for i in range(len_phi,out.shape[-1]):
            out[...,i] = (out[...,i-len_phi:i+1]*np.moveaxis(phi,0,-1)).sum(axis=-1)
        return out[...,len_phi:]
    out[...,len_phi:] = psum@mu + series
    for i in range(len_phi,out.shape[-1]):
        out[...,i] = (np.moveaxis(phi,0,-3) @ np.expand_dims(np.moveaxis(out[...,i-len_phi:i+1],-1,-2),-1)).squeeze(-1).sum(axis=-2)
    return out[...,len_phi:]

def gh(series, w, alpha, beta, x0=None, e0=None, mu=0, initial_var=None):
    series = np.atleast_1d(series)
    beta = np.atleast_1d(beta)
    len_beta = beta.shape[-1]
    beta = beta[...,::-1]
    alpha = np.atleast_1d(alpha)
    len_alpha = alpha.shape[-1]
    alpha = alpha[...,::-1]
    len_max = max(len_alpha, len_beta)
    var = np.empty(series.shape[:-1] + (series.shape[-1]+len_beta,))
    out = np.empty(series.shape[:-1] + (series.shape[-1]+len_alpha,))
    mu = np.asarray(mu)
    mu = np.expand_dims(mu, -1)
    mu = np.broadcast_to(mu, series.shape[:-1] + (1,))
    w = np.asarray(w)
    if initial_var is None:
        var[...,:len_beta] = w/(1-alpha.sum()-beta.sum())
    else:
        var[...,:len_beta] = initial_var
    if x0 is None:
        out[...,:len_alpha] = 0.0
    else:
        x0 = np.atleast_1d(x0)
        if x0.ndim<series.ndim:
            x0 = np.broadcast_to(x0, series.shape[:-1] + (x0.shape[-1],))
        elif series.ndim<x0.ndim:
            series = np.broadcast_to(series, x0.shape[:-1] + (series.shape[-1],))
        if x0.shape[-1]<len_alpha:
            raise RuntimeError("x0.shape[-1] should be greater than or equal to alpha.shape[0]")
        if e0 is not None:
            if x0.shape[-1]<len_max:
                raise RuntimeError("x0.shape[-1] should be greater than or equal to max(alpha.shape[0], beta.shape[0])")
            out[...,:len_alpha] = x0[...,-len_alpha:] - mu
            e0 = np.atleast_1d(e0)
            if e0.ndim<=series.ndim:
                e0 = np.broadcast_to(e0, series.shape[:-1] + (e0.shape[-1],))
            else:
                series = np.broadcast_to(series, e0.shape[:-1] + (series.shape[-1],))
            if e0.shape[-1]<len_beta:
                try:
                    e0 = np.broadcast_to(e0, e0.shape[:-1] + (len_beta,))
                except:
                    raise RuntimeError("e0.shape[-1] should be 1 or equal to beta.shape[0]")
            var[...,:len_beta] = x0[...,-len_beta:]/e0[...,-len_beta:]
    for i in range(out.shape[-1]-len_alpha):
        var[...,len_beta+i] = np.maximum(1e-10, w + (alpha * out[...,i:len_alpha+i]**2).sum(axis=-1) + \
        (beta * var[...,i:len_beta+i]).sum(axis=-1))
        out[...,len_alpha+i] = series[...,i] * var[...,len_beta+i]**0.5
    return out[...,len_alpha:] + mu, var[...,len_beta:]

def ma_inverse(series, theta, mu=0.0):
    series = np.atleast_1d(series)
    theta = -np.atleast_1d(theta)
    mu = np.asarray(mu)
    mu = np.expand_dims(mu, -1)
    mu = np.broadcast_to(mu, series.shape[:-1] + (1,))
    series -= mu
    return ar(series[...,theta.size:], theta, mu=0.0, x0=series[...,:theta.size])

def ar_inverse(series, phi, mu=0.0):
    series = np.atleast_1d(series)
    phi = -np.atleast_1d(phi)
    mu = np.asarray(mu)
    mu = np.expand_dims(mu, -1)
    mu = np.broadcast_to(mu, series.shape[:-1] + (1,))
    series -= mu
    return ma(series[...,phi.size:], phi, mu=0.0, e0=series[...,:phi.size])

def gh_inverse(series, w, alpha, beta, mu=0, initial_var=None):
    series = np.atleast_1d(series)
    beta = np.atleast_1d(beta)
    len_beta = beta.shape[-1]
    beta = beta[::-1]
    alpha = np.atleast_1d(alpha)
    len_alpha = alpha.shape[-1]
    alpha = alpha[::-1]
    len_max = max(len_alpha, len_beta)
    var = np.empty(series.shape[:-1]+(series.shape[-1]-len_alpha+len_beta,))
    out = np.empty(series.shape[:-1]+(series.shape[-1]-len_alpha,))
    mu = np.asarray(mu)
    mu = np.expand_dims(mu, -1)
    mu = np.broadcast_to(mu, series.shape[:-1] + (1,))
    w = np.asarray(w)
    if initial_var is None:
        var[...,:len_beta] = w/(1-alpha.sum()-beta.sum())
    else:
        var[...,:len_beta] = initial_var
    series = series-mu 
    for i in range(out.shape[-1]):
        var[...,len_beta+i] = np.maximum(1e-10, w + (alpha * series[...,i:len_alpha+i]**2).sum(axis=-1) + (beta * var[...,i:len_beta+i]).sum(axis=-1))
        out[...,i] = series[...,len_alpha+i] / var[...,len_beta+i]**0.5
    return out, var[...,len_beta:]

