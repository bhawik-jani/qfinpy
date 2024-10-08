import math
import numpy as np
from matplotlib import pyplot as plt
import statistics
from scipy.optimize import newton
from functools import partial
import scipy.optimize as opt
import copy


def rate_to_continuous(r, t=1, compounded=1):
    if compounded>1 or compounded<0:
        raise RuntimeError("0 <= compounded <= 1, e.x. compounded = 1/4 for quarterly and 0 for continuously")
    return r/t if compounded==0 else (1/compounded) * math.log(1.0 + r*compounded) / t

def rate_to_compounded(r, t):
    return math.exp(r * t) - 1.0

def pvf(r, T):
    return math.exp(-r*T)

def future_value(M, r, T, a=0.0):
    return -a/r + (M + a/r) * pvf(-r, T)

def present_value(M, r, T):
    return M * pvf(r, T)

def normal_cdf(x, mu=0.0, sigma=1.0):
    return statistics.NormalDist(mu=mu, sigma=sigma).cdf(x)

def normal_pdf(x, mu=0.0, sigma=1.0):
    return statistics.NormalDist(mu=mu, sigma=sigma).pdf(x)

def black_scholes_value(type, strike, t, u_price, vol, rf_rate, u_yield=0):
    if type=='C':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        rf_rate_pvf = pvf(rf_rate, t)
        n_d1 = normal_cdf(d1)
        n_d2 = normal_cdf(d2)
        return u_price * u_yield_pvf * n_d1 - strike * rf_rate_pvf * n_d2
    if type=='P':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        rf_rate_pvf = pvf(rf_rate, t)
        n_d1 = normal_cdf(d1)
        n_d2 = normal_cdf(d2)
        return -u_price *u_yield_pvf * (1-n_d1) + strike * rf_rate_pvf * (1-n_d2)
    if type=='BC':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        n_d2 = normal_cdf(d2)
        return rf_rate_pvf * n_d2
    if type=='BP':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        n_d2 = normal_cdf(d2)
        return rf_rate_pvf * (1 - n_d2)

def black_scholes_delta(type, strike, t, u_price, vol, rf_rate, u_yield=0):
    if type=='C':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        n_d1 = normal_cdf(d1)
        return u_yield_pvf * n_d1
    if type=='P':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        n_d1 = normal_cdf(d1)
        return u_yield_pvf * (n_d1 - 1)
    if type=='BC':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        np_d2 = normal_pdf(d2)
        den2 = u_price * den1
        return (rf_rate_pvf * np_d2) / den2
    if type=='BP':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        np_d2 = normal_pdf(d2)
        den2 = u_price * den1
        return -(rf_rate_pvf * np_d2) / den2

def black_scholes_gamma(type, strike, t, u_price, vol, rf_rate, u_yield=0):
    if type=='C':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        np_d1 = normal_pdf(d1)
        den2 = u_price * den1
        return u_yield_pvf * np_d1 / den2
    if type=='P':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        np_d1 = normal_pdf(d1)
        den2 = u_price * den1
        return u_yield_pvf * np_d1 / den2
    if type=='BC':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        np_d2 = normal_pdf(d2)
        den2 = u_price * den1
        return -(rf_rate_pvf * d1 * np_d2) / den2**2
    if type=='BP':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        np_d2 = normal_pdf(d2)
        den2 = u_price * den1
        return (rf_rate_pvf * d1 * np_d2) / den2**2

def black_scholes_theta(type, strike, t, u_price, vol, rf_rate, u_yield=0):
    if type=='C':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        rf_rate_pvf = pvf(rf_rate, t)
        np_d1 = normal_pdf(d1)
        n_d1 = normal_cdf(d1)
        n_d2 = normal_cdf(d2)
        return -(vol*u_price*u_yield_pvf*np_d1)/(2*t**0.5) + \
        u_yield*u_price*n_d1*u_yield_pvf - rf_rate*strike*n_d2*rf_rate_pvf
    if type=='P':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        rf_rate_pvf = pvf(rf_rate, t)
        np_d1 = normal_pdf(d1)
        n_d1 = normal_cdf(d1)
        n_d2 = normal_cdf(d2)
        return -(vol*u_price*u_yield_pvf*np_d1)/(2*t**0.5) - \
        u_yield*u_price*(1-n_d1)*u_yield_pvf + rf_rate*strike*(1-n_d2)*rf_rate_pvf
    if type=='BC':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        n_d2 = normal_cdf(d2)
        np_d2 = normal_pdf(d2)
        return rf_rate_pvf * (rf_rate*n_d2 + np_d2*(0.5*d1/t - (rf_rate-u_yield)/den1))
    if type=='BP':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        n_d2 = normal_cdf(d2)
        np_d2 = normal_pdf(d2)
        return rf_rate_pvf * (rf_rate*(1-n_d2) + np_d2*(0.5*d1/t - (rf_rate-u_yield)/den1))

def black_scholes_speed(type, strike, t, u_price, vol, rf_rate, u_yield=0):
    if type=='C':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        np_d1 = normal_pdf(d1)
        den2 = u_price * den1
        return -u_yield_pvf*np_d1*(d1+den1)/den2**2
    if type=='P':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        np_d1 = normal_pdf(d1)
        den2 = u_price * den1
        return -u_yield_pvf*np_d1*(d1+den1)/den2**2
    if type=='BC':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        np_d2 = normal_pdf(d2)
        den2 = u_price * den1
        return -rf_rate_pvf*np_d2*(-2*d1+(1-d1*d2)/den1)/(u_price*den2**2)
    if type=='BP':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        np_d2 = normal_pdf(d2)
        den2 = u_price * den1
        return rf_rate_pvf*np_d2*(-2*d1+(1-d1*d2)/den1)/(u_price*den2**2)

def black_scholes_vega(type, strike, t, u_price, vol, rf_rate, u_yield=0):
    if type=='C':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        np_d1 = normal_pdf(d1)
        den2 = u_price * den1
        return (den2/vol)*u_yield_pvf*np_d1
    if type=='P':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        np_d1 = normal_pdf(d1)
        den2 = u_price * den1
        return (den2/vol)*u_yield_pvf*np_d1
    if type=='BC':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        np_d2 = normal_pdf(d2)
        return -rf_rate_pvf*np_d2*(den1+d2)/vol
    if type=='BP':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        np_d2 = normal_pdf(d2)
        return rf_rate_pvf*np_d2*(den1+d2)/vol

def black_scholes_rho(type, strike, t, u_price, vol, rf_rate, u_yield=0):
    if type=='C':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        n_d2 = normal_cdf(d2)
        return strike*t*rf_rate_pvf*n_d2
    if type=='P':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        n_d2 = normal_cdf(d2)
        return -strike*t*rf_rate_pvf*(1-n_d2)
    if type=='BC':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        np_d2 = normal_pdf(d2)
        n_d2 = normal_cdf(d2)
        return rf_rate_pvf * t * (-n_d2 + np_d2/den1)
    if type=='BP':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        np_d2 = normal_pdf(d2)
        n_d2 = normal_cdf(d2)
        return rf_rate_pvf * t * (-(1-n_d2) - np_d2/den1)

def black_scholes_yield_sensitivity(type, strike, t, u_price, vol, rf_rate, u_yield=0):
    if type=='C':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        n_d1 = normal_cdf(d1)
        return -t*u_price*u_yield_pvf*n_d1
    if type=='P':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        u_yield_pvf = pvf(u_yield, t)
        n_d1 = normal_cdf(d1)
        return t*u_price*u_yield_pvf*(1-n_d1)
    if type=='BC':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        np_d2 = normal_pdf(d2)
        return -t*rf_rate_pvf*np_d2/den1
    if type=='BP':
        den1 = vol * t**0.5
        d1 = (math.log(u_price/strike) + \
            (rf_rate - u_yield + 0.5 * vol**2) * t) / den1
        d2 = d1 - den1
        rf_rate_pvf = pvf(rf_rate, t)
        np_d2 = normal_pdf(d2)
        return t*rf_rate_pvf*np_d2/den1

def implied_volatility(type, deriv_price, strike, t, u_price, rf_rate, u_yield=0, x0=0.1, tol=1.48e-08):
    if type in ['C', 'P', 'BC', 'BP']:
        def f(vol):
            value = black_scholes_value(type, strike, t, u_price, vol, rf_rate, u_yield=u_yield)
            return value - deriv_price
        def fprime(vol):
            return black_scholes_vega(type, strike, t, u_price, vol, rf_rate, u_yield=u_yield)
        return newton(f, x0, fprime=fprime, tol=tol)
    
def _portfolio_optim_fun(x, c):
    return np.dot(x, x@c)

def portfolio_optim(m, c, expected_return=None, shortable=None, rf_rate=None, allow_borrow=False, max_leverage=1.0e3):
    n = m.size
    m = np.asarray(m)
    c = np.asarray(c)
    if rf_rate is not None:
        m = np.append(m, rf_rate)
        cc = np.zeros((c.shape[0]+1, c.shape[1]+1))
        cc[:-1, :-1] = c
        c = cc
        del(cc)
    if shortable is None:
        shortable = np.zeros(n, dtype=np.float64)
    else:
        shortable = np.asarray(shortable, dtype=np.float64)
        if shortable.size != n:
            raise RuntimeError("size of m and shortable must be equal")
    if rf_rate is not None:
        if allow_borrow:
            shortable = np.append(shortable, 1)
        else:
            shortable = np.append(shortable, 0)
    shortable[shortable>0] = -max_leverage
    x0 = np.ones_like(m)/m.size
    c1 = list(zip(shortable, max_leverage*np.ones_like(shortable)))
    c2 = opt.LinearConstraint(np.ones_like(m), lb=0.99999999, ub=1.0000001)
    if expected_return is not None:
        c3 = opt.LinearConstraint(m, lb=expected_return*(1-0.000001), ub=expected_return*(1+0.000001))
    constraints = [c2, c3] if expected_return is not None else c2
    sol = opt.differential_evolution(partial(_portfolio_optim_fun, c=c), bounds=c1, constraints=constraints, workers=-1, updating='deferred', x0=x0)
    return sol
    
def normal(*n, mu=np.array([0.0]), sigma=np.array([1.0]), bs=None, dtype=np.float64):
    mu = np.asarray(mu, dtype=dtype)
    sigma = np.asarray(sigma, dtype=dtype)
    if mu.ndim>1 or sigma.ndim>1:
        raise RuntimeError("mu, sigma should be 1D")
    if isinstance(bs, int):
        bs = (bs,)
    if bs is None:
        bs = ()
    if n:
        if len(n)>1 or (not isinstance(n[0], int)) or n[0]<1:
            raise RuntimeError("n should be an integer")
        if mu.size>1 and mu.size!=n[0]:
            raise RuntimeError("if n is specified, size of mu should be an 1 or n")
        if sigma.size>1 and sigma.size!=n[0]:
            raise RuntimeError("if n is specified, size of sigma should be an 1 or n")
        out = np.random.normal(loc=np.broadcast_to(mu, n), 
                                scale = np.broadcast_to(sigma, n), size=bs+n)
    else:
        mu, sigma = np.broadcast_arrays(mu, sigma)
        out = np.random.normal(loc=mu, scale = sigma, size=bs+mu.shape)
    if out.ndim==1 and out.size==1:
        return out.item()
    return out

def normal_multivariate(*n, mu=np.zeros(2), cov=np.eye(2), bs=None, dtype=np.float64):
    mu = np.asarray(mu, dtype=dtype)
    cov = np.asarray(cov, dtype=dtype)
    if isinstance(bs, int):
        bs = (bs,)
    if bs is None:
        bs = ()
    if mu.ndim<1 or mu.ndim>2 or cov.ndim>3 or cov.ndim<2:
        raise RuntimeError("1 <= mu.ndim <= 2 , 2 <= cov.ndim <= 3")
    if mu.ndim==1:
        mu = np.expand_dims(mu, -1)
    if cov.ndim==2:
        cov = np.expand_dims(cov, -1)
    if n:
        if len(n)>1 or (not isinstance(n[0], int)) or n[0]<1:
            raise RuntimeError("n should be an integer")
        if mu.shape[-1]!=n[0] and mu.shape[-1]!=1:
            raise RuntimeError("if n is specified and mu.ndim==2, mu.shape[-1] should be an 1 or n")
        if cov.shape[-1]!=n[0] and cov.shape[-1]!=1:
            raise RuntimeError("if n is specified and cov.ndim==3, cov.shape[-1] should be an 1 or n")
    else:
        n = (max(mu.shape[-1], cov.shape[-1]),)
    if mu.shape[-1]==1 and cov.shape[-1]==1:
        return np.ascontiguousarray(np.moveaxis(np.random.multivariate_normal(mean=mu[...,0], cov=cov[...,0], size=bs+n), -2, -1))
    mu = np.broadcast_to(mu, mu.shape[:-1]+n)
    cov = np.broadcast_to(cov, cov.shape[:-1]+n)
    out = np.empty(bs+(cov.shape[-2],)+n)
    for i in range(n[0]):
        out[...,i] = np.random.multivariate_normal(mean=mu[...,i], cov=cov[...,i], size=bs)
    return out

def lognormal(*n, mu=np.array([0.0]), sigma=np.array([1.0]), bs=None, dtype=np.float64):
    if not n:
        return np.exp(normal(mu=mu, sigma=sigma, bs=bs, dtype=dtype))
    return np.exp(normal(n[0], mu=mu, sigma=sigma, bs=bs, dtype=dtype))

def lognormal_multivariate(*n, mu=np.zeros(2), cov=np.eye(2), bs=None, dtype=np.float64):
    if not n:
        return np.exp(normal_multivariate(mu=mu, cov=cov, bs=bs, dtype=dtype))
    return np.exp(normal_multivariate(n[0], mu=mu, cov=cov, bs=bs, dtype=dtype))

def random_walk(series, x0=0.0):
    series = np.atleast_1d(series)
    x0 = np.asarray(x0, dtype=series.dtype)
    if series.ndim>1:
        x0 = np.broadcast_to(x0, series.shape[:-1])
    out = np.empty(series.shape[:-1] + (series.shape[-1]+1,))
    out[...,0] = x0
    out[...,1:] = series
    np.cumsum(out, axis=-1, out=out)
    return out

def random_walk_geometric(series, x0=1.0):
    series = np.atleast_1d(series)
    x0 = np.asarray(x0, dtype=series.dtype)
    if series.ndim>1:
        x0 = np.broadcast_to(x0, series.shape[:-1])
    out = np.empty(series.shape[:-1] + (series.shape[-1]+1,))
    out[...,0] = x0
    out[...,1:] = 1.0 + series
    np.cumprod(out, axis=-1, out=out)
    return out

def sliding_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

class Folio:
    def __init__(self, qty=1):
        self._folios = list()
        self.qty = qty
    def folios(self):
        if not self._folios:
            yield self
        else:
            for x in self._folios:
                yield x
    def extra_repr(self):
        return ''
    def __repr__(self, n=1):
        repr_str = f"{type(self).__name__}(qty={self.qty}{',' if self.extra_repr() else ''} {self.extra_repr()})\n    "
        for folio in self._folios:
            repr_str += n*'    ' + f"{folio.__repr__(n+1)}"
        return repr_str
    def __neg__(self):
        temp = copy.deepcopy(self)
        temp.qty = -temp.qty
        return temp
    def __add__(self, other):
        l1 =[]
        l2 = []
        if not self._folios:
            l1.append(self)
        else:
            l1 = [self.qty*x for x in self._folios]
        if not other._folios:
            l2.append(other)
        else:
            l2 = [other.qty*x for x in other._folios]
        temp = Folio()
        temp._folios = l1 + l2
        return temp
    def __radd__(self, other):
        l1 =[]
        l2 = []
        if not self._folios:
            l1.append(self)
        else:
            l1 = [self.qty*x for x in self._folios]
        if not other._folios:
            l2.append(other)
        else:
            l2 = [other.qty*x for x in other._folios]
        temp = Folio()
        temp._folios = l1 + l2
        return temp
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return (-self) + other
    def __mul__(self, k):
        temp = copy.deepcopy(self)
        temp.qty = temp.qty*k
        return temp
    def __rmul__(self, k):
        temp = copy.deepcopy(self)
        temp.qty = temp.qty*k
        return temp
    
class C(Folio):
    def __init__(self, strike, qty=1):
        super().__init__()
        self.E = strike
    def extra_repr(self):
        return f'strike={self.E}'

class P(Folio):
    def __init__(self, strike, qty=1):
        super().__init__()
        self.E = strike
    def extra_repr(self):
        return f'strike={self.E}'

class S(Folio):
    def __init__(self, qty=1):
        super().__init__()

class BC(Folio):
    def __init__(self, strike, qty=1):
        super().__init__()
        self.E = strike
    def extra_repr(self):
        return f'strike={self.E}'

class BP(Folio):
    def __init__(self, strike, qty=1):
        super().__init__()
        self.E = strike
    def extra_repr(self):
        return f'strike={self.E}'

def options_payoff_diagram(folio, u_price=None):
    for x in folio.folios():
        if (not isinstance(x,C)) and (not isinstance(x,P)) and (not isinstance(x,S)) \
            and (not isinstance(x,BC)) and (not isinstance(x,BP)):
            raise RuntimeError('''Input Folio should be composed of C (European Calls), 
                               P (European Puts), S (stock), BC (Binary Calls), and 
                               BP (Binary Puts)''')
        if isinstance(x,S) and u_price is None:
            raise RuntimeError('u_price (underlying/stock price) parameter is required')
    Es = []
    if u_price is not None:
        Es.append(u_price)
    for x in folio.folios():
        if (not isinstance(x, S)):
            Es.append(x.E - 1e-6)
            Es.append(x.E + 1e-6)
        else:
            Es.append(0)
    Es = list(filter((0).__ne__, Es))
    Es += [0, 1e-6, max(Es)+min(Es) if Es else 0]
    Es.sort()
    ll = []
    for x in folio.folios():
        v =[]
        for e in Es:
            if isinstance(x, C):
                v.append(0 if (e-x.E)<=0 else x.qty*(e-x.E))
            if isinstance(x, P):
                v.append(0 if (e-x.E)>=0 else -x.qty*(e-x.E))
            if isinstance(x, S):
                v.append(x.qty*(e-u_price))
            if isinstance(x, BC):
                v.append(0 if (e-x.E)<=0 else x.qty)
            if isinstance(x, BP):
                v.append(0 if (e-x.E)>=0 else x.qty)
        ll.append(v)   
    s = [0 for _ in range(len(Es))]
    for x in ll:
        for i in range(len(Es)):
            s[i] += x[i]
    fig, ax = plt.subplots()
    ax.axhline(y=0, color='0.5')
    ax.axvline(x=u_price if u_price is not None else 0, color='0.5')
    ax.plot(Es, s)
    #ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax. margins(x=0)
    ax.set_xlabel('underlying_price')
    ax.set_ylabel('Payoff')
    plt.show()