#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stripped down version of chemreac/examples/analytic_diffusion.py
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *


from math import log

import numpy as np

from chemreac import (
    ReactionDiffusion, FLAT, CYLINDRICAL, SPHERICAL, Geom_names
)
from chemreac.integrate import run

np.random.seed(42)


def flat_analytic(x, t, D, mu, x0, xend, v, logy=False, logx=False):
    x = np.exp(x) if logx else x
    a = (4*np.pi*D*t)**-0.5
    b = -(x-mu-v*t)**2/(4*D*t)
    if logy:
        return np.log(a) + b + log(xend-x0)
    else:
        return a*np.exp(b)*(xend-x0)


def cylindrical_analytic(x, t, D, mu, x0, xend, v, logy=False, logx=False):
    x = np.exp(x) if logx else x
    a = (4*np.pi*D*t)**-1
    b = -(x-mu-v*t)**2/(4*D*t)
    if logy:
        return np.log(a) + b + log(xend-x0)
    else:
        return a*np.exp(b)*(xend-x0)


def spherical_analytic(x, t, D, mu, x0, xend, v, logy=False, logx=False):
    x = np.exp(x) if logx else x
    a = (4*np.pi*D)**-0.5 * t**-1.5
    b = -(x-mu-v*t)**2/(4*D*t)
    if logy:
        return np.log(a) + b + log(xend-x0)
    else:
        return a*np.exp(b)*(xend-x0)


def _efield_cb(x):
    """
    Returns a flat efield (-1)
    """
    return -np.ones_like(x)


def integrate_rd(D=2e-3, t0=3., tend=7., x0=0.0, xend=1.0, mu=None, N=64,
                 nt=42, geom='f', logt=False, logy=False, logx=False,
                 random=False, k=0.0, nstencil=3, linterpol=False,
                 rinterpol=False, num_jacobian=False, method='bdf',
                 scale_x=False, atol=1e-6, rtol=1e-6,
                 efield=False, random_seed=42):
    if t0 == 0.0:
        raise ValueError("t0==0 => Dirac delta function C0 profile.")
    if random_seed:
        np.random.seed(random_seed)
    decay = (k != 0.0)
    n = 2 if decay else 1
    mu = float(mu or x0)
    tout = np.linspace(t0, tend, nt)

    assert geom in 'fcs'
    geom = {'f': FLAT, 'c': CYLINDRICAL, 's': SPHERICAL}[geom]
    analytic = {
        FLAT: flat_analytic,
        CYLINDRICAL: cylindrical_analytic,
        SPHERICAL: spherical_analytic
    }[geom]

    # Setup the grid
    _x0 = log(x0) if logx else x0
    _xend = log(xend) if logx else xend
    x = np.linspace(_x0, _xend, N+1)
    if random:
        x += (np.random.random(N+1)-0.5)*(_xend-_x0)/(N+2)

    rd = ReactionDiffusion(
        2 if decay else 1,
        [[0]] if decay else [],
        [[1]] if decay else [],
        [k] if decay else [],
        N,
        D=[D]*(2 if decay else 1),
        z_chg=[1]*(2 if decay else 1),
        mobility=[0.01]*(2 if decay else 1),
        x=x,
        geom=geom,
        logy=logy,
        logt=logt,
        logx=logx,
        nstencil=nstencil,
        lrefl=not linterpol,
        rrefl=not rinterpol,
        xscale=1/(x[1]-x[0]) if scale_x else 1.0
    )

    if efield:
        if geom != FLAT:
            raise ValueError("Only analytic sol. for flat drift implemented.")
        rd.efield = _efield_cb(rd.xcenters)

    # Calc initial conditions / analytic reference values
    t = tout.copy().reshape((nt, 1))
    yref = analytic(rd.xcenters, t, D, mu, x0, xend,
                    0.01 if efield else 0, logy, logx).reshape(nt, N, 1)

    if decay:
        yref = np.concatenate((yref, yref), axis=2)
        if logy:
            yref[:, :, 0] += -k*t
            yref[:, :, 1] += np.log(1-np.exp(-k*t))
        else:
            yref[:, :, 0] *= np.exp(-k*t)
            yref[:, :, 1] *= 1-np.exp(-k*t)

    # Run the integration
    integr = run(rd, yref[0, ...], tout, atol=atol, rtol=rtol,
                 with_jacobian=(not num_jacobian), method=method,
                 C0_is_log=logy)


class TimeThree:
    
    def time_flat(self):
        integrate_rd(N=512, nstencil=3, k=0.1, geom='f', atol=1e-8, rtol=1e-10)

    def time_cylindrical(self):
        integrate_rd(N=512, nstencil=3, k=0.1, geom='c', atol=1e-8, rtol=1e-10)

    def time_spherical(self):
        integrate_rd(N=512, nstencil=3, k=0.1, geom='s', atol=1e-8, rtol=1e-10)

class TimeFive:
    
    def time_flat(self):
        integrate_rd(N=512, nstencil=5, k=0.1, geom='f', atol=1e-8, rtol=1e-10)

    def time_cylindrical(self):
        integrate_rd(N=512, nstencil=5, k=0.1, geom='c', atol=1e-8, rtol=1e-10)

    def time_spherical(self):
        integrate_rd(N=512, nstencil=5, k=0.1, geom='s', atol=1e-8, rtol=1e-10)

class TimeSeven:
    
    def time_flat(self):
        integrate_rd(N=512, nstencil=7, k=0.1, geom='f', atol=1e-8, rtol=1e-10)

    def time_cylindrical(self):
        integrate_rd(N=512, nstencil=7, k=0.1, geom='c', atol=1e-8, rtol=1e-10)

    def time_spherical(self):
        integrate_rd(N=512, nstencil=7, k=0.1, geom='s', atol=1e-8, rtol=1e-10)
