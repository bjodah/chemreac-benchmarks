#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stripped down version of chemreac/examples/aqueous_radiolysis.py
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

# stdlib imports
import json
import os
from math import log, e, exp

# external imports
import numpy as np

# project internal imports
from chemreac import ReactionDiffusion
from chemreac.integrate import run
from chemreac.serialization import load


def integrate_rd(t0=1e-7, tend=.1, doserate=15, N=1000, nt=512, nstencil=3,
                 logy=False, logt=False, num_jacobian=False, savefig='None',
                 verbose=False, plot=False, plot_jacobians=False):
    null_conc = 1e-24

    mu = 1.0  # linear attenuation
    rho = 1.0  # kg/dm3
    rd = load(os.path.join(os.path.dirname(__file__), name+'aqueous_radiolysis.json'),
              ReactionDiffusion, N=N, logy=logy,
              logt=logt, bin_k_factor=[
                  [doserate*rho*exp(-mu*i/N)] for i in range(N)],
              nstencil=nstencil)
    y0_by_name = json.load(open(os.path.join(os.path.dirname(__file__),
                                             name+'.y0.json'), 'rt'))

    # y0 with a H2 gradient
    y0 = np.array([[y0_by_name.get(k, null_conc) if k != 'H2' else
                    1e-3/(i+2) for k in rd.substance_names]
                   for i in range(rd.N)])

    tout = np.logspace(log(t0), log(tend), nt+1, base=e)
    integr = run(rd, y0, tout, with_jacobian=(not num_jacobian))

class TimeAnalyticJacobian:

    def time_untransformed(self):
        integrate_rd(logy=False, logt=False)

    def time_log_transformed(self):
        integrate_rd(logy=True, logt=True, 1e-13)

class TimeNumericalJacobian:

    def time_untransformed(self):
        integrate_rd(logy=False, logt=False, num_jacobian=True)

    def time_log_transformed(self):
        integrate_rd(logy=True, logt=True, t0=1e-13, num_jacobian=True)
