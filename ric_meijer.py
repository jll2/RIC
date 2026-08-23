#!/usr/bin/python3
"""Compute decay of RIC using Meijer G-functions.

Required modules: numpy, mpmath

If to install pip, try:
	pip install numpy
	pip install mpmath
If you use Linux, numpy and mpmath are both available as part of most major
distributions.  Under a Debian/Ubuntu-derived distribution, for example, try:
	apt install python3-numpy python3-mpmath

References
----------
"Decay of radiation–induced–conductivity analyzed as a problem in
one–dimensional ion diffusion," by John L Lawless, Radiation Measurements
(2026):107733.  https://doi.org/10.1016/j.radmeas.2026.107733

"An explanation for why measurements of alkali ion mobility in quartz have
been so inconsistent," by John L Lawless, in preparation (2026)
"""
# pylint: disable=invalid-name

import numpy as np
from numpy.typing import ArrayLike  #pylint:disable=unused-import

from mpmath import meijerg


_ffmt = lambda vars: ' '.join(f'{x:.3g}' for x in vars)  # pretty-print a 1-D array


######################################################
###   Compute probability p for ion survival   #######
######################################################

def probability(t_hat, kmax=10):
	"""Return decay of RIC at non-d time t_hat=t/tau1 after a short burst of radiation.

	args:
		t_hat (ArrayLike) : t/tau
		kmax (int) : number of terms to sum (default=10)

	returns:
		r (ArrayLike) : fraction of ions remaining free after time t_hat

	Examples:
		>>> t_hat=np.array((1e-6, 0.1, 0.3, 1.25, 6))
		>>> _ffmt(probability(t_hat, kmax=1))
		'0.811 0.483 0.304 0.1 0.00972'
		>>> _ffmt(probability(t_hat, kmax=50))
		'0.991 0.5 0.309 0.101 0.00972'
	"""
	t_hat = np.asarray(t_hat)
	if t_hat.shape:
		r = []
		for tt in t_hat:
			r.append(probability(tt, kmax=kmax))
		return np.asarray(r)

	r = 0
	for k in range(1, kmax+1, 2):
		r = r + meijerg(((-1/2, 0, 1), ()), ((), ()), 4/(np.pi**2 * k**2 * t_hat)) / k**2
	return (16 / np.pi**2.5) * float(r)

######################################################
###   Compute L_infty   ##############################
######################################################

def Linf(t_hat, kmax=10):
	"""Return long RIC integral computed up to kmax.

	Assumes radiation started at t=-oo, ended at t=0.  Ions measured
	at non-d time t_hat=t/tau1 later.

	args:
		t_hat (ArrayLike) : t/tau
		kmax (int) : number of terms to sum (default=10)

	returns:
		r (ArrayLike) : fraction of ions remaining free after time t_hat

	Examples:
		>>> t_hat=np.array((1e-9, 0.1, 0.3, 1.25, 6, 10, 20))
		>>> _ffmt(Linf(t_hat, kmax=1))
		'0.493 0.433 0.357 0.195 0.0386 0.0161 0.00332'
		>>> _ffmt(Linf(t_hat, kmax=100))
		'0.5 0.436 0.359 0.195 0.0386 0.0161 0.00332'
	"""
	t_hat = np.asarray(t_hat)
	if t_hat.shape:
		r = []
		for tt in t_hat:
			r.append(Linf(tt, kmax=kmax))
		return np.asarray(r)

	r = 0
	for k in range(1, kmax+1, 2):
		r = r + meijerg(((-3/2, -1, 1), ()), ((), ()), 4/(np.pi**2 * k**2 * t_hat)) / k**4
	return (64 / np.pi**4.5) * float(r)

def L(t_hat_ann, t_hat_irr, kmax=5):
	"""Return ratio of n to Y*t_hat_irr at t_hat_ann after end of radiation lasted t_hat_irr.

	t_hat_ann and t_hat_irr are both non-dimensional stretched times.

	Args:
		t_hat_ann (ArrayLike) : stretched duration of anneal
		t_hat_irr (ArrayLike) : stretched duration of irradiation
		kmax (int) : number of terms to sum (default=5)

	returns:
		Fraction of ions remaining free after time t_hat (ArrayLike)
	"""
	return Linf(t_hat_ann, kmax) - Linf(t_hat_ann + t_hat_irr, kmax)


######################################################
###   When executed from command line, run   #########
###   self-tests or print table of values.   #########
######################################################

def ptable(t_hat=(0.01, 0.1, 1, 10, 100), kmax=(1, 5, 50)):
	"""Print table of values."""
	print("")
	print("Ion survival probability: p")
	print(f'{"t_hat":>12s} ' + ' '.join(f'kmax={x:<7.0f}' for x in kmax))
	for th in t_hat:
		print(f'{th:12.2f} ' + ' '.join('%-12.5g' % probability(th, k) for k in kmax))

def Ltable(t_hat=(0.01, 0.1, 1, 10, 100), kmax=(1, 5, 50)):
	"""Print table of values."""
	print("")
	print("Survival fn for long irradiation: L_infty")
	print(f'{"t_hat":>12s} ' + ' '.join(f'kmax={x:<7.0f}' for x in kmax))
	for th in t_hat:
		print(f'{th:12.2f} ' + ' '.join('%-12.5g' % Linf(th, k) for k in kmax))

def _test():
	"""Perform self tests."""
	import doctest #pylint:disable=import-outside-toplevel
	return doctest.testmod()

if __name__ == '__main__':
	import sys #pylint:disable=import-outside-toplevel
	if '--test' in sys.argv or '__IPYTHON__' in dir(__builtins__):
		_test()
	else:
		ptable()
		print('---')
		Ltable()
