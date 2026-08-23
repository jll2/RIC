#!/usr/bin/python3
"""Compute decay of RIC using adaptive integration.

Required modules: numpy, scipy

References
----------
"Decay of radiation–induced–conductivity analyzed as a problem in
one–dimensional ion diffusion," by John L Lawless, Radiation Measurements
(2026):107733.  https://doi.org/10.1016/j.radmeas.2026.107733

"An explanation for why measurements of alkali ion mobility in quartz have
been so inconsistent," by John L Lawless, in preparation (2026)
"""
# pylint: disable=invalid-name

import warnings
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad, IntegrationWarning


_ffmt = lambda vars: ' '.join(f'{x:.3g}' for x in vars)  # pretty-print a 1-D array

warnings.filterwarnings("error", category=RuntimeWarning)
warnings.filterwarnings("error", category=IntegrationWarning)


######################################################
###   Compute probability p for ion survival   #######
######################################################

def integrand_p(y, z, k=1):
	"""Return integrand for probability given y, z, k

	Args:
		y (number) : variable of integration related to channel width
			y = (1/2) (ptrap * W)**2;  dy = (ptrap * W) d(ptrap * W)
		z (number) : parameter proportional to t_hat
			z = t_hat * (k * pi)^2 = D (ptrap * k * pi)^2 t
		k (number) : Odd integer
	"""
	f = (2 * y)**0.5 + z / (2 * y)
	return (8 / (k * np.pi)**2) * np.exp(-f)

def probability(t_hat, kmax=5):
	"""Use adaptive integration of `integrand` to find ion concentration.

	Args:
		t_hat (ArrayLike) : non-dimensional stretched time
			t_hat = t / tau_1 = D * (ptrap W)^2 * t
		kmax (int) : largest k value to include in sum

	Example
	-------
	>>> t_hat=np.array((1e-6, 0.1, 0.3, 1.25, 6))
	>>> _ffmt(probability(t_hat, kmax=1))
	'0.811 0.483 0.304 0.1 0.00972'
	>>> _ffmt(probability(t_hat, kmax=50))
	'0.991 0.5 0.309 0.101 0.00972'
	"""
	# Uses adaptive integrator `quad`:
	# quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08,
	#      epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None,
	#      wopts=None, maxp1=50, limlst=50, complex_func=False)

	t_hat = np.asarray(t_hat)
	if t_hat.shape:
		r = []
		for tt in t_hat:
			r.append(probability(tt, kmax=kmax))
		return np.asarray(r)

	r = 0
	for k in range(1, kmax+1, 2):
		z = t_hat * (k * np.pi)**2 # z = D t (k pi ptrap)^2
		ypeak = z**(2./3) / 2**(1./3)
		part1 = quad(integrand_p, 1e-9, ypeak, args=(z, k), full_output=0, limit=500)
		part2 = quad(integrand_p, ypeak, np.inf, args=(z, k), full_output=0, limit=500)
		r = r + part1[0] + part2[0]
	return r

######################################################
###   Compute L_infty   ##############################
######################################################

def integrand_Linf(u, c, lnmult=0):
	"""Return integrand."""
	return u**3 * np.exp(lnmult - u - c/u**2)

def Linf(t_hat : ArrayLike, kmax : int = 5, err : float = 1e-9) -> float:
	"""Use return integral for long irradiation.

	Args:
		t_hat : normalized time after irradiation: D * (ptrap W)^2 * t
		kmax : largest k
		err  : epsabs error parameter for fn `quad`

	Example:
		>>> t_hat=np.array((0, 0.1, 0.3, 1.25, 6, 10, 20))
		>>> _ffmt(Linf(t_hat, kmax=1))
		'0.493 0.433 0.357 0.195 0.0386 0.0161 0.00332'
		>>> _ffmt(Linf(t_hat, kmax=100))
		'0.5 0.436 0.359 0.195 0.0386 0.0161 0.00332'

	Returns:
		integral L_infty
	"""
	# Uses adaptive integrator `quad`:
	# quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08,
	#      epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None,
	#      wopts=None, maxp1=50, limlst=50, complex_func=False)

	t_hat = np.asarray(t_hat)
	if t_hat.shape:
		r = []
		for tt in t_hat:
			r.append(Linf(tt, kmax=kmax, err=err))
		return np.asarray(r)

	r = 0
	for k in range(1, kmax+1, 2):
		c = t_hat * (k * np.pi)**2
		ypeak = (2 * c)**(4./3)
		part1 = quad(integrand_Linf, 1e-9, ypeak, args=(c,), full_output=0, epsabs=err)
		part2 = quad(integrand_Linf, ypeak, np.inf, args=(c,), full_output=0, epsabs=err)
		r = r + (part1[0] + part2[0]) / k**4.
	r = r * (8 / np.pi**4)
	return r

def L(t_hat_ann, t_hat_irr, kmax=5):
	"""Return n/Y for radiation that lasted from t_hat_ann to (t_hat_ann + t_hat_irr) ago.

	Args:
		t_hat_ann (ArrayLike) : stretched time of anneal
		t_hat_irr (ArrayLike) : stretched time of irradiation
		kmax (int) : number of terms to sum (default=5)

	returns:
		r (ArrayLike) : fraction of ions remaining free after time ttau
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
