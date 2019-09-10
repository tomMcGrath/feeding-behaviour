import numpy as tt
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import theano
from pymc3.distributions.dist_math import bound

class pause_ll(pm.distributions.Continuous):
	def __init__(self, theta4, theta5, theta6, theta7, theta8, k1):
		self.theta4 = theta4
		self.theta5 = theta5
		self.theta6 = theta6
		self.theta7 = theta7
		self.theta8 = theta8
		self.k1 = k1
		super(pause_ll, self).__init__()

	def logp(self, data):
		"""
		Retrieve thetas
		"""
		theta4 = self.theta4
		theta5 = self.theta5
		theta6 = self.theta6
		theta7 = self.theta7
		theta8 = self.theta8
		k1 = self.k1

		"""
		Parse data
		"""

		f_lengths = data[0]
		g_starts = data[1]
		rates = data[2]
		p_lengths = data[3]
		g_ends = data[4]

		"""
		Transition kernel
		"""
		eps = 0.01
		Q = eps + (1. - 2.*eps)/(1. + tt.exp(-0.1*theta4*(g_ends-20.*theta5)))

		ll_Q = tt.log(Q)
		ll_notQ = tt.log(1-tt.exp(ll_Q))

		"""
		Short pause ll
		"""
		ll_S = bound(tt.log(theta6) - theta6 * p_lengths, p_lengths > 0, theta6 > 0)

		"""
		Long pause ll
		"""
		## Full emptying time
		t_cs = 2.*tt.sqrt(g_ends)/k1

		## ll if time is less than full emptying
		g_pausing_1 = 0.25*tt.sqr(k1*p_lengths) - tt.sqrt(g_ends)*k1*p_lengths + g_ends

		phi_L_1 = 1./(theta7 + theta8*g_pausing_1)

		psi_L_1 = 2.*tt.arctan(0.5*tt.sqrt(theta8/theta7)*(k1*p_lengths - 2.*tt.sqrt(g_ends)))
		psi_L_1 = psi_L_1 - 2.*tt.arctan(0.5*tt.sqrt(theta8/theta7)*(-2.*tt.sqrt(g_ends)))
		psi_L_1 = psi_L_1/(k1*tt.sqrt(theta7*theta8))

		ll_L_1 = tt.log(phi_L_1) - psi_L_1

		## ll if time exceeds full emptying
		phi_L_2 = 1./theta7

		psi_L_2 = 2.*tt.arctan(0.5*tt.sqrt(theta8/theta7)*(k1*t_cs - 2.*tt.sqrt(g_ends)))
		psi_L_2 = psi_L_2 - 2.*tt.arctan(0.5*tt.sqrt(theta8/theta7)*(-2.*tt.sqrt(g_ends)))
		psi_L_2 = psi_L_2/(k1*tt.sqrt(theta7*theta8))

		psi_L_2 = psi_L_2 + (p_lengths-t_cs)/theta7

		ll_L_2 = tt.log(phi_L_2) - psi_L_2

		## Switch based on t_c
		ll_L = tt.switch(tt.lt(p_lengths, t_cs), ll_L_1, ll_L_2)

		## Avoid numerical issues in logaddexp
		ll_short = ll_notQ + ll_S
		ll_long = ll_Q + ll_L

		ll_pause = ll_short + tt.log1p(tt.exp(ll_long - ll_short))

		return ll_pause


class pause_ll_debug(pm.distributions.Continuous):
	def __init__(self, theta5, theta6, theta7, theta8, theta9, k1):
		self.theta5 = theta5
		self.theta6 = theta6
		self.theta7 = theta7
		self.theta8 = theta8
		self.theta9 = theta9
		self.k1 = k1
		super(pause_ll_debug, self).__init__()

	def logp(self, data):
		"""
		Retrieve thetas
		"""
		theta5 = self.theta5
		theta6 = self.theta6
		theta7 = self.theta7
		theta8 = self.theta8
		theta9 = self.theta9
		k1 = self.k1

		#theta5_print = theano.printing.Print('theta5')(theta5)
		#theta6_print = theano.printing.Print('theta6')(theta6)
		#theta7_print = theano.printing.Print('theta7')(theta7)
		#theta8_print = theano.printing.Print('theta8')(theta8)
		#theta9_print = theano.printing.Print('theta9')(theta9)

		"""
		Parse data
		"""

		f_lengths = data[0]
		g_starts = data[1]
		rates = data[2]
		p_lengths = data[3]
		g_ends = data[4]

		p_lengths_print = theano.printing.Print('p_lengths')(p_lengths)
		g_ends_print = theano.printing.Print('g_ends')(g_ends)

		"""
		Transition kernel
		"""
		eps = 0.01
		Q = eps + (1. - 2.*eps)/(1. + tt.exp(-0.1*theta5*(g_ends-20.*theta6)))

		#Q_print = theano.printing.Print('Q')(Q)

		ll_Q = tt.log(Q)

		ll_notQ = tt.log(1-tt.exp(ll_Q))

		"""
		Short pause ll
		"""
		ll_S = bound(tt.log(theta7) - theta7 * p_lengths, p_lengths > 0, theta7 > 0)
		#ll_S = tt.log(theta7) - theta7*p_lengths # just exponential dist

		"""
		Long pause ll
		"""
		## Full emptying time
		t_cs = 2.*tt.sqrt(g_ends)/k1

		#t_cs_print = theano.printing.Print('t_cs')(t_cs)
		#p_lengths_print = theano.printing.Print('p_lengths')(p_lengths)

		## ll if time is less than full emptying
		g_pausing_1 = 0.25*tt.sqr(k1*p_lengths) - tt.sqrt(g_ends)*k1*p_lengths + g_ends

		phi_L_1 = 1./(theta8 + theta9*g_pausing_1)

		psi_L_1 = 2.*tt.arctan(0.5*tt.sqrt(theta9/theta8)*(k1*p_lengths - 2.*tt.sqrt(g_ends)))
		psi_L_1 = psi_L_1 - 2.*tt.arctan(0.5*tt.sqrt(theta9/theta8)*(-2.*tt.sqrt(g_ends)))
		psi_L_1 = psi_L_1/(k1*tt.sqrt(theta8*theta9))

		ll_L_1 = tt.log(phi_L_1) - psi_L_1

		## ll if time exceeds full emptying
		phi_L_2 = 1./theta8

		psi_L_2 = 2.*tt.arctan(0.5*tt.sqrt(theta9/theta8)*(k1*t_cs - 2.*tt.sqrt(g_ends)))
		psi_L_2 = psi_L_2 - 2.*tt.arctan(0.5*tt.sqrt(theta9/theta8)*(-2.*tt.sqrt(g_ends)))
		psi_L_2 = psi_L_2/(k1*tt.sqrt(theta8*theta9))

		psi_L_2 = psi_L_2 + (p_lengths-t_cs)/theta8

		ll_L_2 = tt.log(phi_L_2) - psi_L_2

		## Switch based on t_c
		ll_L = tt.switch(tt.lt(p_lengths, t_cs), ll_L_1, ll_L_2)

		"""
		ll_1_print = theano.printing.Print('ll_1_print')(ll_L_1)
		ll_2_print = theano.printing.Print('ll_2_print')(ll_L_2)
		ll_L_print = theano.printing.Print('ll_L_print')(ll_L)
		"""

		## Assemble 2 likelihood paths

		ll_Q_print = theano.printing.Print('ll_Q')(ll_Q)
		ll_notQ_print = theano.printing.Print('ll_notQ')(ll_notQ)

		ll_L_print = theano.printing.Print('ll_L')(ll_L)
		ll_S_print = theano.printing.Print('ll_S')(ll_S)

		## Avoid numerical issues in logaddexp
		#ll_short = ll_notQ_print + ll_S_print
		#ll_long = ll_Q_print + ll_L_print

		ll_short = ll_notQ + ll_S
		ll_long = ll_Q + ll_L

		"""
		ll_pause = tt.switch(tt.lt(ll_short, ll_long),
							 ll_short + tt.log1p(tt.exp(ll_long - ll_short)),
							 ll_long + tt.log1p(tt.exp(ll_short - ll_long)))
		"""
		#ll_pause = ll_short + tt.log1p(tt.exp(ll_long - ll_short))
		ll_pause = ll_long + tt.log1p(tt.exp(ll_short - ll_long))
		ll_nans = tt.any(tt.isnan(ll_pause))
		ll_nan_print = theano.printing.Print('ll_nans')(ll_nans)
		ll_pause_print = theano.printing.Print('ll_pause')(ll_pause)

		#ll = ll_pause # tt.log(tt.exp(ll_notQ + ll_S) + tt.exp(ll_Q + ll_L))

		#ll_print = theano.printing.Print('ll')(ll)

		return ll_pause_print + ll_nan_print


class longpause_only(pm.distributions.Continuous):
	def __init__(self, theta7, theta8, k1):
		self.theta7 = theta7
		self.theta8 = theta8
		self.k1 = k1
		super(longpause_only, self).__init__()

	def logp(self, data):
		"""
		Retrieve thetas
		"""
		theta7 = self.theta7
		theta8 = self.theta8
		k1 = self.k1

		"""
		Parse data
		"""
		t = data[0]
		x0 = data[1]

		"""
		Long pause ll
		"""
		## Full emptying time
		t_cs = 2.*tt.sqrt(x0)/k1

		## Stomach fullness
		x = 0.25*tt.sqr(k1*t) - k1*t*tt.sqrt(x0) + x0

		## ll if time is less than full emptying
		a_part1 = k1*t*theta8*tt.pow(theta7, 2)*(0.5*theta7 + theta8*(0.25*k1*t*tt.sqrt(x0) - 0.5*x0))
		a_part2_1 = theta7*(tt.pow(theta7, 2) + theta7*theta8*(x + x0) + x0*tt.pow(theta8, 2)*x)
		a_part2_2 = tt.sqrt(theta7*theta8)*(tt.arctan((0.5*k1*t - tt.sqrt(x0))*tt.sqrt(theta8/theta7)) + tt.arctan(tt.sqrt(theta8*x0/theta7)))

		a_denom = k1*tt.pow(theta7, 3)*theta8*(theta7 + x0*theta8)*(theta7 + theta8*x)
		a_psi = (a_part1 + a_part2_1*a_part2_2)/a_denom

		a_phi = 1./(tt.sqr(theta7 + theta8*x))

		ll_L_1 = tt.log(a_phi) - a_psi

		## ll if time exceeds full emptying
		b_phi = 1./tt.sqr(theta7)

		b_part1 = k1*t_cs*theta8*tt.pow(theta7, 2)*(0.5*theta7 + theta8*(0.25*k1*t_cs*tt.sqrt(x0) - 0.5*x0))
		b_part2_1 = theta7*(tt.pow(theta7, 2) + theta7*theta8*(x0))
		b_part2_2 = tt.sqrt(theta7*theta8)*(tt.arctan((0.5*k1*t_cs - tt.sqrt(x0))*tt.sqrt(theta8/theta7)) + tt.arctan(tt.sqrt(theta8*x0/theta7)))

		b_denom = k1*tt.pow(theta7, 3)*theta8*(theta7 + x0*theta8)*(theta7)
		b_psi = (b_part1 + b_part2_1*b_part2_2)/b_denom
		b_psi = b_psi + (t-t_cs)/tt.sqr(theta7)

		ll_L_2 = tt.log(b_phi) - b_psi

		## Switch based on t_c
		ll_L = tt.switch(tt.lt(t, t_cs), ll_L_1, ll_L_2)

		return ll_L
