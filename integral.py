import sympy
import sympy.tensor.tensor
import sympy.tensor.toperators

import warnings
import itertools
import scipy

def _gen_ind_combs(inds):
	"""
	Given a list 'inds' such that len(inds)%2==0, construct permutations of the indices that appear in the corresponding angular integral over len(inds) instances of a unit vector. The number of combinations is listed as 'distinct FICTs' in table 1 of [Kearsley, Fong 1975 - Linearly Independent Sets of Isotropic Cartesian Tensors of Ranks up to Eight]. I believe the formula for the number of elements returned is len(ind_combs) = double_factorial(len(inds)-1).
	
	For our purposes, we do not care about linear independence (see https://mathematica.stackexchange.com/questions/77855/finding-basis-of-isotropic-tensors-of-rank-n for the more complicated approach required in that case)
	"""
	ind_combs = []
	
	if len(inds) > 0:
		ind1 = inds[0]
		for i2, ind2 in enumerate(inds[1:], 1):
			remaining_inds = [inds[i] for i in range(len(inds)) if (i != 0 and i != i2)]
			
			if len(remaining_inds) > 0:
				for comb in _gen_ind_combs(remaining_inds):
					ind_combs.append( [(ind1,ind2)] + comb )
			else:
				ind_combs.append([(ind1,ind2)])
	
	return ind_combs

def _gen_delta_combs(inds, delta):
	"""
	Used in do_angular_integral() to generate combinations of the Kronecker delta that appear in the angular integral over unit vectors.
	"""
	ind_combs = _gen_ind_combs(inds)
	
	delta_combs = []
	for comb in ind_combs:
		this_delta = 1
		for i in range(len(comb)):
			this_delta *= delta(*comb[i])
		
		delta_combs.append(this_delta)
	
	return delta_combs

def create_scalar_integral(expr, var):
	"""
	If isinstance(expr, TensExpr), Integral(expr, var) fails because it tries to convert the tensor to a polynomial and fails. This is a wrapper to take care of that issue.
	
	Arguments
	=========
	expr: TensExpr
	var: Symbol
	
	Examples
	========
	>>> from sympy.tensor.tensor import TensorIndexType, TensorIndex, TensorHead
	>>> from sympy import Function, Symbol
	>>> R3 = TensorIndexType('R3', dim=3)
	>>> p = TensorIndex("p", R3)
	>>> K = TensorHead("K", [R3])
	>>> f = Function('f')
	>>> x = Symbol('x)
	>>> create_scalar_integral( f(x)*K(p), x )
	Integral(f(x), x) * K(p)
	"""
	expr = sympy.expand(expr)
	if isinstance(expr, (sympy.tensor.tensor.TensMul, sympy.tensor.tensor.Tensor)):
		if var in expr.nocoeff.atoms():
			raise ValueError("The tensor part is dependent on the integration variables. Perhaps try expanding expr.")
		return sympy.Integral(expr.coeff, var)*expr.nocoeff
	elif isinstance(expr, sympy.tensor.tensor.TensAdd):
		return expr.func(*[create_scalar_integral(arg, var) for arg in expr.args])
	else:
		return sympy.Integral(expr, var)

class AngularIntegral(sympy.tensor.tensor.TensExpr):
	"""
	AngularIntegral(expr, wavevec): angular integral wrt wavevec over expr
	
	expr: TensExpr
	wavevec: TensorHead
	"""
	def __new__(cls, expr, wavevec):
		"""
		expr: TensExpr
		wavevec: TensorHead, vector with respect to which the angular integral was done.
		"""
		obj = sympy.tensor.tensor.Basic.__new__(cls, expr, wavevec)
		obj.expr = expr
		obj.wavevec = wavevec
		return obj
	
	@property
	def coeff(self):
		return S.One

	@property
	def nocoeff(self):
		return self
	
	def get_indices(self):
		return self.expr.get_indices()
	
	def get_free_indices(self):
		return self.expr.get_free_indices()
	
	def _replace_indices(self, repl):
		return self.func(self.expr._replace_indices(repl), self.wavevec)
	
	def _latex(self, printer):
		expr = printer._print(self.expr)
		# wavevec = printer._print(self.args[1].name) is not good because it just falls back to emptyPrinter (\mathtt{\text{name}})
		wavevec = self.wavevec.name
		return r"\int\mathrm{d}\Omega_{%s} \left(%s\right)" % (wavevec, expr)
	
	def doit(self, **hints):
		deep = hints.get('deep', True)
		if deep:
			expr = self.expr.doit(**hints)
		else:
			expr = self.expr
		
		if isinstance(expr, sympy.tensor.tensor.TensExpr):
			expr = expr.expand()
		
		return _do_angular_integral(expr, self.wavevec)

def do_angular_integral(Expr, wavevec):
	"""
	DEPRECATED
	"""
	return AngularIntegral(Expr, wavevec).doit(deep=False)

def _do_angular_integral(Expr, wavevec):
	"""
	This is the meat of AngularIntegral.doit().
	
	Arguments:
		Expr: sympy expression
		wavevec: sympy.tensor.tensor.TensorHead instance
	
	Returns:
		A sympy expression
	"""
	if isinstance(Expr, sympy.tensor.tensor.TensMul):
		wavevecs = []
		other = []
		
		for arg in Expr.args:
			if hasattr(arg, "component") and arg.component == wavevec:
				wavevecs.append(arg)
			else:
				other.append(arg)
				if arg.has(wavevec):
					raise RuntimeError("Please expand the input expression. Input was {}".format(Expr))
		
		prod_wavevecs = Expr.func(*wavevecs)
		inds = prod_wavevecs.get_indices()
		n = len(inds)
		delta = wavevec.index_types[0].delta #This is the Kronecker delta
		
		if n % 2 == 1:
			angint = 0
		else:
			if n == 0:
				angint = 4*sympy.pi
			elif n <= 6:
				p = sympy.symbols("p")
				prefactor = sympy.tensor.tensor.TensMul(*[ ( wavevec(p) * wavevec(-p) ).as_dummy()  for i in range(int(n/2))]) #multiply appropriate power of the wavenumber
				delta_combs = _gen_delta_combs(inds, delta)
				angint = 4*sympy.pi/int(scipy.special.factorial2(n+1)) * prefactor * sympy.tensor.tensor.TensAdd(*delta_combs)
			else:
				#TODO: I believe the above should work for any order, but am being a bit careful. I should think about this.
				angint = AngularIntegral(prod_wavevecs, wavevec)
		
		newargs = other + [ angint ]
		return Expr.func(*newargs).doit(deep=False)
	elif isinstance(Expr, (sympy.core.add.Add, sympy.tensor.tensor.TensAdd)):
		return Expr.func(*[do_angular_integral(i, wavevec) for i in Expr.args])
	else:
		return 4*sympy.pi*Expr

def do_wave_integral(expr, wavevec, ampl, debug=False, simp=None):
	"""
	expr: TensExpr
	wavevec: TensorHead
	ampl: Symbol
	simp: simplifying function to be applied to the Angular integral before performing the scalar integral. Needs to have signature simp(TensExpr) -> TensExpr
	debug: whether to print debug output.
	"""
	def replace_by_ampl(expr, wavevec, ampl):
		a = sympy.tensor.tensor.WildTensorIndex(True, wavevec.index_types[0])
		w = sympy.Wild('w')
		W = sympy.tensor.tensor.WildTensorHead('W')
		expr = expr.replace( w*W()*wavevec(a)*wavevec(-a), w*W()*ampl**2, repeat=True )
		return expr
	
	if debug:
		import time
		tstart = time.time()
		print(f"do_wave_integral: started @{time.time()-tstart:.2f}s")
	
	ret = expr
	
	if simp is not None:
		ret = simp(ret)
	
	ret = replace_by_ampl(ret, wavevec, ampl)
	
	if debug:
		print(f"do_wave_integral: replacement done @{time.time()-tstart:.2f}s")
	
	ret = AngularIntegral(ret, wavevec).doit().expand()
	
	if len(ret.atoms(AngularIntegral)) > 0:
		raise RuntimeError("Could not do some angular integrals")
	
	if debug:
		print(f"do_wave_integral: AngularIntegral done @{time.time()-tstart:.2f}s")
	
	if simp is not None:
		ret = simp(ret)
		
		if debug:
			print(f"do_wave_integral: applied simplifying function @{time.time()-tstart:.2f}s")
	
	ret = replace_by_ampl(ret, wavevec, ampl)
	
	if debug:
		print(f"do_wave_integral: replacement done @{time.time()-tstart:.2f}s")
	
	ret = create_scalar_integral( ampl**2 * ret, ampl)
	
	if debug:
		print(f"do_wave_integral: scalar integral created @{time.time()-tstart:.2f}s")
	
	return ret
