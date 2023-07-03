"""
TODO Implement
* Vector fields
* Unit vectors (probably just implement as annotation on vector (fields)), but note that I need to be able to differentiate these guys.
* Tensor fields
* Derivative object? Need to be able to represent curl, dive, grad etc. Check how much can be shared with sympy's inbuilt derivative object. Probably should be able to add a derivative method to the Tensor object.

TODO: function that prints an index-based expression in vector notation.
TODO: Function to replicate the 'ind=' functionality from VEST.
TODO: Something similar to VEST's FullSimplifyVectorForm
TODO: Figure out what exactly VEST's userFormSpec does. Sounds useful.
TODO: Check how much of Cadabra's functionality I can reuse. At least Kroenecker delta seems to be implemented there. https://cadabra.science/. AUR package available.
TODO: Can I somehow tell sympy to not worry about the covariant/contravariant distinction?

TODO: May need to define a custom Tensor or TensorHead that becomes a non-tensor when contracted by itself. Currently, you cannot do something like 1/( K(p)*K(-p) ), even though it is mathematically correct. OR figure out if there's a more graceful way to do it. Also would allow one to do something like ( K(p)*K(-p) ) **2 . Another way may be to implement a pow method for TensMul that raises an error if there are any free indices? Or perhaps one may try to override __pow__, __truediv__, and __rtruediv__ in TensExpr (check what TensAdd and TensMul define these methods as). NOTE: See https://github.com/sympy/sympy/issues/18435 for a discussion.
"""

import sympy
import sympy.tensor.tensor
import sympy.tensor.toperators

import warnings
import itertools
import scipy

def replace_by_ampl(expr, wavevec, ampl):
	"""
	expr: TensExpr
	wavevec: TensorHead
	ampl: Symbol
	
	Helper to call replace on expr and replace wavevec(p)*wavevec(-p) by ampl**2
	
	Also see replace_by_ampl_optimized
	"""
	a = sympy.tensor.tensor.WildTensorIndex(True, wavevec.index_types[0], ignore_updown=True)
	w = sympy.Wild('w')
	W = sympy.tensor.tensor.WildTensorHead('W')
	expr = expr.replace( w*W()*wavevec(a)*wavevec(-a), w*W()*ampl**2, repeat=True )
	return expr

def replace_by_ampl_optimized(expr, wavevec, ampl):
	"""
	expr: TensExpr
	wavevec: TensorHead
	ampl: Symbol
	
	Optimized version of replace_by_ampl (helper to call replace on expr and replace wavevec(p)*wavevec(-p) by ampl**2)
	"""
	return expr.replace(sympy.tensor.tensor.TensMul, lambda *args: _replace_by_ampl_for_mul(args, wavevec, ampl))

def _replace_by_ampl_for_mul(args, wavevec, ampl):
	"""
	args: list, args of a TensMul
	wavevec: TensorHead
	ampl: Symbol
	
	To be used as (...).replace(TensMul, lambda *args: _replace_by_ampl_for_mul(args, wavevec, ampl))
	See replace_by_ampl_optimized for a wrapper.
	
	The hope is that by reducing unnecessary walking of the expression tree, this will be more efficient than replace_by_ampl for huge expressions (it seems that replace_by_ampl is very slow on TensAdd instances which have many terms).
	"""
	expr = sympy.tensor.tensor.TensMul(*args).doit(deep=False)
	if wavevec not in expr.atoms(sympy.tensor.tensor.TensorHead):
		return expr
	
	a = sympy.tensor.tensor.WildTensorIndex(True, wavevec.index_types[0], ignore_updown=True)
	w = sympy.Wild('w')
	W = sympy.tensor.tensor.WildTensorHead('W')
	expr = expr.replace( w*W()*wavevec(a)*wavevec(-a), w*W()*ampl**2, repeat=True )
	return expr

def do_epsilon_delta(expr, eps, delta):
	"""
	expr: TensExpr
	eps: TensorHead, TensorIndexType.epsilon
	delta: TensorHead, , TensorIndexType.delta
	
	Perform epsilon-delta replacement by using expr.replace.
	"""
	def _eps_delta_for_mul(args, eps, delta):
		"""
		args: list, args of a TensMul
		eps: TensorHead
		delta: TensorHead
		"""
		expr = sympy.tensor.tensor.TensMul(*args).doit(deep=False)
		if eps not in expr.atoms(sympy.tensor.tensor.TensorHead):
			return expr
		
		index_type = eps.index_types[0]
		_a = sympy.tensor.tensor.WildTensorIndex(True, index_type, ignore_updown=True)
		_b = sympy.tensor.tensor.WildTensorIndex(True, index_type, ignore_updown=True)
		_c = sympy.tensor.tensor.WildTensorIndex(True, index_type, ignore_updown=True)
		_d = sympy.tensor.tensor.WildTensorIndex(True, index_type, ignore_updown=True)
		_e = sympy.tensor.tensor.WildTensorIndex(True, index_type, ignore_updown=True)
		w = sympy.Wild('w')
		W = sympy.tensor.tensor.WildTensorHead('W')
		expr = expr.replace( w*W()*eps(_a, _b, _c)*eps(-_a, _d, _e), w*W()*(delta(_b,_d)*delta(_c,_e) - delta(_b,_e)*delta(_d,_c)), repeat=True )
		return expr
	
	return expr.replace(sympy.tensor.tensor.TensMul, lambda *args: _eps_delta_for_mul(args, eps, delta))

def _do_epsilon_delta(Expr, eps, delta):
	"""
	DEPRECATED: use replace
	
	Performs the replacement corresponding to the epsilon-delta identity.
	
	Arguments:
		Expr: any sympy expression (e.g. TensAdd, Add, Mul, TensMul)
		eps: .epsilon method of a sympy.tensor.tensor.TensorIndexType instance
		delta: .delta method of a sympy.tensor.tensor.TensorIndexType instance
	
	Returns:
		a sympy expression with epsilon-delta replacements performed.
	"""
	if Expr.func == sympy.tensor.tensor.TensMul:
		epsilons = []
		other = []
		
		for arg in Expr.args:
			if hasattr(arg, "component") and arg.component == eps:
				epsilons.append(arg)
			else:
				other.append(arg)
		
		for i1, e1 in enumerate(epsilons):
			for i2, e2 in enumerate(epsilons[i1+1:], start=i1+1):
				if e1 != 1 and e2 != 1: #1 is the placeholder we use to denote eliminated epsilons
					inds1 = e1.get_indices()
					inds2 = e2.get_indices()
					common_inds = [ind for ind in inds1 if -ind in inds2] #Get indices which are in both the epsilons
					if len(common_inds) > 0:
						to_elim = common_inds[0]
						other_indices = (
							[ind for ind in inds1 if ind != to_elim],
							[ind for ind in inds2 if ind != -to_elim]
							)
						
						prefactor = sympy.tensor.tensor.TensMul(e1, e2).canon_bp().coeff #Account for the ordering of indices. Should be +1 or -1
						
						#Make the epsilon-delta replacement. e1 will never be looped over after this, so we can just replace it by the required combination of deltas.
						epsilons[i2] = 1
						epsilons[i1] = prefactor * (
							delta(other_indices[0][0], other_indices[1][0]) * delta(other_indices[0][1], other_indices[1][1])
							- delta(other_indices[0][0], other_indices[1][1]) * delta(other_indices[0][1], other_indices[1][0])
							)
						
						break #We have found the epsilon that is contracted with this e1
		
		newargs = other + epsilons
		return Expr.func(*newargs)
	elif Expr.func == sympy.core.add.Add or sympy.core.mul.Mul or sympy.tensor.tensor.TensAdd:
		return Expr.func(*[do_epsilon_delta(i) for i in Expr.args])
	else:
		return Expr

def partialdiff(Expr, wavevec, ampl=None):
	"""
	Take partial derivative of a tensor expression with respect to a tensor. If the expression contains scalar functions dependent on the amplitude of the wavevector, you should specify that symbol as ampl
	
	Arguments:
		Expr: an instance of sympy.tensor.tensor.TensExpr
		wavevec: an instance of sympy.tensor.tensor.Tensor
		ampl: an instance of sympy.core.symbol.Symbol.
	
	Returns:
		ret: an instance of sympy.tensor.tensor.TensExpr
	"""
	indextype = wavevec.index_types[0]
	Expr = Expr.expand()
	
	if isinstance(Expr, sympy.tensor.tensor.TensAdd) or isinstance(Expr, sympy.core.add.Add):
		return Expr.func(*[ partialdiff(arg, wavevec, ampl=ampl) for arg in Expr.args ])
	else:
		ret = sympy.tensor.toperators.PartialDerivative( Expr, wavevec )
		ret = ret._perform_derivative()
		
		if ampl is not None:
			if indextype is None:
				raise TypeError("indextype needs to be specified to make use of ampl.")
			if len(wavevec.indices) > 1:
				raise NotImplementedError("Unsure how to define amplitude for tensor with more than one index.")
			
			lowered_wavevec = wavevec.head(- wavevec.indices[0] )
			
			if isinstance(Expr, sympy.tensor.tensor.TensMul):
				Expr = Expr.doit(deep=False) #make sure coeff is correctly populated.
				scalarpart = Expr.coeff
				tensorpart = Expr/scalarpart
			else:
				warnings.warn("Could not find any tensor part of {}. Is this correct?".format(Expr), RuntimeWarning)
				scalarpart = Expr
				tensorpart = sympy.S(1)
			
			if scalarpart.has(wavevec.head):
				warnings.warn("Ignoring {} dependence in {}".format(wavevec, scalarpart), RuntimeWarning)
			
			ret += lowered_wavevec/ampl * tensorpart * sympy.Derivative(scalarpart, ampl).doit(deep=False)
			
			if tensorpart.has(ampl):
				raise RuntimeError(f"Expanding the given expression failed; the non-coeff part depends on {ampl}.")
		
		if indextype is not None:
			ret = sympy.tensor.tensor.contract_delta(ret, indextype.delta)
			ret = sympy.tensor.tensor.contract_metric(ret, indextype.metric)
		
		return ret

class PartialVectorDerivative(sympy.tensor.tensor.TensExpr):
	"""
	Unevaluated form of partialdiff
	"""
	def __new__(cls, expr, wavevec, ampl, replace_indices=True):
		"""
		expr: TensExpr
		wavevec: Tensor, vector with respect to which the partial derivative is taken
		ampl: Symbol, variable representing magnitude of wavevec
		"""
		args, indices, free, dum = PartialVectorDerivative._contract_indices_for_derivative(sympy.S(expr), [wavevec], replace_indices=replace_indices)
		
		obj = sympy.tensor.tensor.Basic.__new__(cls, *args, ampl)
		
		obj._indices = indices
		obj._free = free
		obj._dum = dum
		obj._ext_rank = len(free) + 2*len(dum)
		
		return obj
	
	@property
	def expr(self):
		return self.args[0]
	
	@property
	def wavevec(self):
		return self.args[1]
	
	@property
	def ampl(self):
		return self.args[2]
	
	@property
	def variables(self):
		return [self.wavevec]
	
	@property
	def coeff(self):
		return S.One
	
	@property
	def nocoeff(self):
		return self
	
	@property
	def ext_rank(self):
		return self._ext_rank
	
	@property
	def components(self):
		components = []
		if hasattr(self.expr, 'components'):
			components.extend(self.expr.components)
		for arg in self.variables:
			components.extend(arg.components)
		return components
	
	def get_indices(self):
		return self._indices
	
	def get_free_indices(self):
		free = sorted(self._free, key=lambda x: x[1])
		return [i[0] for i in free]
	
	def _replace_indices(self, repl):
		if hasattr(self.expr, "_replace_indices"):
			expr = self.expr._replace_indices(repl)
		else:
			expr = self.expr
		mirrored = {-k: -v for k, v in repl.items()}
		variables = [i.xreplace(mirrored) for i in self.variables]
		return self.func(expr, *variables, self.ampl, replace_indices=False)
	
	def _set_indices(self, *indices):
		repl = dict(zip(self.get_indices(), indices))
		return self._replace_indices(repl)
	
	def commutes_with(self, other):
		return None
	
	def doit(self, **hints):
		deep = hints.get('deep', True)
		if deep:
			expr = self.expr.doit(**hints)
		else:
			expr = self.expr
		
		ret = partialdiff(expr, self.wavevec, self.ampl)
		ret = replace_by_ampl_optimized(ret, self.wavevec.component, self.ampl)
		ret = ret.as_dummy().expand() #to allow cancellation of terms
		
		if isinstance(ret, sympy.tensor.tensor.TensAdd):
			ret = ret.doit(deep=False)
		return ret
	
	@classmethod
	def _contract_indices_for_derivative(cls, expr, variables, replace_indices=True):
		"""
		Copied from PartialDerivative, but set replace_indices to False
		"""
		variables_opposite_valence = []
		
		for i in variables:
			if isinstance(i, sympy.tensor.tensor.Tensor):
				i_free_indices = i.get_free_indices()
				variables_opposite_valence.append(
						i.xreplace({k: -k for k in i_free_indices}))
			elif isinstance(i, sympy.Symbol):
				variables_opposite_valence.append(i)
		
		args, indices, free, dum = sympy.tensor.tensor.TensMul._tensMul_contract_indices(
			[expr] + variables_opposite_valence, replace_indices=replace_indices)
		
		for i in range(1, len(args)):
			args_i = args[i]
			if isinstance(args_i, sympy.tensor.tensor.Tensor):
				i_indices = args[i].get_free_indices()
				args[i] = args[i].xreplace({k: -k for k in i_indices})
		
		return args, indices, free, dum

if __name__ == "__main__":
	sy = sympy
	
	Cartesian = sy.tensor.tensor.TensorIndexType('Cartesian', dim=3)
	Cartesian.set_metric(Cartesian.delta)
	p, q, r, s, t, u = sy.tensor.tensor.tensor_indices("p q r s t u", Cartesian)
	f, g = sympy.symbols('f g', cls=sympy.Function)
	
	K = sy.tensor.tensor.TensorHead("K", [Cartesian])
	k = sympy.symbols("K")
	V = sy.tensor.tensor.TensorHead("V", [Cartesian])
	
