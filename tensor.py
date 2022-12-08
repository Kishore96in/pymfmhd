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
	a = sympy.tensor.tensor.WildTensorIndex(True, wavevec.index_types[0])
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
	
	a = sympy.tensor.tensor.WildTensorIndex(True, wavevec.index_types[0])
	w = sympy.Wild('w')
	W = sympy.tensor.tensor.WildTensorHead('W')
	expr = expr.replace( w*W()*wavevec(a)*wavevec(-a), w*W()*ampl**2, repeat=True )
	return expr

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
				scalarpart = Expr.coeff
				tensorpart = Expr/scalarpart
			else:
				warnings.warn("Could not find any tensor part of {}. Is this correct?".format(Expr), RuntimeWarning)
				scalarpart = Expr
				tensorpart = 1
			
			if scalarpart.has(wavevec.head):
				warnings.warn("Ignoring {} dependence in {}".format(wavevec, scalarpart), RuntimeWarning)
			
			ret += lowered_wavevec/ampl * tensorpart * sympy.Derivative(scalarpart, ampl)
		
		if indextype is not None:
			#NOTE: a separate call to contract_metric does not seem to be needed when we have already set the metric of the TensorIndexType to delta
			ret = ret.contract_delta(indextype.delta).contract_metric(indextype.metric)
		
		return ret

if __name__ == "__main__":
	sy = sympy
	
	Cartesian = sy.tensor.tensor.TensorIndexType('Cartesian', dim=3)
	Cartesian.set_metric(Cartesian.delta)
	p, q, r, s, t, u = sy.tensor.tensor.tensor_indices("p q r s t u", Cartesian)
	f, g = sympy.symbols('f g', cls=sympy.Function)
	
	K = sy.tensor.tensor.TensorHead("K", [Cartesian])
	k = sympy.symbols("K")
	V = sy.tensor.tensor.TensorHead("V", [Cartesian])
	
