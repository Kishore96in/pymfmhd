"""
TODO Implement
* Vector fields
* Unit vectors (probably just implement as annotation on vector (fields)), but note that I need to be able to differentiate these guys.
* Tensor fields
* Derivative object? Need to be able to represent curl, dive, grad etc. Check how much can be shared with sympy's inbuilt derivative object. Probably should be able to add a derivative method to the Tensor object.

TODO: function that prints an index-based expression in vector notation.
TODO: Think about how I would handle, say, a divergenceless vector field. I don't think the assumptions system works yet, so what would be the best way to do it? I don't think subs would work. Workaround could be to add 'divergenceless' and 'curlless' annotations to vector fields.
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

def do_epsilon_delta(Expr, eps, delta):
	"""
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

def gen_ind_combs(inds):
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
				for comb in gen_ind_combs(remaining_inds):
					ind_combs.append( [(ind1,ind2)] + comb )
			else:
				ind_combs.append([(ind1,ind2)])
	
	return ind_combs

def gen_delta_combs(inds, delta):
	"""
	Used in do_angular_integral() to generate combinations of the Kronecker delta that appear in the angular integral over unit vectors.
	"""
	ind_combs = gen_ind_combs(inds)
	
	delta_combs = []
	for comb in ind_combs:
		this_delta = 1
		for i in range(len(comb)):
			this_delta *= delta(*comb[i])
		
		delta_combs.append(this_delta)
	
	return delta_combs

def do_angular_integral(Expr, wavevec, delta):
	"""
	Perform angular integrals over the vector wavevec.
	
	Arguments:
		Expr: sympy expression
		wavevec: sympy.tensor.tensor.TensorHead instance
		delta: .delta method of a sympy.tensor.tensor.TensorIndexType instance
	
	Returns:
		A sympy expression
	"""
	if Expr.func == sympy.tensor.tensor.TensMul:
		wavevecs = []
		other = []
		
		for arg in Expr.args:
			if hasattr(arg, "component") and arg.component == wavevec:
				wavevecs.append(arg)
			else:
				other.append(arg)
		
		#TODO: Need to raise warning/error if any member of other depends on wavevec?
		
		prod_wavevecs = Expr.func(*wavevecs)
		inds = prod_wavevecs.get_indices()
		n = len(inds)
		
		if n % 2 == 1:
			angint = 0
		else:
			delta_combs = gen_delta_combs(inds, delta)
			if n <= 6:
				p = sympy.symbols("p")
				prefactor = sympy.tensor.tensor.TensMul(*[ ( wavevec(p) * wavevec(-p) ).as_dummy()  for i in range(int(n/2))]) #multiply appropriate power of the wavenumber
				angint = 4*sympy.pi/int(scipy.special.factorial2(n+1)) * prefactor * sympy.tensor.tensor.TensAdd(*delta_combs)
			else:
				#TODO: I believe the above should work for any order, but am being a bit careful. I should think about this.
				warnings.RuntimeWarning("Integral over {} wavevectors not implemented.".format(n))
				angint = prod_wavevecs
		
		newargs = other + [ angint ]
		return Expr.func(*newargs)
	elif Expr.func == sympy.core.add.Add or sympy.core.mul.Mul or sympy.tensor.tensor.TensAdd:
		return Expr.func(*[do_angular_integral(i) for i in Expr.args])
	else:
		return Expr

def partialdiff(Expr, wavevec, indextype=None, ampl=None):
	"""
	Take partial derivative of a tensor expression with respect to a tensor. If the expression contains scalar functions dependent on the amplitude of the wavevector, you should specify that symbol as ampl
	
	Arguments:
		Expr: an instance of sympy.tensor.tensor.TensExpr
		wavevec: an instance of sympy.tensor.tensor.Tensor
		ampl: an instance of sympy.core.symbol.Symbol. Requires indextype to be specified.
		indextype: an instance of sympy.tensor.tensor.TensorIndexType
	
	Returns:
		ret: an instance of sympy.tensor.tensor.TensExpr
	"""
	if isinstance(Expr, sympy.tensor.tensor.TensAdd) or isinstance(Expr, sympy.core.add.Add):
		return Expr.func(*[ partialdiff(arg, wavevec, indextype=indextype, ampl=ampl) for arg in Expr.args ])
	elif isinstance(Expr, sympy.tensor.tensor.TensExpr):
		ret = sympy.tensor.toperators.PartialDerivative( Expr, wavevec )
		ret = ret._perform_derivative()
		
		if ampl is not None:
			if indextype is None:
				raise TypeError("indextype needs to be specified to make use of ampl.")
			else:
				if len(wavevec.indices) > 1:
					raise NotImplementedError("Unsure how to define amplitude for tensor with more than one index.")
				
				lowered_wavevec = wavevec.head(- wavevec.indices[0] )
				scalarpart = Expr.coeff
				tensorpart = Expr/scalarpart
				
				if scalarpart.has(wavevec.head):
					warnings.RuntimeWarning("Ignoring {} dependence in {}".format(wavevec, scalarpart))
				
				ret += lowered_wavevec/ampl * tensorpart * sympy.Derivative(scalarpart, ampl)
		
		if indextype is not None:
			#NOTE: a separate call to contract_metric does not seem to be needed when we have already set the metric of the TensorIndexType to delta
			ret = ret.contract_delta(indextype.delta).contract_metric(indextype.metric)
		
		return ret
	else:
		raise NotImplementedError("Don't know how to deal with {}".format(Expr.func))
if __name__ == "__main__":
	sy = sympy
	
	Cartesian = sy.tensor.tensor.TensorIndexType('Cartesian', dim=3)
	Cartesian.set_metric(Cartesian.delta)
	p, q, r, s, t, u = sy.tensor.tensor.tensor_indices("p q r s t u", Cartesian)
	f = sympy.symbols('f', cls=sympy.Function)
	
	K = sy.tensor.tensor.TensorHead("K", [Cartesian])
	k = sympy.symbols("K")
	
	print( partialdiff( K(p) * K(q) * f(k) , K(r), indextype=Cartesian, ampl=k ) )
