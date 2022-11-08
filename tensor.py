"""
TODO Implement
* Vector fields
* Unit vectors (probably just implement as annotation on vector (fields)), but note that I need to be able to differentiate these guys.
* Tensor fields
* Derivative object? Need to be able to represent curl, dive, grad etc. Check how much can be shared with sympy's inbuilt derivative object. Probably should be able to add a derivative method to the Tensor object.
TODO: Currently I can't do something like ( K(p)*K(-p) ) **2 . I would like to be able to do that. Implement a pow method for TensMul that raises an error if there are any free indices?
TODO: function that prints an index-based expression in vector notation.
TODO: Think about how I would handle, say, a divergenceless vector field. I don't think the assumptions system works yet, so what would be the best way to do it? I don't think subs would work. Workaround could be to add 'divergenceless' and 'curlless' annotations to vector fields.
TODO: Function to replicate the 'ind=' functionality from VEST.
TODO: Something similar to VEST's FullSimplifyVectorForm
TODO: Figure out what exactly VEST's userFormSpec does. Sounds useful.
TODO: Check how much of Cadabra's functionality I can reuse. At least Kroenecker delta seems to be implemented there. https://cadabra.science/. AUR package available.
TODO: Can I somehow tell sympy to not worry about the covariant/contravariant distinction?
"""

import sympy
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
		
		#TODO: Need to raise warning/error if any member of other depends on wavevec?
		
		for arg in Expr.args:
			if hasattr(arg, "component") and arg.component == wavevec:
				wavevecs.append(arg)
			else:
				other.append(arg)
		
		prod_wavevecs = Expr.func(*wavevecs)
		inds = prod_wavevecs.get_free_indices()
		#TODO: Not sure what to do about the internally contracted indices (e.g. K(p) * K(-p) ). Just instantiate a scalar symbol with the same name? Perhaps have the user pass the wavenumber variable as an argument? Currently they are just ignored!
		n = len(inds)
		
		if n % 2 == 1:
			angint = 0
		else:
			delta_combs = gen_delta_combs(inds, delta)
			if n <= 6:
				angint = 4*sympy.pi/int(scipy.special.factorial2(n+1)) * sympy.tensor.tensor.TensAdd(*delta_combs)
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

if __name__ == "__main__":
	import sympy.tensor.tensor
	sy = sympy
	
	Cartesian = sy.tensor.tensor.TensorIndexType('Cartesian', dim=3)
	p, q, r, s, t = sy.tensor.tensor.tensor_indices("p q r s t", Cartesian)
	
	K = sy.tensor.tensor.TensorHead("K", [Cartesian])
	
	print( do_angular_integral( K(p) * K(q), K, Cartesian.delta ) )
	print( do_angular_integral( K(p) * K(q) * K(r) * K(s), K, Cartesian.delta ) )
