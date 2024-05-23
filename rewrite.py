"""
Utilities to manipulate/rearrange/rewrite expressions
"""

from sympy import (
	Symbol,
	Add,
	Integral,
	Wild,
	Mul,
	Derivative,
	)
import numpy as np
import warnings

def bring_to_form(expr, term1, term2, wild_properties=()):
	"""
	For all occurrences of term1 in expr, try to convert them to term2 by consolidating them with occurrences of (term2-term1) in expr.
	If you pass in a list wild_properties, that will be used to restrict the coefficient of term1 that is matched (see the properties keyword of sympy.Wild).
	"""
	wild_properties = (lambda expr: not expr.has(term1), *wild_properties)
	w = Wild("__bring_to_form_internal", properties=wild_properties)
	expr = expr.expand()
	repl = w*term2 - (w*term2-w*term1).doit().expand()
	return expr.replace(w*term1, repl).expand()

class CallLimitWarning(RuntimeWarning): pass
class ResultRejectedWarning(RuntimeWarning): pass

def revert_product_rule(expr, var, ratio=1.7, maxcalls=100, call=0):
	"""
	Given an expression containing derivatives of AppliedUndef instances, try to simplify it by pulling out a derivative using the product rule. Assumes all derivatives in the expression are wrt. var.
	
	Arguments:
		expr: Expr
		var: Symbol, variable wrt which the derivative is
		maxcalls: int, maximum number of times this function should be applied on the expression.
		ratio: accept simplification only if ret.count_ops() < ratio*expr.count_ops()
	"""
	if call > maxcalls:
		warnings.warn("revert_product_rule: reached call limit, so giving up.", CallLimitWarning)
		return expr
	
	assert isinstance(var, Symbol)
	
	expr = expr.expand()
	if not isinstance(expr, Add):
		return expr
	
	def der_orders(expr):
		"""
		Return a list containing the order of each Derivative operator in expr. All other objects are considered to have order 0. Returns a list of the same length as expr.args
		"""
		if isinstance(expr, Mul):
			return [(arg.derivative_count if isinstance(arg, Derivative) else 0) for arg in expr.args]
		elif isinstance(expr, Derivative):
			return [expr.derivative_count]
		else:
			raise NotImplementedError
	
	def choose_base_term(expr):
		"""
		Among all args of expr, choose a term suitable for integrating by parts. The term with the highest-order derivative is preferred, and among such terms, the term with the lowest number of Derivative instances is chosen. If nothing could be found, return None
		"""
		args = [arg for arg in expr.args if arg.has(Derivative)]
		args = [arg for arg in args if isinstance(arg, Mul)]
		args = [arg for arg in args if not isinstance(arg.as_coeff_Mul()[1], Derivative)]
		args.sort(key=lambda a: a.count(Derivative))
		args.sort(key=lambda a: np.max(der_orders(a)), reverse=True)
		
		if len(args) > 0:
			return args[0]
		else:
			return None
		
	base_term = choose_base_term(expr)
	if base_term is None:
		return expr
	
	der = base_term.args[np.argmax(der_orders(base_term))]
	rest = base_term/der
	der_int = Integral(der, var).doit(deep=False, manual=True)
	
	#Handle terms of the form F(x)*diff(F(x),x). Special-casing is required because otherwise we would get stuck in an infinite loop.
	x = Symbol("__rpr_internal")
	rest = rest.xreplace({der_int: x})
	d = rest.collect(x, evaluate=False)
	if len(d.keys()) != 1:
		raise RuntimeError
	k = list(d.keys())[0]
	b,p = k.as_base_exp()
	if b == x:
		if p == -1:
			raise NotImplementedError
		rest = rest/((p+1)*x**p)
		der_int = der_int**(p+1)
	rest = rest.xreplace({x: der_int})
	
	#To aid cancellation, numerical coefficients are extracted out of the derivative.
	outside, inside = (der_int*rest).as_coeff_Mul()
	ret = outside*Derivative(inside, var) + revert_product_rule( -der_int*Derivative(rest, var).doit() + expr - base_term, var, maxcalls=maxcalls, call=call+1)
	if ret.count_ops() < ratio*expr.count_ops():
		return ret
	else:
		warnings.warn("revert_product_rule: could not simplify input.", ResultRejectedWarning)
		return expr 

def replace_repeat(expr, pattern, replacement):
	"""
	Repeatedly perform replacement till the expression no longer changes.
	"""
	old_expr = 0
	while expr != old_expr:
		old_expr = expr
		expr = expr.replace(pattern, replacement)
	return expr 
