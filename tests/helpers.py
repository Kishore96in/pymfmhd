"""
Test helpers
"""

def check_tens_eq(expr1, expr2):
	"""
	Canonicalizes the two given tensor expressions and checks equality.
	"""
	diff = expr1 - expr2
	if diff != 0:
		#doit is required to simplify partial derivatives
		diff = diff.doit().canon_bp().simplify()
	assert diff == 0
