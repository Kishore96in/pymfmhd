"""
This allows `python -m pymfmhd` to launch an interactive IPython shell with some useful symbols predefined.

This is a highly simplified version of sympy/interactive/session.py
"""

from IPython.terminal import ipapp
import sympy
import sys

pre_commands = """\
from sympy import *
import sympy.tensor.tensor as tens

w1, w2 = symbols('w_1, w_2', cls=Wild)

R3 = tens.TensorIndexType("R", dim=3)
i, j, i1, i2, i3, i4 = tens.tensor_indices("i, j, i_1, i_2, i_3, i_4", R3)
K, Q = symbols("K, Q", cls=tens.TensorHead, index_types=[R3])
W1, W2 = symbols('W_1, W_2', cls=tens.WildTensorHead)
"""

message = f"""\
These commands were executed:
{pre_commands}
"""

app = ipapp.TerminalIPythonApp()
app.display_banner = False
app.initialize()

app.shell.enable_pylab(import_all=False)
app.shell.run_cell(pre_commands, False)
sympy.init_printing(ip=app.shell)

print(message)
app.shell.mainloop()
sys.exit('Exiting ...')
