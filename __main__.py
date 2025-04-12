"""
This allows `python -m pymfmhd` to launch an interactive IPython shell with some useful symbols predefined.

This is a highly simplified version of sympy/interactive/session.py
"""

from IPython.terminal import ipapp
import sympy
import sys

#Some magic to make sure the pymfmhd module is always available as 'pymfmhd', regardless of how it is actually installed.
pre_commands_silent = f"""\
import importlib
pymfmhd = importlib.import_module("{__name__.removesuffix(".__main__")}")
"""

#Other import commands
pre_commands = """\
from sympy import *
import sympy.tensor.tensor as tens

R3 = tens.TensorIndexType("R", dim=3)
i, j, i1, i2, i3, i4 = tens.tensor_indices("i, j, i_1, i_2, i_3, i_4", R3)

k, q = symbols("k, q", positive=True)
K, Q = symbols("K, Q", cls=tens.TensorHead, index_types=[R3])

w1, w2 = symbols('w_1, w_2', cls=Wild)
W1, W2 = symbols('W_1, W_2', cls=tens.WildTensorHead)
"""

message = f"""\
These commands were executed:
import pymfmhd
{pre_commands}
"""

app = ipapp.TerminalIPythonApp()
app.display_banner = False
app.initialize()

app.shell.enable_pylab(import_all=False)
app.shell.run_cell(pre_commands_silent, store_history=False)
app.shell.run_cell(pre_commands, store_history=False)
sympy.init_printing(ip=app.shell)

print(message)
app.shell.mainloop()
sys.exit('Exiting ...')
