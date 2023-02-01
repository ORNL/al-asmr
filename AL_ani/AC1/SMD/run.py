import os
from lammps import lammps
import torch
import torchani
import numpy as np

os.system("rm geo*.xyz ss*.dat sd.dat")
lmp=lammps()
#lmp.file("bend.in")
lmp.file("smd.in")
#lmp.file("test.in")
#lmp.file("ch4.in")
#lmp.file("in.fix_python_move_nve_melt")
