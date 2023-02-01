import os
from lammps import lammps
import numpy as np

os.system("rm geo*.xyz ss*.dat sd.dat")
lmp=lammps()
lmp.file("smd.in")
