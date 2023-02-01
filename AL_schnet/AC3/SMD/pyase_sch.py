#GSJUNG@ORNL jungg@ornl.gov
from __future__ import print_function
from lammps import lammps
import sys
import numpy as np
from ase import Atoms
from model import Model
from ase.db import connect
import torch
import schnetpack as spk
from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model

#Note that MPI can be available through export below
#export MKL_SERVICE_FORCE_INTEL=1
#export ASE_DFTB_COMMAD="dftb+ > PREFIX.out"
#lmp = lammps()

def end_of_step_callback(lmp):
  L = lammps(ptr=lmp)
  t = L.extract_global("ntimestep", 0)
  print("### END OF STEP ###", t)

def post_force_callback(lmp, v, m):
  #Loading private model from python
  #model = m.model.to(device) #TorchANI based NNP

  L = lammps(ptr=lmp)  
  t = L.extract_global("ntimestep", 0)
  nlocal = L.extract_global("nlocal", 0)
  vol = L.get_thermo("vol")

  atype = L.numpy.extract_atom_iarray("type", nlocal)
  x = L.numpy.extract_atom_darray("x", nlocal, dim=3)
  f = L.numpy.extract_atom_darray("f", nlocal, dim=3)
  
  carray = L.extract_fix("2",2,1,10)
  
  box = L.extract_box()
  boxlo = box[0]
  boxhi = box[1]
  xy = box[2]
  yz = box[3]
  xz = box[4]

  ltype = np.ndarray.tolist(np.transpose(atype)[0])
  tcell = np.array([[boxhi[0]-boxlo[0],0,0],[xy,boxhi[1]-boxlo[1],0],[xz,yz,boxhi[2]-boxlo[2]]])

  #C6H14
  #print(atype)
  systring ="" 
  for i in range(0,len(atype)):
    if(atype[i].item()==6):
      systring+='C'
    if(atype[i].item()==1):
      systring+='H'      
  ats = Atoms(systring,positions=x,cell=tcell,pbc=0)
  model = load_model('best0')
  calc = SpkCalculator(model, device="cpu", energy="energy", forces="forces")
  ats.set_calculator(calc)
  
  energy = ats.get_potential_energy()
  force = ats.get_forces()
  
  f[:][:] = force[:][:]*0.043
  carray[9]=energy*0.043

  

