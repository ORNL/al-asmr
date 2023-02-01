from __future__ import print_function
from lammps import lammps
import sys
import numpy as np
from ase.build import bulk
from ase.calculators.espresso import Espresso
from ase import Atoms
from model import Model

#Note that MPI can be available through export below
#export MKL_SERVICE_FORCE_INTEL=1
#export ASE_ESPRESSO_COMMAND="mpirun -n 12 pw.x -in PREFIX.pwi > PREFIX.pwo"
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

  #Graphene
  """
  pseudopotentials ={'C': 'C.pbe-n-kjpaw_psl.1.0.0.UPF'}
  graphene = Atoms('C4',positions=x,cell=tcell,pbc=1)
  input={'control':{'pseudo_dir':'./'},
       'system': {'ecutwfc':40,'ecutrho':400,'occupations':'smearing','degauss':0.001},
       'electrons': {'conv_thr': 1.0e-9,'mixing_beta': 0.3,'electron_maxstep':200}
  }
  """

  #MoS
  """
  pseudopotentials ={'Mo': 'Mo.pbe-spn-kjpaw_psl.1.0.0.UPF','S':'S.pbe-nl-kjpaw_psl.1.0.0.UPF'}
  model = Atoms('Mo3S3Mo3S3Mo3S3Mo3S3Mo3S3Mo3S3',positions=x,cell=tcell,pbc=1)
  input={'control':{'pseudo_dir':'./'},
       'system': {'ecutwfc':50,'ecutrho':500,'occupations':'smearing','degauss':0.001},
       'electrons': {'conv_thr': 1.0e-9,'mixing_beta': 0.3,'electron_maxstep':300}
  }
  calc = Espresso(pseudopotentials=pseudopotentials,
                tstress=True, tprnfor=True,kpts=(1,1,1),input_data = input)
  model.calc = calc
  """
  
  #C6H14
  model = Atoms('C6',positions=x,cell=tcell,pbc=1)
  calc = Dftb(Hamiltonian_ = 'DFTB',
              Hamiltonian_SCC='Yes',
              Hamiltonian_SCCTorlerance = 1e-8,
              HamiltonianMaxAngularMomentum_H='s',
              HamiltonianMaxAngularMomentum_C='p')

  
  


  energy = model.get_potential_energy()
  force = model.get_forces()
  sig = model.get_stress()  
  
  f[:][:] = force[:][:]
  carray[9]=energy
  carray[0]=-sig[0]*vol #xx
  carray[1]=-sig[3]*vol #xy
  carray[2]=-sig[4]*vol #xz
  carray[3]=-sig[3]*vol #xy
  carray[4]=-sig[1]*vol #yy
  carray[5]=-sig[5]*vol #yz
  carray[6]=-sig[4]*vol #xz
  carray[7]=-sig[5]*vol #yz
  carray[8]=-sig[2]*vol #yz  
  

