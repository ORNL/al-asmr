from __future__ import print_function
from lammps import lammps
import torch,os
import torchani
import sys
import numpy as np
from torchani.utils import ChemicalSymbolsToInts
from model import Model

hartree2ev = np.float64(27.211386024367243)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_str = 'cpu'
device = torch.device(device_str)
dtype_str = 'float64'

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

#lmp = lammps()

def scale_cell(cell,scaling_factor):
  return torch.matmul(scaling_factor,cell)

def real_coordinates(cell,scoordinates):
  coordinates = torch.matmul(scoordinates,cell)
  return coordinates

def scaled_coordinates(cell,coordinates,pbc):
  inv_cell = torch.inverse(cell)
  scoordinates = torch.matmul(coordinates,inv_cell)
  #wrappig
  #scoordinates -= scoordinates.floor()*pbc
  return scoordinates

def end_of_step_callback(lmp):
  L = lammps(ptr=lmp)
  t = L.extract_global("ntimestep", 0)
  print("### END OF STEP ###", t)

def post_force_callback(lmp, v, m):
  #Loading model from python 
  #nn = m.nn
  species_to_tensor=m.species_to_tensor
  
  emodel = m

  
  
  #aev_computer = m.aev_computer
  #energy_shifter = m.energy_shifter

  L = lammps(ptr=lmp)  
  #extract_global variable_name flags(0 = integer, 2 integer)
  t = L.extract_global("ntimestep", 0)
  nlocal = L.extract_global("nlocal", 0)
  atype = L.numpy.extract_atom_iarray("type", nlocal)
  x = L.numpy.extract_atom_darray("x", nlocal, dim=3)
  f = L.numpy.extract_atom_darray("f", nlocal, dim=3)
  
  #carray = L.extract_fix("2",1,1,10)
  carray = L.extract_fix("2",2,1,10)
  
  box = L.extract_box()
  boxlo = box[0]
  boxhi = box[1]
  xy = box[2]
  yz = box[3]
  xz = box[4]

  slist=[]
  #for i in range(0,nlocal):
  #    slist.append('C')
  #species
  for i in range(0,22):
    slist.append('C')
    slist.append('C')
    slist.append('C')
    slist.append('H')
    slist.append('H')
    slist.append('H')  
  
  species = species_to_tensor(slist).unsqueeze(0).to(device)     
  lcoord=np.ndarray.tolist(x)

  coordinates = torch.tensor([lcoord],requires_grad=True,dtype=torch.double,device=device)
  
  ltype = np.ndarray.tolist(np.transpose(atype)[0])
  cell=torch.tensor([[boxhi[0]-boxlo[0],0,0],[xy,boxhi[1]-boxlo[1],0],[xz,yz,boxhi[2]-boxlo[2]]],dtype=torch.float64,device=device) # check version2   may be this is right
  pbc = torch.tensor(box[5],dtype=torch.bool,device=device)

  emodel.gvalues(species,coordinates,cell,pbc)

  energy = torch.mean(emodel.nenergy,dim=0)
  force = torch.mean(emodel.nforces,dim=0).numpy()
  stress =  torch.mean(emodel.nstress,dim=0)
  
  """
  displacement = torch.zeros(3,3,requires_grad=True,dtype=torch.float64)
  scaling_factor = torch.eye(3,dtype=torch.float64)+displacement
  
  scoordinates=scaled_coordinates(cell,coordinates,pbc)
  new_cell = scale_cell(cell,scaling_factor)
  new_coordinates=real_coordinates(new_cell,scoordinates)

  energy = model((species, new_coordinates),new_cell,pbc).energies
  derivative = torch.autograd.grad(energy.sum(),coordinates,retain_graph=True)[0]
  force = -derivative.numpy()[0]
  stress = torch.autograd.grad(energy.sum(),displacement)[0]
  """
  """
  print("energy",energy)  
  print("forces",force[:][:])
  print("stress",stress) 
  """  
  f[:][:] = force[:][:]*hartree2ev
  carray[9]=energy.sum()*hartree2ev
  for i in range(0,3):
    for j in range(0,3):
      carray[i*3+j]=-stress[i][j]*hartree2ev


