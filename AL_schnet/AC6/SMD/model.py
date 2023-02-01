from __future__ import print_function
import torch,os
import sys,copy
#from ase import Atoms
import numpy as np
import schnetpack as spk
from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model

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

class Model:
    def __init__(self):
        #define the number of models
        count = 0
        path = './best'+str(count)
        while(os.path.isfile(path)):
            count+=1
            path='./best'+str(count)
        self.nmodels = count
        #Customized models
        models=[]
        for i in range(0,self.nmodels):
            fname = 'best'+str(i)
            schmodel = load_model(fname)            
            calc = SpkCalculator(schmodel, device="cpu", energy="energy", forces="forces")
            models.append(calc)

        self.models = models
        self.count =0

    #obsolete; function without MPI (neigh nlocal); This gets energy in the MPI domain with ghost atoms
    def gvalues(self,ats):
        tenergy=[]
        tforces=[]
        tstress=[]

        #Building graph for autograd

        ndisplacement = []
        for calc in self.models:
            sat = copy.deepcopy(ats)
            sat.set_calculator(calc)
            energy=sat.get_potential_energy()
            
            tenergy.append(energy)
            
            force = sat.get_forces()
            tforces.append(force)
            
        #self.nenergy = torch.cat(tenergy,dim=0)
        #self.nforces = torch.cat(tforces,dim=0)
        self.nenergy = np.array(tenergy)
        self.nforces = np.array(tforces)        

    def lvalues(self,species,coordinates,cell,pbc,mylist):
        tlenergy=[]
        tenergy=[]
        tforces=[]
        tstress=[]

        aev_computer = self.aev_computer
        energy_shifter = self.energy_shifter
        
        #List for local atoms
        tlist=[]
        for iatom, neighs in mylist:
            tlist.append(iatom)


        for nn in self.nns:
            displacement = torch.zeros(3,3,requires_grad=True,dtype=torch.float64)
            scaling_factor = torch.eye(3,dtype=torch.float64)+displacement
            
            scoordinates=scaled_coordinates(cell,coordinates,pbc)
            new_cell = scale_cell(cell,scaling_factor)
            new_coordinates=real_coordinates(new_cell,scoordinates)

            #For atomic energy
            sspecies,saevs = aev_computer((species,new_coordinates),cell,pbc)
            atomic_energies = nn._atomic_energies((sspecies,saevs))
            shift_energies = energy_shifter.self_energies.clone().to(species.device)
            shift_energies = shift_energies[species]
            shift_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
            assert shift_energies.shape == atomic_energies.shape
            atomic_energies+=shift_energies
            
            loclist=torch.tensor(tlist,dtype=torch.int64,device=species.device)
            local_energies=torch.index_select(atomic_energies,1,loclist)
            
            #energy=nnp((species,new_coordinates),new_cell,pbc).energies
            #derivative = torch.autograd.grad(energy.sum(), new_coordinates,retain_graph=True)[0]
            #force = -derivative
            #stress = torch.autograd.grad(energy.sum(), displacement)[0]

            derivative = torch.autograd.grad(atomic_energies.sum(),coordinates,retain_graph=True)[0]
            #force = torch.index_select(-derivative,1,loclist)
            force=-derivative
            #stress = np.array(torch.autograd.grad(local_energies.sum(),displacement)[0].cpu())
            stress = torch.autograd.grad(local_energies.sum(),displacement)[0]

            energy =torch.sum(local_energies,dim=1)

            #print("lenergy",local_energies)
            #print("energy",energy)
            
            tlenergy.append(local_energies)
            tenergy.append(energy)
            tforces.append(force)
            tstress.append(stress[None,:])

        self.nlenergy = torch.cat(tlenergy,dim=0)
        self.nenergy = torch.cat(tenergy,dim=0)
        self.nforces = torch.cat(tforces,dim=0)
        self.nstress = torch.cat(tstress,dim=0)                
        

