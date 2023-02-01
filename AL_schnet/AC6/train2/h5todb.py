import ase,os,math
import h5py
import numpy as np
from ase.io import read
from schnetpack import AtomsData
from ase import Atoms
from ase.units import Hartree

import schnetpack as spk
from schnetpack.data import AtomsDataError
from schnetpack.datasets import DownloadableAtomsData

hartree2ev = np.float32(27.211386024367243)
ev2hartree = np.float32(1.0/hartree2ev)
hartree2kcalmol = np.float32(627.5094738898777)
class HDF5(DownloadableAtomsData):
    """
    MD17 benchmark data set for molecular dynamics of small molecules
    containing molecular forces.

    Args:
        dbpath (str): path to database
        molecule (str or None): Name of molecule to load into database. Allowed are:
                            aspirin
                            benzene
                            ethanol
                            malonaldehyde
                            naphthalene
                            salicylic_acid
                            toluene
                            uracil
        subset (list, optional): Deprecated! Do not use! Subsets are created with
            AtomsDataSubset class.
        download (bool): set true if dataset should be downloaded
            (default: True)
        collect_triples (bool): set true if triples for angular functions
            should be computed (default: False)
        load_only (list, optional): reduced set of properties to be loaded
        environment_provider (spk.environment.BaseEnvironmentProvider): define how
            neighborhood is calculated
            (default=spk.environment.SimpleEnvironmentProvider).


    See: http://quantum-machine.org/datasets/
    """

    energy = "energy"
    forces = "forces"

    def __init__(
        self,
        dbpath,
        download=False,
        subset=None,
        collect_triples=False,
        high_energies=False,            
        load_only=None,
        environment_provider=spk.environment.SimpleEnvironmentProvider(),
    ):

        available_properties = [HDF5.energy, HDF5.forces]
        units =[1.0, 1.0]

        super().__init__(
            dbpath=dbpath,
            subset=subset,
            load_only=load_only,
            collect_triples=collect_triples,
            download=download,
            available_properties=available_properties,
            environment_provider=environment_provider,
        )

    def _load_data(self):
        file_name ='data.h5'
        self._load_h5_file(file_name)
        
    def load_h5file(self,file_name):
        coords,tforces,cells,energies,species,virial=ReadHDF5('data.h5',0)
        for i in range(0,len(energies)):
            #atm = Atoms(species,coords[i],cells[i],pbc=1)
            atm = Atoms(species,coords[i])
            energy = energies[i]#*self.units[self.energy]
            force = tforces[i]#*self.units[self.forces]
            #properties ={self.energy:energy,self.forces:force}
            properties ={self.energy:energy}
            atoms_list.append(atm)
            properties_list.append(properties)

        self.add_system(atoms_list,properties_list)


def ReadHDF5(filename,cut):
    h5fr = h5py.File(filename,'r')    
    mols =h5fr['mols']
    keys=list(mols.keys())
    mol =mols[keys[0]]        
    coordinates = np.array(mol['coordinates'][:])
    forces = np.array(mol['forces'][:])
    cell = np.array(mol['cell'][:])
    energies = np.array(mol['energies'][:])
    species = np.array(mol['species'][:])
    virial = np.array(mol['virial'][:])    
    print('########## Read %s ###############' %filename)
    print('######### Species %s ' %len(species))
    print('######### Data # %s ' %len(coordinates))    
    h5fr.close()    
    return coordinates,forces,cell,energies,species,virial

def SaveHDF5(filename,coordinates,forces,energy,species,cell,virial):
    h5f = h5py.File(filename,'w')    
    mols = h5f.create_group('mols')
    mol = mols.create_group('mol')
    mol.create_dataset('coordinates',data=coordinates)
    mol.create_dataset('forces',data=forces)
    mol.create_dataset('cell',data=cell)
    mol.create_dataset('energies',data=energy)
    mol.create_dataset('species',data=species)
    mol.create_dataset('virial',data=virial)
    print('Saved')
    h5f.close()

def CheckHDF5(filename):
    h5fr = h5py.File(filename,'r')
    mols =h5fr['mols']
    keys=list(mols.keys())
    print(keys)
    mol =mols[keys[0]]    
    print("Keys: %s" %mol.keys())
    species = np.array(mol['species'][:])    
    print(mol['species'],species)

    print(mol['cell'])
    print(mol['coordinates'])
    print(mol['forces'])    
    print(mol['energies'])
    print(mol['virial'])    
    h5fr.close()
    return species
    
def H5todbeV(filename):
    coords,tforces,cells,energies,species,virial=ReadHDF5(filename,0)

    atoms=[]
    property_list =[]

    print(tforces[0])

    for i in range(0,len(coords)):
        #mol = Atoms('CH4',positions=coords[i],cell=cells[i],pbc=1)
        mol = Atoms('CH4',positions=coords[i],cell=cells[i],pbc=1)
        atoms.append(mol)
        property_list.append(
            {'energy':np.array([energies[i]*hartree2ev],dtype=np.float32),'forces':np.array([tforces[i]*hartree2ev],dtype=np.float32)}
        )

    os.system('rm ./new_dataset.db')
    new_dataset = AtomsData('./new_dataset.db', available_properties=['energy','forces'])
    new_dataset.add_systems(atoms, property_list)

    example = new_dataset[0]
    print('Properties of molecule with id 0:')

    for k, v in example.items():
        print('-', k, ':', v.shape,v)


def H5todb(filename,comp):
    #Hartree energy
    coords,tforces,cells,energies,species,virial=ReadHDF5(filename,0)

    atoms=[]
    property_list =[]

    print(tforces[0])

    for i in range(0,len(coords)):
        #mol = Atoms('CH4',positions=coords[i],cell=cells[i],pbc=1)
        mol = Atoms(comp,positions=coords[i],cell=cells[i],pbc=1)        
        atoms.append(mol)
        property_list.append(
            {'energy':np.array([energies[i]],dtype=np.float32),'forces':np.array([tforces[i]],dtype=np.float32)}
        )

    os.system('rm ./new_dataset.db')
    new_dataset = AtomsData('./new_dataset.db', available_properties=['energy','forces'])
    new_dataset.add_systems(atoms, property_list)

    example = new_dataset[0]
    print('Properties of molecule with id 0:')

    for k, v in example.items():
        print('-', k, ':', v.shape,v)

def H5todbkcal(filehead,comp):
    #Hartree energy
    filename = filehead+".h5"
    coords,tforces,cells,energies,species,virial=ReadHDF5(filename,0)

    atoms=[]
    property_list =[]

    for i in range(0,len(coords)):
        tcell = np.transpose(cells[i])
        mol = Atoms(comp,positions=coords[i],cell=tcell,pbc=1)        
        atoms.append(mol)
        property_list.append(
            {'energy':np.array([energies[i]*hartree2kcalmol],dtype=np.float32),'forces':np.array(tforces[i]*hartree2kcalmol,dtype=np.float32)}
        )

    dbname=filehead+".db"
    com="rm " +dbname
    os.system(com)
    new_dataset = AtomsData(dbname, available_properties=['energy','forces'])
    new_dataset.add_systems(atoms, property_list)

    example = new_dataset[0]
    #print('Properties of molecule with id 0:')

    #for k, v in example.items():
    #    print('-', k, ':', v.shape,v)        

species=CheckHDF5('validation.h5')

systring ="" 
for i in range(0,len(species)):
    if(species[i].decode('utf-8')=='C'):
        systring+='C'
    if(species[i].decode('utf-8')=='H'):        
        systring+='H'      
print(systring)

H5todbkcal('train',systring) #heptane
H5todbkcal('validation',systring) #heptane
#H5todbkcal('smd',systring) #heptane

    
