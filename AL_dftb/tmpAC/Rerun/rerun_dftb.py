#This script for LAMMPS using python interface to generate coordinates, forces, energy from DFTB+
#jungg@ornl.gov
from lammps import lammps, LAMMPS_INT, LMP_STYLE_GLOBAL, LMP_VAR_EQUAL, LMP_VAR_ATOM
import numpy as np
import copy
import h5py
import graphene2cg as Gra

hartree2ev = np.float64(27.211386024367243)
ev2hartree = np.float64(1.0/hartree2ev)

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
    mol =mols['mol']    
    print("Keys: %s" %mol.keys())
    spec = mol['species']
    print(mol['species'],spec[:])
    print(mol['cell'])
    print(mol['coordinates'])
    print(mol['forces'])    
    print(mol['energies'])
    print(mol['virial'])    
    h5fr.close()            

def ReadHDF5(filename,cut):
    h5fr = h5py.File(filename,'r')    
    mols =h5fr['mols']
    mol =mols['mol']
    coordinates = np.array(mol['coordinates'][cut:])
    forces = np.array(mol['forces'][cut:])
    cell = np.array(mol['cell'][cut:])
    energies = np.array(mol['energies'][cut:])
    species = np.array(mol['species'][:])
    virial = np.array(mol['virial'][cut:])    
    print('########## Read %s ###############' %filename)
    print('######### Species %s ' %len(species))
    print('######### Data # %s ' %len(coordinates))    
    h5fr.close()    
    return coordinates,forces,cell,energies,species,virial

#


lmp = lammps()

#Initial relaxation with 0 pressure
lmp.file("rerun_dftb.in")
lmp.command(0)
nlocal = lmp.extract_global("nlocal",0)
natom = nlocal
#Species array, in this case all carbon
species = np.chararray(nlocal)
atype = lmp.numpy.extract_atom_iarray("type",nlocal)
for i in range(0,len(atype)):
    if(atype[i]==6):
        species[i] = 'C'
    if(atype[i]==1):
        species[i] = 'H'
    if(atype[i]==8):
        species[i] = 'O'
    if(atype[i]==7):
        species[i] = 'N'                        

box = lmp.extract_box()
boxlo = box[0]
boxhi = box[1]
xy = box[2]
yz = box[3]
xz = box[4]

lmp.close()
#Relaxed structure
#np.float64 = double in C

#place holder
coordinates=np.empty(shape=[0,natom,3])
cell=np.empty(shape=[0,3,3])
virial=np.empty(shape=[0,6])
energy=np.empty(shape=[0])
forces=np.empty(shape=[0,natom,3])

out=""
gra = Gra.Graphene(8,12)
hdfcoordinates,hdfforces,hdfcell,hdfenergies,hdfspecies,hdfvirial = ReadHDF5('selected.h5',0)


for i in range(0,len(hdfcell)):
#for i in range(0,1):
    tmpcoords = hdfcoordinates[i]
    tmpcell = hdfcell[i]

    gra.HDF(tmpcoords,tmpcell,species)
    Gra.GenDataANI(gra,"molani.data")
    Gra.GenDataDFTB(gra,"geo_end.gen",False)

    print("########################## %d #######################################" %i)
    lmp = lammps()
    lmp.file("rerun_dftb.in")

    lmp.command("run 0")

    #box, coords, forces, energy
    nlocal = lmp.extract_global("nlocal",0)
    box = lmp.extract_box()
    boxlo = box[0]
    boxhi = box[1]
    xy = box[2]
    yz = box[3]
    xz = box[4]    
    tcell = np.array([[[boxhi[0]-boxlo[0],0,0],[xy,boxhi[1]-boxlo[1],0],[xz,yz,boxhi[2]-boxlo[2]]]],dtype=np.float64)
    tx = np.array([lmp.numpy.extract_atom_darray("x",nlocal,dim=3)])
    tf = np.array([lmp.numpy.extract_atom_darray("f",nlocal,dim=3)*ev2hartree])
    tpe = np.array([lmp.get_thermo("pe")*ev2hartree])
    tvir = np.zeros(6)

    for i in range(0,6):
        vname="vir"+str(i)
        tmp=lmp.extract_variable(vname,None,LMP_VAR_EQUAL)
        tvir[i]=tmp*ev2hartree
    #out+="\n"
    vir = np.array([tvir])
    print(np.shape(cell),np.shape(tcell))
    cell=np.append(cell,tcell,axis=0)
    coordinates = np.append(coordinates,tx,axis=0)
    energy = np.append(energy,tpe,axis=0)
    forces = np.append(forces,tf,axis=0)
    virial = np.append(virial,vir,axis=0)
    lmp.close()

SaveHDF5('data.h5',coordinates,forces,energy,species,cell,virial)
CheckHDF5('data.h5')



