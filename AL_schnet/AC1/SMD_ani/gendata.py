#This script for LAMMPS using python interface to generate coordinates, forces, energy from DFTB+
#jungg@ornl.gov
from lammps import lammps, LAMMPS_INT, LMP_STYLE_GLOBAL, LMP_VAR_EQUAL, LMP_VAR_ATOM
import numpy as np
import os,sys,math,torch
import copy
import h5py
from numpy import linalg as LA
import torch
import torchani

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
    print(mol['species'])
    print(mol['cell'])
    print(mol['coordinates'])
    print(mol['forces'])    
    print(mol['energies'])
    print(mol['virial'])    
    h5fr.close()            


os.system("rm geo*.xyz ss*.dat sd.dat")
lmp=lammps()
lmp.file("smd.in")

#tot_step = 240  #Let's make 10 points
tot_step = 50  #Let's make 10 points


#tot_step = 2  #Let's make 10 points
t_step = 1

nlocal = lmp.extract_global("nlocal", 0)
#You cannot assign the array with the same name used in torch ANI
aspecies = np.chararray(nlocal)

atype = copy.deepcopy(lmp.numpy.extract_atom_iarray("type",nlocal))


for i in range(0,len(atype)):
    if(atype[i]==1):
        aspecies[i] = 'H'
    if(atype[i]==6):
        aspecies[i] = 'C'

#
tmpcoordinates = copy.deepcopy(lmp.numpy.extract_atom_darray("x", nlocal, dim=3))
tmpforces = copy.deepcopy(lmp.numpy.extract_atom_darray("f", nlocal, dim=3))
box = copy.deepcopy(lmp.extract_box())
boxlo = box[0]
boxhi = box[1]
xy = box[2]
yz = box[3]
xz = box[4]


#Save Total Atoms
dcoordinates=np.empty(shape=[0,nlocal,3])
dcell=np.empty(shape=[0,3,3])
dvirial=np.empty(shape=[0,6])
denergy=np.empty(shape=[0])
dforces=np.empty(shape=[0,nlocal,3])


#Save Local Atoms
"""
dcoordinates=np.empty(shape=[0,36,3])
dcell=np.empty(shape=[0,3,3])
dvirial=np.empty(shape=[0,6])
denergy=np.empty(shape=[0])
dforces=np.empty(shape=[0,36,3])
"""
#Generating data
#Uqc = 0.30

Uqc = 0.022498292627798327+0.022826418399720882*3

outuq=""


for i in range(0,tot_step):
    #box, coords, forces, energy
    lmp.command("run 200") #20000/10 = 2000
    #lmp.command("run 10") #20000/10 = 2000
    nlocal = lmp.extract_global("nlocal",0)
    box = lmp.extract_box()
    boxlo = box[0]
    boxhi = box[1]
    xy = box[2]
    yz = box[3]
    xz = box[4]    
    tcell = np.array([[[boxhi[0]-boxlo[0],0,0],[xy,boxhi[1]-boxlo[1],0],[xz,yz,boxhi[2]-boxlo[2]]]],dtype=np.float64) # check version2   may be this is right    

    tx = np.array([lmp.numpy.extract_atom_darray("x",nlocal,dim=3)])
    tf = np.array([lmp.numpy.extract_atom_darray("f",nlocal,dim=3)*ev2hartree])
    tpe = np.array([lmp.get_thermo("pe")*ev2hartree])
    
    tuq = lmp.numpy.extract_atom_darray("q",nlocal,dim=1)    
    #ind = np.argmax(tuq[12:])
    ind = np.argmax(tuq)
    vuq = tuq[ind]

    print("UQ max id and value: ", ind, tuq[ind], Uqc)
    
    #outuq+=str(ind)+" "+str(tuq[ind])+" "+np.array2string(tuq,precision=3)+"\n"
    outuq+=str(ind)+" "+str(tuq[ind].item())+" "+str(np.mean(tuq)) +" "+str(np.std(tuq))+ "\n"
    
    if(vuq > Uqc):
        print("New coordinates are restored!")
        unit = int(ind/6)
        if(unit>19):unit=19
        elif(unit<3):unit=3

        #save local atoms
        istart = (unit-3)*6
        iend = (unit+3)*6

        #save total atoms
        istart = 0
        iend = nlocal
        
        tvir = np.array([np.zeros(6)])

        indx = np.arange(istart,iend).reshape(1,iend-istart,1)
        sx=np.take_along_axis(tx,indx,1)
        sf=np.take_along_axis(tf,indx,1)

        print(aspecies[istart:iend])
        print(np.shape(sx),np.shape(sf))
        
        dcoordinates = np.append(dcoordinates,sx,axis=0)
        dforces = np.append(dforces,sf,axis=0)

        denergy = np.append(denergy,tpe,axis=0)
        dcell=np.append(dcell,tcell,axis=0)
        dvirial = np.append(dvirial,tvir,axis=0)
        
        
    #Without UQ
    #tpe = np.array([lmp.get_thermo("pe")*ev2hartree])
    #tvir = np.array([np.zeros(6)])
    #dcell=np.append(dcell,tcell,axis=0)
    #dcoordinates = np.append(dcoordinates,tx,axis=0)
    #denergy = np.append(denergy,tpe,axis=0)
    #dforces = np.append(dforces,tf,axis=0)
    #dvirial = np.append(dvirial,tvir,axis=0)

f=open('uqhist.dat','w')
f.write(outuq)
f.close()
    
outfile = "data.h5"

#sdcoordinates=dcoordinates[:][istart:iend]
#print(np.shape(dcoordinates))
#sdcoordinates=dcoordinates[:][istart:iend][:]

#indx = np.arange(istart,iend).reshape(1,iend-istart,1)
#print(indx)
#sdcoordinates=np.take_along_axis(dcoordinates,indx,1)
#sdforces=np.take_along_axis(dforces,indx,1)
saspecies=aspecies[0:36]

#SaveHDF5(outfile,sdcoordinates,sdforces,denergy,saspecies,dcell,dvirial)
SaveHDF5(outfile,dcoordinates,dforces,denergy,saspecies,dcell,dvirial)
CheckHDF5(outfile)

lmp.close()

