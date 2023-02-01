import os,sys,math
import h5py
import torch
import torchani
import numpy as np
import numpy.ma as ma
import random as rand
import copy
#import matplotlib.pyplot as plt

from model import Model

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error as MAE
import torch.utils.tensorboard
import tqdm

from torchani.units import hartree2kcalmol    
hartree2ev = np.float64(27.211386024367243)
ev2hartree = np.float64(1.0/hartree2ev)

# helper function to convert energy unit from Hartree to kcal/mol
#device = torch.device('cpu')
device = torch.device('cuda:0')
batch_size = 100

def ReadFF(filename):
    f=open(filename)
    L=f.readlines()
    f.close()
    pe=[]
    dist=[]
    for i in range(1,len(L)):
        tmp=L[i].split()
        pe.append(float(tmp[7]))
        dist.append(float(tmp[5]))        

    print("# of data",len(pe))
    return pe,dist

def CheckCoord(coords):
    nlist=[]    
    for i in range(0,len(coords)):
        nlist.append([])
        for j in range(0,len(coords)):
            if(i!=j):
                iatom = coords[i]
                jatom = coords[j]
            
                #dist = DistAtoms(iatom,jatom)
                #print(np.shape(iatom),np.shape(jatom))
                dist = np.linalg.norm(iatom-jatom)
                if(dist <3.0):
                    nlist[i].append(j)

    check = True
    
    for i in range(0,len(coords)):
        nenum = len(nlist[i])
        if(nenum==0):
            #print("Geo corrupted!")
            check=False
        
    return check 

def ChangeSpecies(filein,fileout):
    filename=filein+".h5"
    tcoordinates,tforces,tcell,tenergies,tspecies,tvirial,enthalpies,order,volume,aenergies = ReadHDF5MOMT(filename,0)
    natom = np.shape(tspecies)
    species = np.chararray(natom,itemsize=1)    
    for i in range(0,len(species)):
        species[i] = 'C'
    fnameout = fileout+".h5"
    SaveHDF5MOMT(fnameout,tcoordinates,tforces,tenergies,species,tcell,tvirial,enthalpies,order,volume,aenergies)
    

def ReduceData(data,mask):
    rd_data = data[mask]
    return rd_data

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

def SaveHDF5MOMT(filename,coordinates,forces,energy,species,cell,virial,enthalpies,order,volume,aenergies):
    h5f = h5py.File(filename,'w')    
    mols = h5f.create_group('mols')
    mol = mols.create_group('mol')
    mol.create_dataset('coordinates',data=coordinates)
    mol.create_dataset('forces',data=forces)
    mol.create_dataset('cell',data=cell)
    mol.create_dataset('energies',data=energy)
    mol.create_dataset('species',data=species)
    mol.create_dataset('virial',data=virial)
    mol.create_dataset('enthalpies',data=enthalpies)
    mol.create_dataset('order',data=order)
    mol.create_dataset('volume',data=volume)
    mol.create_dataset('aenergies',data=aenergies)        
    print('Saved')
    h5f.close()    

def ReadHDF5(filename,cut):
    h5fr = h5py.File(filename,'r')    
    mols =h5fr['mols']
    keys=list(mols.keys())
    mol =mols[keys[0]]        
    #mol =mols['mol']
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

def ReadHDF5MOMT(filename,cut):
    h5fr = h5py.File(filename,'r')    
    mols =h5fr['mols']
    keys=list(mols.keys())
    mol =mols[keys[0]]        
    #mol =mols['mol']
    coordinates = np.array(mol['coordinates'][cut:])
    forces = np.array(mol['forces'][cut:])
    cell = np.array(mol['cell'][cut:])
    energies = np.array(mol['energies'][cut:])
    species = np.array(mol['species'][:])
    virial = np.array(mol['virial'][cut:])
    enthalpies = np.array(mol['enthalpies'][cut:])
    order = np.array(mol['order'][cut:])
    volume = np.array(mol['volume'][cut:])
    aenergies = np.array(mol['aenergies'][cut:])
    
    print('########## Read %s ###############' %filename)
    print('######### Species %s ' %len(species))
    print('######### Data # %s ' %len(coordinates))    
    h5fr.close()    
    return coordinates,forces,cell,energies,species,virial,enthalpies,order,volume,aenergies


def CheckHDF5Mol(filehead):
    filename=filehead+".h5"
    h5fr = h5py.File(filename,'r')
    mols =h5fr['mols']
    mol =mols['mol']    
    print("Keys: %s" %mol.keys())
    print(mol['species'])
    print(mol['cell'])
    print(mol['coordinates'])
    print(mol['forces'])    
    print(mol['energies'])
    h5fr.close()       

"""
def CheckHDF5(filename):
    h5fr = h5py.File(filename,'r')
    mols =h5fr['mols']
    keys=list(mols.keys())
    print(keys)
    mol =mols[keys[0]]    
    print("Keys: %s" %mol.keys())
    print(mol['species'])
    print(mol['cell'])
    print(mol['coordinates'])
    print(mol['forces'])    
    print(mol['energies'])
    print(mol['virial'])    
    h5fr.close()            
"""

def CheckHDF5(filehead):
    filename=filehead+".h5"
    h5fr = h5py.File(filename,'r')
    mols =h5fr['mols']
    mol =mols['mol']    
    print("Keys: %s" %mol.keys())
    print(mol['species'])
    print(mol['cell'])
    print(mol['coordinates'])
    print(mol['forces'])    
    print(mol['energies'])
    print(mol['enthalpies'])    
    print(mol['order'])    
    print(mol['virial'])
    print(mol['aenergies'])        
    h5fr.close()            
    
def LoadANI(path,mname):
    sae_file = os.path.join(path, 'sae_linfit_dftb.dat')  # noqa: E501
    const_file = os.path.join(path, 'rC.params')

    consts = torchani.neurochem.Constants(const_file)
    aev_computer = torchani.AEVComputer(**consts)

    min_cell=torch.tensor([[9.0,0,0],[0,9.0,0],[0,0,9.0]],requires_grad=True,dtype=torch.float64,device=device)
    pbc = torch.tensor([1,1,1],dtype=torch.bool,device=device)
    aev_computer.setMinCell(min_cell)

    energy_shifter = torchani.neurochem.load_sae(sae_file)
    species_order = ['C']
    aev_dim = aev_computer.aev_length
    C_network = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 224),
        torch.nn.GELU(),
        torch.nn.Linear(224, 192),
        torch.nn.GELU(),
        torch.nn.Linear(192, 160),
        torch.nn.GELU(),
        torch.nn.Linear(160, 1)
    )

    nn = torchani.ANIModel([C_network])
    ptname = mname+'.pt'
    #nn.load_state_dict(torch.load('force-training-best.pt',map_location='cpu'))
    nn.load_state_dict(torch.load(ptname,map_location='cpu'))
    model = torchani.nn.Sequential(aev_computer, nn, energy_shifter).to(device)    

    return model

def Eval(filehead,mname,datanum):
    logout="Evaluation of trained model under given data\n"
    filename=filehead+".h5"
    logout+="Data file name: "+filename+"\n"    
    
    species_order = ['C']

    #Load Data
    tcoordinates,tforces,tcell,tenergies,species,tvirial = ReadHDF5(filename,0)
    tdata = torchani.data.load(filename,
    additional_properties=('forces','cell')
).species_to_indices(species_order) #.subtract_self_energies(energy_shifter, species_order).species_to_indices(species_order)
    tdata = tdata.collate(batch_size).cache()

    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0

    natoms=len(species)
    #place holder for
    error = np.empty([0]) #error
    ref_en = np.empty([0]) #energy from reference
    exp_en = np.empty([0]) #energy from prediction
    
    ref_f = np.empty(shape=[0,natoms,3]) #force from reference
    exp_f = np.empty(shape=[0,natoms,3]) #force from prediction

    #Load model
    try:
        path = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        path = os.getcwd()
    
    model = LoadANI(path,mname)
    model.train(False)

    new_coords=np.empty(shape=[0,natoms,3])
    new_cell=np.empty(shape=[0,3,3])
    new_virial=np.empty(shape=[0,6])#dummy virial
    new_en=np.empty(shape=[0])
    new_f=np.empty(shape=[0,natoms,3])
    
    #with torch.no_grad():
    pbc = torch.tensor([1,1,1],dtype=torch.bool,device=device)
    for properties in tdata:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
    
        true_energies = properties['energies'].to(device).float()
        true_forces = properties['forces'].to(device).float()
        cell = properties['cell'].to(device).float()
        _, predicted_energies = model((species, coordinates),cell,pbc)
        predicted_forces = -torch.autograd.grad(predicted_energies.sum(), coordinates)[0]#, create_graph=True, retain_graph=True)[0]    
        tvir = np.array([[0,0,0,0,0,0]])
    
        ref_en = np.append(ref_en,true_energies.cpu().detach().numpy(),axis=0)
        exp_en = np.append(exp_en,predicted_energies.cpu().detach().numpy(),axis=0)        
        ref_f = np.append(ref_f,true_forces.cpu().detach().numpy(),axis=0)
        exp_f = np.append(exp_f,predicted_forces.cpu().detach().numpy(),axis=0)    
    
        total_mse += mse_sum(predicted_energies, true_energies).item()    
        count += predicted_energies.shape[0]

    #refmse is not reliable because we did not subtract the self energy. 
    refmse = hartree2kcalmol(math.sqrt(total_mse / count))
    model.train(True)

    logout+="# of data to evaluate: "+str(count)+"\n"
    
    #################################Force unit conversion#######################################    
    ref_f = hartree2kcalmol(ref_f.ravel()[:,np.newaxis])
    exp_f = hartree2kcalmol(exp_f.ravel()[:,np.newaxis])
    
    #Total # of frames
    nframe =ref_en.shape[0]
    if(datanum < ref_en.shape[0]):
        pv = float(datanum)/ref_en.shape[0]
    else:
        pv =1.0
    maskf = np.random.choice(a=[False,True], size =(nframe*3*natoms), p=[1-pv,pv])
    mask = np.random.choice(a=[False,True], size =nframe, p=[1-pv,pv])    
    minf = np.min(ref_f)
    maxf = np.max(ref_f)

    #print(ref_f.shape)
    #print(ref_en.shape)
    
    msef = mean_squared_error(ref_f,exp_f)
    maef = MAE(ref_f,exp_f)    
    print("Min and Max values of force (kcal/molA)",minf,maxf)
    logout+="Min and Max values of force (kcal/molA): "+str(minf)+" "+str(maxf)+"\n"
    print("Mean Square Error of force (kcal/molA)^2 per component:",msef)
    logout+="Mean Square Error of force (kcal/molA)^2 per component: "+str(msef)+"\n"
    print("Root Mean Square Error of force (kcal/molA) per component:",math.sqrt(msef))
    logout+="Root Mean Square Error of force (kcal/molA) per component: "+str(math.sqrt(msef))+"\n"
    print("Mean Absolute Error of force (kcal/molA) per component:",maef)
    logout+="Mean Absolute Error of force (kcal/molA) per component: "+str(maef)+"\n"
    print("Mean Absolute Error of force (eV/A) per component:",maef*0.043)    
    logout+="Mean Absolute Error of force (eV/A) per component: "+str(maef*0.043)+"\n"    
    
    out=""
    ref_fr=ref_f[maskf]
    exp_fr=exp_f[maskf]
    for i in range(0,len(ref_fr)):
        out+=str(ref_fr[i][0])+" "+str(exp_fr[i][0])+"\n"

    ffile="force_"+filehead+".data"
    fout = open(ffile,'w')
    fout.write(out)
    fout.close()

    #Energy
    ref_en = ref_en[:,np.newaxis]
    exp_en = exp_en[:,np.newaxis]
    
    ref_mean = np.mean(ref_en)
    exp_mean = np.mean(exp_en)
    diff_mean = ref_mean-exp_mean


    #################################Energy unit conversion#######################################
    nref_en = hartree2kcalmol(ref_en)
    nexp_en = hartree2kcalmol(exp_en +diff_mean) #+ regr.intercept_
    
    nref_enr=nref_en[mask]
    nexp_enr=nexp_en[mask]
    out=""    
    for i in range(0,len(nref_enr)):
        out+=str(nref_enr[i][0]/natoms)+" "+str(nexp_enr[i][0]/natoms)+" "
        out+=str(nref_enr[i][0])+" "+str(nexp_enr[i][0])+"\n"

    efile="energy_"+filehead+".data"
    fout = open(efile,'w')
    fout.write(out)
    fout.close()
    
    mine = np.min(nref_en)
    maxe = np.max(nref_en)
    mse = mean_squared_error(nref_en,nexp_en)

    maee = MAE(nref_en,nexp_en)        
    print("Self energy: (hartree) and kcal/mol",diff_mean/natoms,hartree2kcalmol(diff_mean/natoms))
    logout+="Self energy: (hartree) and kcal/mol: "+str(diff_mean/natoms)+" "+str(hartree2kcalmol(diff_mean/natoms))+"\n"
    print("Min and Max values of energy (kcal/mol)",mine,maxe)
    logout+="Min and Max values of energy (kcal/mol): "+str(mine)+str(maxe)+"\n"
    print("Mean Square Error of energy (kcal/mol): ",mse,refmse*refmse)
    logout+="Mean Square Error of energy (kcal/mol): "+str(mse)+" "+str(refmse*refmse)+"\n"
    print("Root Mean Square Error of energy (kcal/mol):",math.sqrt(mse))
    logout+="Root Mean Square Error of energy (kcal/mol): "+str(math.sqrt(mse))+"\n"
    print("Root Mean Square Error of energy (kcal/mol) per atom:",math.sqrt(mse)/natoms)
    logout+="Root Mean Square Error of energy (kcal/mol) per atom: "+str(math.sqrt(mse)/natoms)+"\n"
    print("Mean Absolute Error of energy (kcal/mol) and meV per atom:",maee/natoms,maee/natoms*0.043*1000)
    logout+="Mean Absolute Error of energy (kcal/mol) and meV per atom: "+str(maee/natoms)+" "+str(maee/natoms*0.043*1000)+"\n"

    logfile = filehead+"_"+mname+".log"
    fout = open(logfile,'w')
    fout.write(logout)
    fout.close()    
    
def Sample(filehead,mname,datanum):
    logout="Sampling for visualization\n"
    filename=filehead+".h5"
    logout+="Data file name: "+filename+"\n"    
    
    species_order = ['C']

    #Load Data
    tcoordinates,tforces,tcell,tenergies,species,tvirial,enthalpies,order,volume,aenergies = ReadHDF5MOMT(filename,0)
    tdata = torchani.data.load(filename,
    additional_properties=('forces','cell')
).species_to_indices(species_order) #.subtract_self_energies(energy_shifter, species_order).species_to_indices(species_order)
    tdata = tdata.collate(batch_size).cache()

    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0

    natoms=len(species)

    
    #place holder for
    error = np.empty([0]) #error
    ref_en = np.empty([0]) #energy from reference
    exp_en = np.empty([0]) #energy from prediction
    
    ref_f = np.empty(shape=[0,natoms,3]) #force from reference
    exp_f = np.empty(shape=[0,natoms,3]) #force from prediction

    #Load model
    try:
        path = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        path = os.getcwd()
    
    model = LoadANI(path,mname)
    model.train(False)

    new_coords=np.empty(shape=[0,natoms,3])
    new_cell=np.empty(shape=[0,3,3])
    new_virial=np.empty(shape=[0,6])#dummy virial
    new_en=np.empty(shape=[0])
    new_f=np.empty(shape=[0,natoms,3])
    
    #with torch.no_grad():
    pbc = torch.tensor([1,1,1],dtype=torch.bool,device=device)
    for properties in tdata:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
    
        true_energies = properties['energies'].to(device).float()
        true_forces = properties['forces'].to(device).float()
        cell = properties['cell'].to(device).float()
        _, predicted_energies = model((species, coordinates),cell,pbc)
        predicted_forces = -torch.autograd.grad(predicted_energies.sum(), coordinates)[0]#, create_graph=True, retain_graph=True)[0]    
        tvir = np.array([[0,0,0,0,0,0]])
    
        ref_en = np.append(ref_en,true_energies.cpu().detach().numpy(),axis=0)
        exp_en = np.append(exp_en,predicted_energies.cpu().detach().numpy(),axis=0)        
        ref_f = np.append(ref_f,true_forces.cpu().detach().numpy(),axis=0)
        exp_f = np.append(exp_f,predicted_forces.cpu().detach().numpy(),axis=0)    
    
        total_mse += mse_sum(predicted_energies, true_energies).item()    
        count += predicted_energies.shape[0]

    #refmse is not reliable because we did not subtract the self energy. 
    refmse = hartree2kcalmol(math.sqrt(total_mse / count))
    model.train(True)

    logout+="# of data to evaluate: "+str(count)+"\n"
    
    #################################Force unit conversion#######################################    
    #Total # of frames
    nframe =ref_en.shape[0]
    if(datanum < ref_en.shape[0]):
        pv = float(datanum)/ref_en.shape[0]
    else:
        pv =1.0
    mask = np.random.choice(a=[False,True], size =nframe, p=[1-pv,pv])    



    #################################Energy unit conversion#######################################
    #Energy
    ref_en = ref_en[:,np.newaxis]
    exp_en = exp_en[:,np.newaxis]

    order=order[:,np.newaxis]
    enthalpies=enthalpies[:,np.newaxis]
    
    ref_mean = np.mean(ref_en)
    exp_mean = np.mean(exp_en)
    diff_mean = ref_mean-exp_mean

    #print("Linear fit value: ",diff_mean)

    print(np.shape(ref_en),np.shape(exp_en),np.shape(order),np.shape(enthalpies))
    
    nref_en = hartree2kcalmol(ref_en)
    nexp_en = hartree2kcalmol(exp_en +diff_mean) #+ regr.intercept_
    #nexp_en = hartree2kcalmol(exp_en) #+ regr.intercept_
    
    nref_enr=nref_en[mask]
    nexp_enr=nexp_en[mask]
    norder = order[mask]
    nenth =enthalpies[mask]

    """
    out=""    
    for i in range(0,len(nref_enr)):
        out+=str(nref_enr[i][0]/natoms)+" "+str(nexp_enr[i][0]/natoms)+" "
        out+=str(nref_enr[i][0])+" "+str(nexp_enr[i][0])+"\n"

    efile="energy_"+filehead+".data"
    fout = open(efile,'w')
    fout.write(out)
    fout.close()
    """

    out=""    
    for i in range(0,len(nref_enr)):
        out+=str(nenth[i][0])+" "+str(norder[i][0])+" "+str(abs(nexp_enr[i][0]-nref_enr[i][0]))+" "+str(abs(nexp_enr[i][0]-nref_enr[i][0])/natoms*0.043*1000)+"\n"
    efile="emap_"+filehead+".data"
    fout = open(efile,'w')
    fout.write(out)
    fout.close()

def UQ(filehead,nmodels,datanum):
    nmodels=3
    emodel = Model(nmodels)

    logout="Evaluation of trained model for UQ\n"
    filename=filehead+".h5"
    logout+="Data file name: "+filename+"\n"    
    
    species_order = ['C']

    #Load Data
    
    tcoordinates,tforces,tcell,tenergies,species,tvirial,enthalpies,order,volume,aenergies = ReadHDF5MOMT(filename,0)
    tdata = torchani.data.load(filename,
                               additional_properties=('forces','cell')
    ).species_to_indices(species_order) #.subtract_self_energies(energy_shifter, species_order).species_to_indices(species_order)
    tdata = tdata.collate(batch_size).cache()

    #place holder for
    natoms=len(species)
    error = np.empty([0]) #error
    ref_en = np.empty([0]) #energy from reference
    exp_en = np.empty([0]) #energy from prediction
    std_en = np.empty([0]) #std of ensemble from prediction
    ref_f = np.empty(shape=[0,natoms,3]) #force from reference
    exp_f = np.empty(shape=[0,natoms,3]) #force from prediction

    new_coords=np.empty(shape=[0,natoms,3])
    new_cell=np.empty(shape=[0,3,3])
    new_virial=np.empty(shape=[0,6])#dummy virial
    new_en=np.empty(shape=[0])
    new_f=np.empty(shape=[0,natoms,3])

    pbc = torch.tensor([1,1,1],dtype=torch.bool,device=device)
    mse_sum = torch.nn.MSELoss(reduction='sum')
    count=0
    total_mse=0.0
    for properties in tdata:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
    
        true_energies = properties['energies'].to(device).float()
        true_forces = properties['forces'].to(device).float()
        cell = properties['cell'].to(device).float()
        #_, predicted_energies = model((species, coordinates),cell,pbc)
        #predicted_forces = -torch.autograd.grad(predicted_energies.sum(), coordinates)[0]#, create_graph=True, retain_graph=True)[0]    
        #tvir = np.array([[0,0,0,0,0,0]])
        
        emodel.gvalues(species,coordinates,cell,pbc)
        #print(true_energies.size(),true_forces.size())
        edata = true_energies.size()
        fdata =true_forces.size()
        
        ensemble_energies = emodel.nenergy.view(3,edata[0]) #return list of energy from each model
        ensemble_forces = emodel.nforces.view(3,fdata[0],fdata[1],fdata[2])

        #print(predicted_energies.size(),predicted_forces.size())
        predicted_energies = torch.mean(ensemble_energies,dim=0)
        predicted_forces = torch.mean(ensemble_forces,dim=0)
        
        std_energies = torch.std(ensemble_energies,dim=0)    
    
        ref_en = np.append(ref_en,true_energies.cpu().detach().numpy(),axis=0)
        exp_en = np.append(exp_en,predicted_energies.cpu().detach().numpy(),axis=0)
        std_en = np.append(std_en,std_energies.cpu().detach().numpy(),axis=0)            
        ref_f = np.append(ref_f,true_forces.cpu().detach().numpy(),axis=0)
        exp_f = np.append(exp_f,predicted_forces.cpu().detach().numpy(),axis=0)    
        
        total_mse += mse_sum(predicted_energies, true_energies).item()    
        count += predicted_energies.shape[0]

        #refmse is not reliable because we did not subtract the self energy. 
        refmse = hartree2kcalmol(math.sqrt(total_mse / count))
    
    #Total # of frames// DATA SAMPLING
    nframe =ref_en.shape[0]
    datanum=1000
    if(datanum < ref_en.shape[0]):
        pv = float(datanum)/ref_en.shape[0]
    else:
        pv =1.0
    mask = np.random.choice(a=[False,True], size =nframe, p=[1-pv,pv])    
    #################################Energy unit conversion#######################################
    ref_en = ref_en[:,np.newaxis]
    exp_en = exp_en[:,np.newaxis]
    std_en = std_en[:,np.newaxis]

    order=order[:,np.newaxis]
    enthalpies=enthalpies[:,np.newaxis]

    ref_mean = np.mean(ref_en)
    exp_mean = np.mean(exp_en)
    diff_mean = ref_mean-exp_mean

    #print("Linear fit value: ",diff_mean)
    print(np.shape(ref_en),np.shape(exp_en),np.shape(order),np.shape(enthalpies))
    
    nref_en = hartree2kcalmol(ref_en)
    nexp_en = hartree2kcalmol(exp_en +diff_mean) #+ regr.intercept_
    #nexp_en = hartree2kcalmol(exp_en) #+ regr.intercept_
    
    nref_enr=nref_en[mask]
    nexp_enr=nexp_en[mask]
    norder = order[mask]
    nenth =enthalpies[mask]
    nstd_en=std_en[mask]

    out=""    
    for i in range(0,len(nref_enr)):
        out+=str(nenth[i][0])+" "+str(norder[i][0])+" "+str(abs(nexp_enr[i][0]-nref_enr[i][0]))+" "+str(abs(nexp_enr[i][0]-nref_enr[i][0])/natoms*0.043*1000)+" "+str(std_en[i][0]/natoms*0.043*1000)+"\n"
    efile="emap_"+filehead+".data"
    fout = open(efile,'w')
    fout.write(out)
    fout.close()

    out=""    
    for i in range(0,len(nref_enr)):
        out+=str(abs(nexp_enr[i][0]-nref_enr[i][0])/natoms*0.043*1000)+" "+str(std_en[i][0]/natoms*0.043*1000)+"\n"
    efile="corr_"+filehead+".data"
    fout = open(efile,'w')
    fout.write(out)
    fout.close()

    print(logout)

def UQatomic(filehead,nmodels,datanum):
    #nmodels=3
    emodel = Model(nmodels)

    logout="Evaluation of trained model for UQ\n"
    filename=filehead+".h5"
    logout+="Data file name: "+filename+"\n"    
    
    species_order = ['C']

    #Load Data
    
    tcoordinates,tforces,tcell,tenergies,tspecies,tvirial,enthalpies,order,volume,aenergies = ReadHDF5MOMT(filename,0)

    #This data should not be shuffled!
    tdata = torchani.data.load(filename,
                               additional_properties=('forces','cell')
    ).species_to_indices(species_order) #.subtract_self_energies(energy_shifter, species_order).species_to_indices(species_order)
    tdata = tdata.collate(batch_size).cache()

    #total number of frame
    nframe = len(tenergies)
    
    #place holder for
    natoms=len(tspecies)
    error = np.empty([0]) #error
    ref_en = np.empty([0]) #energy from reference
    exp_en = np.empty([0]) #energy from prediction
    std_en = np.empty([0]) #std of ensemble from prediction
    ref_f = np.empty(shape=[0,natoms,3]) #force from reference
    exp_f = np.empty(shape=[0,natoms,3]) #force from prediction
    std_aen = np.empty(shape=[0,natoms]) #atomic energy
    exp_aen=np.empty(shape=[0,natoms]) #atomic energy

    #Let's make it sure. At least, the key data would not be messed 
    ref_coords=np.empty(shape=[0,natoms,3])
    ref_cell=np.empty(shape=[0,3,3])
    pbc = torch.tensor([1,1,1],dtype=torch.bool,device=device)
    mse_sum = torch.nn.MSELoss(reduction='sum')
    count=0
    total_mse=0.0
    for properties in tdata:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
    
        true_energies = properties['energies'].to(device).float()
        true_forces = properties['forces'].to(device).float()
        cell = properties['cell'].to(device).float()
        #_, predicted_energies = model((species, coordinates),cell,pbc)
        #predicted_forces = -torch.autograd.grad(predicted_energies.sum(), coordinates)[0]#, create_graph=True, retain_graph=True)[0]    
        #tvir = np.array([[0,0,0,0,0,0]])

        #total energy
        #emodel.gvalues(species,coordinates,cell,pbc)
        emodel.atomic(species,coordinates,cell,pbc)

        #atomic energy

        edata = true_energies.size()
        fdata =true_forces.size()
        
        ensemble_energies = emodel.nenergy.view(nmodels,edata[0]) #return list of energy from each model
        ensemble_forces = emodel.nforces.view(nmodels,fdata[0],fdata[1],fdata[2])
        ensemble_aenergies = emodel.nlenergy.view(nmodels,fdata[0],fdata[1]) #return list of energy from each model
        
        #print(predicted_energies.size(),predicted_forces.size())
        predicted_aenergies = torch.mean(ensemble_aenergies,dim=0)        
        predicted_energies = torch.mean(ensemble_energies,dim=0)
        predicted_forces = torch.mean(ensemble_forces,dim=0)

        print("Atomic energy array",predicted_aenergies.size(),np.shape(aenergies),np.shape(exp_aen))
        
        std_energies = torch.std(ensemble_energies,dim=0)    
        std_aenergies = torch.std(ensemble_aenergies,dim=0)
        
        #print(std_aenergies.size())
        
        ref_en = np.append(ref_en,true_energies.cpu().detach().numpy(),axis=0)
        exp_en = np.append(exp_en,predicted_energies.cpu().detach().numpy(),axis=0)
        std_en = np.append(std_en,std_energies.cpu().detach().numpy(),axis=0)
        std_aen = np.append(std_aen,std_aenergies.cpu().detach().numpy(),axis=0)
        ref_f = np.append(ref_f,true_forces.cpu().detach().numpy(),axis=0)
        exp_f = np.append(exp_f,predicted_forces.cpu().detach().numpy(),axis=0)    
        exp_aen = np.append(exp_aen,predicted_aenergies.cpu().detach().numpy(),axis=0)

        ref_coords = np.append(ref_coords,coordinates.cpu().detach().numpy(),axis=0)
        ref_cell = np.append(ref_cell,cell.cpu().detach().numpy(),axis=0)                    
        
        total_mse += mse_sum(predicted_energies, true_energies).item()    
        count += predicted_energies.shape[0]

        #refmse is not reliable because we did not subtract the self energy. 
        refmse = hartree2kcalmol(math.sqrt(total_mse / count))
    
    #Total # of frames// DATA SAMPLING
    nframe =ref_en.shape[0]
    if(datanum < ref_en.shape[0]):
        pv = float(datanum)/ref_en.shape[0]
    else:
        pv =1.0
    mask = np.random.choice(a=[False,True], size =nframe, p=[1-pv,pv])    
    #################################Energy unit conversion#######################################
    ref_en = ref_en[:,np.newaxis]
    exp_en = exp_en[:,np.newaxis]
    std_en = std_en[:,np.newaxis]

    order=order[:,np.newaxis]
    enthalpies=enthalpies[:,np.newaxis]

    ref_mean = np.mean(ref_en)
    exp_mean = np.mean(exp_en)
    diff_mean = ref_mean-exp_mean

    #print("Linear fit value: ",diff_mean,hartree2kcalmol(diff_mean))
    #diff_mean=0.0
    print(np.shape(ref_en),np.shape(exp_en),np.shape(order),np.shape(enthalpies))
    
    nref_en = hartree2kcalmol(ref_en)
    nexp_en = hartree2kcalmol(exp_en +diff_mean) #+ regr.intercept_
    #nexp_en = hartree2kcalmol(exp_en) #+ regr.intercept_

    max_auq = np.max(std_aen,axis=1)
    min_auq = np.min(std_aen,axis=1)    
    mean_auq = np.mean(std_aen,axis=1)
    std_auq = np.std(std_aen,axis=1)        

    #std_aen is atomic uq from atomic energy
    alpha=1.0

    #Atomic values
    mean_auq =mean_auq[:,np.newaxis]
    max_auq=max_auq[:,np.newaxis]
    min_auq=min_auq[:,np.newaxis]    
    std_auq=std_auq[:,np.newaxis]    
    cri_auq = mean_auq+std_auq*5.0
    #cri_auq = min_auq+std_auq*5.0

    print("UQ matrix from atomic",np.shape(std_aen),np.shape(max_auq),np.shape(mean_auq))

    mae = abs(nref_en-nexp_en)/natoms*0.043*1000
    c = ma.masked_where(mae>10.0,mae)
    lowmask = c.mask

    d=ma.masked_where(std_aen>cri_auq,std_aen)
    newmask = d.mask
    newref=std_aen[newmask]

    sel_mask = np.any(newmask,axis=1)
    sel_en = ref_en[sel_mask]
    print("# of data selected atoms for the next iteratoin",np.shape(std_aen),np.shape(newref),np.shape(newmask))
    print("# of data selected frames for the next iteratoin",np.shape(sel_en))

    #print(np.shape(lowacc),np.shape(highacc),np.mean(lowacc),np.std(lowacc),mu,sigma)
    
    nref_enr=nref_en[mask]
    nexp_enr=nexp_en[mask]
    norder = order[mask]
    nenth =enthalpies[mask]
    nstd_en=std_en[mask]
    nmean_auq=mean_auq[mask]
    nmax_auq=max_auq[mask]
    nmin_auq=min_auq[mask]    
    ncri_auq=cri_auq[mask]    
    

    out=""
    #1 enthalpy
    #2 order
    #3 error kcal/mol
    #4 error meV/atom
    #5 UQ (meV)
    #6 Mean of UQ (at each data) (meV)
    #7 MAX of UQ (at each data) (meV)
    #8 Criteria Mean+5*STD
    #9 
    for i in range(0,len(nref_enr)):
        out+=str(nenth[i][0])+" "+str(norder[i][0])+" "+str(abs(nexp_enr[i][0]-nref_enr[i][0]))+" "+str(abs(nexp_enr[i][0]-nref_enr[i][0])/natoms*0.043*1000)+" "+str(hartree2kcalmol(std_en[i][0])/natoms*0.043*1000)+" "+str(hartree2kcalmol(nmean_auq[i][0])*0.043*1000)+" "+str(hartree2kcalmol(nmax_auq[i][0])*0.043*1000)+" "+str(hartree2kcalmol(ncri_auq[i][0])*0.043*1000)+" "+str(hartree2kcalmol(nmax_auq[i][0]-ncri_auq[i][0])*0.043*1000)+"\n"
    efile="emap_"+filehead+".data"
    fout = open(efile,'w')
    fout.write(out)
    fout.close()

    out=""    
    for i in range(0,len(nref_enr)):
        out+=str(abs(nexp_enr[i][0]-nref_enr[i][0])/natoms*0.043*1000)+" "+str(hartree2kcalmol(std_en[i][0])/natoms*0.043*1000)+" "+str(hartree2kcalmol(nmean_auq[i][0])*0.043*1000)+" "+str(hartree2kcalmol(nmax_auq[i][0])*0.043*1000)+" "+str(hartree2kcalmol(nmax_auq[i][0]-ncri_auq[i][0])*0.043*1000)+"\n"
    efile="corr_"+filehead+".data"
    fout = open(efile,'w')
    fout.write(out)
    fout.close()


    #max_auq = np.max(std_aen,axis=1)
    #mean_auq = np.mean(std_aen,axis=1)
    #std_auq = np.std(std_aen,axis=1)   
    #atomic energy

    print("Total data of atomic energy",np.shape(aenergies),np.shape(exp_aen),np.shape(aenergies.reshape(nframe,108)))
    true_aen=aenergies.reshape(108*nframe)
    exp_aen=exp_aen.reshape(108*nframe)
    std_aen=std_aen.reshape(108*nframe)
    mean_auq=mean_auq.reshape(nframe)
    std_auq=std_auq.reshape(nframe)
    print("For new critiera",np.shape(exp_aen),np.shape(std_aen),np.shape(mean_auq),np.shape(std_auq))
    
    out=""
    for i in range(0,len(exp_aen)):
        tval=hartree2kcalmol(true_aen[i])*0.043*1000
        eval=hartree2kcalmol(exp_aen[i]+diff_mean/108)*0.043*1000
        atomic_uq=hartree2kcalmol(std_aen[i])*0.043*1000
        cri = hartree2kcalmol(mean_auq[int(i/108)]+5.0*std_auq[int(i/108)])*0.043*1000
        out+=str(tval)+" "+str(eval)+" "+str(math.sqrt((eval-tval)*(eval-tval)))+" "+str(atomic_uq)+" "+str(atomic_uq-cri)+"\n"
    efile="aenergy_"+filehead+".data"
    fout = open(efile,'w')
    fout.write(out)
    fout.close()

    print(logout)

    #Let's save the selected data
    #ref_en = ref_en[:,np.newaxis]
    #exp_en = exp_en[:,np.newaxis]    
    #selref_en = ref_en[newmask]

    sel_coords = ref_coords[sel_mask][:] #from torch data load           
    sel_en = ref_en[sel_mask] # from torch data load
    sel_force = ref_f[sel_mask][:] # from torch data load       
    sel_cell = ref_cell[sel_mask][:] # from torch data load       

    sel_order = order[sel_mask] #from HDF, should not be messed
    sel_enthalpies = enthalpies[sel_mask] #from HDF, should not be messed
    sel_volume = volume[sel_mask] #from HDF, should not be messed    
    sel_aenergies = aenergies[sel_mask][:] #from HDF, should not be messed
    sel_virial = tvirial[sel_mask][:]

    sel_en=sel_en.squeeze()
    sel_enthalpies=sel_enthalpies.squeeze()     
    print(np.shape(sel_en),np.shape(sel_cell),np.shape(sel_coords),np.shape(sel_force))

    SaveHDF5MOMT('selected.h5',sel_coords,sel_force,sel_en,tspecies,sel_cell,sel_virial,sel_enthalpies,sel_order,sel_volume,sel_aenergies)  
    CheckHDF5('selected')


def Dist(coords,ia,ja):
    icoord = coords[ia]
    jcoord = coords[ja]
    dx=icoord[0]-jcoord[0]
    dy=icoord[1]-jcoord[1]
    dz=icoord[2]-jcoord[2]    

    return math.sqrt(dx*dx+dy*dy+dz*dz)
    
def UQatomicMol(filehead,nmodels,datanum):
    #nmodels=3
    emodel = Model(nmodels)

    logout="Evaluation of trained model for UQ\n"
    filename=filehead+".h5"
    logout+="Data file name: "+filename+"\n"    
    
    species_order = ['H','C']

    #Load Data
    
    tcoordinates,tforces,tcell,tenergies,tspecies,tvirial = ReadHDF5(filename,0)
    
    #This data should not be shuffled!
    tdata = torchani.data.load(filename,
                               additional_properties=('forces','cell')
    ).species_to_indices(species_order) #.subtract_self_energies(energy_shifter, species_order).species_to_indices(species_order)
    tdata = tdata.collate(batch_size).cache()

    #total number of frame
    nframe = len(tenergies)
    
    #place holder for
    natoms=len(tspecies)
    error = np.empty([0]) #error
    ref_en = np.empty([0]) #energy from reference: This values are dummy
    exp_en = np.empty([0]) #energy from prediction
    std_en = np.empty([0]) #std of ensemble from prediction
    ref_f = np.empty(shape=[0,natoms,3]) #force from reference
    exp_f = np.empty(shape=[0,natoms,3]) #force from prediction
    std_aen = np.empty(shape=[0,natoms]) #atomic energy
    exp_aen=np.empty(shape=[0,natoms]) #atomic energy

    #Let's make it sure. At least, the key data would not be messed 
    ref_coords=np.empty(shape=[0,natoms,3])
    ref_cell=np.empty(shape=[0,3,3])
    pbc = torch.tensor([1,1,1],dtype=torch.bool,device=device)
    mse_sum = torch.nn.MSELoss(reduction='sum')
    count=0
    total_mse=0.0
    for properties in tdata:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
    
        true_energies = properties['energies'].to(device).float() # dummy
        true_forces = properties['forces'].to(device).float() #dummy
        cell = properties['cell'].to(device).float()

        #total energy
        emodel.atomic(species,coordinates,cell,pbc)
        #atomic energy
        edata = true_energies.size()
        fdata =true_forces.size()
        
        ensemble_energies = emodel.nenergy.view(nmodels,edata[0]) #return list of energy from each model
        ensemble_forces = emodel.nforces.view(nmodels,fdata[0],fdata[1],fdata[2])
        ensemble_aenergies = emodel.nlenergy.view(nmodels,fdata[0],fdata[1]) #return list of energy from each model
        
        #print(predicted_energies.size(),predicted_forces.size())
        predicted_aenergies = torch.mean(ensemble_aenergies,dim=0)        
        predicted_energies = torch.mean(ensemble_energies,dim=0)
        predicted_forces = torch.mean(ensemble_forces,dim=0)

        print("Atomic energy array",predicted_aenergies.size(),np.shape(exp_aen))
        
        std_energies = torch.std(ensemble_energies,dim=0)    
        std_aenergies = torch.std(ensemble_aenergies,dim=0)

        #zeros=np.zeros(np.shape(species.cpu().detach().numpy()))
        #print(species)
        #hmask = np.ma.masked_where(species==1,std_aenergies.cpu().detach().numpy())
        
        #print(std_aenergies.size())
        
        ref_en = np.append(ref_en,true_energies.cpu().detach().numpy(),axis=0)
        exp_en = np.append(exp_en,predicted_energies.cpu().detach().numpy(),axis=0)
        std_en = np.append(std_en,std_energies.cpu().detach().numpy(),axis=0)
        std_aen = np.append(std_aen,std_aenergies.cpu().detach().numpy(),axis=0)
        ref_f = np.append(ref_f,true_forces.cpu().detach().numpy(),axis=0)
        exp_f = np.append(exp_f,predicted_forces.cpu().detach().numpy(),axis=0)    
        exp_aen = np.append(exp_aen,predicted_aenergies.cpu().detach().numpy(),axis=0)

        ref_coords = np.append(ref_coords,coordinates.cpu().detach().numpy(),axis=0)
        ref_cell = np.append(ref_cell,cell.cpu().detach().numpy(),axis=0)                    
        
        total_mse += mse_sum(predicted_energies, true_energies).item()    
        count += predicted_energies.shape[0]

        #refmse is not reliable because we did not subtract the self energy. 
        refmse = hartree2kcalmol(math.sqrt(total_mse / count))
    
    #Total # of frames// DATA SAMPLING
    nframe =ref_en.shape[0]
    if(datanum < ref_en.shape[0]):
        pv = float(datanum)/ref_en.shape[0]
    else:
        pv =1.0
    mask = np.random.choice(a=[False,True], size =nframe, p=[1-pv,pv])    
    #################################Energy unit conversion#######################################
    ref_en = ref_en[:,np.newaxis]
    exp_en = exp_en[:,np.newaxis]
    std_en = std_en[:,np.newaxis]

    ref_mean = np.mean(ref_en)
    exp_mean = np.mean(exp_en)
    diff_mean = ref_mean-exp_mean

    ref_min = np.min(ref_en)
    exp_min = np.min(exp_en)
    diff_min = ref_min-exp_min

    #print("Linear fit value: ",diff_mean,hartree2kcalmol(diff_mean))
    #diff_mean=0.0
    print(np.shape(ref_en),np.shape(exp_en))
    
    nref_en = hartree2kcalmol(ref_en)
    nexp_en = hartree2kcalmol(exp_en +diff_mean) #+ regr.intercept_

    zeros = np.zeros(np.shape(std_aen))
    scom = tspecies[np.newaxis]
    sarray= np.repeat(scom,np.shape(zeros)[0],axis=0)
    print("tspecies",sarray.dtype,np.shape(tspecies),np.shape(sarray))
    print("Characters in species",sarray[0])

    Hchar=np.fromstring('H',dtype='|S1')
    Cchar=np.fromstring('C',dtype='|S1')    
    
    hmask = np.ma.masked_where(sarray==Hchar,std_aen)
    cmask = np.ma.masked_where(sarray==Cchar,std_aen)
    min = cmask.min()
    #cmean = cmask.mean()
    cmean = np.mean(cmask,axis=1)
    hmean = np.mean(hmask,axis=1)
    cmean = cmean[:,np.newaxis]
    hmean = hmean[:,np.newaxis]    

    cmean = np.repeat(cmean[:,:],20,axis=1)
    hmean = np.repeat(hmean[:,:],20,axis=1)    
    
    #hmin = hmask.min()
    #hmean = hmask.mean()
    #hsum =hmask.sum()
    #csum =cmask.sum()

    print("Cmean and Hmean",np.shape(cmean),np.shape(hmean),cmean[0],hmean[0])

    #hshift = np.ma.filled(np.ma.masked_where(sarray==Hchar,zeros),fill_value=hmean) 
    #cshift = np.ma.filled(np.ma.masked_where(sarray==Cchar,zeros),fill_value=cmean)

    hshift = np.ma.filled(np.ma.masked_where(sarray!=Hchar,hmean),fill_value=0.0) 
    cshift = np.ma.filled(np.ma.masked_where(sarray!=Cchar,cmean),fill_value=0.0)
    print("Cmean and Hmean",np.shape(cmean),np.shape(hmean),cshift[0],hshift[0])    

    #print("min/mean/sum of atomic UQ for H and C: ",hmin,cmin,hmean,cmean,hsum,csum,np.shape(hmask)) #confirmed that mean is actually mean for each atom
    print("Atomic UQ of molecules",np.shape(std_aen),np.shape(tspecies),np.shape(zeros),np.shape(cmean),np.shape(hmean))
    #print("masked ", hmask[0],cmask[0],std_aen[0])

    #std_aen=std_aen-cmean-hmean #all atoms shifted auq
    std_aen=std_aen-cshift-hshift #all atoms shifted auq
    
    max_auq = np.max(std_aen-hshift,axis=1)
    min_auq = np.min(std_aen,axis=1)    
    mean_auq = np.mean(std_aen,axis=1)
    std_auq = np.std(std_aen,axis=1)        

    #std_aen is atomic uq from atomic energy
    alpha=1.0

    #Atomic values
    mean_auq =mean_auq[:,np.newaxis] # mean of each frame
    max_auq=max_auq[:,np.newaxis]
    min_auq=min_auq[:,np.newaxis]    
    std_auq=std_auq[:,np.newaxis]    
    cri_auq = mean_auq+std_auq*1.0
    #cri_auq = min_auq+std_auq*5.0

    print("UQ matrix from atomic",np.shape(std_aen),np.shape(max_auq),np.shape(mean_auq))

    mae = abs(nref_en-nexp_en)/natoms*0.043*1000
    c = ma.masked_where(mae>10.0,mae)
    lowmask = c.mask

    #Atomic criteria, Not useful when the system size is small.
    #d=ma.masked_where(std_aen>cri_auq,std_aen)
    d=ma.masked_where(std_aen>cri_auq,std_aen)
    newmask = d.mask
    newref=std_aen[newmask] #atom selection
    sel_mask1 = np.any(newmask,axis=1) #frame selection

    
    fmean = np.mean(mean_auq)
    fstd = np.std(mean_auq)
    print("Frame average",fmean,fstd)
    df=ma.masked_where(mean_auq>(fmean),mean_auq)    
    sel_mask = df.mask.squeeze()

    d=ma.masked_where(std_aen>(fmean+2.0*fstd),std_aen)
    newmask = d.mask
    newref=std_aen[newmask] #atom selection
    sel_mask = np.any(newmask,axis=1) #frame selection        
    
    selnum = np.shape(exp_en[sel_mask])
    print("Initial masked energy shape frame and atom",selnum,np.shape(newmask),np.shape(newref),np.shape(mean_auq),np.shape(sel_mask),np.shape(sel_mask1))
    selout=""
    torder=[]
    for i in range(0,len(exp_en)):
        tdist = Dist(ref_coords[i],9,16)
        pe = exp_en[i].item()

        if(sel_mask[i]):
            selout+=str(tdist)+" "+str(pe) +" "+ str(pe+1)+"\n"
            torder.append(tdist)            
        else:
            selout+=str(tdist)+" "+str(pe) +" "+str(pe)+"\n"        

    fsel=open('iselected.data','w')
    fsel.write(selout)
    fsel.close()
    
    #UQ screened geometries
    #Grid

    #dE = 1.0/hartree2kcalmol # Energy cut for Total # of atoms, eV
    dE = 1.0/hartree2kcalmol(1.0) # Energy cut for Total # of atoms, 1 kcal/mol
    dr = 0.01
    #pe,dist =ReadFF('ff.dat')
    #print("FF data #",len(pe),len(dist))
    cdist = Dist(ref_coords[0],9,16)
    cE = exp_en[0]
    selnum =0

    for i in range(1,len(exp_en)):
        tdist = Dist(ref_coords[i],9,16)
        xd = abs(tdist-cdist)
        Ed = abs(exp_en[i]-cE)
        
        geocheck = CheckCoord(ref_coords[i])
        if(xd > dr and Ed > dE and geocheck):
            cdist = tdist
            cE=exp_en[i]
            selnum+=1
        else:
            sel_mask[i]=False
            
    selout=""
    for i in range(0,len(exp_en)):
        tdist = Dist(ref_coords[i],9,16)
        pe = exp_en[i].item()
        if(sel_mask[i]):
            selout+=str(tdist)+" "+str(pe) +" "+ str(pe+1)+"\n"
        else:
            selout+=str(tdist)+" "+str(pe) +" "+str(pe)+"\n"        

    fsel=open('selected.data','w')
    fsel.write(selout)
    fsel.close()

    
    check_mask=sel_mask[True,:] #if there is no selected frame
    print("# of selected frame:",selnum,(~sel_mask).all(),np.shape(check_mask),np.shape(sel_mask))    
    if(~(sel_mask).all()):
        sel_mask[0]=True #if there is no selected frame add one 

    sel_en = ref_en[sel_mask]
    print("# of data selected atoms for the next iteratoin",np.shape(std_aen),np.shape(newref),np.shape(newmask))
    print("# of data selected frames for the next iteratoin",np.shape(sel_en))

    nref_enr=nref_en[mask]
    nexp_enr=nexp_en[mask]
    nstd_en=std_en[mask]
    nmean_auq=mean_auq[mask]
    nmax_auq=max_auq[mask]
    nmin_auq=min_auq[mask]    
    ncri_auq=cri_auq[mask]
    nstd_auq=std_auq[mask]        

    print("Check shape:",np.shape(std_en),np.shape(nmean_auq))
    
    out=""
    #1 index
    #2 error kcal/mol
    #3 error meV/atom
    #4 UQ of Total energy(meV)
    #5 Mean of aUQ (at each data) (meV)
    #6 Std of aUQ (at each data) (meV)    
    #7 MAX of aUQ (at each data) (meV)
    #8 Criteria Mean+2*STD
    #8 values 
    for i in range(0,len(nref_enr)):
        out+=str(i)+" "+str(abs(nexp_enr[i][0]-nref_enr[i][0]))+" "+str(abs(nexp_enr[i][0]-nref_enr[i][0])/natoms*0.043*1000)+" "+str(hartree2kcalmol(std_en[i][0])/natoms*0.043*1000)+" "+str(hartree2kcalmol(nmean_auq[i][0])*0.043*1000)+" "+str(hartree2kcalmol(nstd_auq[i][0])*0.043*1000)+" "+str(hartree2kcalmol(nmax_auq[i][0])*0.043*1000)+" "+str(hartree2kcalmol(ncri_auq[i][0])*0.043*1000)+" "+str(hartree2kcalmol(nmax_auq[i][0]-ncri_auq[i][0])*0.043*1000)+"\n"
    efile="emap_"+filehead+".data"
    fout = open(efile,'w')
    fout.write(out)
    fout.close()

    out=""    
    for i in range(0,len(nref_enr)):
        out+=str(abs(nexp_enr[i][0]-nref_enr[i][0])/natoms*0.043*1000)+" "+str(hartree2kcalmol(std_en[i][0])/natoms*0.043*1000)+" "+str(hartree2kcalmol(nmean_auq[i][0])*0.043*1000)+" "+str(hartree2kcalmol(nmax_auq[i][0])*0.043*1000)+" "+str(hartree2kcalmol(nmax_auq[i][0]-ncri_auq[i][0])*0.043*1000)+"\n"
    efile="corr_"+filehead+".data"
    fout = open(efile,'w')
    fout.write(out)
    fout.close()

    exp_aen=exp_aen.reshape(natoms*nframe)
    std_aen=std_aen.reshape(natoms*nframe)
    mean_auq=mean_auq.reshape(nframe)
    std_auq=std_auq.reshape(nframe)
    print("For new critiera",np.shape(exp_aen),np.shape(std_aen),np.shape(mean_auq),np.shape(std_auq))
    
    print(logout)

    #Let's save the selected data
    sel_coords = ref_coords[sel_mask][:] #from torch data load           
    sel_en = ref_en[sel_mask] # from torch data load
    sel_force = ref_f[sel_mask][:] # from torch data load       
    sel_cell = ref_cell[sel_mask][:] # from torch data load       
    sel_virial = tvirial[sel_mask][:] # dummy

    sel_en=sel_en.squeeze()
    #sel_enthalpies=sel_enthalpies.squeeze()     
    print(np.shape(sel_en),np.shape(sel_cell),np.shape(sel_coords),np.shape(sel_force))
    
    #SaveHDF5MOMT('selected.h5',sel_coords,sel_force,sel_en,tspecies,sel_cell,sel_virial,sel_enthalpies,sel_order,sel_volume,sel_aenergies)
    SaveHDF5('selected.h5',sel_coords,sel_force,sel_en,tspecies,sel_cell,sel_virial)  
    CheckHDF5Mol('selected')

def UQani(filehead,nmodels,sig,cutnum,datanum):
    cutnum=1000
    emodel = Model(nmodels)

    logout="Evaluation of trained model for UQ\n"
    filename=filehead+".h5"
    logout+="Data file name: "+filename+"\n"    
    
    species_order = ['H','C']

    #Load Data
    
    tcoordinates,tforces,tcell,tenergies,tspecies,tvirial = ReadHDF5(filename,0)
    
    #This data should not be shuffled!
    tdata = torchani.data.load(filename,
                               additional_properties=('forces','cell')
    ).species_to_indices(species_order) #.subtract_self_energies(energy_shifter, species_order).species_to_indices(species_order)
    tdata = tdata.collate(batch_size).cache()

    #total number of frame
    nframe = len(tenergies)
    
    #place holder for
    natoms=len(tspecies)
    error = np.empty([0]) #error
    ref_en = np.empty([0]) #energy from reference: This values are dummy
    exp_en = np.empty([0]) #energy from prediction
    std_en = np.empty([0]) #std of ensemble from prediction
    ref_f = np.empty(shape=[0,natoms,3]) #force from reference
    exp_f = np.empty(shape=[0,natoms,3]) #force from prediction


    #Let's make it sure. At least, the key data would not be messed 
    ref_coords=np.empty(shape=[0,natoms,3])
    ref_cell=np.empty(shape=[0,3,3])
    pbc = torch.tensor([1,1,1],dtype=torch.bool,device=device)
    mse_sum = torch.nn.MSELoss(reduction='sum')
    count=0
    total_mse=0.0
    for properties in tdata:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
    
        true_energies = properties['energies'].to(device).float() # dummy
        true_forces = properties['forces'].to(device).float() #dummy
        cell = properties['cell'].to(device).float()

        #total energy
        emodel.atomic(species,coordinates,cell,pbc)
        #atomic energy
        edata = true_energies.size()
        fdata =true_forces.size()
        
        ensemble_energies = emodel.nenergy.view(nmodels,edata[0]) #return list of energy from each model
        ensemble_forces = emodel.nforces.view(nmodels,fdata[0],fdata[1],fdata[2])

        predicted_energies = torch.mean(ensemble_energies,dim=0)
        predicted_forces = torch.mean(ensemble_forces,dim=0)

        print("Energy array",predicted_energies.size())
        
        std_energies = torch.std(ensemble_energies,dim=0)    

        #zeros=np.zeros(np.shape(species.cpu().detach().numpy()))
        #print(species)
        #hmask = np.ma.masked_where(species==1,std_aenergies.cpu().detach().numpy())
        
        #print(std_aenergies.size())
        
        ref_en = np.append(ref_en,true_energies.cpu().detach().numpy(),axis=0)
        exp_en = np.append(exp_en,predicted_energies.cpu().detach().numpy(),axis=0)
        std_en = np.append(std_en,std_energies.cpu().detach().numpy(),axis=0)
        ref_f = np.append(ref_f,true_forces.cpu().detach().numpy(),axis=0)
        exp_f = np.append(exp_f,predicted_forces.cpu().detach().numpy(),axis=0)    

        ref_coords = np.append(ref_coords,coordinates.cpu().detach().numpy(),axis=0)
        ref_cell = np.append(ref_cell,cell.cpu().detach().numpy(),axis=0)                    
        
        total_mse += mse_sum(predicted_energies, true_energies).item()    
        count += predicted_energies.shape[0]

        refmse = hartree2kcalmol(math.sqrt(total_mse / count))
    
    #Total # of frames// DATA SAMPLING
    nframe =ref_en.shape[0]
    if(datanum < ref_en.shape[0]):
        pv = float(datanum)/ref_en.shape[0]
    else:
        pv =1.0
    mask = np.random.choice(a=[False,True], size =nframe, p=[1-pv,pv])    
    #################################Energy unit conversion#######################################
    ref_en = ref_en[:,np.newaxis]
    exp_en = exp_en[:,np.newaxis]
    std_en = std_en[:,np.newaxis]

    ref_mean = np.mean(ref_en)
    exp_mean = np.mean(exp_en)
    diff_mean = ref_mean-exp_mean

    ref_min = np.min(ref_en)
    exp_min = np.min(exp_en)
    diff_min = ref_min-exp_min

    print(np.shape(ref_en),np.shape(exp_en))
    
    nref_en = hartree2kcalmol(ref_en)
    nexp_en = hartree2kcalmol(exp_en +diff_mean) #+ regr.intercept_

    mean_uq = np.mean(std_en[:cutnum])
    std_uq = np.std(std_en[:cutnum])    
    cri_uq = mean_uq+std_uq*sig
    
    print("Mean and STD of UQ",hartree2kcalmol(mean_uq),hartree2kcalmol(std_uq),hartree2kcalmol(cri_uq),np.shape(mean_uq),np.shape(std_uq))
    d=ma.masked_where(std_en>cri_uq,std_en)
    newmask = d.mask

    selout=""
    torder=[]
    for i in range(0,len(exp_en)):
        tdist = Dist(ref_coords[i],9,16)
        pe = exp_en[i].item()

        if(newmask[i]):
            selout+=str(tdist)+" "+str(pe*0.043) +" "+ str(pe*0.043+1)+"\n"
            torder.append(tdist)            
        else:
            selout+=str(tdist)+" "+str(pe*0.043) +" "+str(pe*0.043)+"\n"        

    fsel=open('iselected.data','w')
    fsel.write(selout)
    fsel.close()
    
    #UQ screened geometries
    #Grid

    dE = 1 # Energy cut for Total # of atoms, 1 kcal/mol
    dr = 0.01
    
    cdist = Dist(ref_coords[0],9,16)
    cE = hartree2kcalmol(exp_en[0].item())
    selnum =0

    for i in range(1,len(exp_en)):
        tdist = Dist(ref_coords[i],9,16)
        xd = abs(tdist-cdist)
        Ed = abs(hartree2kcalmol(exp_en[i])-cE)
        
        geocheck = CheckCoord(ref_coords[i])
        if(xd > dr and Ed > dE and geocheck):
            cdist = tdist
            cE=hartree2kcalmol(exp_en[i].item())
            selnum+=1
        else:
            newmask[i]=False
            
    selout=""
    
    for i in range(0,len(exp_en)):
        tdist = Dist(ref_coords[i],9,16)
        pe = hartree2kcalmol(exp_en[i].item())
        if(newmask[i]):
            selout+=str(tdist)+" "+str(pe) +" "+ str(pe+10)+"\n"
        else:
            selout+=str(tdist)+" "+str(pe) +" "+str(pe)+"\n"        

    fsel=open('selected.data','w')
    fsel.write(selout)
    fsel.close()

    new_en=exp_en[newmask] #Configuration Selection
    new_coords=ref_coords[newmask.squeeze()] #Configuration Selection
    new_forces=exp_f[newmask.squeeze()] #Configuration Selection
    new_cell=ref_cell[newmask.squeeze()] #Configuration Selection            
    print("Masked",np.shape(new_en),np.shape(new_coords),np.shape(new_forces),np.shape(new_cell))

    out=""
    #1 index
    #2 Mean of UQ upto 1000 (at each data) (kcal/mol)
    #6 Std of UQ upto 1000 (at each data) (kcal/mol)
    #7 MAX of aUQ (at each data) (meV)
    #8 Criteria Mean+2*STD
    #8 values
    
    for i in range(0,len(std_en)):
        #out+=str(i)+" "+str(std_en[i][0])*0.043*1000)+" "+str((avg_en[i][0])*0.043*1000)+"\n"
        out+=str(i)+" "+str(hartree2kcalmol(exp_en[i][0]))+" "+str(hartree2kcalmol(std_en[i][0]))+"\n"
    efile="emap_"+filehead+".data"
    fout = open(efile,'w')
    fout.write(out)
    fout.close()
    

    new_en=new_en.squeeze()
    
    #SaveHDF5MOMT('selected.h5',sel_coords,sel_force,sel_en,tspecies,sel_cell,sel_virial,sel_enthalpies,sel_order,sel_volume,sel_aenergies)
    SaveHDF5('selected.h5',new_coords[:maxnum],new_forces[:maxnum],new_en[:maxnum],tspecies,new_cell[:maxnum],tvirial[:maxnum])        
    #SaveHDF5('selected.h5',sel_coords,sel_force,sel_en,tspecies,sel_cell,sel_virial)  
    CheckHDF5Mol('selected')

sig = 3.0
navg= 1000
nensem=5
maxnum=50

if(len(sys.argv)==4):
    sig=float(sys.argv[1])
    nensem=int(sys.argv[2])
    maxnum=int(sys.argv[3])        
    
dataname = 'smd'
CheckHDF5Mol(dataname)
#UQatomicMol(dataname,5,1000)
UQani(dataname,nensem,sig,maxnum,1000)

