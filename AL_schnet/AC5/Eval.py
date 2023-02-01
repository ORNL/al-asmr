import os,math,sys
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

def LoadANI1(path,mname):
    sae_file = os.path.join(path, 'sae_linfit_zero.dat')  # noqa: E501
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

def LoadANI2(path,mname):
    sae_file = os.path.join(path, 'sae_linfit_zero.dat')  # noqa: E501
    const_file = os.path.join(path, 'rC.params')

    consts = torchani.neurochem.Constants(const_file)
    aev_computer = torchani.AEVComputer(**consts)

    min_cell=torch.tensor([[100.0,0,0],[0,100.0,0],[0,0,100.0]],requires_grad=True,dtype=torch.float64,device=device)
    pbc = torch.tensor([1,1,1],dtype=torch.bool,device=device)
    aev_computer.setMinCell(min_cell)

    energy_shifter = torchani.neurochem.load_sae(sae_file)
    species_order = ['H','C']
    aev_dim = aev_computer.aev_length
    H_network = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 192),
        torch.nn.GELU(),
        torch.nn.Linear(192, 160),
        torch.nn.GELU(),
        torch.nn.Linear(160, 1)
    )
    C_network = torch.nn.Sequential(
        torch.nn.Linear(aev_dim, 224),
        torch.nn.GELU(),
        torch.nn.Linear(224, 192),
        torch.nn.GELU(),
        torch.nn.Linear(192, 160),
        torch.nn.GELU(),
        torch.nn.Linear(160, 1)
    )

    nn = torchani.ANIModel([H_network,C_network])
    ptname = mname+'.pt'
    #nn.load_state_dict(torch.load('force-training-best.pt',map_location='cpu'))
    nn.load_state_dict(torch.load(ptname,map_location='cpu'))
    model = torchani.nn.Sequential(aev_computer, nn, energy_shifter).to(device)    

    return model

def EvalMol(filehead,mname,datanum):
    logout="Evaluation of trained model under given data\n"
    filename=filehead+".h5"
    logout+="Data file name: "+filename+"\n"    
    
    species_order = ['H','C']

    #Load Data
    tcoordinates,tforces,tcell,tenergies,species,tvirial = ReadHDF5(filename,0)

    Ccount=0
    Hcount=0
    Hen = -0.2385986511 #Reference from dftb calculation
    Cen = -1.3984917919 #Reference from dftb calculation
    
    for i in range(0,len(species)):
        tmpc =species[i].decode('utf-8') 
        if(tmpc=='C'):Ccount+=1
        elif(tmpc=='H'):Hcount+=1

    SelE = Hen*Hcount + Cen*Ccount

    
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
    
    model = LoadANI2(path,mname)
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
    print(Ccount,Hcount,SelE,diff_mean)
    newCen=diff_mean/SelE*Cen
    newHen=diff_mean/SelE*Hen

    print("New self energy: ",newHen,newCen)
    fname = "sae_"+filehead+"_"+mname+".dat"
    saeout = "H,0="+str(newHen)+"\n"
    saeout += "C,1="+str(newCen)+"\n"    
    fsae=open(fname,'w')
    fsae.write(saeout)
    fsae.close()
    
    
    #################################Energy unit conversion#######################################
    nref_en = hartree2kcalmol(ref_en)
    nexp_en = hartree2kcalmol(exp_en +diff_mean) #+ regr.intercept_
    bexp_en = hartree2kcalmol(exp_en) #+ regr.intercept_
    
    nref_enr=nref_en[mask]
    nexp_enr=nexp_en[mask]
    bexp_enr=bexp_en[mask]    
    
    out=""    
    for i in range(0,len(nref_enr)):
        out+=str(nref_enr[i][0]/natoms)+" "+str(nexp_enr[i][0]/natoms)+" "
        out+=str(nref_enr[i][0]*0.043)+" "+str(nexp_enr[i][0]*0.043)+" "        
        out+=str(nref_enr[i][0]*0.043)+" "+str(bexp_enr[i][0]*0.043)+"\n"

    efile="energy_"+filehead+".data"
    fout = open(efile,'w')
    fout.write(out)
    fout.close()
    
    mine = np.min(nref_en)
    maxe = np.max(nref_en)
    mse = mean_squared_error(nref_en,nexp_en)

    maee = MAE(nref_en,nexp_en)
    #self energy for single type atom
    #print("Self energy: (hartree) and kcal/mol",diff_mean/natoms,hartree2kcalmol(diff_mean/natoms))
    #logout+="Self energy: (hartree) and kcal/mol: "+str(diff_mean/natoms)+" "+str(hartree2kcalmol(diff_mean/natoms))+"\n"
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

    #return maee/natoms*0.043*1000,maef*0.043 #for meV/atom
    #return maee/natoms,maef # for kcal/mol/atom
    #return maee*0.043,maef*0.043 # for kcal/mol/atom
    return maee,maef # for kcal/mol

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

    print("Linear fit value: ",diff_mean)

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

    efile="energy_"+filehead+".dat"
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

    print("Linear fit value: ",diff_mean)
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

    print("Linear fit value: ",diff_mean,hartree2kcalmol(diff_mean))
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
    cri_auq = mean_auq*alpha+std_auq*5.0
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

    print("Total data of atomic energy",np.shape(aenergies),np.shape(exp_aen),np.shape(aenergies.reshape(nframe,natoms)))
    true_aen=aenergies.reshape(natoms*nframe)
    exp_aen=exp_aen.reshape(natoms*nframe)
    std_aen=std_aen.reshape(natoms*nframe)
    mean_auq=mean_auq.reshape(nframe)
    std_auq=std_auq.reshape(nframe)
    print("For new critiera",np.shape(exp_aen),np.shape(std_aen),np.shape(mean_auq),np.shape(std_auq))
    
    out=""
    for i in range(0,len(exp_aen)):
        tval=hartree2kcalmol(true_aen[i])*0.043*1000
        eval=hartree2kcalmol(exp_aen[i]+diff_mean/natoms)*0.043*1000
        atomic_uq=hartree2kcalmol(std_aen[i])*0.043*1000
        cri = hartree2kcalmol(mean_auq[int(i/natoms)]+5.0*std_auq[int(i/natoms)])*0.043*1000
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
    
#mname='momt_10.1_1'
dataname = 'data'
CheckHDF5Mol(dataname)

if(len(sys.argv)==2):dataname=str(sys.argv[1])

#UQ(dataname,3,1000)
#UQatomic(dataname,5,1000)
filename ='accuracy_'+dataname+".data"


f=open(filename,'w')
out="\n"
for i in range(0,5):
    mname = 'best'+str(i)
    maee,maef=EvalMol(dataname,mname,1000)
    out+=str(i)+" "+str(maee)+" "+str(maef)+"\n"
f.write(out)
f.close()

