import numpy as np
import h5py
import sys

def SplitData(data,mask_tr,mask_val):
    tr_data = data[mask_tr]
    val_data = data[mask_val]
    return tr_data,val_data
    
def CombineDataConstN(flist,fname,N):
    tr_coord=np.empty(shape=[0,N,3])
    tr_cell=np.empty(shape=[0,3,3])
    tr_virial=np.empty(shape=[0,6])#dummy virial
    tr_en=np.empty(shape=[0])
    tr_f=np.empty(shape=[0,N,3])
    
    for i in range(0,len(flist)):
        dirname=flist[i]
        cut=cutlist[i]
        
        filename = flist[i] +'/data.h5'
        tcoordinates,tforces,tcell,tenergies,species,tvirial = ReadHDF5(filename,cut)

        tr_coord = np.append(tr_coord,tcoordinates,axis=0)
        tr_f = np.append(tr_f,tforces,axis=0)    
        tr_cell=np.append(tr_cell,tcell,axis=0)
        tr_en = np.append(tr_en,tenergies,axis=0)
        tr_virial = np.append(tr_virial,tvirial,axis=0)


    SaveHDF5(fname,tr_coord,tr_f,tr_en,species,tr_cell,tr_virial)    
    CheckHDF5(fname)

def HDF5(mols,fname,coordinates,forces,energy,species,cell,virial):
    #mols = h5f.create_group()
    mol = mols.create_group(fname)
    mol.create_dataset('coordinates',data=coordinates)
    mol.create_dataset('forces',data=forces)
    mol.create_dataset('cell',data=cell)
    mol.create_dataset('energies',data=energy)
    mol.create_dataset('species',data=species)
    mol.create_dataset('virial',data=virial)
    print('Saved %s' %fname)
    #h5f.close()
    
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

    #for empty values
    #dims=np.shape(coordinates)    
    #cell=np.empty(shape=[dims[0],3,3])
    #virial=np.zeros(shape=[dims[0],6])#dummy virial
    #forces=np.zeros(shape=dims)
    
    print('########## Read %s ###############' %filename)
    print('######### Species %s ' %len(species))
    print('######### Data # %s ' %len(coordinates))    
    h5fr.close()    
    return coordinates,forces,cell,energies,species,virial
    
def CheckHDF5(filename):
    h5fr = h5py.File(filename,'r')
    mols =h5fr['mols']
    keys=list(mols.keys())
    for i in range(0,len(keys)):
        key = keys[i]
        print("Saved key",key)
        mol=mols[key]
        print(mol['species'],np.array(mol['species'][:]))
        print(mol['cell'])
        print(mol['coordinates'])
        print(mol['forces'])    
        print(mol['energies'])
        print(mol['virial'])
        
#flist = ['data_0.2_5','v1_0.2_5','v2_0.2_5']
#flist = ['data_0.2_10','v1_0.2_10','v2_0.2_10']

#flist = ['data_0.1_5','v1_0.1_5','v2_0.1_5']
#flist = ['data_0.1_10','v1_0.1_10','v2_0.1_10']

#flist = ['data_0.1_10','dv1_0.1_10']
#flist = ['data_0.1_10','mix_0.1_10']

#flist = ['rnv1_0.2_10','rn_0.2_10','rnv2_0.2_10','data_0.2_10']
#flist = ['rnv1_0.2_5','rn_0.2_5','rnv2_0.2_5','vx','vy','vxy']

#flist = ['rn_0.2_5','rnv1_0.2_5','rnv2_0.2_5','vx','vy','vxy']

#flist = ['rnv1_0.1_10','rn_0.1_10','rnv2_0.1_10','vx','vy','vxy']

#flist = ['rn_0.1_10','rnv1_0.1_10','rnv2_0.1_10']

#flist = ['rnv1_0.1_5','rn_0.1_5','rnv2_0.1_5','vx','vy','vxy']

#flist = ['qdata_0.1_10','cdata_0.02_5']
#flist = ['qdata_0.4_1','cdata_0.02_5']

#flist = ['Rd10.0','Sel10.0']
#flist = ['Rd','Sel','Sel1','Sel2','Sel3','Sel4','Sel5','Sel6']
flist = ['Ref','AC0','AC1','AC2','AC3','AC4','AC5','AC6']
#flist =['NVT500K','SMD10K']

#flist = ['AC1']
#flist = ['Rd']


cutlist = [0,0,0,0,0,0,0,0,0]

for i in range(0,len(flist)):
    if(i<5):cutlist.append(0)
    else:cutlist.append(0)

trname = 'train.h5'
valname = 'validation.h5'
h5f_tr = h5py.File(trname,'w')
h5f_val = h5py.File(valname,'w')
mols_tr = h5f_tr.create_group('mols')
mols_val = h5f_val.create_group('mols')

pv = 0.2 # the portion of validation

flenth=len(flist)
if(len(sys.argv)==2):flenth=int(sys.argv[1])

for i in range(0,flenth):
    np.random.seed()
    
    dirname=flist[i]
    cut=cutlist[i]

    filename = flist[i] +'/data.h5'
    #filename=flist[i]+'.h5'
    tcoordinates,tforces,tcell,tenergies,tspecies,tvirial = ReadHDF5(filename,0)

    natom = np.shape(tspecies)
    species = np.chararray(natom)

    #print(natom)


    #for i in range(0,len(species)):
    #    species[i] = 'C'
        
    """
    for i in range(0,len(species)):
        if(i%6==0 or i%6==1 or i%6==2):
            species[i] = 'C'
        if(i%6==3 or i%6==4 or i%6==5):            
            species[i] = 'H'            
    """
    #for 
    
    dim = np.shape(tcoordinates)
    N = dim[0]

    mask_tr = np.random.choice(a=[False,True], size =(N), p=[pv,1-pv])
    mask_val = np.invert(mask_tr)
    tr_coord,val_coord = SplitData(tcoordinates,mask_tr,mask_val)
    tr_cell,val_cell = SplitData(tcell,mask_tr,mask_val)
    tr_virial,val_virial = SplitData(tvirial,mask_tr,mask_val)
    tr_en,val_en = SplitData(tenergies,mask_tr,mask_val)
    tr_f,val_f = SplitData(tforces,mask_tr,mask_val)
    
    HDF5(mols_tr,dirname,tr_coord,tr_f,tr_en,species,tr_cell,tr_virial)
    HDF5(mols_val,dirname,val_coord,val_f,val_en,species,val_cell,val_virial)    

h5f_tr.close()
h5f_val.close()

print("check training")
CheckHDF5(trname)
print("check validating")
CheckHDF5(valname)


