#!/usr/bin/env python
# coding: utf-8

# # ScheNet

# In[1]:


import math
import os

import ase
import h5py
import numpy as np
from ase import Atoms
from ase.units import Hartree

import schnetpack as spk

# from ase.io import read
# from schnetpack import AtomsData
from schnetpack.data import ASEAtomsData

# from schnetpack.data import AtomsDataError
# from schnetpack.datasets import DownloadableAtomsData


# In[ ]:


hartree2ev = np.float32(27.211386024367243)
ev2hartree = np.float32(1.0 / hartree2ev)
hartree2kcalmol = np.float32(627.5094738898777)


# In[ ]:


def ReadHDF5(filename, cut):
    h5fr = h5py.File(filename, "r")
    mols = h5fr["mols"]
    mol = mols["data"]
    coordinates = np.array(mol["coordinates"][cut:])
    forces = np.array(mol["forces"][cut:])
    cell = np.array(mol["cell"][cut:])
    energies = np.array(mol["energies"][cut:])
    species = np.array(mol["species"][:])
    virial = np.array(mol["virial"][cut:])
    print("########## Read %s ###############" % filename)
    print("######### Species %s " % len(species))
    print("######### Data # %s " % len(coordinates))
    h5fr.close()
    return coordinates, forces, cell, energies, species, virial


# In[ ]:


def CheckHDF5(filename):
    h5fr = h5py.File(filename, "r")
    mols = h5fr["mols"]
    keys = list(mols.keys())
    print(keys)
    mol = mols[keys[0]]
    print("Keys: %s" % mol.keys())
    print(mol["species"], np.array(mol["species"]))
    print(mol["cell"])
    print(mol["coordinates"])
    print(mol["forces"])
    print(mol["energies"])
    print(mol["virial"])
    h5fr.close()


# In[ ]:


def H5todbkcal(filename, comp):
    # Hartree energy
    coords, tforces, cells, energies, species, virial = ReadHDF5(filename, 0)

    atoms = []
    property_list = []
    numbers = np.array(
        [6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 1, 6, 1, 1, 1], dtype=np.uint8
    )

    print(tforces[0])
    # atome=-1.3984936983
    atome = -1.3984936983
    for i in range(0, len(coords)):

        tcell = np.transpose(cells[i])
        # print(coords[i])
        # print(cells[i])
        # print(tcell)
        # mol = Atoms('C24',positions=coords[i],cell=cells[i],pbc=1)
        # mol = Atoms(comp,positions=coords[i],cell=tcell,pbc=1)
        mol = Atoms(comp, positions=coords[i], cell=tcell, pbc=1)
        atoms.append(mol)
        property_list.append(
            {
                "energy": np.array([energies[i] * hartree2kcalmol], dtype=np.float32),
                "forces": np.array(tforces[i] * hartree2kcalmol, dtype=np.float32),
            }
        )

    os.system("rm ./new_dataset.db")
    new_dataset = ASEAtomsData.create(
        "./new_dataset.db",
        distance_unit="Ang",
        property_unit_dict={"energy": "kcal/mol", "forces": "kcal/mol/Ang"},
    )
    new_dataset.add_systems(property_list, atoms)

    example = new_dataset[0]
    print("Properties of molecule with id 0:")

    for k, v in example.items():
        print("-", k, ":", v.shape, v)


# We check data:

# In[2]:


CheckHDF5("train.h5")


# ## Conversion
# We convert our data to create ".db" that can be used for SchNet

# In[7]:


H5todbkcal("train.h5", "CH2CH2CH2CH2CH3CH3")


# In[8]:


new_dataset = ASEAtomsData("new_dataset.db")


# In[9]:


print("Number of reference calculations:", len(new_dataset))
print("Available properties:")

for p in new_dataset.available_properties:
    print("-", p)
print()

example = new_dataset[0]
print("Properties of molecule with id 0:")

for k, v in example.items():
    print("-", k, ":", v.shape)


# ## Training

# In[10]:


import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.transform import ASENeighborList

forcetest = "./forcetest"
if not os.path.exists(forcetest):
    os.makedirs(forcetest)


# In[11]:

os.system('rm split.npz')
data = spk.data.AtomsDataModule(
    "new_dataset.db",
    batch_size=100,
    num_train=1000,
    num_val=1000,
    num_test=1000,
    num_workers=1,
    load_properties=["energy", "forces"],
    transforms=[
        trn.ASENeighborList(cutoff=5.0),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32(),
    ],
)

data.prepare_data()
data.setup()


# In[12]:


print("Number of reference calculations:", len(data.dataset))
print("Number of train data:", len(data.train_dataset))
print("Number of validation data:", len(data.val_dataset))
print("Number of test data:", len(data.test_dataset))
print("Available properties:")

for p in data.dataset.available_properties:
    print("-", p)


# In[13]:


properties = data.dataset[0]
print("Loaded properties:\n", *["{:s}\n".format(i) for i in properties.keys()])


# In[14]:


for batch in data.train_dataloader():
    print(batch.keys())
    break


# In[15]:


cutoff = 5.0
n_atom_basis = 30

pairwise_distance = (
    spk.atomistic.PairwiseDistances()
)  # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis,
    n_interactions=3,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff),
)


# In[16]:


pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="energy")
pred_forces = spk.atomistic.Forces(energy_key="energy", force_key="forces")


# In[17]:


nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy, pred_forces],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets("energy", add_mean=True, add_atomrefs=False),
    ],
)


# In[18]:


output_energy = spk.task.ModelOutput(
    name="energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.01,
    metrics={"MAE": torchmetrics.MeanAbsoluteError()},
)

output_forces = spk.task.ModelOutput(
    name="forces",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.99,
    metrics={"MAE": torchmetrics.MeanAbsoluteError()},
)


# In[19]:


task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy, output_forces],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4},
)


# In[20]:


logger = pl.loggers.TensorBoardLogger(save_dir=forcetest)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(forcetest, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss",
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=forcetest,
    max_epochs=1,  # for testing, we restrict the number of epochs
)
trainer.fit(task, datamodule=data)


# In[28]:


from ase import Atoms

H5todbkcal("validation.h5", "CH2CH2CH2CH2CH3CH3")
test_dataset = ASEAtomsData("new_dataset.db")

# load model
model_path = os.path.join(forcetest, "best_inference_model")
best_model = torch.load(model_path)

# set up converter
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32
)

# create atoms object from dataset
structure = test_dataset[2]
atoms = Atoms(
    numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
)

# convert atoms to SchNetPack inputs and perform prediction
inputs = converter(atoms)
results = best_model(inputs)

calculator = spk.interfaces.SpkCalculator(
    model_file=model_path,
    neighbor_list=trn.ASENeighborList(cutoff=5.0),
    energy_key="energy",
    force_key="forces",
    energy_unit="kcal/mol",
    position_unit="Ang",
)

atoms.set_calculator(calculator)

print("Prediction:")
print("energy:", atoms.get_total_energy())
print("forces:", atoms.get_forces())


# In[ ]:




