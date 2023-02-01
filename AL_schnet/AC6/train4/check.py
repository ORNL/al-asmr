from ase.db import connect
import torch
import schnetpack as spk
from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model
import os,math

forcetut = './forcetut'
if not os.path.exists(forcetut):
    os.makedirs(forcetut)

from schnetpack.datasets import MD17

val = MD17(os.path.join('validation.db'))
val_loader = spk.AtomsLoader(val, batch_size=100) 

means, stddevs = val_loader.get_statistics(
    spk.datasets.MD17.energy, divide_by_atoms=True
)

print('Mean atomization energy / atom:      {:12.4f} [kcal/mol]'.format(means[MD17.energy][0]))
print('Std. dev. atomization energy / atom: {:12.4f} [kcal/mol]'.format(stddevs[MD17.energy][0]))

n_features = 128

schnet = spk.representation.SchNet(
    n_atom_basis=n_features,
    n_filters=n_features,
    n_gaussians=25,
    n_interactions=3,
    cutoff=5.0,
    cutoff_network=spk.nn.cutoff.CosineCutoff
)

energy_model = spk.atomistic.Atomwise(
    n_in=n_features,
    property=MD17.energy,
    mean=means[MD17.energy],
    stddev=stddevs[MD17.energy],
    derivative=MD17.forces,
    negative_dr=True
)

model = spk.AtomisticModel(representation=schnet, output_modules=energy_model)


# tradeoff
rho_tradeoff = 0.1


# check if a GPU is available and use a CPU otherwise
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Construct figure
#model = torch.load(os.path.join(forcetut, 'best_model'))
model = load_model('best_model')
conn = connect('validation.db')
ats = conn.get_atoms(1)

calc = SpkCalculator(model, device="cpu", energy="energy", forces="forces")
ats.set_calculator(calc)

print(ats)

print("forces:", ats.get_forces())
print("total_energy", ats.get_total_energy())

energy_error = 0.0
forces_error = 0.0



