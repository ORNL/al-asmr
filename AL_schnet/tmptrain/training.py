
import schnetpack as spk
import os
import sys
import time

forcetut = './forcetut'
if not os.path.exists(forcetut):
    os.makedirs(forcetut)

from schnetpack.datasets import MD17

import argparse

#Parse
parser = argparse.ArgumentParser(description='Schnet Training')
parser.add_argument('--nepoch',type=int,default=300)
args = parser.parse_args()

train = MD17(os.path.join('train.db'))
val = MD17(os.path.join('validation.db'))

atoms, properties = val.get_properties(0)

#print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])
#print('Atoms:\n', atoms) 
#print('Positions: ',atoms.positions)
#print('Forces:\n', properties[MD17.forces])
#print('Shape:\n', properties[MD17.forces].shape)

train_loader = spk.AtomsLoader(train, batch_size=100, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=100) 

means, stddevs = train_loader.get_statistics(
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

import torch

# tradeoff
rho_tradeoff = 0.1

# loss function
def loss(batch, result):
    # compute the mean squared error on the energies
    diff_energy = batch[MD17.energy]-result[MD17.energy]
    err_sq_energy = torch.mean(diff_energy ** 2)
    
    # compute the mean squared error on the forces
    diff_forces = batch[MD17.forces]-result[MD17.forces]
    err_sq_forces = torch.mean(diff_forces ** 2)

    # build the combined loss function
    err_sq = rho_tradeoff*err_sq_energy + (1-rho_tradeoff)*err_sq_forces
    
    return err_sq


from torch.optim import Adam

# build optimizer
optimizer = Adam(model.parameters(), lr=5e-4)

os.system('rm ./forcetut/checkpoints -rf')
os.system('rm ./forcetut/log.csv')
os.system('rm ./forcetut/best_model')


import schnetpack.train as trn

## Hook for both CSV and screen
class MyHook(trn.LoggingHook):
    def __init__(
            self,
            log_path,
            metrics,
            log_train_loss=True,
            log_validation_loss=True,
            log_learning_rate=True,
            every_n_epochs=1,
        ):    
        log_path = os.path.join(log_path, "log.csv")
        super(MyHook, self).__init__(
            log_path, metrics, log_train_loss, log_validation_loss, log_learning_rate
        )
        self._offset = 0
        self._restart = False
        self.every_n_epochs = every_n_epochs

    def on_train_begin(self, trainer):
        if os.path.exists(self.log_path):
            remove_file = False
            with open(self.log_path, "r") as f:
                # Ensure there is one entry apart from header
                lines = f.readlines()
                if len(lines) > 1:
                    self._offset = float(lines[-1].split(",")[0]) - time.time()
                    self._restart = True
                else:
                    remove_file = True

            # Empty up to header, remove to avoid adding header twice
            if remove_file:
                os.remove(self.log_path)
        else:
            self._offset = -time.time()
            # Create the log dir if it does not exists, since write cannot
            # create a full path
            log_dir = os.path.dirname(self.log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        if not self._restart:
            log = ""
            log += "Time"

            if self.log_learning_rate:
                log += ",Learning rate"

            if self.log_train_loss:
                log += ",Train loss"

            if self.log_validation_loss:
                log += ",Validation loss"

            if len(self.metrics) > 0:
                log += ","

            for i, metric in enumerate(self.metrics):
                log += str(metric.name)
                if i < len(self.metrics) - 1:
                    log += ","

            with open(self.log_path, "a+") as f:
                f.write(log + os.linesep)

    def on_validation_end(self, trainer, val_loss):
        if trainer.epoch % self.every_n_epochs == 0:
            ctime = time.time() + self._offset
            log = str(ctime)

            if self.log_learning_rate:
                log += "," + str(trainer.optimizer.param_groups[0]["lr"])

            if self.log_train_loss:
                log += "," + str(self._train_loss / self._counter)

            if self.log_validation_loss:
                log += "," + str(val_loss)

            if len(self.metrics) > 0:
                log += ","

            for i, metric in enumerate(self.metrics):
                m = metric.aggregate()
                if hasattr(m, "__iter__"):
                    log += ",".join([str(j) for j in m])
                else:
                    log += str(m)
                if i < len(self.metrics) - 1:
                    log += ","

            with open(self.log_path, "a") as f:
                f.write(log + os.linesep)
            
            print (log)


# set up metrics
metrics = [
    spk.metrics.MeanAbsoluteError(MD17.energy),
    spk.metrics.MeanAbsoluteError(MD17.forces)
]

# construct hooks
hooks = [
    # trn.CSVHook(log_path=forcetut, metrics=metrics), 
    MyHook(log_path=forcetut, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer, 
        patience=5, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    )
]

trainer = trn.Trainer(
    model_path=forcetut,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)


# check if a GPU is available and use a CPU otherwise
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#device=torch.device('cuda:1')
# determine number of epochs and train
n_epochs = args.nepoch
trainer.train(device=device, n_epochs=n_epochs)

import numpy as np
import matplotlib.pyplot as plt
from ase.units import kcal, mol

# Load logged results
results = np.loadtxt(os.path.join(forcetut, 'log.csv'), skiprows=1, delimiter=',')

# Determine time axis
time = results[:,0]-results[0,0]

# Load the validation MAEs
energy_mae = results[:,4]
forces_mae = results[:,5]

# Get final validation errors
print('Validation MAE:')
print('    energy: {:10.3f} kcal/mol'.format(energy_mae[-1]))
print('    forces: {:10.3f} kcal/mol/\u212B'.format(forces_mae[-1]))

# Construct figure

"""
plt.figure(figsize=(14,5))

# Plot energies
plt.subplot(1,2,1)
plt.plot(time, energy_mae)
plt.title('Energy')
plt.ylabel('MAE [kcal/mol]')
plt.xlabel('Time [s]')

# Plot forces
plt.subplot(1,2,2)
plt.plot(time, forces_mae)
plt.title('Forces')
plt.ylabel('MAE [kcal/mol/\u212B]')
plt.xlabel('Time [s]')

plt.show()
"""

"""
best_model = torch.load(os.path.join(forcetut, 'best_model'))
test_loader = spk.AtomsLoader(test, batch_size=100)

energy_error = 0.0
forces_error = 0.0

for count, batch in enumerate(test_loader):    
    # move batch to GPU, if necessary
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # apply model
    pred = best_model(batch)
    
    # calculate absolute error of energies
    tmp_energy = torch.sum(torch.abs(pred[MD17.energy] - batch[MD17.energy]))
    tmp_energy = tmp_energy.detach().cpu().numpy() # detach from graph & convert to numpy
    energy_error += tmp_energy
    
    # calculate absolute error of forces, where we compute the mean over the n_atoms x 3 dimensions
    tmp_forces = torch.sum(
        torch.mean(torch.abs(pred[MD17.forces] - batch[MD17.forces]), dim=(1,2))
    )
    tmp_forces = tmp_forces.detach().cpu().numpy() # detach from graph & convert to numpy
    forces_error += tmp_forces
    
    # log progress
    percent = '{:3.2f}'.format(count/len(test_loader)*100)
    print('Progress:', percent+'%'+' '*(5-len(percent)), end="\r")

energy_error /= len(test)
forces_error /= len(test)

print('\nTest MAE:')
print('    energy: {:10.3f} kcal/mol'.format(energy_error))
print('    forces: {:10.3f} kcal/mol/\u212B'.format(forces_error))

"""
