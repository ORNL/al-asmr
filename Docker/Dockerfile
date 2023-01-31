# syntax = docker/dockerfile:1.4.0

FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

COPY schnet.py schnet.py
COPY train.h5 train.h5
COPY validation.h5 validation.h5
COPY new_dataset.db new_dataset.db

RUN apt update && \
    apt install -y git && apt install -y build-essential

## TorchANI
RUN git clone https://github.com/aiqm/torchani.git
RUN cd torchani && pip install .

## SchNet
RUN git clone https://github.com/atomistic-machine-learning/schnetpack.git schnetpack
RUN cd schnetpack && pip install -r requirements.txt
RUN cd schnetpack && pip install .

## NequIP
RUN git clone https://github.com/mir-group/nequip.git
RUN cd nequip && sed -i 's/torch>=1.8,<=1.12/torch>=1.8,<=1.13/' setup.py 
RUN cd nequip && pip install .

## mpi and mpi4py
RUN apt install -y mpich 
RUN git clone -b 3.0.3 https://github.com/mpi4py/mpi4py.git
RUN cd mpi4py && CC=mpicc CXX=mpicxx pip install .

## LAMMPS
RUN git clone -b stable_29Oct2020 https://github.com/lammps/lammps.git
RUN cd lammps && sed -i 's/python_SYSINC = .*/python_SYSINC = $(shell python3-config --includes)/' lib/python/Makefile.lammps
RUN cd lammps && sed -i 's/python_SYSLIB = .*/python_SYSLIB = $(shell python3-config --ldflags) -lpython3.9/' lib/python/Makefile.lammps
RUN cd lammps && sed -i 's/PYTHON=.*/PYTHON = python3/' lib/python/Makefile.lammps
COPY fix_python_torch.cpp fix_python_torch.h lammps/src
RUN cd lammps/src && make yes-python
RUN cd lammps/src && make mode=shared mpi && make install-python

## DFTB+
#/opt/conda/lib/libmkl_gf_lp64.so;

RUN apt install wget
RUN git clone -b 21.2 https://github.com/dftbplus/dftbplus.git
RUN mkdir dftbplus/_build
RUN dftbplus/utils/get_opt_externals
RUN cd dftbplus/_build && CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=OFF -DWITH_API=OFF -DWITH_PYTHON=OFF .. && make install
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/conda/lib:$LD_LIBRARY_PATH
RUN pip install tensorboard
