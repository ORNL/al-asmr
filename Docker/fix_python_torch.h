/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(python/torch,FixPythonTorch)
FixStyle(python,FixPythonTorch)

#else

#ifndef LMP_FIX_PYTHON_TORCH_H
#define LMP_FIX_PYTHON_TORCH_H

#include "fix.h"

namespace LAMMPS_NS {

class FixPythonTorch : public Fix {
 public:
  FixPythonTorch(class LAMMPS *, int, char **);
  virtual ~FixPythonTorch();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void setup(int);
  void min_setup(int);
  void setup_pre_reverse(int, int);
  void initial_integrate(int);
  void pre_reverse(int, int);
  //void post_force(int);
  virtual void post_force(int);
  double compute_vector(int);
  
  void min_post_force(int);
  void final_integrate();
  void reset_dt();
  double compute_scalar();
  double memory_usage();

  void write_restart(FILE *);
  void restart(char *);

 private:
  char *id_pe;
  void * pFunc;
  void * pModel;
  
  int selected_callback;

  int eflag_caller;
  
  double const sunitconv=1.0/0.367493245336341E-01;
  double const funitconv=1.0/0.194469064593167E-01;
  double const eunitconv=1.0/0.367493245336341E-01;

  double nnani_energy;
  
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Unsupported callback name for fix python/torch

UNDOCUMENTED

E: Could not initialize embedded Python

UNDOCUMENTED

E: Could not find Python function

UNDOCUMENTED

*/
