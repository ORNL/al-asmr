import numpy as np
import os, sys, math
import copy
import h5py
from numpy import linalg as LA
import argparse
import logging

import parsl
from parsl import python_app
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parsl.launchers import WrappedLauncher

# from libsubmit.providers.local.local import Local
from parsl.executors import HighThroughputExecutor
from parsl.executors import ThreadPoolExecutor
from mos import mos

import glob
import sh

# export MKL_THREADING_LAYER=1
# python active.py --enum=5 --ACnum=10 --nboost=3 --sig=3.0 --maxnum=50
global cmdbaked


@python_app
def run_ensem(i, j, nepoch, cmdbaked):
    import sh
    import random
    import time

    def runcmd(cmd, _out=None):
        print("cmd:", cmd, file=_out, flush=True)
        execname = os.path.basename(cmdbaked._path)
        if (execname == b"python") or (execname == "python"):
            return cmdbaked(cmd.split(), _out=_out, _err_to_out=True)
        elif (
            (execname == b"docker")
            or (execname == b"singularity")
            or (execname == "docker")
            or (execname == "singularity")
        ):
            return cmdbaked(
                ["bash", "-c", "cd /workspace; %s" % (cmd)], _out=_out, _err_to_out=True
            )
        else:
            return cmdbaked(cmd, _out=_out, _err_to_out=True)

    h = open("train%d-%d.log" % (i, j), "a")
    print("Model Training in AC %d loop and %d model" % (i, j), file=h, flush=True)
    dirname = "train%d-%d" % (i, j)
    if os.path.isdir(dirname):
        sh.rm("-rf", dirname)
    sh.cp("-r", "tmptrain", dirname)

    with sh.pushd(dirname):
        sh.ln("-snf", "../DataSplit/tot.h5")
        sh.ln("-snf", "../DataSplit/comdiv.py")
        sh.ln("-snf", "../DataSplit/div.py")
        sh.ln("-snf", "../DataSplit/Ref")

    ## Frequent BlockingIOError with parallel running. Try to rerun.
    while True:
        try:
            runcmd("cd %s; python comdiv.py %d" % (dirname, i + 1), _out=h)
            break
        except:
            t = random.random() * 10
            print("error on comdiv.py. Sleep", t, file=h, flush=True)
            time.sleep(t)
            continue

    runcmd(
        "cd %s; python nnp_training_force.py --nepoch=%d" % (dirname, nepoch), _out=h
    )


@python_app
def run_ac(i, sig, nensem, maxnum, nboost, cmdbaked):
    import sh

    def runcmd(cmd, _out=None):
        print("cmd:", cmd, file=_out, flush=True)
        if os.path.basename(cmdbaked._path) == "python":
            return cmdbaked(cmd.split(), _out=_out, _err_to_out=True)
        else:
            return cmdbaked(
                ["bash", "-c", "cd /workspace; %s" % (cmd)], _out=_out, _err_to_out=True
            )

    print("Running AC %d" % (i))
    dirname = "AC%d" % (i)

    if os.path.isdir(dirname):
        sh.rm("-rf", dirname)

    # Active Learning directory
    sh.cp("-r", "tmpAC", dirname)

    with sh.pushd(dirname):
        for j in range(0, nensem):
            sh.ln("-snf", "../train%d-%d" % (i, j), "train%d" % (j))
            sh.ln("-snf", "train%d/force-training-best.pt" % (j), "best%d.pt" % (j))

    ## Select Best model
    h = open("AC%d.log" % (i), "a")
    runcmd("cd %s; python Sel.py %d" % (dirname, nensem), _out=h)

    ## SMD directory
    with sh.pushd("%s/SMD" % (dirname)):
        sh.cp("../bestbest.pt", "best0.pt")

    runcmd("cd %s/SMD; python run.py" % (dirname), _out=h)

    with sh.pushd("%s/SMD" % (dirname)):
        sh.cp("vmd.xyz", "../")
        sh.cp("ff.dat", "../")
        sh.cp("ff.dat", "../../ff%d.dat" % (i))

    ## UQ
    runcmd("cd %s; python xyz2h5.py" % (dirname), _out=h)
    runcmd("cd %s; python UQ.py %f %d %d" % (dirname, sig, nensem, maxnum), _out=h)

    ## Rerun directory
    with sh.pushd("%s/Rerun" % (dirname)):
        sh.cp("../selected.h5", "iselected.h5")

    runcmd("cd %s/Rerun; python boost.py %d" % (dirname, nboost), _out=h)
    runcmd("cd %s/Rerun; python rerun.py" % (dirname), _out=h)

    with sh.pushd("%s/Rerun" % (dirname)):
        sh.cp("data.h5", "../next.h5")

    with sh.pushd(dirname):
        sh.mkdir("-p", "../DataSplit/%s" % (dirname))
        sh.cp("next.h5", "../DataSplit/%s/" % (dirname))

    runcmd("cd %s; python Eval.py data %d" % (dirname, nensem), _out=h)


if __name__ == "__main__":
    # Parse
    parser = argparse.ArgumentParser(
        description="Active Learning with Steered Molecular Dynamics"
    )
    parser.add_argument("--enum", type=int, default=3)
    parser.add_argument("--ACnum", type=int, default=10)
    parser.add_argument("--nboost", type=int, default=3)
    parser.add_argument("--sig", type=float, default=3.0)
    parser.add_argument("--maxnum", type=int, default=50)
    parser.add_argument(
        "--nepoch", type=int, default=300
    )  ## set nepoch for NNP training
    parser.add_argument("--restarti", type=int, default=0)  ## set initial i (AC#)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--docker", action="store_const", dest="container", const="docker"
    )
    group.add_argument(
        "--singularity", action="store_const", dest="container", const="singularity"
    )
    group.add_argument(
        "--nocontainer", action="store_const", dest="container", const="none"
    )
    parser.set_defaults(container="docker")
    parser.add_argument("--container_name", default="activeml", help="container name")
    parser.add_argument(
        "--bind_dir",
        default=os.path.dirname(__file__),
        help="directory location to bind",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="number of parallel workers"
    )
    parser.add_argument("--srun", action="store_true", help="use srun")
    args = parser.parse_args()

    nensem = args.enum
    nAC = args.ACnum
    nboost = args.nboost
    sig = args.sig
    maxnum = args.maxnum
    restarti = args.restarti

    print("Input Check...")
    print("# of ensembles: ", nensem)
    print("# of total AC run: ", nAC)
    print("# of boosting: ", nboost)
    print("Value of sigma: ", sig)
    print("# of epochs for training: ", args.nepoch)

    logging.basicConfig(level=logging.INFO)

    config = Config(executors=[ThreadPoolExecutor(max_threads=args.num_workers)])
    # config = Config(
    #     executors=[
    #         HighThroughputExecutor(
    #             provider=LocalProvider(launcher=SimpleLauncher(debug=True)),
    #             max_workers=args.num_workers,
    #         )
    #     ]
    # )
    parsl.load(config)

    # mos.chdir('DataSplit')
    itrain = True
    if restarti != 0:
        itrain = False

    if itrain == True:
        try:
            sh.rm(glob.glob("AC*/data.h5"))
        except:
            pass

    # wrapcmd = 'docker run -it -v "/Users/jyc/project/activeml/AL_ani:/workspace" jychoihpc/activeml bash -c "%s"'
    # wrapcmd='srun -u singularity exec --bind "/lustre/or-scratch/cades-birthright/jyc/activeml/AL_ani:/workspace" --nv /lustre/or-scratch/cades-birthright/jyc/activeml/activeml bash -c "cd /workspace; %s"'

    if args.container == "docker":
        cmdbaked = sh.docker.bake(
            "run",
            "-v",
            "%s:/workspace" % (args.bind_dir),
            args.container_name,
        )
    elif args.container == "singularity":
        if args.srun:
            cmdbaked = sh.srun.bake(
                "--ntasks=1",
                "--gres=gpu:1",
                "--overlap",
                "singularity",
                "exec",
                "--bind",
                "%s:/workspace" % (args.bind_dir),
                "--nv",
                args.container_name,
            )
        else:
            cmdbaked = sh.singularity.bake(
                "exec",
                "--bind",
                "%s:/workspace" % (args.bind_dir),
                "--nv",
                args.container_name,
            )
    else:
        cmdbaked = sh.bash.bake("-c")

    for i in range(restarti, nAC):
        future_list = list()
        for j in range(0, nensem):
            future = run_ensem(i, j, args.nepoch, cmdbaked)
            future_list.append(future)

        for j in range(0, nensem):
            try:
                future_list[j].result()
            except Exception as e:
                print("ERROR:", j, e)
                sys.exit()

        future = run_ac(i, sig, nensem, maxnum, nboost, cmdbaked)
        try:
            future.result()
        except Exception as e:
            print("ERROR:", e)

        sys.exit()
