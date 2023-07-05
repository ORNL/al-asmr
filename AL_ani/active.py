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
global pythonbaked


@python_app
def run_ensem(i, j, nepoch, pythonbaked):
    import sh

    def runpython(cmd, _out=None):
        if os.path.basename(pythonbaked._path) == "python":
            return pythonbaked(cmd.split(), _out=_out, _err_to_out=True)
        else:
            return pythonbaked(
                ["bash", "-c", "cd /workspace; %s" % (cmd)], _out=_out, _err_to_out=True
            )

    print("Model Training in Ac %d loop and %d model" % (i, j))
    dirname = "train%d-%d" % (i, j)
    if os.path.isdir(dirname):
        sh.rm("-rf", dirname)
    sh.cp("-r", "tmptrain", dirname)

    h = open("train%d-%d.log" % (i, j), "a")
    with sh.pushd(dirname):
        sh.ln("-snf", "../DataSplit/tot.h5")
        sh.ln("-snf", "../DataSplit/comdiv.py")
        sh.ln("-snf", "../DataSplit/div.py")
        sh.ln("-snf", "../DataSplit/Ref")

    runpython("cd %s; python comdiv.py %d" % (dirname, i + 1), _out=h)
    runpython(
        "cd %s; python nnp_training_force.py --nepoch=%d" % (dirname, nepoch), _out=h
    )


@python_app
def run_ac(i, sig, nensem, maxnum, nboost, pythonbaked):
    import sh

    def runpython(cmd, _out=None):
        if os.path.basename(pythonbaked._path) == "python":
            return pythonbaked(cmd.split(), _out=_out, _err_to_out=True)
        else:
            return pythonbaked(
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
    runpython("cd %s; python Sel.py %d" % (dirname, nensem), _out=h)

    ## SMD directory
    with sh.pushd("%s/SMD" % (dirname)):
        sh.cp("../bestbest.pt", "best0.pt")

    runpython("cd %s/SMD; python run.py" % (dirname), _out=h)

    with sh.pushd("%s/SMD" % (dirname)):
        sh.cp("vmd.xyz", "../")
        sh.cp("ff.dat", "../")
        sh.cp("ff.dat", "../../ff%d.dat" % (i))

    ## UQ
    runpython("cd %s; python xyz2h5.py" % (dirname), _out=h)
    runpython("cd %s; python UQ.py %f %d %d" % (dirname, sig, nensem, maxnum), _out=h)

    ## Rerun directory
    with sh.pushd("%s/Rerun" % (dirname)):
        sh.cp("../selected.h5", "iselected.h5")

    runpython("cd %s/Rerun; python boost.py %d" % (dirname, nboost), _out=h)
    runpython("cd %s/Rerun; python rerun.py" % (dirname), _out=h)

    with sh.pushd("%s/Rerun" % (dirname)):
        sh.cp("data.h5", "../next.h5")

    with sh.pushd(dirname):
        sh.mkdir("-p", "../DataSplit/%s" % (dirname))
        sh.cp("next.h5", "../DataSplit/%s/" % (dirname))

    runpython("cd %s; python Eval.py data %d" % (dirname, nensem), _out=h)


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
    group.add_argument("--none", action="store_const", dest="container", const="none")
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
    parser.add_argument("--debug", action="store_true", help="debug log")
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

    if args.debug:
        logging.basicConfig(level=logging.INFO)

    config = Config(executors=[ThreadPoolExecutor(max_threads=args.num_workers)])
    # config = Config(
    #     executors=[
    #         HighThroughputExecutor(
    #             provider=LocalProvider(
    #                 launcher=SimpleLauncher(debug=True)
    #             ),
    #             max_workers=1
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
        pythonbaked = sh.docker.bake(
            "run",
            "-v",
            "%s:/workspace" % (args.bind_dir),
            args.container_name,
        )
    elif args.container == "singularity":
        pythonbaked = sh.docker.bake(
            "exec",
            "--bind",
            "%s:/workspace" % (args.bind_dir),
            args.container_name,
        )
    else:
        pythonbaked = sh.python.bake()

    for i in range(restarti, nAC):
        future_list = list()
        for j in range(0, nensem):
            future = run_ensem(i, j, args.nepoch, pythonbaked)
            future_list.append(future)
        try:
            future.result()
        except Exception as e:
            print("ERROR:", e)

        future = run_ac(i, sig, nensem, maxnum, nboost, pythonbaked)
        try:
            future.result()
        except Exception as e:
            print("ERROR:", e)

        sys.exit()

        # Select Best model
        com = "cd %s; python Sel.py %d" % (dirname, nensem)
        mos.system(com, wrapcmd=wrapcmd)

        # in the SMD directory
        com = "cd %s/SMD; cp ../bestbest.pt best0.pt" % (dirname)
        mos.system(com)

        com = "cd %s/SMD; python run.py" % (dirname)
        mos.system(com, wrapcmd=wrapcmd)

        com = "cd %s/SMD; cp vmd.xyz ../" % (dirname)
        mos.system(com)

        com = "cd %s/SMD; cp ff.dat ../" % (dirname)
        mos.system(com)

        com = "cd %s/SMD; cp ff.dat ../../ff%d.dat" % (dirname, i)
        mos.system(com)

        # UQ
        com = "cd %s; python xyz2h5.py" % (dirname)
        mos.system(com, wrapcmd=wrapcmd)

        com = "cd %s; python UQ.py %f %d %d" % (dirname, sig, nensem, maxnum)
        mos.system(com, wrapcmd=wrapcmd)

        # in the Rerun directory
        com = "cd %s/Rerun; cp ../selected.h5 iselected.h5" % (dirname)
        mos.system(com)

        com = "cd %s/Rerun; boost.py %d" % (dirname, nboost)
        mos.system(com, wrapcmd=wrapcmd)

        com = "cd %s/Rerun; rerun.py %d" % (dirname, nboost)
        mos.system(com, wrapcmd=wrapcmd)

        com = "cd %s/Rerun; cp data.h5 ../next.h5" % (dirname)
        mos.system(com)

        # in the AC directory
        com = "cd %s; mkdir ../DataSplit/%s" % (dirname, dirname)
        mos.system(com)

        com = "cd %s; cp next.h5 ../DataSplit/%s/data.h5" % (dirname, dirname)
        mos.system(com)

        com = "cd %s; python Eval.py data %d" % (dirname, nensem)
        mos.system(com, wrapcmd=wrapcmd)

    sys.exit()

    for i in range(restarti, nAC):
        for j in range(0, nensem):
            mos.system("pwd")
            print("Model Training in Ac %d loop and %d model" % (i, j))
            dirname = "../train" + str(j)
            if os.path.isdir(dirname):
                com = "rm " + dirname + " -rf"
                mos.system(com)

            com = "cp ../tmptrain " + dirname + " -rf"
            mos.system(com)

            com = "python comdiv.py " + str(i + 1)
            mos.system(com)

            com = "cp *.h5 " + dirname
            mos.system(com)
            mos.chdir(dirname)
            mos.system("pwd")
            mos.system("python nnp_training_force.py --nepoch=%d" % (args.nepoch))
            com = "cp force-training-best.pt ../best" + str(j) + ".pt"
            mos.system(com)
            mos.chdir("../DataSplit")

        mos.chdir("../")
        dirname = "AC" + str(i)
        if os.path.isdir(dirname):
            com = "rm " + dirname + " -rf"
            mos.system(com)

        # Active Learning directory
        com = "cp tmpAC " + dirname + " -rf"
        mos.system(com)

        mos.chdir(dirname)
        mos.system("pwd")

        # move all trained values and train directory
        mos.system("mv ../best*.pt .")
        mos.system("mv ../train* . ")

        # Select Best model
        mos.system("python Sel.py " + str(nensem))

        # in the SMD directory
        mos.chdir("SMD")
        # mos.system("cp ../best*.pt .")
        mos.system("cp ../bestbest.pt best0.pt")

        mos.system("python run.py")
        mos.system("cp vmd.xyz ../")
        mos.system("cp ff.dat ../")
        com = "cp ff.dat ../../ff" + str(i) + ".dat"
        mos.system(com)
        mos.chdir("../")
        mos.system("python xyz2h5.py")
        mos.system(
            "python UQ.py " + str(sig) + " " + str(nensem) + " " + str(maxnum)
        )  # selected.h5
        # mos.system("python UQ.py "+str(sig)+" "+str(nensem)+" "+str(maxnum)) #selected.h5

        # in the Rerun directory
        mos.chdir("./Rerun")
        mos.system("cp ../selected.h5 iselected.h5")
        # mos.system("python boost.py")
        mos.system("python boost.py " + str(nboost))
        mos.system("python rerun.py")
        mos.system("cp data.h5 ../next.h5")

        # in the AC directory
        mos.chdir("../")
        com = "mkdir ../DataSplit/" + dirname
        mos.system(com)
        com = "cp next.h5 ../DataSplit/" + dirname + "/data.h5"
        mos.system(com)
        mos.system("python Eval.py data " + str(nensem))
        mos.chdir("../DataSplit")
