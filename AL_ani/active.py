import numpy as np
import os,sys,math
import copy
import h5py
from numpy import linalg as LA
import argparse

import parsl
from parsl import python_app
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.providers import LocalProvider
from parsl.launchers import WrappedLauncher
# from libsubmit.providers.local.local import Local
from parsl.executors import HighThroughputExecutor
from parsl.executors import ThreadPoolExecutor
from mos import mos

#export MKL_THREADING_LAYER=1
#python active.py --enum=5 --ACnum=10 --nboost=3 --sig=3.0 --maxnum=50 

class MyShifterSRunLauncher:
    def __init__(self):
        self.srun_launcher = SrunLauncher()

    def __call__(self, command, tasks_per_node, nodes_per_block):
        new_command="worker-wrapper {}".format(command)
        return self.srun_launcher(new_command, tasks_per_node, nodes_per_block)
    
@python_app
def run_ensem(i,j,nepoch,wrapcmd="%s"):
    from mos import mos
    mos.system("pwd")        
    print("Model Training in Ac %d loop and %d model"%(i, j))
    dirname = "train%d-%d"%(i, j)
    # if(os.path.isdir(dirname)):
    #     com="rm -rf "+dirname
    #     os.system(com)

    com = "cp -r tmptrain "+dirname
    mos.system(com)        

    com="cd %s; cp ../DataSplit/tot.h5 ."%(dirname)
    mos.system(com)

    com="cd %s; ln -snf ../DataSplit/*.py ."%(dirname)
    mos.system(com)

    com="cd %s; ln -snf ../DataSplit/Ref ."%(dirname)
    mos.system(com)

    com="cd %s; python comdiv.py %d"%(dirname, i+1)
    mos.system(com, wrapcmd=wrapcmd)

    com="cd %s; python nnp_training_force.py --nepoch=%d"%(dirname, nepoch)
    mos.system(com, wrapcmd=wrapcmd)

if __name__ == "__main__":
    #Parse
    parser = argparse.ArgumentParser(description='Active Learning with Steered Molecular Dynamics')
    parser.add_argument('--enum',type=int,default=3)
    parser.add_argument('--ACnum',type=int,default=10)
    parser.add_argument('--nboost',type=int,default=3)
    parser.add_argument('--sig',type=float,default=3.0)
    parser.add_argument('--maxnum',type=int,default=50)
    parser.add_argument('--nepoch',type=int,default=300) ## set nepoch for NNP training
    parser.add_argument('--restarti',type=int,default=0) ## set initial i (AC#)
    parser.add_argument('--wrapcmd') ## set initial i (AC#)
    #parser.add_argument('--navg',type=int,default=1000) #of data for mean and std
    args = parser.parse_args()

    nensem=args.enum
    nAC = args.ACnum
    nboost=args.nboost
    sig=args.sig
    maxnum=args.maxnum
    restarti=args.restarti

    print("Input Check...")
    print("# of ensembles: ",nensem)
    print("# of total AC run: ",nAC)
    print("# of boosting: ", nboost)
    print("Value of sigma: ", sig)
    print("# of epochs for training: ", args.nepoch)

    config = Config(
        executors=[
            ThreadPoolExecutor(
                max_threads=1
            )
        ]
    )
    parsl.load(config)

    # mos.chdir('DataSplit')
    itrain = True
    if(restarti!=0):
        itrain = False

    if(itrain==True):
        mos.system('rm AC*/data.h5', exit_on_error=False)
    
    #wrapcmd='docker run -it -v "/Users/jyc/project/activeml/AL_ani:/workspace" jychoihpc/activeml bash -c "%s"'
    wrapcmd='srun -u singularity exec --bind "/lustre/or-scratch/cades-birthright/jyc/activeml/AL_ani:/workspace" --nv /lustre/or-scratch/cades-birthright/jyc/activeml/activeml bash -c "cd /workspace; %s"'

    for i in range(restarti,nAC):
        future_list = list()
        for j in range(0,nensem):
                future = run_ensem(i, j, args.nepoch, wrapcmd=wrapcmd)
                future_list.append(future)
        try:
            future.result()
        except Exception as e:
            print('ERROR:', e)

        dirname="AC"+str(i)
        # if(os.path.isdir(dirname)):
        #     com="rm -rf "+dirname
        #     mos.system(com)

        #Active Learning directory
        com="cp -r tmpAC "+dirname
        mos.system(com)

        for j in range(0,nensem):
            com="cd %s; ln -snf ../AC%d-train%d train%d"%(dirname, i, j, j)
            mos.system(com)
            com="cd %s; ln -snf train%d/force-training-best.pt best%d.pt"%(dirname, j, j)
            mos.system(com)
        
        com="cd %s; python Sel.py %d"%(dirname, nensem)
        mos.system(com, wrapcmd=wrapcmd)

        com="cd %s/SMD; cp ../bestbest.pt best0.pt"%(dirname)
        mos.system(com)

        com="cd %s/SMD; python run.py"%(dirname)
        mos.system(com, wrapcmd=wrapcmd)

        com="cd %s; python xyz2h5.py"%(dirname)
        mos.system(com, wrapcmd=wrapcmd)

        com="cd %s; python UQ.py %f %d %d"%(dirname, sig, nensem, maxnum)
        mos.system(com, wrapcmd=wrapcmd)

        #in the Rerun directory
        com="cd %s/Rerun; cp ../selected.h5 iselected.h5"%(dirname)
        mos.system(com)

        com="cd %s/Rerun; boost.py %d"%(dirname, nboost)
        mos.system(com, wrapcmd=wrapcmd)

        com="cd %s/Rerun; rerun.py %d"%(dirname, nboost)
        mos.system(com, wrapcmd=wrapcmd)

        com="cd %s/Rerun; cp data.h5 ../next.h5"%(dirname)
        mos.system(com)

        #in the AC directory
        com="cd %s; mkdir ../DataSplit/%s"%(dirname, dirname)
        mos.system(com)

        com="cd %s; cp next.h5 ../DataSplit/%s/data.h5"%(dirname, dirname)
        mos.system(com)

        com="cd %s; python Eval.py data %d"%(dirname, nensem)
        mos.system(com, wrapcmd=wrapcmd)

    sys.exit()

    for i in range(restarti,nAC):
        for j in range(0,nensem):
            mos.system("pwd")        
            print("Model Training in Ac %d loop and %d model"%(i,j))
            dirname = "../train"+str(j)
            if(os.path.isdir(dirname)):
                com="rm "+dirname +" -rf"
                mos.system(com)

            com = "cp ../tmptrain "+dirname+" -rf"
            mos.system(com)        

            com="python comdiv.py "+str(i+1)
            mos.system(com)

            com="cp *.h5 "+dirname
            mos.system(com)
            mos.chdir(dirname)
            mos.system("pwd")        
            mos.system("python nnp_training_force.py --nepoch=%d"%(args.nepoch))
            com="cp force-training-best.pt ../best"+str(j)+".pt"
            mos.system(com) 
            mos.chdir("../DataSplit")
            
        mos.chdir("../")
        dirname="AC"+str(i)
        if(os.path.isdir(dirname)):
            com="rm "+dirname +" -rf"
            mos.system(com)

        #Active Learning directory
        com="cp tmpAC "+dirname+" -rf"
        mos.system(com)    

        mos.chdir(dirname)
        mos.system("pwd") 

        #move all trained values and train directory
        mos.system("mv ../best*.pt .")
        mos.system("mv ../train* . ")

        #Select Best model
        mos.system("python Sel.py "+str(nensem))    
        
        #in the SMD directory
        mos.chdir("SMD")
        #mos.system("cp ../best*.pt .")
        mos.system("cp ../bestbest.pt best0.pt")    


        mos.system("python run.py")
        mos.system("cp vmd.xyz ../")
        mos.system("cp ff.dat ../")
        com ="cp ff.dat ../../ff"+str(i)+".dat"
        mos.system(com)        
        mos.chdir("../")
        mos.system("python xyz2h5.py")
        mos.system("python UQ.py "+str(sig)+" "+str(nensem)+" "+str(maxnum)) #selected.h5
        #mos.system("python UQ.py "+str(sig)+" "+str(nensem)+" "+str(maxnum)) #selected.h5
        
        #in the Rerun directory    
        mos.chdir("./Rerun")
        mos.system("cp ../selected.h5 iselected.h5")
        #mos.system("python boost.py")
        mos.system("python boost.py "+str(nboost))    
        mos.system("python rerun.py")
        mos.system("cp data.h5 ../next.h5")

        #in the AC directory        
        mos.chdir("../")
        com="mkdir ../DataSplit/"+dirname
        mos.system(com)
        com ="cp next.h5 ../DataSplit/"+dirname+"/data.h5"
        mos.system(com)
        mos.system("python Eval.py data "+str(nensem))    
        mos.chdir("../DataSplit")    

