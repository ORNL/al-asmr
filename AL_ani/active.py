import numpy as np
import os,sys,math
import copy
import h5py
from numpy import linalg as LA
import argparse

## Replacement of os.chdir and os.system
class mos:
    def print(*args, sep=" "):
        print ("\u001b[31m", sep.join(map(str, args)), "\u001b[0m")

    def chdir(cmd):
        os.chdir(cmd)
        mos.print (">>> chdir:", cmd, "(cwd: %s)"%(os.getcwd()))

    def system(cmd, exit_on_error=True):
        mos.print (">>>", cmd, "(cwd: %s)"%(os.getcwd()))
        ret = os.system(cmd)

        if exit_on_error and ret > 0:
            mos.print (">>> exec error:", ret)
            sys.exit(ret)

#export MKL_THREADING_LAYER=1
#python active.py --enum=5 --ACnum=10 --nboost=3 --sig=3.0 --maxnum=50 

#Parse
parser = argparse.ArgumentParser(description='Active Learning with Steered Molecular Dynamics')
parser.add_argument('--enum',type=int,default=3)
parser.add_argument('--ACnum',type=int,default=10)
parser.add_argument('--nboost',type=int,default=3)
parser.add_argument('--sig',type=float,default=3.0)
parser.add_argument('--maxnum',type=int,default=50)
parser.add_argument('--nepoch',type=int,default=300) ## set nepoch for NNP training
parser.add_argument('--restarti',type=int,default=0) ## set initial i (AC#)


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

mos.chdir('DataSplit')
itrain = True
if(restarti!=0):
    itrain = False

if(itrain==True):
    mos.system('rm AC*/data.h5', exit_on_error=False)


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

