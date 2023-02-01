import numpy as np
import os,sys,math
import copy
import h5py
from numpy import linalg as LA
import argparse
#export MKL_THREADING_LAYER=1
#python active.py --enum=5 --ACnum=10 --nboost=3 --sig=3.0 --maxnum=50 

#Parse
parser = argparse.ArgumentParser(description='Active Learning with Steered Molecular Dynamics')
parser.add_argument('--enum',type=int,default=5)
parser.add_argument('--ACnum',type=int,default=10)
parser.add_argument('--nboost',type=int,default=1)
parser.add_argument('--sig',type=float,default=3.0)
parser.add_argument('--maxnum',type=int,default=50)
#parser.add_argument('--navg',type=int,default=1000) #of data for mean and std

args = parser.parse_args()

nensem=args.enum
nAC = args.ACnum
nboost=args.nboost
sig=args.sig
maxnum=args.maxnum

print("Input Check...")
print("# of ensembles: ",nensem)
print("# of total AC run: ",nAC)
print("# of boosting: ", nboost)
print("Value of sigma: ", sig)


os.chdir('DataSplit')
itrain = True
os.system('rm AC*/data.h5')
for i in range(0,nAC):
    #Training
    for j in range(0,nensem):
        os.system("pwd")        
        print("Model Training in Ac %d loop and %d model"%(i,j))
        dirname = "../train"+str(j)
        if(os.path.isdir(dirname)):
            com="rm "+dirname +" -rf"
            os.system(com)

        com = "cp ../tmptrain "+dirname+" -rf"
        os.system(com)        

        com="python div.py "+str(i+1)
        os.system(com)

        com="cp *.h5 "+dirname
        os.system(com)
        os.chdir(dirname)
        os.system("pwd")        
        os.system("python nnp_training_force.py")
        com="cp force-training-best.pt ../best"+str(j)+".pt"
        os.system(com) 
        os.chdir("../DataSplit")
        
    os.chdir("../")
    dirname="AC"+str(i)
    if(os.path.isdir(dirname)):
        com="rm "+dirname +" -rf"
        os.system(com)

    #Active Learning directory
    com="cp tmpAC "+dirname+" -rf"
    os.system(com)    

    os.chdir(dirname)
    os.system("pwd") 

    #move all trained values and train directory
    os.system("mv ../best*.pt .")
    os.system("mv ../train* . ")

    #in the SMD directory
    os.chdir("SMD")
    os.system("cp ../best*.pt .")
    os.system("python run.py")
    os.system("cp vmd.xyz ../")
    os.system("cp ff.dat ../")
    com ="cp ff.dat ../../ff"+str(i)+".dat"
    os.system(com)        
    os.chdir("../")
    os.system("python xyz2h5.py")
    os.system("python UQ.py "+str(sig)+" "+str(nensem)+" "+str(maxnum)) #selected.h5
    #os.system("python UQ.py "+str(sig)+" "+str(nensem)+" "+str(maxnum)) #selected.h5
    
    #in the Rerun directory    
    os.chdir("./Rerun")
    os.system("cp ../selected.h5 iselected.h5")
    #os.system("python boost.py")
    os.system("python boost.py "+str(nboost))    
    os.system("python rerun.py")
    os.system("cp data.h5 ../next.h5")

    #in the AC directory        
    os.chdir("../")
    com ="cp next.h5 ../DataSplit/"+dirname+"/data.h5"
    os.system(com)
    os.system("python Eval.py data "+str(nensem))    
    os.chdir("../DataSplit")    
