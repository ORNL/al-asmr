import numpy as np
import os,sys,math
import copy
import h5py
from numpy import linalg as LA
#import torch
#import torchani


nensem=5
aclist=[0]
os.chdir('DataSplit')
itrain = True
os.system('rm AC*/data.h5')
#for i in range(0,len(aclist)):
for i in range(0,10):
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

    #in the Rerun directory    
    os.chdir("../Rerun")
    os.system("cp ../SMD/vmd.xyz .")
    os.system("cp ../SMD/ff.dat .")
    com ="cp ff.dat ../../ff"+str(i)+".dat"
    os.system(com)    
    os.system("python selxyz.py")
    os.system("python xyz2h5.py")
    os.system("python rerun.py")
    os.system("cp data.h5 ../next.h5")

    #in the AC directory        
    os.chdir("../")
    os.system("python UQ.py")
    com ="cp selected.h5 ../DataSplit/"+dirname+"/data.h5"
    os.system(com)
    os.system("python Eval.py")    
    os.chdir("../DataSplit")    

