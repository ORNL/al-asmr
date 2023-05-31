import os,sys,math
import numpy as np

def ReadFF(filename):
    f=open(filename)
    L=f.readlines()
    f.close()
    pe=[]
    dist=[]
    Eb =0.0

    tmp0=L[1].split()
    tmp1=L[len(L)-1].split()


    for i in range(1,len(L)):
        tmp=L[i].split()
        pe.append(float(tmp[6]))
        dist.append(float(tmp[5]))        

    Ef = np.mean(np.array(pe[-100]))
    Eb=Ef-float(tmp0[6])    
    return pe,dist,Eb

def MAE(true,pred):
    return np.mean(np.abs(true-pred))

def ReadAcc(filename):
    f=open(filename)
    L=f.readlines()
    f.close()

    maeelist=[]
    maeflist=[]

    for i in range(1,len(L)):
        tmp=L[i].split()
        maeelist.append(float(tmp[1]))
        maeflist.append(float(tmp[2]))                

    av_mae = np.mean(np.array(maeelist))
    std_mae = np.std(np.array(maeelist))    

    return av_mae,std_mae

def ReadEval(filename):
    f=open(filename)
    L=f.readlines()
    f.close()

    maeelist=[]
    maeflist=[]

    for i in range(0,len(L)):
        tmp=L[i].split()
        maeelist.append(float(tmp[0]))
        maeflist.append(float(tmp[1]))                

    av_mae = np.mean(np.array(maeelist))
    std_mae = np.std(np.array(maeelist))    

    return av_mae,std_mae

## evaluation of energy barrier
ndata = 6500
pe_ref,tdist,Eb_ref=ReadFF('ref.dat')
pe_ref=np.array(pe_ref[:ndata])

out=""
for i in range(0,10):
    pe,tdist,Eb=ReadFF('ff'+str(i)+'.dat')
    pe=np.array(pe[:ndata])
    tmae = MAE(pe_ref,pe)
    out+=str(i)+" "+str(tmae)+" "+str(Eb)+" "+str(abs(Eb-Eb_ref))+"\n"

f=open('summary_gap.dat','w')
f.write(out)
f.close()

## evaluation of train+validation data set## eval.dat
out=""
for i in range(0,10):
    filename = 'AC'+str(i)+'/eval.dat'
    mae,stmae=ReadEval(filename)
    out+=str(i)+" "+str(mae)+" "+str(stmae)+"\n"
f=open('summary_eval.dat','w')
f.write(out)
f.close()

## evaluation of SMD 6500 data set##
out=""
for i in range(0,10):
    filename = 'AC'+str(i)+'/accuracy_data.data'
    mae,stmae=ReadAcc(filename)
    out+=str(i)+" "+str(mae)+" "+str(stmae)+"\n"


f=open('summary_smd.dat','w')
f.write(out)
f.close()
