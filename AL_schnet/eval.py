import os,sys,math
import numpy as np

def ReadFF(filename):
    f=open(filename)
    L=f.readlines()
    f.close()
    pe=[]
    dist=[]
    for i in range(1,len(L)):
        tmp=L[i].split()
        pe.append(float(tmp[6]))
        dist.append(float(tmp[5]))        

    #print("# of data",len(pe))
    return pe,dist

def MAE(true,pred):
    return np.mean(np.abs(true-pred))

ndata = 6500
pe_ref,tdist=ReadFF('ref.dat')
pe_ref=np.array(pe_ref[:ndata])

out=""
for i in range(0,10):
    pe,tdist=ReadFF('ff'+str(i)+'.dat')
    pe=np.array(pe[:ndata])
    tmae = MAE(pe_ref,pe)
    out+=str(i)+" "+str(tmae)+"\n"

f=open('summary.dat','w')
f.write(out)
f.close()

