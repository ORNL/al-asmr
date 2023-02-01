import os,sys,math
#GS JUNG@ORNL

class Atom:
    def __str__(self):
        return "Position " + str(self.x) +" " +str(self.y) + " " +str(self.z)
    def __init__(self,coor,aname):
        self.x = coor[0]
        self.y = coor[1]
        self.z = coor[2]
        self.eps =0.0
        self.aname =aname
        if(aname=='H'):
            self.atype =1
        if(aname=='C'):
            self.atype =2            
            
    def coord(self):
        return [self.x,self.y,self.z]

def DistAtoms(atom1,atom2,box=[1000000,1000000,100000]):
    x1=atom1.x
    y1=atom1.y
    z1=atom1.z

    x2=atom2.x
    y2=atom2.y
    z2=atom2.z

    dx=abs(x1-x2)
    dy=abs(y1-y2)
    dz=abs(z1-z2)

    while(dx>0.5*box[0]):dx-=box[0]
    while(dy>0.5*box[1]):dy-=box[1]
    while(dz>0.5*box[2]):dz-=box[2]

    r2=dx*dx+dy*dy+dz*dz
    return math.sqrt(r2)    
    
def GenVMD(frames,filename):
    
    
    numAtom=len(frames[0])
    f=open(filename,'w')

    for j in range(0,len(frames)):
        atoms=frames[j]
        vmdout=str(numAtom)+"\n"
        vmdout+="Atoms. Timestep: 0\n"
        for i in range(0,len(atoms)):
            if(atoms[i].atype>0):
                vmdout+= atoms[i].aname+" "+str(atoms[i].x) +" "+str(atoms[i].y)+" "+str(atoms[i].z)+"\n"
        f.write(vmdout)
    f.close()    

def CheckCoord(atoms):
    nlist=[]    
    for i in range(0,len(atoms)):
        nlist.append([])
        for j in range(0,len(atoms)):
            if(i!=j):
                iatom = atoms[i]
                jatom = atoms[j]
            
                dist = DistAtoms(iatom,jatom)
                if(dist <3.0):
                    nlist[i].append(j)

    check = True
    
    for i in range(0,len(atoms)):
        nenum = len(nlist[i])
        if(nenum==0):
            #print("Geo corrupted!")
            check=False
        
    return check 
    
def ReadXYZ(filename):
    frames=[]
    f=open(filename)
    L=f.readlines()
    f.close()    

    natoms= int((L[0].split()[0]))
    nframe = int(len(L)/(natoms+2))

    print("# Atoms and # of Frames: ",natoms,nframe)
    for j in range(0,nframe):
        atoms=[]        
        for i in range(2+j*(natoms+2),2+j*(natoms+2)+natoms):
            tmp=L[i].split()
            aname=tmp[0]
            x=float(tmp[1])
            y=float(tmp[2])
            z=float(tmp[3])
            coord=(x,y,z)
            atoms.append(Atom(coord,aname))
        frames.append(atoms)

    return frames

def ReadFF(filename):
    f=open(filename)
    L=f.readlines()
    f.close()
    pe=[]
    dist=[]
    for i in range(1,len(L)):
        tmp=L[i].split()
        pe.append(float(tmp[7]))
        dist.append(float(tmp[5]))        

    print("# of data",len(pe))
    return pe,dist

frames=ReadXYZ('vmd.xyz')
pe,dist =ReadFF('ff.dat')

newframes = []
newpe = []

#Controling params
dE = 0.001*(6+14) # Energy cut for Total # of atoms
dr = 0.03

check=[]
for i in range(0,len(frames)):
    check.append(True)

cdist = dist[0]
cE = pe[0]
selnum =0
for i in range(1,len(dist)):
    xd = abs(dist[i]-cdist)
    Ed = abs(pe[i]-cE)
    geocheck = CheckCoord(frames[i])
    if(xd > dr and Ed > dE and geocheck):
        cdist = dist[i]
        cE=pe[i]
        selnum+=1
    else:
        check[i]=False


print("# of selected frame:",selnum)

out=""
for i in range(0,len(pe)):
    if(check[i]):
        out+=str(i)+" "+str(pe[i]) +" "+ str(pe[i]+1)+"\n"
    else:
        out+=str(i)+" "+str(pe[i]) +" "+str(pe[i])+"\n"        

f=open('selected.data','w')
f.write(out)
f.close()

for i in range(0,len(frames)):
    if(check[i]):
        newframes.append(frames[i])

GenVMD(newframes,'selected.xyz')
#dE = 0.001*(6) # Total # of carbon
#for i in range(0,len(pe)):
    
