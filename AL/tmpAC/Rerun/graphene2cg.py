import os,sys,math,copy
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import h5py

#GANG SEOB JUNG at ORNL 03.02.2021, PYTHON3
#Graphene data file generation for lammps based on different ranked coarse grains
#Email:jungg@ornl.gov or gs4phone@gmail.com

def SaveHDF5(filename,coordinates,forces,energy,species,cell,virial):
    h5f = h5py.File(filename,'w')    
    mols = h5f.create_group('mols')
    mol = mols.create_group('mol')
    mol.create_dataset('coordinates',data=coordinates)
    mol.create_dataset('forces',data=forces)
    mol.create_dataset('cell',data=cell)
    mol.create_dataset('energies',data=energy)
    mol.create_dataset('species',data=species)
    mol.create_dataset('virial',data=virial)    
    print('Saved')
    h5f.close()

def CheckHDF5(filename):
    h5fr = h5py.File(filename,'r')
    mols =h5fr['mols']
    mol =mols['mol']
    print("Filename: %s" %filename)
    print("Keys: %s" %mol.keys())
    print(mol['species'])
    print(mol['cell'])
    print(mol['coordinates'])
    print(mol['forces'])    
    print(mol['energies'])
    print(mol['virial'])    
    h5fr.close()            

def ReadHDF5(filename,cut):
    h5fr = h5py.File(filename,'r')    
    mols =h5fr['mols']
    mol =mols['mol']
    coordinates = np.array(mol['coordinates'][cut:])
    forces = np.array(mol['forces'][cut:])
    cell = np.array(mol['cell'][cut:])
    energies = np.array(mol['energies'][cut:])
    species = np.array(mol['species'][:])
    virial = np.array(mol['virial'][cut:])    
    print('########## Read %s ###############' %filename)
    print('######### Species %s ' %len(species))
    print('######### Data # %s ' %len(coordinates))    
    h5fr.close()    
    return coordinates,forces,cell,energies,species,virial

class Atom:
    def __str__(self):
        return "Position " + str(self.x) +" " +str(self.y) + " " +str(self.z)
    def __init__(self,coor,aname):
        self.x = coor[0]
        self.y = coor[1]
        self.z = coor[2]
        self.eps =0.0
        self.aname =aname
        if(aname=='C'):
            self.atype =1
        else:self.atype=2
    def coord(self):
        return [self.x,self.y,self.z]

class Box:
    def __str__(self):
        tmp="xlo " + str(self.xmin) +" xhi " +str(self.xmax)+"\n"
        tmp+="ylo " + str(self.ymin) +" yhi " +str(self.ymax)+"\n"
        tmp+="zlo " + str(self.zmin) +" zhi " +str(self.zmax)+"\n"
        return tmp
    def __init__(self,tbox):
        self.xmin = tbox[0]
        self.xmax = tbox[1]
        self.ymin = tbox[2]
        self.ymax = tbox[3]
        self.zmin = tbox[4]
        self.zmax = tbox[5]

class Graphene:
    def __str__(self):
        tmp="lx " + str(self.lx) +" ly " +str(self.ly)+"\n"
        return tmp
    def __init__(self,lx,ly,n=1):
        sqrt3=math.sqrt(3)
        bondl=1.42*(2**(n-1))
        #self.atype=["C","H"]
        self.atype=["C"]
        self.bondl=bondl
        self.xlo=0.0
        self.ylo=0.0
        self.zlo=-16.75
        self.zhi=16.75
        self.lx=lx
        self.ly=ly
        self.lz=33.5
        self.xy = 0.0
        self.yz = 0.0
        self.xz = 0.0
        self.latticex = sqrt3*bondl
        self.latticey = 3*bondl
        self.latticez = 3.35
        self.atoms=[]
        self.clist=[]
        self.blist=[]
        self.bondlist=[]
        self.anglist=[]
        self.dilist=[]
        self.cx=1
        self.cy=1
        self.mass=12.0*(4**(n-1))
        
        x1 = 0.0;
        y1 = 0.5*bondl;
        
        x2 = 0.5*sqrt3*bondl;
        y2 = bondl;
        
        x3 = 0.5*sqrt3*bondl;
        y3 = 2*bondl;
        
        x4 = 0.0;
        y4 = 2.5*bondl;

        self.replix = int(self.lx/self.latticex)
        self.repliy = int(self.ly/self.latticey)

        self.xhi=self.replix*self.latticex
        self.yhi=self.repliy*self.latticey

        self.lx=self.xhi-self.xlo
        self.ly=self.yhi-self.ylo

        for i in range(0,self.replix):
            for j in range(0,self.repliy):
                x0 = i*self.latticex;
                y0 = j*self.latticey;
                self.atoms.append(Atom([x0+x1,y0+y1,0.0],"C"))
                self.atoms.append(Atom([x0+x2,y0+y2,0.0],"C"))
                self.atoms.append(Atom([x0+x3,y0+y3,0.0],"C"))
                self.atoms.append(Atom([x0+x4,y0+y4,0.0],"C"))

    def Hexagonal(self,size,offsets,dists):
        line1=0.0

    def HDF(self,coord,cell):
        atoms=[]
        self.xlo=0
        self.ylo=0
        self.zlo=0
        self.xhi=cell[0][0]
        self.yhi=cell[1][1]
        self.zhi=cell[2][2]                
        self.xy=cell[0][1]
        self.xz=cell[0][2]
        self.yz=cell[1][2]
        self.lx = self.xhi-self.xlo
        self.ly = self.yhi-self.ylo
        self.lz = self.zhi-self.zlo                
        
        for i in range(0,len(coord)):
            atom = Atom(coord[i],'C')
            atoms.append(atom)
        self.atoms=atoms
            
    def Crack(self,size,offsets=[0,0]):
        x0 = self.xlo+0.5*self.lx
        y0 = self.ylo        

        for i in range(0,len(self.atoms)):
            coord=self.atoms[i].coord()
            tmpx=coord[0]-x0
            tmpy=coord[1]-y0            
            if(tmpy<0.1*self.ly and tmpx<0.01*self.lx and tmpx>-0.01*self.lx):
                #print "crack"
                self.atoms[i].atype=-1
                
        self.Update()

    def Replicate(self,nx,ny):
        box=[]
        a=[self.lx,0.0,0.0]
        b=[self.xy,self.ly,0.0]
        c=[self.xz,self.yz,self.lz]
        box.append([a[0],b[0],c[0]])
        box.append([a[1],b[1],c[1]])
        box.append([a[2],b[2],c[2]])
        h = np.array(box)
        invh=np.linalg.inv(h)
        #scaling = np.array([[nx, 0, 0],[0,ny,0],[0.0,0.0,1]]) # deformation lammps    
        newbox = []
        newbox.append([a[0]*nx,b[0]*ny,c[0]])
        newbox.append([a[1]*nx,b[1]*ny,c[1]])
        newbox.append([a[2]*nx,b[2]*ny,c[2]])
        newh = np.array(newbox)
        
        newatoms=[]
        atoms=self.atoms
        for i in range(0,nx):
            for j in range(0,ny):
                for ai in range(0,len(atoms)):
                    coords = atoms[ai].coord()
                    tmp = np.dot(invh,coords)
                    scoords=[tmp[0]+i,tmp[1]+j,tmp[2]]
                    
                    coords=np.dot(h,scoords)
                    atom = Atom(coords,"C")
                    newatoms.append(atom)

        #ax bx cx
        #0  by cy
        #0  0  cz

        self.atoms=newatoms
        #lx = a
        #xy = b
        self.lx = newh[0][0]
        self.ly = newh[1][1]
        self.lz = newh[2][2]         

        self.xy = newh[0][1]
        self.xz = newh[0][2]
        self.yz = newh[1][2]

        self.xlo = 0.0
        self.xhi = self.lx

        self.ylo = 0.0
        self.yhi = self.ly

        self.zlo = 0.0
        self.zhi = self.lz        

        
    def CoordCellNP(self):

        box=[]
        a=[self.lx,0.0,0.0]
        b=[self.xy,self.ly,0.0]
        c=[self.xz,self.yz,self.lz]
        box.append([a[0],b[0],c[0]])
        box.append([a[1],b[1],c[1]])
        box.append([a[2],b[2],c[2]])
        h = np.array(box)
        atoms=self.atoms
        natom = len(atoms)

        species = np.chararray(natom)
        for i in range(0,natom):
            species[i]='C'
            
        coord = np.empty(shape=[0,3])

        for i in range(0,natom):
            tcoord = np.array([atoms[i].coord()])
            coord = np.append(coord,tcoord,axis=0)
        return species,coord,h
        
    def Triangle(self,size,dists,offsets=[0,0]):
        sqrt3=math.sqrt(3)
        centers=[]
        dlx=size+dists[0]
        dly=size*sqrt3*0.5+dists[1]

        spanx=int(self.lx/dlx)
        spany=int(self.ly/dly)

        if(offsets==[0,0]):
            offsets[0]=0.5*(self.lx-dlx*spanx)
            offsets[1]=0.5*(self.ly-dly*spany)   

        coords=[0.5*dists[0]+offsets[0],0.5*dists[1]+offsets[1]]     

        for i in range(0,spanx):
            for j in range(0,spany):
                centers.append([coords[0]+dlx*i+0.5*size,coords[1]+dly*j,0.0])

        for i in range(0,len(self.atoms)):
            coord=self.atoms[i].coord()
            for j in range(0,len(centers)):
                tmpx=coord[0]-centers[j][0]
                tmpy=coord[1]-centers[j][1]
                if(tmpy>0.0):
                    if(tmpy-sqrt3*tmpx-sqrt3*0.5*size < 0.0 and tmpy+sqrt3*tmpx-sqrt3*0.5*size<0.0):
                        self.atoms[i].atype=-1

        self.Update()

    def TriangleInv(self,size,dists,offsets=[0,0]):
        sqrt3=math.sqrt(3)
        centers=[]
        dlx=size+dists[0]
        dly=size*sqrt3*0.5+dists[1]

        spanx=int(self.lx/dlx)
        spany=int(self.ly/dly)

        if(offsets==[0,0]):
            offsets[0]=0.5*(self.lx-dlx*spanx)
            offsets[1]=0.5*(self.ly-dly*spany)   

        coords=[0.5*dists[0]+offsets[0],0.5*dists[1]+offsets[1]]     

        for i in range(0,spanx):
            for j in range(0,spany):
                centers.append([coords[0]+dlx*i+0.5*size,coords[1]+dly*j+0.5*size*sqrt3,0.0])

        for i in range(0,len(self.atoms)):
            coord=self.atoms[i].coord()
            for j in range(0,len(centers)):
                tmpx=coord[0]-centers[j][0]
                tmpy=coord[1]-centers[j][1]
                if(tmpy<0.0):
                    if(tmpy-sqrt3*tmpx+sqrt3*0.5*size > 0.0 and tmpy+sqrt3*tmpx+sqrt3*0.5*size>0.0):
                        self.atoms[i].atype=-1

        self.Update()

    def Rectangle(self,sizes,dists,offsets=[0,0]):
        centers=[]
        dlx=sizes[0]+dists[0]
        dly=sizes[1]+dists[1]
        
        spanx=int(self.lx/dlx)
        spany=int(self.ly/dly)

        if(offsets==[0,0]):
            offsets[0]=0.5*(self.lx-dlx*spanx)
            offsets[1]=0.5*(self.ly-dly*spany)

        coords=[0.5*dists[0]+offsets[0],0.5*dists[1]+offsets[1]]

        for i in range(0,spanx):
            for j in range(0,spany):
                centers.append([coords[0]+dlx*i,coords[1]+dly*j,0.0])

        for i in range(0,len(self.atoms)):
            coord=self.atoms[i].coord()
            for j in range(0,len(centers)):
                if(coord[0]-centers[j][0] < sizes[0] and coord[0]-centers[j][0]>0):
                    if(coord[1]-centers[j][1] < sizes[1] and coord[1]-centers[j][1]>0):
                        self.atoms[i].atype=-1

        self.Update()

    def Circle(self,radius,dists,offsets=[0,0]):
        centers=[]
        dlx=radius*2+dists[0]
        dly=radius*2+dists[1]

        spanx=int(self.lx/dlx)
        spany=int(self.ly/dly)

        if(offsets==[0,0]):
            offsets[0]=0.5*(self.lx-dlx*spanx)
            offsets[1]=0.5*(self.ly-dly*spany)

        coords=[radius+0.5*dists[0]+offsets[0],radius+0.5*dists[1]+offsets[1]]
        
        for i in range(0,spanx):
            for j in range(0,spany):
                centers.append([coords[0]+dlx*i,coords[1]+dly*j,0.0])

        for i in range(0,len(self.atoms)):
            coord=self.atoms[i].coord()
            for j in range(0,len(centers)):
                if(DistVec(coord,centers[j]) < radius):
                    self.atoms[i].atype=-1

        self.Update()

    def Update(self):
        tatoms=[]
        for i in range(0,len(self.atoms)):
            if (self.atoms[i].atype>0):
                tatoms.append(self.atoms[i])

        self.atoms=tatoms

    def Rotate(self):
        for i in range(0,len(self.atoms)):
            tx = self.atoms[i].x
            ty = self.atoms[i].y
            self.atoms[i].x=ty
            self.atoms[i].y=tx                        

        txhi=self.xhi
        tyhi=self.yhi        

        self.xhi=tyhi
        self.yhi=txhi

        self.lx = self.xhi-self.xlo
        self.ly = self.yhi-self.ylo

    def Noise(self,es):
        #let nosie except index 0 
        ex=es[0]
        ey=es[1]
        ez=es[2]        
        atoms=self.atoms
        for i in range(1,len(atoms)):
            dx = rand.uniform(-ex,ex)
            dy = rand.uniform(-ey,ey)
            dz = rand.uniform(-ez,ez)            

            atoms[i].x+=dx
            atoms[i].y+=dy
            atoms[i].z+=dz

    def Vacancy(self,nvac):
        #let nosie except index 0 
        atoms=self.atoms
        natom=len(atoms)-nvac
        
        for i in range(0,nvac):
            vid = rand.randint(1,natom-1)
            del atoms[vid]

        self.atoms =atoms
        
    def CellList(self):
        self.clist=[]
        clist=[]
        dr=self.bondl*2
        lx=self.lx
        ly=self.ly
        cx=int(lx/dr)+1
        cy=int(ly/dr)+1
        
        for i in range(0,cx):
            clist.append([])
            for j in range(0,cy):
                clist[i].append([])

        self.cx=cx
        self.cy=cy
        atoms=self.atoms
        numAtom=len(atoms)
        
        for i in range(0,numAtom):
            coord=atoms[i].coord()
                
            ci=int(coord[0]/dr)
            cj=int(coord[1]/dr)
            clist[ci][cj].append(i)

        self.clist=clist

    def AngleList(self):
        anglist=[]
        blist=self.blist
        atoms=self.atoms
        if(len(blist)<1):print ("Hey, the bondlist seems not generated!")
        numAtom=len(atoms)
        
        for i in range(0,numAtom):
            for j in range(0,len(blist[i])):
                ja=blist[i][j]
                for k in range(0,len(blist[ja])):
                    ka=blist[ja][k]
                    if(i<ka):
                        anglist.append([i,ja,ka])

        self.anglist=anglist

    def DihedralList(self):
        dilist=[]
        anglist=self.anglist
        atoms=self.atoms
        blist=self.blist
        if(len(anglist)<1):print ("Hey, the angle list seems not generated!")
        
        numAngle=len(anglist)
        for i in range (0,numAngle):
            ja=anglist[i][2]
            ca=anglist[i][1]
            for j in range (0,len(blist[ja])):
                if(ca!=blist[ja][j]):
                    dilist.append([anglist[i][0],anglist[i][1],anglist[i][2],blist[ja][j]])

        self.dilist=dilist

    def BondList(self,pbc=True):
        blist=[]
        bondlist=[]
        clist=self.clist
        self.blist=[]
        atoms=self.atoms
        bondl=self.bondl
        cx=self.cx
        cy=self.cy

        box=[self.lx,self.ly,self.lz]
        
        numAtom=len(atoms)
        for i in range(0,numAtom):
            blist.append([])
            bondlist.append([])

        for c_i in range(0,cx):
            for c_j in range(0,cy):
                nca=len(clist[c_i][c_j])
                for cai in range(0,nca):
                    i = clist[c_i][c_j][cai]
                    for ci in range(c_i-1,c_i+2):
                        for cj in range(c_j-1,c_j+2):
                            if(ci<0):ci=cx-1
                            if(cj<0):cj=cy-1
                            if(ci==cx):ci=0
                            if(cj==cy):cj=0
                            if(pbc!=True): box=[100000,100000,100000]

                            if(ci>-1 and cj >-1 and ci<cx and cj<cy):
                                for ca in range(0,len(clist[ci][cj])):
                                    ja = clist[ci][cj][ca]
                                    if(i!=ja and i<ja):
                                        dist=DistAtoms(atoms[i],atoms[ja],box)
                                        if(dist < bondl*1.5):
                                            bondlist[i].append(ja)
                                    if(i!=ja):
                                        dist=DistAtoms(atoms[i],atoms[ja],box)
                                        if(dist < bondl*1.5):
                                            blist[i].append(ja)

        """for i in range(0,len(blist)):
            print i,blist[i]
            """        
        self.blist=blist
        self.bondlist=bondlist

    def Strain(self,epx,epy):
        atoms=self.atoms
        lx=self.lx
        ly=self.ly        
        xlo=self.xlo
        ylo=self.ylo
        zlo=self.zlo
        xhi=self.xhi
        yhi=self.yhi
        zhi=self.zhi

        for i in range(0,len(atoms)):
            atom=atoms[i]
            atom.x = atom.x*(1.0+epx)
            atom.y = atom.y*(1.0+epy)            

        self.lx=(1.0+epx)*lx
        self.ly=(1.0+epy)*ly        
        self.xhi=(1.0+epx)*xhi
        self.yhi=(1.0+epy)*yhi        

    def ShearStrain(self,exy):
        atoms=self.atoms
        lx=self.lx
        ly=self.ly        
        xlo=self.xlo
        ylo=self.ylo
        zlo=self.zlo
        xhi=self.xhi
        yhi=self.yhi
        zhi=self.zhi
        lxy=ly*exy*2.0

        for i in range(0,len(atoms)):
            atom=atoms[i]
            atom.x = atom.x*(1.0+epx)
            atom.y = atom.y*(1.0+epy)            

        self.lx=(1.0+epx)*lx
        self.ly=(1.0+epy)*ly        
        self.xhi=(1.0+epx)*xhi
        self.yhi=(1.0+epy)*yhi


    def Deform(self,scaling):
        #scaling = np.array([[sx, 0.0, 0.0],[xy,sy,0.0],[xz,yz,sz]])

        box=[]
        a=[self.lx,0.0,0.0]
        b=[self.xy,self.ly,0.0]
        c=[self.xz,self.yz,self.lz]
        box.append([a[0],b[0],c[0]])
        box.append([a[1],b[1],c[1]])
        box.append([a[2],b[2],c[2]])
        h = np.array(box)
        invh=np.linalg.inv(h)
        
        newh = np.matmul(scaling,h)
        atoms = self.atoms

        for i in range(0,len(atoms)):
            coords = atoms[i].coord()
            tmp = np.dot(invh,coords)
            scoords=[tmp[0],tmp[1],tmp[2]]
            coords=np.dot(newh,scoords)
            atoms[i].x=coords[0]
            atoms[i].y=coords[1]
            atoms[i].z=coords[2]    

        self.atoms = atoms
        self.lx = newh[0][0]
        self.ly = newh[1][1]
        self.lz = newh[2][2]         

        self.xy = newh[0][1]
        self.xz = newh[0][2]
        self.yz = newh[1][2]

        self.xlo = 0.0
        self.xhi = self.lx

        self.ylo = 0.0
        self.yhi = self.ly

        self.zlo = 0.0
        self.zhi = self.lz
        
        #print(newh)

        
        
def DistVec(coord1,coord2):
    dx=coord1[0]-coord2[0]
    dy=coord1[1]-coord2[1]
    dz=coord1[2]-coord2[2]

    r2=dx*dx+dy*dy+dz*dz
    return math.sqrt(r2)    

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

def ReadXYZ(filename):
    f=open(filename)
    L=f.readlines()

    atoms=[]

    for i in range(2,len(L)):
        tmp=L[i].split()
        aname=tmp[0]
        x=float(tmp[1])
        y=float(tmp[2])
        z=float(tmp[3])
        coord=(x,y,z)
        atoms.append(Atom(coord,aname))
            
    f.close()
    return atoms

def GetDim(atoms):
    xmin=0
    xmax=0
    ymin=0
    ymax=0
    zmin=0
    zmax=0
    for i in range(0,len(atoms)):
        x=atoms[i].x
        y=atoms[i].y
        z=atoms[i].z
        if(x>xmax):xmax=x
        if(x<xmin):xmin=x
        if(y>ymax):ymax=y
        if(y<ymin):ymin=y
        if(z>zmax):zmax=z
        if(z<zmin):zmin=z
                       
    tbox=[xmin,xmax,ymin,ymax,zmin,zmax]
    return Box(tbox)

def GenVMD(atoms,filename):
    numAtom=len(atoms)
    f=open(filename,'w')
    vmdout=str(numAtom)+"\n"
    vmdout+="Atoms. Timestep: 0\n"
    for i in range(0,len(atoms)):
        if(atoms[i].atype>0):
            vmdout+= atoms[i].aname+" "+str(atoms[i].x) +" "+str(atoms[i].y)+" "+str(atoms[i].z)+"\n"
    f.write(vmdout)
    f.close()

def GenDataREBO(graphene,filename):
    atoms=graphene.atoms
    numAtom=len(atoms)
    dataout="LAMMPS DATA FILE by G. Jung@ORNL\n\n"
    dataout+=str(numAtom) + " atoms\n"
    dataout+="2 atom types\n"
    dataout+=str(graphene.xlo)+" "+str(graphene.xhi)+" xlo xhi\n"
    dataout+=str(graphene.ylo)+" "+str(graphene.yhi)+" ylo yhi\n"
    if(graphene.xy!=0.0 or graphene.xz !=0.0 or graphene.yz!=0.0):
        dataout+=str(graphene.zlo)+" "+str(graphene.zhi)+" zlo zhi\n"                
        dataout+=str(graphene.xy)+" "+str(graphene.xz)+" "+str(graphene.yz)+" xy xz yz\n\n"
    else:dataout+=str(graphene.zlo)+" "+str(graphene.zhi)+" zlo zhi\n\n"

    dataout+="Masses\n\n"
    dataout+="1 "+str(graphene.mass)+"\n"
    dataout+="2 1.00\n"
    dataout+="\nAtoms\n\n"

    for i in range(0,numAtom):
        dataout+=str(i+1)+" "+ str(atoms[i].atype)+" 0 "+ str(atoms[i].x)+" "+ str(atoms[i].y)+" "+ str(atoms[i].z)+"\n"

    f=open(filename,'w')
    f.write(dataout)
    f.close()

def GenDataANI(graphene,filename):
    atoms=graphene.atoms
    numAtom=len(atoms)
    dataout="LAMMPS DATA FILE by G. Jung@ORNL\n\n"
    dataout+=str(numAtom) + " atoms\n"
    dataout+="8 atom types\n"
    dataout+=str(graphene.xlo)+" "+str(graphene.xhi)+" xlo xhi\n"
    dataout+=str(graphene.ylo)+" "+str(graphene.yhi)+" ylo yhi\n"
    #dataout+=str(0.0)+" "+str(graphene.zhi-graphene.zlo)+" zlo zhi\n\n"
    if(graphene.xy!=0.0 or graphene.xz !=0.0 or graphene.yz!=0.0):
        dataout+=str(graphene.zlo)+" "+str(graphene.zhi)+" zlo zhi\n"                
        dataout+=str(graphene.xy)+" "+str(graphene.xz)+" "+str(graphene.yz)+" xy xz yz\n\n"
    else:dataout+=str(graphene.zlo)+" "+str(graphene.zhi)+" zlo zhi\n\n"        
    dataout+="Masses\n\n"
    dataout+="1 1.01\n"    
    dataout+="2 4.00\n"
    dataout+="3 6.94\n"
    dataout+="4 9.01\n"
    dataout+="5 10.81\n"
    dataout+="6 12.01\n"
    dataout+="7 14.01\n"
    dataout+="8 16.0\n"        
    dataout+="\nAtoms\n\n"

    for i in range(0,numAtom):
        if(atoms[i].aname =="C"):
            dataout+=str(i+1)+" 6 0 "+ str(atoms[i].x)+" "+ str(atoms[i].y)+" "+ str(atoms[i].z-graphene.zlo)+"\n"

    f=open(filename,'w')
    f.write(dataout)
    f.close()    

def GenDataDFTB(graphene,filename,pbc=True):
    A2Bohr=1.889725989
    atoms=graphene.atoms
    numAtom=len(atoms)
    dataout=""
    dataout+=str(numAtom)
    if(pbc==True):
        dataout+=" S\n" #PBC with cartesian coordinates
    else:
        dataout+=" C\n" #Cluster

    atypes=graphene.atype
    for i in range(0,len(atypes)):
        dataout+=atypes[i]+" "
    dataout+="\n"

    atom_string="%6d %3d %18.10E %18.10E %18.10E\n"            
    for i in range(0,numAtom):
        #dataout+=(atom_string %(i+1,atoms[i].atype,atoms[i].x*A2Bohr,atoms[i].y*A2Bohr,atoms[i].z*A2Bohr))
        dataout+=(atom_string %(i+1,atoms[i].atype,atoms[i].x-graphene.xlo,atoms[i].y-graphene.ylo,atoms[i].z-graphene.zlo))
    box_string="%18.10E %18.10E %18.10E\n"        
    #dataout+=(box_string %(graphene.xlo*A2Bohr,graphene.ylo*A2Bohr,graphene.zlo*A2Bohr))
    dataout+=(box_string %(0,0,0))        
    dataout+=(box_string %(graphene.xhi-graphene.xlo,0,0))    
    dataout+=(box_string %(0,graphene.yhi-graphene.ylo,0))    
    dataout+=(box_string %(0,0,graphene.zhi-graphene.zlo))

    f=open(filename,'w')
    f.write(dataout)
    f.close()
    
def GenData(graphene,filename):
    atoms=graphene.atoms
    bondlist=graphene.bondlist
    bondl=graphene.bondl
    anglist=graphene.anglist
    dilist=graphene.dilist
    numAtom=len(atoms)
    numBond=0
    for i in range(0,numAtom):
        numBond+=len(bondlist[i])
    numAngle=len(anglist)
    numDihedral=len(dilist)

    dataout="LAMMPS DATA FILE by G. Jung@ORNL\n\n"
    dataout+=str(numAtom) + " atoms\n"
    dataout+=str(numBond) + " bonds\n"
    dataout+=str(numAngle)+" angles\n"
    dataout+=str(numDihedral)+ " dihedrals\n"
    dataout+="2 atom types\n"
    dataout+="1 bond types\n"
    dataout+="1 angle types\n"
    dataout+="1 dihedral types\n\n"
    dataout+=str(graphene.xlo)+" "+str(graphene.xhi)+" xlo xhi\n"
    dataout+=str(graphene.ylo)+" "+str(graphene.yhi)+" ylo yhi\n"
    dataout+=str(graphene.zlo)+" "+str(graphene.zhi)+" zlo zhi\n\n"
    dataout+="Masses\n\n"
    dataout+="1 "+str(graphene.mass)+"\n"
    dataout+="2 1.00\n"
    dataout+="\nAtoms\n\n"

    for i in range(0,numAtom):
        dataout+=str(i+1)+" "+ str(atoms[i].atype)+" 1 0 "+ str(atoms[i].x)+" "+ str(atoms[i].y)+" "+ str(atoms[i].z)+"\n"

    #dataout+="\nBond Coeffs\n\n"
    #dataout+="1 1.0 "+str(bondl)+"\n"

    dataout+="\nBonds\n\n"
    count=1
    for i in range(0,numAtom):
        for j in range(0,len(bondlist[i])):
            dataout+=str(count) + " 1 " + str(i+1) +" "+ str(bondlist[i][j]+1)+"\n"
            count+=1

    dataout+="\nAngles\n\n"
    
    count=1
    for i in range(0,len(anglist)):
        dataout+=str(count) + " 1 " + str(anglist[i][0]+1) +" "+ str(anglist[i][1]+1)+" "+ str(anglist[i][2]+1)+"\n"
        count+=1

    dataout+="\nDihedrals\n\n"
    
    count=1
    for i in range(0,len(dilist)):
        dataout+=str(count) + " 1 " + str(dilist[i][0]+1) +" "+ str(dilist[i][1]+1)+" "+ str(dilist[i][2]+1)+" "+ str(dilist[i][3]+1)+"\n"
        count+=1

    f=open(filename,'w')
    f.write(dataout)
    f.close()

def GenDataNano(graphene,filename):
    atoms=graphene.atoms
    bondlist=graphene.bondlist
    bondl=graphene.bondl
    anglist=graphene.anglist
    dilist=graphene.dilist
    numAtom=len(atoms)
    numBond=0
    for i in range(0,numAtom):
        numBond+=len(bondlist[i])
    numAngle=len(anglist)
    numDihedral=len(dilist)

    amu=1.660539e-24 #gram

    eunit=1.60318e-19 #eV to Joule
    conv=eunit*1.0e21
    
    dataout="LAMMPS DATA FILE by G. Jung@ORNL\n\n"
    dataout+=str(numAtom) + " atoms\n"
    dataout+=str(numBond) + " bonds\n"
    dataout+=str(numAngle)+" angles\n"
    dataout+=str(numDihedral)+ " dihedrals\n"
    dataout+="2 atom types\n"
    dataout+="1 bond types\n"
    dataout+="1 angle types\n"
    dataout+="1 dihedral types\n\n"
    dataout+=str(graphene.xlo*0.1)+" "+str(graphene.xhi*0.1)+" xlo xhi\n"
    dataout+=str(graphene.ylo*0.1)+" "+str(graphene.yhi*0.1)+" ylo yhi\n"
    dataout+=str(graphene.zlo)+" "+str(graphene.zhi)+" zlo zhi\n\n"
    dataout+="Masses\n\n"
    dataout+="1 "+str(graphene.mass*amu/1.0e-18)+"\n"
    dataout+="2 1.00\n"
    dataout+="\nAtoms\n\n"

    for i in range(0,numAtom):
        dataout+=str(i+1)+" "+ str(atoms[i].atype)+" 1 0 "+ str(atoms[i].x*0.1)+" "+ str(atoms[i].y*0.1)+" "+ str(atoms[i].z*0.1)+"\n"

    #dataout+="\nBond Coeffs\n\n"
    #dataout+="1 1.0 "+str(bondl)+"\n"

    dataout+="\nBonds\n\n"
    count=1
    for i in range(0,numAtom):
        for j in range(0,len(bondlist[i])):
            dataout+=str(count) + " 1 " + str(i+1) +" "+ str(bondlist[i][j]+1)+"\n"
            count+=1

    dataout+="\nAngles\n\n"
    
    count=1
    for i in range(0,len(anglist)):
        dataout+=str(count) + " 1 " + str(anglist[i][0]+1) +" "+ str(anglist[i][1]+1)+" "+ str(anglist[i][2]+1)+"\n"
        count+=1

    dataout+="\nDihedrals\n\n"
    
    count=1
    for i in range(0,len(dilist)):
        dataout+=str(count) + " 1 " + str(dilist[i][0]+1) +" "+ str(dilist[i][1]+1)+" "+ str(dilist[i][2]+1)+" "+ str(dilist[i][3]+1)+"\n"
        count+=1

    f=open(filename,'w')
    f.write(dataout)
    f.close()

    
def Shift(system,dx,dy,dz):
    atoms=system.atoms

    lx = system.lx
    ly = system.ly
    lz = system.lz
    
    for i in range(0,len(atoms)):
        atom = atoms[i]
        atom.x += dx
        atom.y += dy
        atom.z += dz        

        if(atom.x>lx):atom.x-=lx
        if(atom.y>ly):atom.y-=ly
        if(atom.y>lz):atom.z-=lz
        
    return atoms

def AdjustZ(system,dz):
    atoms=system.atoms

    for i in range(0,len(atoms)):
        atom = atoms[i]
        atom.z = dz 

    return atoms
        
def GenDataMicro(graphene,filename):
    atoms=graphene.atoms
    bondlist=graphene.bondlist
    bondl=graphene.bondl
    anglist=graphene.anglist
    dilist=graphene.dilist
    numAtom=len(atoms)
    numBond=0
    for i in range(0,numAtom):
        numBond+=len(bondlist[i])
    numAngle=len(anglist)
    numDihedral=len(dilist)

    amu=1.660539e-24 #gram

    eunit=1.60318e-19 #eV to Joule
    conv=eunit*1.0e21
    
    dataout="LAMMPS DATA FILE by G. Jung@LAMM\n\n"
    dataout+=str(numAtom) + " atoms\n"
    dataout+=str(numBond) + " bonds\n"
    dataout+=str(numAngle)+" angles\n"
    dataout+=str(numDihedral)+ " dihedrals\n"
    dataout+="2 atom types\n"
    dataout+="1 bond types\n"
    dataout+="1 angle types\n"
    dataout+="1 dihedral types\n\n"
    dataout+=str(graphene.xlo*0.0001)+" "+str(graphene.xhi*0.0001)+" xlo xhi\n"
    dataout+=str(graphene.ylo*0.0001)+" "+str(graphene.yhi*0.0001)+" ylo yhi\n"
    dataout+=str(graphene.zlo)+" "+str(graphene.zhi)+" zlo zhi\n\n"
    dataout+="Masses\n\n"
    dataout+="1 "+str(graphene.mass*amu/1.0e-12)+"\n"
    dataout+="2 1.00\n"
    dataout+="\nAtoms\n\n"

    for i in range(0,numAtom):
        dataout+=str(i+1)+" "+ str(atoms[i].atype)+" 1 0 "+ str(atoms[i].x*0.0001)+" "+ str(atoms[i].y*0.0001)+" "+ str(atoms[i].z*0.0001)+"\n"

    #dataout+="\nBond Coeffs\n\n"
    #dataout+="1 1.0 "+str(bondl)+"\n"

    dataout+="\nBonds\n\n"
    count=1
    for i in range(0,numAtom):
        for j in range(0,len(bondlist[i])):
            dataout+=str(count) + " 1 " + str(i+1) +" "+ str(bondlist[i][j]+1)+"\n"
            count+=1

    dataout+="\nAngles\n\n"
    
    count=1
    for i in range(0,len(anglist)):
        dataout+=str(count) + " 1 " + str(anglist[i][0]+1) +" "+ str(anglist[i][1]+1)+" "+ str(anglist[i][2]+1)+"\n"
        count+=1

    dataout+="\nDihedrals\n\n"
    
    count=1
    for i in range(0,len(dilist)):
        dataout+=str(count) + " 1 " + str(dilist[i][0]+1) +" "+ str(dilist[i][1]+1)+" "+ str(dilist[i][2]+1)+" "+ str(dilist[i][3]+1)+"\n"
        count+=1

    f=open(filename,'w')
    f.write(dataout)
    f.close()

def RandomStrain():
    xy = rand.uniform(0.0,0.3)
    xz = 0.0
    yz = 0.0
    sx = rand.uniform(0.95,1.2)
    sy = rand.uniform(0.95,1.2)
    sz = 1.0
    scaling = np.array([[sx, xy, xz],[0,sy,yz],[0.0,0.0,sz]]) # deformation lammps    
    return scaling 
