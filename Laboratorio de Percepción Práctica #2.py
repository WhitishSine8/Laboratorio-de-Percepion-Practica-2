#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from random import *
from math import *
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from matplotlib.pyplot import plot,show

#Cálculos
def normalizar(r,lb,ub):
    return (r-lb)/(ub-lb);

def desnormalizar(n,lb,ub):
    return n*(ub-lb)+lb;

def maxp(V):
    #(val,pos)=maxp(V)
    n = len(V);
    pos = 0;
    val = V[pos];
    for e in range(n):
        if V[e] > val:
            val = V[e];
            pos = e;
    return val, pos

def minp(V):
    n = len(V);
    pos = 0;
    val = V[pos];
    for e in range(n):
        if V[e] < val:
            val = V[e];
            pos = e;
    return val, pos

def calcD(V1,V2):
    n = len(V1);
    s = 0;
    for e in range(n):
        s = s + (V1[e]-V2[e])**2;
    d = sqrt(s);
    return d

def DatabaseRead():
    df = pd.read_excel(r'C:\Users\migue\Desktop\datos.xls')
    Nrows = len(df); 
    Ncols = len(df.columns);
    DataBrute = [[0 for i in range(Ncols)] for j in range(Nrows)]; 
    for r in range(Nrows):
        for c in range(Ncols):
            DataBrute[r][c]=df[df.columns[c]][r];
    return DataBrute

def plotgraph(V):
    N = len(V)
    D = [j for j in range(N)]
    grafica.figure(1)
    grafica.plot(D, V, "b:", linewidth = 2)
    grafica.xlabel("Número de datos")
    grafica.ylabel("Señal")
    grafica.show()
    
def NormalData(DataExp):
    Trows=len(DataExp);
    Tcols=len(DataExp[0]);
    V = [0 for i in range(Trows)];
    MRange = [[0 for i in range(2)] for j in range(Tcols)]; 
    DataNorm = [[0 for i in range(Tcols)] for j in range(Trows)]; 
    for c in range(Tcols):
        for r in range(Trows):
            V[r]=DataExp[r][c];
        (valmax,posmax)=maxp(V);
        (valmin,posmin)=minp(V)    
        for r in range(Trows):
            DataNorm[r][c] = normalizar(DataExp[r][c],valmin,valmax);
        MRange[c][0]=valmin;
        MRange[c][1]=valmax;
    return DataNorm, MRange

def scrambling(DataBase):
    Trows=len(DataBase);
    DataBaseS=DataBase;
    for i in range(Trows*10):
        pos1 = floor(random()*Trows);
        pos2 = floor(random()*Trows);
        temp = DataBaseS[pos1];
        DataBaseS[pos1] = DataBaseS[pos2];
        DataBaseS[pos2] = temp;
    return DataBaseS

def GenTrainVal(DataExp,percent):
    DataBaseS=scrambling(DataExp);
    Trows=len(DataBaseS);
    Tcols=len(DataBaseS[0]);
    DataTrain = [[0 for i in range(Tcols)] for j in range(Trows-floor(Trows*percent))]; 
    DataVal = [[0 for i in range(Tcols)] for j in range(floor(Trows*percent))]; 
    for dd in range(Trows-floor(Trows*percent)):
        DataTrain[dd]=DataBaseS[dd];
    for dd in range(Trows-floor(Trows*percent),Trows):
        DataVal[dd-(Trows-floor(Trows*percent))]=DataBaseS[dd];
    return DataTrain,DataVal
    
#RandomWeights
def RandomWeights(TINP, TMID, TOUT):
    m =  [[random()-0.5 for i in range(TINP)] for j in range(TMID)]
    o =  [[random()-0.5 for i in range(TMID)] for j in range(TOUT)]
    ma = [[random()-0.5 for i in range(TINP)] for j in range(TMID)]
    oa = [[random()-0.5 for i in range(TMID)] for j in range(TOUT)]    
    return m, ma, o, oa

def fa(x):
    if (x > 20):
        x=20;    
    if (x < -20):
        x=-20;
    x = exp(-x);
    return 1 / ( 1 + x );

#Forward block
def ForwardBKG(VI, m, o):
    TMID = len(m);
    TINP = len(m[0]);
    TOUT = len(o);
    neto = [0 for j in range(TOUT)];
    netm = [0 for j in range(TMID)];
    so = [0 for j in range(TOUT)];
    sm = [0 for j in range(TMID)];
    for y in range(TMID):
        for x in range(TINP):
                  netm[y] = netm[y] + m[y][x] * VI[x];
        sm[y] = fa(netm[y]);
    for z in range(TOUT):
         for y in range(TMID):
            neto[z] = neto[z] + o[z][y] * sm[y];      
         so[z] = fa(neto[z]);
    return sm, so, neto, netm

def fad(x):
     return fa(x)*(1 - fa(x))
       
#Backward block
def BackwardBKG(DO, netm, m, o, so, neto):
    TMID = len(m);
    TINP = len(m[0]);
    TOUT = len(o);
    eo = [0 for j in range(TOUT)];
    em = [0 for j in range(TMID)];
    sum1 = 0;
    for z in range(TOUT):
          eo[z] = (DO[z] - so[z])*fad(neto[z]);
    for y in range(TMID):
          sum1 = 0;
          for z in range(TOUT):
              sum1 = sum1 + eo[z]*o[z][y];
          em[y] = fad(netm[y])*sum1;
    return em,eo

#Bloque de aprendizaje
def  LearningBKG(VI, m, ma, sm, em, o, oa, eo, ETA, ALPHA):
    TMID=len(m);
    TINP=len(m[0]);
    TOUT=len(o);
    for z in range(TOUT):
         for y in range(TMID):
             o[z][y] = o[z][y] + ETA*eo[z]*sm[y] + ALPHA*oa[z][y];
             oa[z][y] = ETA*eo[z]*sm[y];
    
    for y in range(TMID):
        for x in range(TINP):
            m[y][x] = m[y][x] + ETA*em[y] * VI[x] + ALPHA*ma[y][x];
            ma[y][x] = ETA*em[y]*VI[x];
    return m, ma, o, oa  

#TrainingNNBK
def TrainingNNBKni10(NTEpochs,DataTrain):
    TData = len(DataTrain);
    Tcols = len(DataTrain[0]);
    TINP = Tcols;
    TMID = 5;
    TOUT = 1;
    ETA = 0.5;
    ALPHA = 0.125;
    #Random weights
    (m, ma, o, oa) = RandomWeights(TINP, TMID, TOUT);
    Errg = [0 for i in range (NTEpochs)];
    emin = 10000000000
    VI = [0 for i in range(TINP)];
    for epochs in range(NTEpochs):
         DataTrain=scrambling(DataTrain);
         etotal = 0;
         for data in range(TData):
             #Take first data
             for ii in range(TINP):
                 VI[ii] = DataTrain[data][ii];
             VI[TINP-1] = 1;
             DO = [DataTrain[data][Tcols-1]];
             (sm,so,neto,netm) = ForwardBKG(VI,m,o);
             (em,eo) =  BackwardBKG(DO, netm, m, o, so, neto);
             (m, ma, o, oa) = LearningBKG(VI, m, ma, sm, em, o, oa, eo, ETA, ALPHA);
             #Error gradient calculation
             etotal = eo[0]*eo[0] + etotal;
         errcm = 0.5 * sqrt(etotal)
         if errcm < emin:
             emin = errcm;
         Errg[epochs] = emin;
    return Errg,m,o;
 
#Nota: esto quiza deba quitarse del código
"""def SetDatabases():
    DataBrute = DatabaseRead()
    (DataNorm,MRange) = NormalData(DataBrute)
    (DataTrain,DataVal) = GenTrainVal(DataNorm,0.2)
    return DataTrain,DataVal"""

#ValidationNNBK
def ValidationNNBKni10(DataVal,m,o):
    TData=len(DataVal);
    Tcols=len(DataVal[0]);    
    TINP = len(m[0]) 
    TMID = len(m) 
    TOUT = 1 
    Ynn = [[0 for i in range(2)] for j in range(TData)];
    VI = [0 for i in range(TINP)];
    etotal = 0;
    for data in range(TData):
        #take input data
        for ii in range(TINP):
            VI[ii] = DataVal[data][ii];
        VI[TINP-1] = 1; #Bias
        DO=[DataVal[data][Tcols-1]];
        (sm,so,neto,netm)=ForwardBKG(VI,m,o);
        Ynn[data][0] = DO[0];
        Ynn[data][1] = so[0];
    return Ynn

#Main code
def Agente(X, DB, MRange):
    TR = len(DB)
    TC = len(DB[0])
    
    D = [-1 for j in range (TR)]
    V1 = [0 for j in range(TC_1)]
    V2 = [0 for j in range(TC-1)]
    Xn = [0 for j in range (TC-1)]
    for c in range(TC-1):
        Xn[c] = normalizar(X[c], MRange[c][0], MRange[c][1])
    for i in range(TR):
        for c in range(TC-1):
            V1[c] = Xn[c]
            V2[c] = DB[r][c]
        D[r] = calcD(V1, V2)
    (val, pos) = minp(D)
    R = DB[pos][TC-1]
    return R

#USENNBK
def usennbk(X,MRange,m,o):
    # R=usennbk(X,MRange,m,o);
    TINP = len(X); #bias included Input neurons
    Xn = [0 for i in range(TINP+1)] #plus bias
    Tcols = len(MRange);
    for i in range(TINP):
        Xn[i] = normalizar(X[i],MRange[i][0],MRange[i][1])
    Xn[TINP]=1

    (sm,so,neto,netm)=ForwardBKG(Xn,m,o)

    #Desnormalization     
    R = desnormalizar(so[0],MRange[TINP][0],MRange[TINP][1])
    return R

DataBrute = DatabaseRead();

(DataNorm, MRange) = NormalData(DataBrute)

DatabaseS = scrambling(DataNorm)

(DataTrain, DataVal) = GenTrainVal(DatabaseS, 0.1);

(Errg, m, o) = TrainingNNBKni10(5000, DataTrain);

plot(Errg)

Ynn = ValidationNNBKni10(DataVal,m,o);

#plot(Ynn)