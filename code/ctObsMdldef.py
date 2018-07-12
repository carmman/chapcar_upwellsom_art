# -*- coding: cp1252 -*-
import sys
import time as time
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm
#
TRIEDPY = "C:/Users/Charles/Documents/FAD/FAD_Charles/WORKZONE/Python3"
sys.path.append(TRIEDPY);
from   triedpy  import triedtools as tls; # triedpy rep should have been appended
from   localdef import *;                 # this may be true for others rep
#
#
#======================================================================
# Table(s) des Modèles
#Tmodels_rmean_ Tmodels_raverage_ (in _SV0, but caduc now)
Tmodels_anyall_OUT = np.array([     # Mettre de coté certains modèles; on peut
        ["Observations",    ""],    # aussi les déplacer dans un repertoire sst_OUT/     
]);
if 0
    Tmodels_anyall = np.array([
        ["bcc-csm1-1",      ""],    #( 3)
        ["bcc-csm1-1-m",    ""],    #( 4)
        #["BNU-ESM",         ""],        # non fourni
        ["CanCM4",          ""],    #( 5)
        ["CanESM2",         ""],    #( 6)
        ["CMCC-CESM",       ""],    #(13)
        ["CMCC-CM",         ""],    #(14)
        ["CMCC-CMS",        ""],    #(15) 
        ["CNRM-CM5",        ""],    #(17)
        ["CNRM-CM5-2",      ""],    #(16)
        ["ACCESS1-0",       ""],    #(01)
        ["ACCESS1-3",       ""],    #( 2)       
        ["CSIRO-Mk3-6-0",   ""],    #(18)
        ["inmcm4",          ""],    #(35)
        ["IPSL-CM5A-LR",    ""],    #(36)
        ["IPSL-CM5A-MR",    ""],    #(37)
        ["IPSL-CM5B-LR",    ""],    #(38)
        ["FGOALS-g2",       ""],    #(20)
        ["FGOALS-s2",       ""],    # AAMMfin=200412 (360, 25, 36) -> manque une année 2005
        ["MIROC-ESM",       ""],    #(40)
        ["MIROC-ESM-CHEM",  ""],    #(22)
        ["MIROC5",          ""],    #(39)
        ["HadCM3",          ""],    #(31)
        ["HadGEM2-CC",      ""],    #(33)
        ["HadGEM2-ES",      ""],    #(34)
        ["MPI-ESM-LR",      ""],    #(41)
        ["MPI-ESM-MR",      ""],    #(42)
        ["MPI-ESM-P",       ""],    #(43)   
        ["MRI-CGCM3",       ""],    #(44)
        ["MRI-ESM1",        ""],    #(45)   
        ["GISS-E2-H",       ""],    #(28)
        ["GISS-E2-H-CC",    ""],    #(27)
        ["GISS-E2-R",       ""],    #(30)
        ["GISS-E2-R-CC",    ""],    #(29)
        ["CCSM4",           ""],    #(07)
        ["NorESM1-M",       ""],    #(46)
        ["NorESM1-ME",      ""],    #(47)
        ["HadGEM2-AO",      ""],    #(32) 
        ["GFDL-CM2p1",      ""],    #(23)
        ["GFDL-CM3",        ""],    #(24)
        ["GFDL-ESM2G",      ""],    #(25)
        ["GFDL-ESM2M",      ""],    #(26)
        ["CESM1-BGC",       ""],    #( 8)
        ["CESM1-CAM5",      ""],    #(10)  
        ["CESM1-CAM5-1-FV2",""],    #( 9)        
        ["CESM1-FASTCHEM",  ""],    #(11)
        ["CESM1-WACCM",     ""],    #(12)
        #["FIO-ESM",         ""],    #(??)]);
        #["OBS",             ""],    #(??)    # par exemple.
        ]);
else :
    Tmodels_and_institut_anyall = np.array([
        ['BCC',          'bcc-csm1-1',       ''],   #( 3)
        ['BCC',          'bcc-csm1-1-m',     ''],   #( 4)
        #['BNU',          'BNU-ESM',          ''],   # non fourni
        ['CCCma',        'CanCM4',           ''],
        ['CCCma',        'CanESM2',          ''],
        ['CMCC',         'CMCC-CESM',        ''],
        ['CMCC',         'CMCC-CM',          ''],
        ['CMCC',         'CMCC-CMS',         ''],
        ['CNRM-CERFACS', 'CNRM-CM5',         ''],
        ['CNRM-CERFACS', 'CNRM-CM5-2',       ''],
        ['CSIRO-BOM',    'ACCESS1-0',        ''],
        ['CSIRO-BOM',    'ACCESS1-3',        ''],
        ['CSIRO-QCCCE',  'CSIRO-Mk3-6-0',    ''],
        ['INM',          'inmcm4',           ''],
        ['IPSL',         'IPSL-CM5A-LR',     ''],
        ['IPSL',         'IPSL-CM5A-MR',     ''],
        ['IPSL',         'IPSL-CM5B-LR',     ''],
        ['LASG-CESS',    'FGOALS-g2',        ''],
        ['LASG-IAP',     'FGOALS-s2',        ''],   # AAMMfin=200412 (360, 25, 36) -> manque une année 2005
        ['MIROC',        'MIROC-ESM',        ''],
        ['MIROC',        'MIROC-ESM-CHEM',   ''],
        ['MIROC',        'MIROC5',           ''],
        ['MOHC',         'HadCM3',           ''],
        ['MOHC',         'HadGEM2-CC',       ''],
        ['MOHC',         'HadGEM2-ES',       ''],
        ['MPI-M',        'MPI-ESM-LR',       ''],
        ['MPI-M',        'MPI-ESM-MR',       ''],
        ['MPI-M',        'MPI-ESM-P',        ''],
        ['MRI',          'MRI-CGCM3',        ''],
        ['MRI',          'MRI-ESM1',         ''],
        ['NASA-GISS',    'GISS-E2-H',        ''],
        ['NASA-GISS',    'GISS-E2-H-CC',     ''],
        ['NASA-GISS',    'GISS-E2-R',        ''],
        ['NASA-GISS',    'GISS-E2-R-CC',     ''],
        ['NCAR',         'CCSM4',            ''],
        ['NCC',          'NorESM1-M',        ''],
        ['NCC',          'NorESM1-ME',       ''],
        ['NIMR-KMA',     'HadGEM2-AO',       ''],
        ['NOAA-GFDL',    'GFDL-CM2p1',       ''],
        ['NOAA-GFDL',    'GFDL-CM3',         ''],
        ['NOAA-GFDL',    'GFDL-ESM2G',       ''],
        ['NOAA-GFDL',    'GFDL-ESM2M',       ''],
        ['NSF-DOE-NCAR', 'CESM1-BGC',        ''],
        ['NSF-DOE-NCAR', 'CESM1-CAM5',       ''],
        ['NSF-DOE-NCAR', 'CESM1-CAM5-1-FV2', ''],
        ['NSF-DOE-NCAR', 'CESM1-FASTCHEM',   ''],
        ['NSF-DOE-NCAR', 'CESM1-WACCM',      ''],
        #['FIO',          'FIO-ESM',          ''],    # pas de donnees 'tos', en 'historical', uniquement 'so', mais a des donnees en scenarios ...
        #['',             'OBS',              ''],    #(??)    # par exemple.
    ]);
    Tinstitut_anyall = Tmodels_and_institut_anyall[:,0]
    Tmodels_anyall = Tmodels_and_institut_anyall[:,(1,2)]
#----------------------------------------------------------------------
def pentes(X) : # Courbes des pentes (b1) par pîxel
    N,L,C = np.shape(X);
    tps   = np.arange(N)+1;
    Tb1   = []
    plt.figure();
    for i in np.arange(C) :
        for j in np.arange(L) :
            y = X[:,j,i]
            b0,b1,s,R2,sigb0,sigb1= tls.linreg(tps,y);
            Tb1 = np.append(Tb1, b1)  #print(b0, b1)
    plt.plot(Tb1); plt.axis('tight');
    
def trendless(X) : # Suppression de la tendance
    N,L,C = np.shape(X);
    tps   = np.arange(N)+1;
    X_ = np.empty(np.shape(X))
    for i in np.arange(C) :
        for j in np.arange(L) :
            y = X[:,j,i]
            b0,b1,s,R2,sigb0,sigb1= tls.linreg(tps,y);
            ycor = y - b1*tps;
            X_[:,j,i] = ycor
    return X_

def anomalies(X) :
    N,L,C = np.shape(X);
    Npix  = L*C;
    X_    = np.reshape(X, (N,Npix)); 
    for i in np.arange(Npix) :
        for j in np.arange(0,N,12) :  # 0,   12,   24,   36, ...
            moypiaj = np.mean(X_[j:j+12, i]); # moyenne du pixel i année j
            X_[j:j+12, i] = X_[j:j+12, i] - moypiaj
    X_ = np.reshape(X_, (N,L,C));
    return X_

def Dpixmoymens(data,visu=None, climato=None, douze=12, stat=None) :
    global vvmin, vvmax
    #data may include nan
    #
    Ndata, Ldata, Cdata = np.shape(data)
    if Ndata%douze != 0 :
        print("Le nombre de données doit etre un multiple de %d"%douze);
    #
    # On réorganise les données par pixel (Mise à plat des données)
    Npix = Ldata*Cdata; # 
    Data = np.reshape(data, (Ndata,Npix));
    #
    if climato==None : # Calcul des moyennes mensuelles par pixels
        Data_mmoy = np.zeros((Npix,douze));
        for m in np.arange(douze) :            # Pour chaque mois m
            imois = np.arange(m,Ndata,douze);  # les indices d'un mois m
            if stat == None : # moyenne par défaut
                for i in np.arange(Npix) :         # Pour chaque pixel
                    Data_mmoy[i,m] = np.mean(Data[imois,i]);    # on calcule la moyenne du mois
            elif stat == 'std' : 
                for i in np.arange(Npix) :         # Pour chaque pixel
                    Data_mmoy[i,m] = np.std(Data[imois,i]);    # on calcule l'écart type du mois
            else :
                print("Dpixmoyens : check your stat statement")
                plt.show(); sys.exit(0);
    #
    elif climato=="GRAD" : # Calcul des gradients mensuelles par pixels
        tm = np.arange(Ndata/douze)+1;
        Data_mmoy = np.zeros((Npix,douze));
        for m in np.arange(douze) :            # Pour chaque mois m
            imois = np.arange(m,Ndata,douze);  # les indices d'un mois m     
            for i in np.arange(Npix) :         # Pour chaque pixel
                y = Data[imois,i];
                b0,b1,s,R2,sigb0,sigb1= tls.linreg(tm,y);    # on calcule la régression
                Data_mmoy[i,m] = b1;
    else :
        print("Dpixmoymens : choose a good climato");
        sys.exit(0)

    if visu is not None : # Une visu (2D) pour voir si ca ressemble à quelque chose.
        if climato=="GRAD" :
            vmin=np.nanmin(Data_mmoy);
            vmax=np.nanmax(Data_mmoy);
        else :
            vmin = vvmin; vmax = vvmax;
        showimgdata(Data_mmoy.T.reshape(douze,1,Ldata,Cdata),n=douze,fr=0,Labels=varnames,
                    cmap=cm.gist_ncar,interp='none',
                    figsize=(12, 9), wspace=0.0, hspace=0.0,
                    vmin=vmin,vmax=vmax);
                    #vmin=vvmin,vmax=vvmax);
                    #vmin=12.,vmax=32.);
                    #vmin=np.nanmin(Data),vmax=np.nanmax(Data));
        plt.suptitle("Dpixmoymens: visu to check : %s \nmin=%f, max=%f"
                    %(visu,np.nanmin(Data_mmoy),np.nanmax(Data_mmoy)));
                    #%(visu,np.nanmin(Data),np.nanmax(Data)));
    #
    # On vire les pixels qui sont nan (données manquantes)
    Data = np.reshape(Data_mmoy,(Npix*douze)); 
    Inan = np.where(np.isnan(Data))[0];  # Is Not a Number
    Iisn = np.where(~np.isnan(Data))[0]; # IS a Number
    Data = Data[Iisn]
    Data = Data.reshape(int(len(Iisn)/douze),douze);
    return Data, Iisn, Inan

def aff2D(XD,L,C,isnum,isnan,varnames=None,wvmin=None,wvmax=None,figsize=(9,9)) :
    ND,p      = np.shape(XD);
    X_        = np.empty((L*C,p));   
    X_[isnum] = XD   
    X_[isnan] = np.nan
    showimgdata(X_.T.reshape(p,1,L,C),n=p,fr=0,Labels=varnames,interp='none',
                cmap=cm.gist_ncar,figsize=figsize, wspace=0.0, hspace=0.1,
                vmin=wvmin,vmax=wvmax);
    
def refbmusD(sm, bmus, Lig, Col, Iisn, Inan) :
    Ndata = len(bmus);
    #obs_bmus  = bmus2O[:,0];
    Dbmus    = sm.codebook[bmus,]; 
    X_       = np.empty(Lig*Col*12);  
    X_[Iisn] = Dbmus.reshape(Ndata*12);
    X_[Inan] = np.nan;
    X_       =  X_.reshape(Lig*Col,12); 
    showimgdata(X_.T.reshape(12,1,Lig,Col),n=12,fr=0,Labels=varnames,
                cmap=cm.gist_ncar,interp='none',
                figsize=(12, 9), wspace=0.0, hspace=0.0,
                vmin=wvmin,vmax=wvmax);
    return X_

def moybmusD(X_,Lig,Col): # Visu des moyennes des pixels
    moyX_ = np.nanmean(X_,axis=1)
    plt.figure()
    plt.imshow(moyX_.reshape(Lig,Col), cmap=cm.gist_ncar,interpolation='none', 
               vmin=wvmin,vmax=wvmax);

def classgeo(X, classe_X, nanval=np.nan) :
    N,L,C = np.shape(X);  
    X_    = np.empty((L*C)); #.astype(int);
    axi   = X[0].reshape(L*C)
    iko   = np.where(np.isnan(axi))[0];
    iok   = np.where(~np.isnan(axi))[0];
    X_[iok] = classe_X;
    #X_[iko] = np.nan;
    X_[iko] = nanval;
    X_    =  np.reshape(X_,(L,C));
    return X_;

def dto2d(X1D,L,C,isnum,missval=np.nan) :
    # Pour les classes par exemple, passer de 1D (cad N pixels
    # valides en ligne à une image 2D (LxC) des classes)
    # ps : devrait remplacer classgeo)
    X = np.ones(L*C)*missval;
    X[isnum] = X1D;
    X = np.reshape(X,(L,C));
    return X

def indsc(X, C=20) :
    # Climatologie glissante par une année
    # X : N images-mensuelles de dimension py, px
    # C : Nombre d'années pour une Climatologie
    N, py, px = np.shape(X);
    P = py*px;
    X = X.reshape(N,py*px);  
    K = 1 + (N-C*12) / 12; # Nombre de climatologies
    K = int(K); # because python 2.7 et 3.4 c'est pas pareil
    IndSC = np.zeros((K,P)); # Init
    for k in np.arange(K) :
        MoyCp = np.zeros((12,P));   # Init Moyenne mensuelle sur les C années pour les pixels i
        for m in np.arange(12) :
            Im = np.arange(m,12*C,12);                      #(1)
            Im = Im + 12*k;         # print(Im)
            for i in np.arange(P) : # Pour chaque pixel
                MoyCp[m,i] = np.mean(X[Im,i])               #(2) SST20[x,y,12]
        #tprin(MoyCp, " %.2f ");
        #
        # Now, Calcul de IndSC    
        for i in np.arange(P) : # Pour chaque pixel
            IndSC[k,i] =  np.max(MoyCp[:,i]) - np.min(MoyCp[:,i])   
    IndSC = IndSC.reshape(K,py,px);
    return IndSC
#----------------------------------------------------------------------
def inan(X) :
    # On fait l'hypothèse que tous les nans d'une image à l'autre sont
    # à la même place
    L,C   = np.shape(X); #print("X.shape : ", N, L, C);
    X_    = np.reshape(X, L*C); 
    Inan  = np.where(np.isnan(X_))[0];
    return Inan
def imaxnan() :
    X_       = np.load("Datas/sst_obs_1854a2005_Y60X315.npy");
    Inan     = inan(X_[0])
    X_       = np.load("Datas/CM5A-LR_rm_1854a2005_25L36C.npy")
    Inan_mdl = inan(X_[0])
    Inan     = np.unique(np.concatenate((Inan,Inan_mdl)))
    X_       = np.load("Datas/CM5A-MR_rm_1854a2005_25L36C.npy")
    Inan_mdl = inan(X_[0])
    Inan     = np.unique(np.concatenate((Inan,Inan_mdl)))
    X_       = np.load("Datas/GISS-E2-R_rm_1854a2005_25L36C.npy")
    Inan_mdl = inan(X_[0])
    Inan     = np.unique(np.concatenate((Inan,Inan_mdl)))
    return Inan;
#----------------------------------------------------------------------
def nan2moy (X, Left=True, Above=True, Right=True, Below=True) :
    # On supose que quelle que soit l'image de X tous les nan son calé 
    # sur les memes pixels; on a choisi de s'appuyer sur l'image 0
    # (pompé de C:\...\DonneesUpwelling\Upwell2_predictions\sst_nc_X.py)
    Nim, Lim, Cim = np.shape(X);
    ctrnan = 0;
    for L in np.arange(Lim) :             #Lim=25:=> [0, 1, ..., 23, 24]
        for C in np.arange(Cim) :         #Cim=36:=> [0, 1, ..., 34, 35]
    #for C in np.arange(Cim) :             #Cim=36:=> [0, 1, ..., 34, 35]
    #    for L in np.arange(Lim) :         #Lim=25:=> [0, 1, ..., 23, 24]
    #for L in np.arange(Lim-1,-1,-1) :     #Lim=25:=> [24, 23, ..., 1, 0]
    #    for C in np.arange(Cim-1,-1,-1) : #Cim=36:=> [35, 34, ..., 1, 0]
    #for C in np.arange(Cim-1,-1,-1) :     #Cim=36:=> [35, 34, ..., 1, 0]
    #    for L in np.arange(Lim-1,-1,-1) : #Lim=25:=> [24, 23, ..., 1, 0]
    #for L in np.arange(Lim) :             #Lim=25:=> [0, 1, ..., 23, 24]
    #    for C in np.arange(Cim-1,-1,-1) : #Cim=36:=> [35, 34, ..., 1, 0]
    #for C in np.arange(Cim-1,-1,-1) :     #Cim=36:=> [35, 34, ..., 1, 0]
    #    for L in np.arange(Lim) :         #Lim=25:=> [0, 1, ..., 23, 24]
    #for L in np.arange(Lim-1,-1,-1) :     #Lim=25:=> [24, 23, ..., 1, 0]
    #    for C in np.arange(Cim) :         #Cim=36:=> [0, 1, ..., 34, 35]
    #for C in np.arange(Cim) :              #Cim=36:=> [0, 1, ..., 34, 35]
    #    for L in np.arange(Lim-1,-1,-1) :  #Lim=25:=> [24, 23, ..., 1, 0]
    #      
            if np.isnan(X[0,L,C]) :   # tous calé sur les memes pixels
                ctrnan = ctrnan+1;
                nok = 0; som=np.zeros(Nim)
                if Left and C > 0 :
                    if ~np.isnan(X[0,L,  C-1]) :  # carré de gauche
                        som = som + X[:,L,  C-1]; nok = nok + 1;
                if Above and L > 0 :
                    if ~np.isnan(X[0,L-1,C]) :    # carré du dessus
                        som = som + X[:,L-1,C  ]; nok = nok + 1;
                if Right and C+1 < Cim :
                    if ~np.isnan(X[0,L,  C+1]) :  # carré de droite
                        som = som + X[:,L,  C+1]; nok = nok + 1;
                if Below and L+1 < Lim :
                    if ~np.isnan(X[0,L+1,C]) :    # carré du dessous
                        som = som + X[:,L+1,C  ]; nok = nok + 1;
                if nok > 0 :
                    X[:,L,C] = som/nok;
    #print("nombre de nan de chaque image : %d" %(ctrnan));
    return X, ctrnan;
#======================================================================
#----------------------------------------------------------------------
def grid() :
    axes = plt.axis(); # (-0.5, 35.5, 24.5, -0.5)
    for i in np.arange(axes[0], axes[1], 1) :
        for j in np.arange(axes[3], axes[2], 1) :
            plt.plot([axes[0], axes[1]],[j, j],'k-',linewidth=0.5);
            plt.plot([i, i],[axes[2], axes[3]],'k-',linewidth=0.5);
    plt.axis('tight');
def transcoclass_algo2(classe_Dobs, classe_Dmdl, nb_class) :
    # Tentative d'harminisation des N° de classe (en attribuant
    # la même classe en fonction du nombre de pixels communs) 
    klasse_Dmdl = np.zeros(len(classe_Dobs));
    TO=[]; TM=[]; # ca va etre les tableaux des indices de classe
    TO_cptr  = np.zeros(nb_class); # compteur des elt par classe
    TM_cptr  = np.zeros(nb_class);
    for i in np.arange(nb_class) :
        ii = np.where(classe_Dobs==i+1)[0];
        TO.append(ii);
        TO_cptr[i] = len(ii);
        ii = np.where(classe_Dmdl==i+1)[0];
        TM.append(ii);
        TM_cptr[i] = len(ii);
    for i in np.arange(nb_class) :
        IOmax = np.argmax(TO_cptr);
        bestj = -1; bestlen = 0;
        for j in np.arange(nb_class) :
            ij = np.intersect1d(TO[IOmax], TM[j]);
            if len(ij) > bestlen :
                bestlen = len(ij);
                bestj = j;
        if bestj > -1 :
            klasse_Dmdl[TM[bestj]] = IOmax+1;
            TM[bestj] = [];
            TM_cptr[bestj]=0;                  
        else : # par défaut je prend la max de TM restante, sans justification ?
            IMmax = np.argmax(TM_cptr);
            klasse_Dmdl[TM[IMmax]] = IOmax+1;
            TM[IMmax] = [];
            TM_cptr[IMmax]=0;             
        TO_cptr[IOmax]=0;
    return klasse_Dmdl
def transcoclass_algo1(classe_Dobs, classe_Dmdl, nb_class) :
    # Tentative d'harminisation des N° de classe (en attribuant
    # la même classe en fonction de l'importance de la classe)
    klasse_Dmdl = np.zeros(len(classe_Dobs));
    TO_cptr  = np.zeros(nb_class); # compteur des elt par classe
    TM_cptr  = np.zeros(nb_class);
    for i in np.arange(nb_class) :
        ii = np.where(classe_Dobs==i+1)[0];
        TO_cptr[i] = len(ii);
        ii = np.where(classe_Dmdl==i+1)[0];
        TM_cptr[i] = len(ii);
    for i in np.arange(nb_class) :
        IOmax = np.argmax(TO_cptr);
        IMmax = np.argmax(TM_cptr);
        ii = np.where(classe_Dmdl==IMmax+1)[0];
        klasse_Dmdl[ii] = IOmax+1;
        TO_cptr[IOmax]=0;
        TM_cptr[IMmax]=0;
    return klasse_Dmdl
#----------------------------------------------------------------------
def transco_class(class_ref,codebook,crit='') :
    nb_class = max(class_ref)
    if isinstance(crit, str) :
        Tvalcrit = np.zeros(nb_class); 
        for c in np.arange(nb_class) :
            Ic = np.where(class_ref==c+1)[0];
            if crit=='GAP' :
                Tvalcrit[c] = np.max(codebook[Ic])-np.min(codebook[Ic]);
                #print(c+1, np.max(codebook[Ic]),np.min(codebook[Ic]),np.max(codebook[Ic])-np.min(codebook[Ic]))
            elif crit=='GAPNM' : # NormMax par curiosité pour voir
                Tvalcrit[c] = (np.max(codebook[Ic])-np.min(codebook[Ic])) / np.max(codebook[Ic]);
            elif crit=='STD' :
                Tvalcrit[c] = np.std(codebook[Ic]);
            elif crit=='MOY' : # does not ok for anomalie ?
                Tvalcrit[c] = np.mean(codebook[Ic]);
            elif crit=='MAX' : # does not ok for anomalie ?
                Tvalcrit[c] = np.max(codebook[Ic]);
            elif crit=='MIN' : # does not ok for anomalie ?
                Tvalcrit[c] = np.min(codebook[Ic]);
            elif crit=='GRAD' : # does not ok for anomalie ?
                tm = np.arange(12)+1;  #!!! 12 en dure
                sompente = 0.0
                for p in np.arange(len(Ic)) :
                    y = codebook[Ic[p]]
                    b0,b1,s,R2,sigb0,sigb1= tls.linreg(tm,y);
                    sompente = sompente + b1;
                Tvalcrit[c] = sompente / len(Ic); # la moyenne des pentes sur les CB de cette classe 
            else :
                print("transco_class : bad criterium, should be one of this : \
                       'GAP', 'GAPNM', 'STD', 'MOY', 'MAX', 'MIN', 'GRAD'; found %s"%crit);
                sys.exit(0);
        Icnew = np.argsort(Tvalcrit);
    else :
        Icnew = np.array(crit)-1;
    #    
    cref_new = np.zeros(len(class_ref)).astype(int);               
    cc = 1;
    for c in Icnew :
        Ic = np.where(class_ref==c+1)[0];
        cref_new[Ic] = cc;
        cc = cc+1; 
    return cref_new
def moymensclass (varN2D,isnum,classe_D,nb_class) :
    # Moyennes mensuelles par classe
    N,L,C   = np.shape(varN2D);
    #!!!X   = np.reshape(sst_obs,(N,L*C))
    X       = np.reshape(varN2D,(N,L*C))
    MoyMens = np.zeros((12,L*C))
    for m in np.arange(12) :
        Im = np.arange(m,N,12);
        Dm = X[Im];
        MoyMens[m] = np.mean(Dm, axis=0);
    MoyMens = MoyMens[:,isnum]
    #        
    Tmoymensclass = np.zeros((12,nb_class))
    for c in np.arange(nb_class) :
        Ic = np.where(classe_D==c+1)[0];
        Tmoymensclass[:,c] = np.mean(MoyMens[:,Ic], axis=1)
    return Tmoymensclass
def deltalargeN2D(X) :
    # Différence de sst par rapport à la valeur au large.
    X_ = np.copy(X);
    N, L, C = np.shape(X_);
    for i in np.arange(N) :
        for l in np.arange(L) : 
            for c in np.arange(C-1,-1,-1) : # [35, 34, ..., 1, 0]
                X_[i,l,c] = X_[i,l,c] - X_[i,l,0];
    return X_

def deltalargeM12(X,L,C,isnum) :
    # Différence de sst par rapport à la valeur au large.
    # (Doit redonner le même résultat que old, ce n'est
    # qu'une question d'optimisation)
    M, m = np.shape(X);    # (743, 12)
    Xm_  = np.zeros((L*C, m))
    Xm_[isnum,:] = X;
    Xm_  = Xm_.reshape(L,C, m);
    for l in np.arange(L) : 
        for c in np.arange(C-1,-1,-1) : # [35, 34, ..., 1, 0]
            Xm_[l,c,:] = Xm_[l,c,:] - Xm_[l,0,:];
    Xm_ = Xm_.reshape(L*C,m);
    return Xm_[isnum,:]
#======================================================================
def perfbyclass (classe_Dobs,classe_Dmdl,nb_class) :
    NDobs = len(classe_Dobs);
    Tperf = [];
    classe_DD = np.ones(NDobs)*np.nan; 
    for c in np.arange(nb_class)+1 :      
        iobsc = np.where(classe_Dobs==c)[0]; # Indices des classes c des obs
        imdlc = np.where(classe_Dmdl==c)[0]; # Indices des classes c des mdl
        #igood= np.where(imdlc==iobsc)[0];   # np.where => même dim
        igood = np.intersect1d(imdlc,iobsc);
        classe_DD[igood] = c;
        #       
        Niobsc=len(iobsc); Nigood=len(igood);
        if Niobsc>0 : # because avec red_class... on peut avoir écarté une classe
            perfc = Nigood/Niobsc; 
        else :
            perfc = 0.0; # ... à voir ... 
        Tperf.append(perfc)
    return classe_DD, Tperf
#
def red_classgeo(X,isnum,classe_D,frl,tol,frc,toc) :
    # Doit retourner les classes sur une zone réduite en faisant
    # attention aux nans
    N,L,C = np.shape(X);  
    XC_   = np.ones(L*C)*np.nan;
    XC_[isnum] = classe_D;
    XC_   = XC_.reshape(L,C)
    XC_   = XC_[frl:tol,frc:toc];
    l,c   = np.shape(XC_); 
    oC_   = XC_.reshape(l*c)
    isnum_red   = np.where(~np.isnan(oC_))[0]
    oC_   = oC_[isnum_red]
    X_     = X[:,frl:tol,frc:toc];
    return X_, XC_, oC_, isnum_red;
#
#======================================================================
def Dmdlmoy4CT (TDmdl4CT,igroup,pond=None) :
    # Modèle Moyen d'un groupe\cluster des données 4CT
    # Si pond, il doit avoir la meme longueur que TDmdl4CT
    if pond is None : # Non pondéré
        CmdlMoy = np.mean(TDmdl4CT[igroup],axis=0); # Dmdl moyen d'un cluster
    else : # Modèle Moyen Pondéré
        pond       = pond[igroup]; # dans igroup y'a pas l'indice qui correspond aux Obs
        TDmdl4CTi_ = TDmdl4CT[igroup];        # (11,743,12)
        CmdlMoy    = TDmdl4CTi_[0] * pond[0]; # init du modele moyen
        for kk in np.arange(len(pond)-1)+1 :
            CmdlMoy = CmdlMoy + (TDmdl4CTi_[kk] * pond[kk])
        CmdlMoy    = CmdlMoy / np.sum(pond);
    return CmdlMoy; # Cluster modèle Moyen
#
#{{{{{{{{{{{{{{{{{{{{{{{{{{{
import UW3_triedctk   as ctk
#
from   ParamCas   import *
douze = 12; #KKKKKK
#}}}}}}}}}}}}}}}}}}}}}}}}}}}
#
def datacodification4CT_old(data) :
    # Codification des données to CT
    Ndata, Ldata, Cdata = np.shape(data);
    #
    #if INDSC : # Indicateur de Saisonalité Climatologique
    #
    if TRENDLESS : # Suppression de la tendance pixel par pixel
        data = trendless(data);
    #
    if WITHANO :    # Anomalies : supression moy mens annuelle par pixel
        if 1 :
            data = anomalies(data)
        elif 0 : # Apres Trendless, ca revient quasi au meme de Centrer
            X_ = data.reshape(Ndata,Ldata*Cdata);
            X_ = tls.centree(X_)
            data = X_.reshape(Ndata,Ldata,Cdata)
            del X_;
    #
    #if 0 : # Pour le brute, je tenterais bien un normalisation entre 0 et 1 !?
    #    mindata = np.nanmin(data);  maxdata = np.nanmax(data);
    #    data = data - mindata / (maxdata - mindata)
    #
    #-----
    #if UISST == "before" :  data = deltalargeN2D(data);
    #-----
    # Climatologie: Moyennes mensuelles par pixel
    Ddata, Iisn_data, Inan_data  = Dpixmoymens(data,climato=climato,douze=douze);
    #-----
    if UISST : # == "after" :
        #Ddata = deltalargeM12old(Ddata,Ldata,Cdata,isnumobs);
        Ddata = deltalargeM12(Ddata,Ldata,Cdata,isnumobs);
    #----
    NDdata = len(Ddata);
    #----
    # Transfo Après mise sous forme de pixels Moyens Mensuels
    if NORMMAX == True :
        Maxi = np.max(Ddata, axis=1);
        Ddata = (Ddata.T / Maxi).T;
    #if NORMMAX == 2 :
    if CENTRED :
        Ddata  = tls.centred(Ddata,biais=0); # mais en fait ...
    #----
    return data, Ddata, NDdata;

def datacodif4CT_beforeclimato(data) :
    # Codification des données 4CT avant la climatologie
    Ndata, Ldata, Cdata = np.shape(data);
    #
    #if INDSC : # Indicateur de Saisonalité Climatologique
    #
    if TRENDLESS : # Suppression de la tendance pixel par pixel
        data = trendless(data);
    #
    if WITHANO :    # Anomalies : supression moy mens annuelle par pixel
        if 1 :
            data = anomalies(data)
        elif 0 : # Apres Trendless, ca revient quasi au meme de Centrer
            X_ = data.reshape(Ndata,Ldata*Cdata);
            X_ = tls.centree(X_)
            data = X_.reshape(Ndata,Ldata,Cdata)
            del X_;
    #
    #if 0 : # Pour le brute, je tenterais bien un normalisation entre 0 et 1 !?
    #    mindata = np.nanmin(data);  maxdata = np.nanmax(data);
    #    data = data - mindata / (maxdata - mindata)
    #
    #-----
    #if UISST == "before" :  data = deltalargeN2D(data);
    #-----
    return data
def datacodification4CT(data) :
    # Codification des données to CT
    #-----
    # Codification des données 4CT avant la climatologie
    data = datacodif4CT_beforeclimato(data)
    #-----
    # Climatologie: Moyennes mensuelles par pixel
    Ddata, Iisn_data, Inan_data  = Dpixmoymens(data,climato=climato,douze=douze);
    #-----
    if UISST : # == "after" :
        Ddata = deltalargeM12(Ddata,Ldata,Cdata,isnumobs);
    #----
    NDdata = len(Ddata);
    #----
    # Transfo Après mise sous forme de pixels Moyens Mensuels
    if NORMMAX == True :
        Maxi = np.max(Ddata, axis=1);
        Ddata = (Ddata.T / Maxi).T;
    #if NORMMAX == 2 :
    if CENTRED :
        Ddata  = tls.centred(Ddata,biais=0); # mais en fait ...
    #----
    return data, Ddata, NDdata;
#
#======================================================================
def myki2(M) :
    Msquare = M**2
    sumlig = np.sum(M, axis=1);
    sumcol = np.sum(M, axis=0);
    Nstar  = np.outer(sumlig,sumcol);
#?  ki2    = np.sum(Msquare / Nstar);
    ki2    = np.nansum(Msquare / Nstar);
    ki2    = np.sum(M)*(ki2 -1); #print(ki2);
    return ki2;
#----------------------------------------------------------------------
#TypePerf = ["GlobalAccuracy","MeanClassAccuracy","Index2Rand","SpearmanCorr","ContengencyCoef"];
def perfglobales(TypePerf, classe_Dobs, classe_Dmdl, nb_class) :
    # GlobalAccuracy  : Nombre de pixels  biens classés / Nombre de pixels total (kperf=1)
    # MeanlAccuracy   : Poucentage moyen des biens classés par classe (kperf=2)
    # Index2Rand      : Index de Monsieur Rand
    # SpearmanCorr    : Index de corrélation de Spearman
    # ContengencyCoef : chi2
    NDobs     = len(classe_Dobs);
    Tperfglob = [];
    #
    for i in np.arange(len(TypePerf)) :
        
        if TypePerf[i] == "GlobalAccuracy" : #(kperf=1)
            # Tous les pixels biens classés / nombre de pixels total
            perfglob = len(np.where(classe_Dmdl==classe_Dobs)[0])/NDobs;

        elif TypePerf[i] == "MeanClassAccuracy" : #(kperf=2)
            Tperfc = [];
            for c in np.arange(nb_class)+1 :      
                iobsc = np.where(classe_Dobs==c)[0]; # Indices des classes c des obs
                imdlc = np.where(classe_Dmdl==c)[0]; # Indices des classes c des mdl
                #igood= np.where(imdlc==iobsc)[0];   # np.where => même dim
                igood = np.intersect1d(imdlc,iobsc);
                #       
                Niobsc=len(iobsc); Nigood=len(igood);
                #perfc = Nigood/Niobsc; #
                if Niobsc>0 : # because avec red_class... on peut avoir écarté une classe
                    perfc = Nigood/Niobsc; # erratum: perfc = Nigood/Nimdlc;
                else :
                    perfc = 0.0; # ... à voir ... 
                Tperfc.append(perfc)
            perfglob = np.mean(Tperfc);

        elif TypePerf[i] == "Index2Rand" :
            #from sklearn.metrics.cluster import adjusted_rand_score
            perfglob = adjusted_rand_score(classe_Dobs, classe_Dmdl);   

        elif TypePerf[i] == "SpearmanCorr" :
            #from scipy.stats import spearmanr
            SpearCorr = spearmanr(classe_Dobs, classe_Dmdl)
            perfglob  = SpearCorr[0];

        elif TypePerf[i] == "ContengencyCoef"  :
            #from scipy.stats import chi2_contingency
            # CHI2 or CC or MCC sur la matrice de confusion.
            pvalue = 0; #chi2 = 0; # Init ...
            MC = tls.matconf(classe_Dobs, classe_Dmdl,visu=False)
            MC = MC[1:nb_class+1,1:nb_class+1]; # On enlève la classe nulle
            if 1 : # Suppression des colonnes = 0; est-ce licite !!!???
                Smarg = np.sum(MC,axis=0);  # sum marginale en colonne
                Inot0 = np.where(Smarg!=0)[0];
                MC    = MC[:,Inot0];
                #MC   = MC*10;  # *1000 ou *100 ou *10 -> =~1 qqs Mdl
                #MC   = MC*1.5; # *1.5  ou 2 : la forme se retrouve mais sur un intervalle + petit
            elif 0 : # Autre alternative, ajouter 1 partout pour etre sur de
                   # ne pas avoir 0 ...
                MC    = MC + 1; # Ca redonne la meme chose
            elif 0 : # Mettre 1 que pour la diagonale de la classe nulle
                Smarg = np.sum(MC,axis=0);  # sum marginale en colonne
                Iis0 = np.where(Smarg==0)[0];
                for i in Iis0 :
                    MC[i,i]=1;  # Ca redonne la meme chose
            if 0 : # PLM #bok
                # Suppression des ligne = 0; est-ce licite !!!???
                # (dans le cas 'All') ca ne devrait pas être utile, car les lignes
                # correspondent aux classes des Obs qui ont toujours des effectifs > 0
                # par définition. Mais dans le cas 'RED', je ne sais plus comment
                # ca se présente, et ce sera à ré-étudier si besoin ...
                Smarg = np.sum(MC,axis=1);  # sum marginale en ligne
                Inot0 = np.where(Smarg!=0)[0];
                MC    = MC[Inot0,:];
            try :
                chi2, pvalue, degrees, expected = chi2_contingency(MC); #print("chi2_");
            except :
                try :
                    chi2 = myki2(MC); 
                except :
                    chi2 = 0; 
            if 0 : # Le CHI2 itself
                perfglob = chi2;
            else : # ou plutot, le Coefficient de Contingence
                CC  = np.sqrt( chi2 / (NDobs+chi2));    perfglob = CC;
                #MCC= np.sqrt( chi2 / NDobs);           perfglob = MCC;
                #Tchi2_pvalue.append(pvalue);
        #
        Tperfglob.append(perfglob);
    #
    return Tperfglob;
#
#----------------------------------------------------------------------
