# -*- coding: cp1252 -*-
import sys
import time as time
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm
from   scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.io
from   scipy.stats import spearmanr, chi2_contingency
from   sklearn.metrics.cluster import adjusted_rand_score
#
from   triedpy import triedtools as tls
from   triedpy import triedsompy as SOM
from   triedpy import triedacp   as acp
#from  triedpy import triedctk   as ctk
import UW3_triedctk   as ctk
#
from   localdef    import *
from   ctObsMdldef import *
#
#%=====================================================================
def afcnuage (CP,cpa,cpb,Xcol,K,xoomK=500,linewidths=1,indname=None,
              cmap=cm.jet,holdon=False) :
# pompé de WORKZONE ... TPA05
    if holdon == False :
        # j'ai un pb obs \ pas obs qui apparaissent dans la même couleur que le dernier cluster
        # quand bien même il ne participe pas à la clusterisation
        lenCP = len(CP); lenXcol = len(Xcol);
        if lenCP > lenXcol : # hyp : lenCP>lenXcol
            # Je considère que les (LE) surnuméraire de CP sont les obs (ou aut chose), je l'enlève,
            # et le met de coté
            CPobs = CP[lenXcol:lenCP,:];
            CP    = CP[0:lenXcol,:];
            # K et indname suivent CP
            if np.ndim(K) == 1 :
                K = K.reshape(len(K),1)
            Kobs  = K[lenXcol:lenCP,:];
            K     = K[0:lenXcol,:];
            obsname = indname[lenXcol:lenCP];
            indname = indname[0:lenXcol];
        #
        plt.figure();
        my_norm = plt.Normalize()
        my_normed_data = my_norm(Xcol)
        ec_colors = cmap(my_normed_data) # a Nx4 array of rgba value
        #? if np.ndim(K) > 1 : # On distingue triangle à droite ou vers le haut selon l'axe
        n,p = np.shape(K);
        if p > 1 : # On distingue triangle à droite ou vers le haut selon l'axe 
            plt.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K[:,cpa-1]*xoomK,
                            marker='>',edgecolors=ec_colors,facecolor='none',linewidths=linewidths)
            plt.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K[:,cpb-1]*xoomK,
                            marker='^',edgecolors=ec_colors,facecolor='none',linewidths=linewidths)
            if lenCP > lenXcol : # cas des surnumeraire, en principe les obs
                plt.scatter(CPobs[:,cpa-1],CPobs[:,cpb-1],s=Kobs[:,cpa-1]*xoomK,
                                marker='>',edgecolors='k',facecolor='none',linewidths=linewidths)
                plt.scatter(CPobs[:,cpa-1],CPobs[:,cpb-1],s=Kobs[:,cpb-1]*xoomK,
                                marker='^',edgecolors='k',facecolor='none',linewidths=linewidths)            
        else :
            plt.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K*xoomK,
                            marker='s',edgecolors=ec_colors,facecolor='none',linewidths=linewidths);
            if lenCP > lenXcol : # ? cas des surnumeraire, en principe les obs
                plt.scatter(CPobs[:,cpa-1],CPobs[:,cpb-1],s=Kobs*xoomK,
                            marker='s',edgecolors='k',facecolor='none',linewidths=linewidths);
                
    else : #(c'est pour les colonnes)
        plt.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K[:,cpa-1]*xoomK,
                        marker='o',facecolor='m')
        plt.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K[:,cpb-1]*xoomK,
                        marker='o',facecolor='c',alpha=0.5)
    #plt.axis('tight')
    plt.xlabel('axe %d'%cpa); plt.ylabel('axe %d'%cpb)
    
    if 0 : # je me rapelle plus tres bien à quoi ca sert; do we need a colorbar here ? may be
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(Xcol), vmax=np.max(Xcol)))
        sm.set_array([])
        #if holdon == False :
        #    plt.colorbar(sm);

    # Labelisation des points, if not empty
    if indname is not None :
        N,p = np.shape(CP);
        for i in np.arange(N) :
            plt.text(CP[i,cpa-1],CP[i,cpb-1],indname[i])
    if holdon == False and lenCP > lenXcol :
        N,p = np.shape(CPobs);
        for i in np.arange(N) :
            plt.text(CPobs[i,cpa-1],CPobs[i,cpb-1],obsname[i])
            
    # Tracer les axes
    xlim = plt.xlim(); plt.xlim(xlim);
    plt.plot(xlim, np.zeros(2));
    ylim = plt.ylim(); plt.ylim(ylim);
    plt.plot(np.zeros(2),ylim);

    # Plot en noir des triangles de référence en bas à gauche
    if holdon == False :
        dx = xlim[1] - xlim[0];
        dy = ylim[1] - ylim[0];
        px = xlim[0] + dx/(xoomK) + dx/20; # à ajuster +|- en ...
        py = ylim[0] + dy/(xoomK) + dy/20; # ... fonction de xoomK
        plt.scatter(px,py,marker='>',edgecolors='k', s=xoomK,     facecolor='none');
        plt.scatter(px,py,marker='>',edgecolors='k', s=xoomK*0.5, facecolor='none');
        plt.scatter(px,py,marker='>',edgecolors='k', s=xoomK*0.1, facecolor='none');
#----------------------------------------------------------------------
def Dgeoclassif(sMap,Data,L,C,isnum,MajorPerf,visu=True) :
    bmus_   = ctk.mbmus (sMap,Data); 
    classe_ = class_ref[bmus_].reshape(len(bmus_));   
    X_Mgeo_ = dto2d(classe_,L,C,isnum); # Classification géographique
    #plt.figure(); géré par l'appelant car ce peut être une fig déjà définie
    #et en subplot ... ou pas ...
    #classe_DD_, Tperf_, Perfglob_ = perfbyclass(classe_Dobs,classe_,nb_class,kperf=kperf);
    classe_DD_, Tperf_ = perfbyclass(classe_Dobs, classe_, nb_class);
    Perfglob_ = perfglobales([MajorPerf], classe_Dobs, classe_, nb_class)[0];
    if visu :
        plt.imshow(X_Mgeo_, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
        Tperf_ = np.round([iperf*100 for iperf in Tperf_]).astype(int); #print(Tperf_)    
        hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(Tperf_);
        hcb.ax.tick_params(labelsize=8);
        plt.axis('off');
        #grid(); # for easier check
    return Perfglob_

#%% ----------------------------------------------------------------------
# Des truc qui pourront servir
tpgm0 = time();
plt.ion()
varnames = np.array(["JAN","FEV","MAR","AVR","MAI","JUI",
                    "JUI","AOU","SEP","OCT","NOV","DEC"]);
obs_data_path = '../Datas'
#######################################################################
#
# PARAMETRAGE (#1) DU CAS
from ParamCas import *
#
#======================================================================

#%%
#######################################################################
# ACQUISITION DES DONNEES D'OBSERVATION (et application des codifications)
#======================================================================
#
# Lecture des Obs____________________________________
if 0 : # Ca c'était avant
    #lat: 29.5 à 5.5 ; lon: -44.5 à -9.5
    sst_obs  = np.load(os.path.join(obs_data_path,"sst_obs_1854a2005_Y60X315.npy"));
    # Selection___________________________________________
    Nobs,Lobs,Cobs = np.shape(sst_obs);
    if Nda > 0 : # Ne prendre que les Nda dernières années (rem ATTENTION, toutes les ne commencent
        sst_obs = sst_obs[Nobs-(Nda*12):Nobs,]; #  pas à l'année 1850 ou 1854 ni au mois 01 !!!!!!!
#
if DATAOBS == "raverage_1975_2005" :  
    obs_filename = os.path.join(obs_data_path,"Donnees_1975-2005","Obs",
                                "ersstv3b_1975-2005_extract_LON-315-351_LAT-30-5.nc");
elif DATAOBS == "raverage_1930_1960" :  
    obs_filename = os.path.join(obs_data_path,"Donnees_1930-1960","Obs",
                                "ersstv3b_1930-1960_extract_LON-315-351_LAT-30-5.nc");   
elif DATAOBS == "raverage_1944_1974" :
    obs_filename = os.path.join(obs_data_path,"Donnees_1944-1974","Obs",
                                "ersstv3b_1944-1974_extract_LON-315-351_LAT-30-5.nc");  
elif DATAOBS == "rcp_2006_2017" :
    obs_filename = os.path.join(obs_data_path,"Donnees_2006-2017","Obs",
                                "ersstv3b_2006-2017_extrac_LON-315-351_LAT-30-5.nc");
#
import netCDF4
nc      = netCDF4.Dataset(obs_filename);
liste_var = nc.variables;       # mois par mois de janvier 1930 à decembre 1960 I guess
sst_var   = liste_var['sst']    # 1960 - 1930 + 1 = 31 ; 31 * 12 = 372
sst_obs   = sst_var[:];         # np.shape = (372, 1, 25, 36)
Nobs,Ncan,Lobs,Cobs = np.shape(sst_obs);
if 0 : # visu obs
    showimgdata(sst_obs,fr=0,n=Nobs);
    plt.suptitle("Obs raverage 1930 à 1960")
    plt.show(); sys.exit(0)
sst_obs   = sst_obs.reshape(Nobs,Lobs,Cobs); # np.shape = (372, 25, 36)
sst_obs   = sst_obs.filled(np.nan);
#    
lat      = np.arange(29.5, 4.5, -1);
lon      = np.arange(-44.5, -8.5, 1);
Nobs,Lobs,Cobs = np.shape(sst_obs); print("obs.shape : ", Nobs,Lobs,Cobs);
#
# Paramétrage : _____________________________________
# Définition d'une zone plus petite
if SIZE_REDUCTION == 'sel' or SIZE_REDUCTION == 'RED':
    frl = int(np.where(lat == frlat)[0]);
    tol = int(np.where(lat == tolat)[0]); # pour avoir 12.5, faut mettre 11.5
    frc = int(np.where(lon == frlon)[0]);
    toc = int(np.where(lon == tolon)[0]); # pour avoir 12.5, faut mettre 11.5
    #
    lat = lat[frl:tol];
    lon = lon[frc:toc];
if SIZE_REDUCTION == 'sel' :
    # Prendre d'entrée de jeu une zone plus petite
    sst_obs = sst_obs[:,frl:tol,frc:toc];
#
Nobs, Lobs, Cobs = np.shape(sst_obs); print("obs.shape : ", Nobs, Lobs, Cobs);
Npix = Lobs*Cobs; # C'est sensé etre la même chose pour tous les mdl
#
# Définir une fois pour toutes, les indices des nan et non nan pour UNE SEULE
# image (sachant qu'on fait l'hypothese que pour toutes les images, les nans
# sont aux memes endroits). En principe ici les modèles sont alignés sur les Obs
X_       = sst_obs[0].reshape(Lobs*Cobs);
isnanobs = np.where(np.isnan(X_))[0];
isnumobs = np.where(~np.isnan(X_))[0];
del X_;
#%%
#_________________________
# Codification des Obs 4CT 
sst_obs, Dobs, NDobs = datacodification4CT(sst_obs);
#-------------------------
#
if WITHANO :
    #wvmin=-3.9; wvmax = 4.9; # ok pour obs 1975-2005 : ANO 4CT: min=-3.8183; max=4.2445 (4.9 because ...)
    #wvmin=-4.3; wvmax = 4.9; # ok pour obs 2006-2017 : ANO 4CT: min=-4.2712; max=4.3706
    wvmin = -4.9; wvmax = 4.9; # pour mettre tout le monde d'accord ?
else : # On suppose qu'il s'agit du brute ...
    wvmin =16.0; wvmax =30.0; # ok pour obs 1975-2005 : SST 4CT: min=16.8666; max=29.029
#    
if Visu_ObsStuff : # Visu (et sauvegarde éventuelle de la figure) des données
    # telles qu'elles vont etre utilisées par la Carte Topologique
    minDobs = np.min(Dobs);   maxDobs=np.max(Dobs);
    moyDobs = np.mean(Dobs);  stdDobs=np.std(Dobs);
    if climato != "GRAD" :
        aff2D(Dobs,Lobs,Cobs,isnumobs,isnanobs,wvmin=wvmin,wvmax=wvmax,
              figsize=(12,9),varnames=varnames); #...
    else :
        aff2D(Dobs,Lobs,Cobs,isnumobs,isnanobs,wvmin=0.0,wvmax=0.042,
              figsize=(12,9),varnames=varnames); #...
    plt.suptitle("%sSST%d-%d). Obs for CT\nmin=%f, max=%f, moy=%f, std=%f"
                 %(fcodage,andeb,anfin,minDobs,maxDobs,moyDobs,stdDobs));
    if 0 : #SAVEFIG : # sauvegarde de la figure
        plt.savefig("%sObs4CT"%(fshortcode))
    #X_ = np.mean(Dobs, axis=1); X_ = X_.reshape(743,1); #rem = 0.0 when anomalies
    #plt.show(); sys.exit(0);
    #
    # ECARTS TYPES par mois et par pixel
    # Ici sst_obs correspond aux anomalies (si c'est uniquement ca qu'on
    # a demandé) et plus généralement aux données avant climatologie
    Dstd_, pipo_, pipo_  = Dpixmoymens(sst_obs, stat='std');
    if ecvmin >= 0 : 
        aff2D(Dstd_,Lobs,Cobs,isnumobs,isnanobs, figsize=(12,9),varnames=varnames,
              wvmin=ecvmin,wvmax=ecvmax);
    else :
        aff2D(Dstd_,Lobs,Cobs,isnumobs,isnanobs, figsize=(12,9),varnames=varnames,
              wvmin=np.nanmin(Dstd_),wvmax=np.nanmax(Dstd_));
    plt.suptitle("%sSST(%s)). Obs Before Climatologie\nEcarts Types par mois et par pixel" \
                     %(fcodage,DATAMDL));
    if 0 : #SAVEFIG : # sauvegarde de la figure
        plt.savefig("std%sObs"%(fshortcode))
    del Dstd_, pipo_
#
#######################################################################
#
if STOP_BEFORE_CT :
    plt.show(); sys.exit(0);
#######################################################################
#                       Carte Topologique
#======================================================================
tseed = 0; #tseed = 9; #tseed = np.long(time());
print("tseed=",tseed); np.random.seed(tseed);
#----------------------------------------------------------------------
# Création de la structure de la carte_______________
norm_method = 'data'; # je n'utilise pas 'var' mais je fais centred à
                      # la place (ou pas) qui est équivalent, mais qui
                      # me permetde garder la maitrise du codage
sMapO = SOM.SOM('sMapObs', Dobs, mapsize=[nbl, nbc], norm_method=norm_method, \
                initmethod='random', varname=varnames)
#
print("NDobs(sm.dlen)=%d, dim(Dapp)=%d\nCT : %dx%d=%dunits" \
      %(sMapO.dlen,sMapO.dim,nbl,nbc,sMapO.nnodes));
#!EU-T-IL FALLUT NORMALISER LES DONNEES ; il me semble me rappeler que
#ca à peut etre a voir avec norm_method='data' ci dessus
#
# Apprentissage de la carte _________________________
etape1=[epoch1,radini1,radfin1];    etape2=[epoch2,radini2,radfin2];
sMapO.train(etape1=etape1,etape2=etape2, verbose='on');
# + err topo maison
bmus2O = ctk.mbmus (sMapO, Data=None, narg=2);
etO    = ctk.errtopo(sMapO, bmus2O); # dans le cas 'rect' uniquement
print("Obs, erreur topologique = %.4f" %etO)
#
# Visualisation______________________________________
if Visu_CTStuff : #==>> la U_matrix
    a=sMapO.view_U_matrix(distance2=2, row_normalized='No', show_data='Yes', \
                      contooor='Yes', blob='No', save='No', save_dir='');
    plt.suptitle("Obs, The U-MATRIX", fontsize=16);
if Visu_CTStuff : #==>> La carte
    ctk.showmap(sMapO,sztext=11,colbar=1,cmap=cm.rainbow,interp=None);
    plt.suptitle("Obs, Les Composantes de la carte", fontsize=16);
#
# Other stuffs ______________________________________
bmusO     = ctk.mbmus (sMapO, Data=Dobs); # déjà vu ? conditionnellement ?
minref    = np.min(sMapO.codebook);
maxref    = np.max(sMapO.codebook);
Z_        = linkage(sMapO.codebook, method_cah, dist_cah);
class_ref = fcluster(Z_,nb_class,'maxclust'); # Classes des referents
#
if Visu_Dendro :
    plt.figure();
    #R_ = dendrogram(Z_,sMapO.dlen,'lastp');
    dendrogram(Z_,sMapO.dlen,'lastp');
    plt.title("CAH on SOM(%dx%d), %s, nb_class=%d"%(nbl,nbc,method_cah,nb_class));
del Z_
#
coches = np.arange(nb_class)+1;   # ex 6 classes : [1,2,3,4,5,6]
ticks  = coches + 0.5;            # [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
bounds = np.arange(nb_class+1)+1; # pour bounds faut une frontière de plus [1, 2, 3, 4, 5, 6, 7]
sztitle = 10;
#
# Transcodage des indices des classes
if TRANSCOCLASSE is not '' :
    class_ref = transco_class(class_ref,sMapO.codebook,crit=TRANSCOCLASSE);
#
classe_Dobs = class_ref[bmusO].reshape(NDobs); #(sMapO.dlen)
XC_Ogeo     = dto2d(classe_Dobs,Lobs,Cobs,isnumobs); # Classification géographique
#
# Nombre de pixels par classe (pour les obs)
Nobsc = np.zeros(nb_class)
for c in np.arange(nb_class)+1 :
    iobsc = np.where(classe_Dobs==c)[0]; # Indices des classes c des obs
    Nobsc[c-1] = len(iobsc);
#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Pour différencier la zone entiere de la zone REDuite, je conviens que le o
# de obs sera en majuscule pour la zone entière (selectionnée).
# du coup, je duplique.
sst_Obs     = np.copy(sst_obs); NObs=Nobs; LObs=Lobs; CObs=Cobs;
isnumObs    = np.copy(isnumobs);
XC_ogeo     = np.copy(XC_Ogeo);
classe_DObs = np.copy(classe_Dobs);
#
if SIZE_REDUCTION == 'RED' :
    #sst_obs, XC_Ogeo, classe_DObs, isnumobs = red_classgeo(sst_Obs,isnumObs,classe_DObs,frl,tol,frc,toc);
    sst_obs, XC_ogeo, classe_Dobs, isnumobs = red_classgeo(sst_Obs,isnumObs,classe_DObs,frl,tol,frc,toc);
    # si on ne passe pas ici, les petits o et les grand O sont égaux
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#plt.figure(); plt.imshow(XC_ogeo, interpolation='none',vmin=1,vmax=nb_class)
Nobs, Lobs, Cobs = np.shape(sst_obs)
NDobs  = len(classe_Dobs)
fond_C = np.ones(NDobs)
fond_C = dto2d(fond_C,Lobs,Cobs,isnumobs,missval=0.5)
#!!!!!!>
# Pour SIZE_REDUCTION=='RED', il faut peut être redéfinir isnanobs et isnumobs isn'it
X_       = sst_obs[0].reshape(Lobs*Cobs);
isnanobs = np.where(np.isnan(X_))[0];
isnumobs = np.where(~np.isnan(X_))[0];
del X_;
#!!!!!!<
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
if Visu_ObsStuff : # Visualisation de truc liés au Obs
    # Classification
    plt.figure();
    plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
    hcb    = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
    hcb.set_ticklabels(coches);
    hcb.ax.tick_params(labelsize=8)
    plt.title("obs, classe géographique Method %s"%(method_cah),fontsize=16); 
    nticks = 1; # 4
    plt.xticks(np.arange(0,Cobs,nticks), lon[np.arange(0,Cobs,nticks)], rotation=45, fontsize=10)
    plt.yticks(np.arange(0,Lobs,nticks), lat[np.arange(0,Lobs,nticks)], fontsize=10)
    #grid(); # for easier check
    #
    # Courbes des Moyennes Mensuelles par Classe
    plt.figure();
    TmoymensclassObs = moymensclass(sst_obs,isnumobs,classe_Dobs,nb_class)
    #plt.plot(TmoymensclassObs); plt.axis('tight');
    for i in np.arange(nb_class) :
            plt.plot(TmoymensclassObs[:,i],'.-',color=pcmap[i]);
    plt.axis('tight');
    plt.xlabel('mois');
    plt.legend(np.arange(nb_class)+1,loc=2,fontsize=8);
    plt.title("obs, Moy. Mens. par Classe Method %s"%(method_cah),fontsize=16); #,fontweigth='bold');
    #plt.show(); sys.exit(0)
#
if Visu_CTStuff : # Visu des profils des référents de la carte
    ctk.showprofils(sMapO, Data=Dobs,visu=3, scale=2,Clevel=class_ref-1,Gscale=0.5,
                ColorClass=pcmap);
#
#######################################################################
#
if STOP_BEFORE_MDLSTUFF :
    plt.show(); sys.exit(0)
#%%
#######################################################################
#                        MODELS STUFFS START HERE
#======================================================================
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#       INITILISATIONS EN AMONT de LA BOUCLE SUR LES MODELES
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# For (sub)plot by modele
nsub   = 49; # actuellement au plus 48 modèles + 1 pour les obs.      
#nsub  = 9;  # Pour MICHEL (8+1pour les obs)
def lcsub(nsub) :
    nbsubc = np.ceil(np.sqrt(nsub));
    nbsubl = np.ceil(1.0*nsub/nbsubc);
    return nbsubc, nbsubl
nbsubc, nbsubl = lcsub(nsub);
isubplot=0;
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
min_moymensclass = 999999.999999; # sert pour avoir tous les ...
max_moymensclass = 000000.000000; # ... subplots à la même échelles
#
Tperfglob        = np.zeros((Nmodels,len(TypePerf))); # Tableau des Perf globales des modèles
if NIJ==1 :
    TNIJ         = [];
TTperf           = [];
#
TDmdl4CT         = []; # Stockage des modèles 4CT pour AFC-CAH ...
#
Nmdlok           = 0;  # Pour si y'a cumul ainsi connaitre l'indice de modèle valide 
                       # courant, puis au final, le nombre de modèles valides
                       # quoique ca dependra aussi que SUMi(Ni.) soit > 0                   
Tperfglob4Sort   = [];
Tclasse_DMdl     = [];
Tmdlname         = []; # Table des modèles
Tmoymensclass    = [];
#
def red_climato(Dmdl,L,C,isnum,isnum_red,frl,tol,frc,toc) :
    X_ = np.empty((L*C,12))
    X_[isnum] = Dmdl;
    X_ = X_.reshape(L,C,12);
    X_ = X_[frl:tol,frc:toc,:]
    l,c,m = np.shape(X_);
    X_ = X_.reshape(l*c,m)
    X_ = X_[isnum_red]
    return X_
#
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#           PREMIERE BOUCLE SUR LES MODELES START HERE
#           PREMIERE BOUCLE SUR LES MODELES START HERE
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
print("ooooooooooooooooooooooooooooo first loop ooooooooooooooooooooooooooooo");
# >>>> Pour selectionner (filtrer) uniquement certains modèles (par exemple
# ceux d'un cluster ou un Sopt quelconque) et pour éviter de le faire par
# le biais de la table des modèles ou par le déplacement des fichier dans
# un autre répertoire, et ainsi pouvoir produire les figures 107, 109, 101
# ou autre à voir ...
# et aussi les données 4CT à condition de mettre ces memes modèles pour 
# la généralisation.
#Sfiltre = ['CMCC-CM'];
#Sfiltre = ['CanCM4'];
# 13x12
# best cluster Methode AFC
#Sfiltre = ['CNRM-CM5', 'CMCC-CMS', 'CNRM-CM5-2', 'GFDL-CM3', 'FGOALS-s2'];
# best cum Methode des Regroupements Icrementaux
#Sfiltre = ['CanCM4', 'CNRM-CM5', 'CMCC-CMS', 'CNRM-CM5-2','GFDL-CM3',
#           'CanESM2', 'NorESM1-ME']
# 25x36
# best cluster Methode AFC
Sfiltre = ['CMCC-CM', 'HadGEM2-ES', 'HadGEM2-AO', 'HadGEM2-CC', 'CMCC-CMS',
           'CNRM-CM5-2', 'CanESM2', 'CanCM4', 'GFDL-CM3', 'CNRM-CM5',
           'FGOALS-s2', 'CSIRO-Mk3-6-0', 'CMCC-CESM'];
# best cum Methode des Regroupements Icrementaux
#Sfiltre = ['CMCC-CM'];
#<<<<<<<<
#
for imodel in np.arange(Nmodels) :
    instname = Tinstit[imodel]; #print("-> ", mdlname)
    mdlname = Tmodels[imodel,0]; #print("-> ", mdlname)
    anstart = Tmodels[imodel,1]; # (utile pour rmean seulement)
    #
    # >>> Filtre (selection)de modèles en entrée ; Mettre 0 dans le if pour ne pas filtrer
    if 0 and mdlname not in Sfiltre :
        continue;
    print(mdlname)
    # <<<<< 
    #______________________________________________________
    # Lecture des données (fichiers.mat générés par Carlos)
    if  DATAMDL=="raverage_1975_2005" : 
        mdl_filename = os.path.join(obs_data_path,"Donnees_1975-2005",
                                "all_data_historical_raverage_1975-2005",
                                'Data',
                                instname+'_'+mdlname,
                                "sst_"+mdlname+"_raverage_1975-2005.mat")
    elif DATAMDL=="raverage_1930_1960" : 
        mdl_filename = os.path.join(obs_data_path,"Donnees_1930-1960",
                                "all_data_historical_raverage_1930-1960",
                                'Data',
                                instname+'_'+mdlname,
                                "sst_"+mdlname+"_raverage_1930-1960.mat")
    elif DATAMDL=="raverage_1944_1974" : 
        mdl_filename = os.path.join(obs_data_path,"Donnees_1944-1974",
                                "all_data_historical_raverage_1944-1974",
                                'Data',
                                instname+'_'+mdlname,
                                "sst_"+mdlname+"_raverage_1944-1974.mat")
    elif DATAMDL == "rcp_2006_2017":
        mdl_filename = os.path.join(obs_data_path,"Donnees_2006-2017",
                                "all_data_"+scenar+"_raverage_2006-2017",
                                'Data',
                                instname+'_'+mdlname,
                                "sst_"+scenar+mdlname+"_raverage_2006-2017.mat")
    elif DATAMDL == "rcp_2070_2100":
        mdl_filename = os.path.join(obs_data_path,"Donnees_2070-2100",
                                "all_data_"+scenar+"_raverage_2070-2100",
                                'Data',
                                instname+'_'+mdlname,
                                "sst_"+scenar+mdlname+"_raverage_2070-2100.mat")
    try :
        sst_mat = scipy.io.loadmat(mdl_filename);
    except :
        print("modèle %s not found"%mdlname);
        continue;
    sst_mdl = sst_mat['SST'];
    #
    Nmdl, Lmdl, Cmdl = np.shape(sst_mdl); #print("mdl.shape : ", Nmdl, Lmdl, Cmdl);
    #
    Nmdlok = Nmdlok + 1; # Pour si y'a cumul ainsi connaitre l'indice de modèle 
             # valide courant, puis au final, le nombre de modèles valides.
             # (mais ...)
    #
    if MDLCOMPLETION : # Complémentation des données modèles de sorte à ce que seul
        nnan=1;        # le mappage des nans d'obs soit utilisé
        while nnan > 0 :
            sst_mdl, nnan = nan2moy(sst_mdl, Left=1, Above=0, Right=0, Below=0)
    #________________________________________________
    if SIZE_REDUCTION=='sel' : # Prendre une zone plus petite
        sst_mdl = sst_mdl[:,frl:tol,frc:toc];
    Nmdl, Lmdl, Cmdl = np.shape(sst_mdl); #print("mdl.shape : ", Nmdl, Lmdl, Cmdl); 
    #
    #________________________________________________
    if 1 : # Y'a plus simple (si ca marche)
        X_ = sst_mdl.reshape(Nmdl, Lmdl*Cmdl);
        X_[:,isnanobs] = np.nan;
        sst_mdl = X_.reshape(Nmdl,Lmdl,Cmdl);
    if mdlname == "FGOALS-s2" and DATAMDL == "raverage_1975_2005" :
        mdlname = "FGOALS-s2(2004)" # au passage
    #________________________________________________________
    # Codification du modèle (4CT)            
    sst_mdl, Dmdl, NDmdl = datacodification4CT(sst_mdl);
    #________________________________________________________
    # Ecart type des modèles en entrées cumulés et moyennés
    # (événtuellement controlé par Sfiltre ci-dessus)
    if OK101 :
        # Moyenne
        Dmoy_, pipo_, pipo_  = Dpixmoymens(sst_mdl);
        if Nmdlok == 1 :     
            Smoy_ = Dmoy_;        
        else :
            Smoy_ = Smoy_ + Dmoy_;
        #
        # Ecart type (à cause de FGOALS-s2 ca complique tout ...
        # (Il y aura un décalage entre moyenne et ecart type qui sera un peu
        # faussé, mais ainsi je retrouverai les même résultats qu'avant ...
        sst_ = sst_mdl;
        if mdlname == "FGOALS-s2(2004)" and DATAMDL == "raverage_1975_2005" :
            sst_ = np.concatenate((sst_, sst_[360-12:360]))
        if Nmdlok == 1 :
            Tsst_ = sst_;        
        else :
            Tsst_ = Tsst_ + sst_;
    #________________________________________________________
    TDmdl4CT.append(Dmdl);  # stockage des modèles 4CT pour AFC-CAH ...
    Tmdlname.append(Tmodels[imodel,0])
    #calcul de la perf glob du modèle et stockage pour tri
    bmusM       = ctk.mbmus (sMapO, Data=Dmdl);
    classe_DMdl = class_ref[bmusM].reshape(NDmdl);
    perfglob = perfglobales([TypePerf[0]], classe_Dobs, classe_DMdl, nb_class)
    Tperfglob4Sort.append(perfglob[0])
    Tclasse_DMdl.append(classe_DMdl)
    #
    if OK106 : # Stockage (if required) pour la Courbes des moyennes mensuelles par classe
        Tmoymensclass.append(moymensclass(sst_mdl,isnumobs,classe_Dobs,nb_class)); ##!!?? 
#
# Fin de la PREMIERE boucle sur les modèles
#
#____________________________
if OK101 :
    if len(Tmdlname) > 6 :            # Sous forme de liste, la liste des noms de modèles
        Tnames_ = Tmdlname;           # n'est pas coupé dans l'affichage du titre de la figure
    else :                            # par contre il l'est sous forme d'array; selon le cas, ou 
        Tnames_ = np.array(Tmdlname); # le nombre de modèles, il faut adapter comme on peut
    #
    # Moyenne des modèles en entrée, moyennées
    # (Il devrait suffire de refaire la même chose pour Sall-cum)
    Smoy_ = Smoy_ / Nmdlok; # Moyenne des moyennes cumulées
    aff2D(Smoy_,Lobs,Cobs,isnumobs,isnanobs, figsize=(12,9),varnames=varnames,
            wvmin=wvmin,wvmax=wvmax);
    #plt.suptitle("MCUMMOY%s\n%sSST(%s)). Moyenne par mois et par pixel (Before Climatologie)\nmin=%f, max=%f, moy=%f, std=%f" \
    plt.suptitle("Mdl_MOY%s\n%sSST(%s)). Moyenne par mois et par pixel (Before Climatologie)\nmin=%f, max=%f, moy=%f, std=%f" \
                     %(Tnames_,fcodage,DATAMDL,
                       np.nanmin(Smoy_),np.nanmax(Smoy_),np.nanmean(Smoy_),np.nanstd(Smoy_),));
    #
    # Ecart type des modèles en entrée moyennées
    Tsst_ = Tsst_ / Nmdlok; # Moyenne des cumuls des animalies (modèle moyen des annomalies mais en fait avant climato))
    Dstd_, pipo_, pipo_  = Dpixmoymens(Tsst_,stat='std'); # cliamtologie
    if ecvmin >= 0 : 
        aff2D(Dstd_,Lobs,Cobs,isnumobs,isnanobs, figsize=(12,9),varnames=varnames,
              wvmin=ecvmin,wvmax=ecvmax);
    else :
        aff2D(Dstd_,Lobs,Cobs,isnumobs,isnanobs, figsize=(12,9),varnames=varnames,
              wvmin=np.nanmin(Dstd_),wvmax=np.nanmax(Dstd_));
    #plt.suptitle("MCUMMOY%s\n%sSST(%s)). \nEcarts Types par mois et par pixel (Before Climatologie)" \
    plt.suptitle("Mdl_MOY%s\n%sSST(%s)). \nEcarts Types par mois et par pixel (Before Climatologie)" \
                     %(Tnames_,fcodage,DATAMDL));
    #
    del Smoy_, Tsst_, Dstd_, Tnames_, pipo_
    #plt.show(); sys.exit(0)
#_____________________________________________________________________
######################################################################
# Reprise de la boucle (avec les modèles valides du coup).
# (question l'emplacement des modèles sur les figures ne devrait pas etre un problème ?)
#*****************************************
del Tmodels ##!!??
#_________________________________________
# TRI et Reformatage des tableaux si besoin
Nmodels = len(TDmdl4CT);            ##!!??
if 1 : # Sort On Perf
    IS_ = np.argsort(Tperfglob4Sort)
    IS_= np.flipud(IS_)
else : # no sort
    IS_ = np.arange(Nmodels)
Tperfglob4Sort = np.array(Tperfglob4Sort)
Tperfglob4Sort = Tperfglob4Sort[IS_];

X1_ = np.copy(TDmdl4CT)
X2_ = np.copy(Tmdlname)
X3_ = np.copy(Tclasse_DMdl)
for i in np.arange(Nmodels) : # TDmdl4CT = TDmdl4CT[I_];
    TDmdl4CT[i]     = X1_[IS_[i]]
    Tmdlname[i]     = X2_[IS_[i]]  
    Tclasse_DMdl[i] = X3_[IS_[i]]
del X1_, X2_, X3_;
if OK106 :
    X1_ = np.copy(Tmoymensclass);
    for i in np.arange(Nmodels) :
        Tmoymensclass[i] = X1_[IS_[i]]
    del X1_   
    Tmoymensclass    = np.array(Tmoymensclass);
    min_moymensclass = np.nanmin(Tmoymensclass); ##!!??
    max_moymensclass = np.nanmax(Tmoymensclass); ##!!??
#*****************************************
MaxPerfglob_Qm  = 0.0; # Utilisé pour savoir les quels premiers modèles
IMaxPerfglob_Qm = 0;   # prendre dans la stratégie du "meilleur cumul moyen"
#*****************************************
# l'Init des figures à produire doit pouvoir etre placé ici (sauf la 106)
if OK104 : # Classification avec, "en transparance", les mals classés
           # par rapport aux obs
    plt.figure(104,figsize=(18,9),facecolor='w')
    plt.subplots_adjust(wspace=0.0, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.90)
    suptitle104="%sSST(%s)). %s Classification of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,method_cah);
if OK105 : #Classification
    plt.figure(105,figsize=(18,9))
    plt.subplots_adjust(wspace=0.0, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.90)
    suptitle105="%sSST(%s)). %s Classification of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,method_cah);
if OK106 : # Courbes des moyennes mensuelles par classe
    plt.figure(106,figsize=(18,9),facecolor='w'); # Moyennes mensuelles par classe
    plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.90)
    if 0 : #MICHEL
        suptitle106="MoyMensClass(%sSST(%s)).) %s Classification of Completed Models (vs Obs)" \
                     %(fcodage,DATAMDL,method_cah);
    else : # pas MICHEL
        suptitle106="MOY - MoyMensClass(%sSST(%s)).) %s Classification of Completed Models (vs Obs)" \
                     %(fcodage,DATAMDL,method_cah);
if OK107 : # Variance (not 'RED' compatible)
    suptitle107="VARiance(%sSST(%s)).) Variance (by pixel) on Completed Models" \
                 %(fcodage,DATAMDL);
    Dmdl_TVar  = np.ones((Nmodels,NDobs))*np.nan; # Tableau des Variance par pixel sur climatologie
                   # J'utiliserais ainsi showimgdata pour avoir une colorbar commune
if MCUM > 0 :
    # Moyenne CUMulative
    if OK108 : # Classification en Model Cumulé Moyen
        plt.figure(108,figsize=(18,9),facecolor='w'); # Moyennes mensuelles par classe
        plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.90)
        suptitle108="MCUM - %sSST(%s)). %s Classification of Completed Models (vs Obs)" \
                     %(fcodage,DATAMDL,method_cah);
    #
    DMdl_Q  = np.zeros((NDmdl,12));  # f(modèle climatologique Cumulé) #+
    DMdl_Qm = np.zeros((NDmdl,12));  # f(modèle climatologique Cumulé moyen) #+
    #
    # Variance CUMulative
    if OK109 : # Variance sur les Models Cumulés Moyens (not 'RED' compatible)
        Dmdl_TVm = np.ones((Nmodels,NDobs))*np.nan; # Tableau des Variance sur climatologie
                   # cumulée, moyennée par pixel. J'utiliserais ainsi showimgdata pour avoir
                   # une colorbar commune
        suptitle109="VCUM - %sSST(%s)). Variance sur la Moyenne Cumulée de Modeles complétés" \
                     %(fcodage,DATAMDL);
#
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#           DEUXIEME BOUCLE SUR LES MODELES START HERE
#           DEUXIEME BOUCLE SUR LES MODELES START HERE
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
isubplot = 0;
print("ooooooooooooooooooooooooooooo 2nd loop ooooooooooooooooooooooooooooo")
for imodel in np.arange(Nmodels) :
    isubplot=isubplot+1;
    #
    Dmdl    = TDmdl4CT[imodel];
    mdlname = Tmdlname[imodel];
    # 
    classe_DMdl = Tclasse_DMdl[imodel];
    XC_Mgeo     = dto2d(classe_DMdl,LObs,CObs,isnumObs); # Classification géographique
    #
    #>>>>>>>>>>>
    classe_Dmdl = np.copy(classe_DMdl); # ... because RED ... du coup je duplique           pour avoir les memes
    XC_mgeo     = np.copy(XC_Mgeo);     # ... because RED ... du coup je duplique           noms de variables.
    #
    if SIZE_REDUCTION == 'RED' :
        print("RED> %s"%(mdlname), np.shape(sst_mdl), len(isnumObs),len(classe_DMdl))
        pipo, XC_mgeo, classe_Dmdl, isnum_red = red_classgeo(sst_mdl,isnumObs,classe_DMdl,frl,tol,frc,toc); 
        print("<RED %s"%mdlname)
    #<<<<<<<<<<<
    #
    # Perf par classe
    classe_DD, Tperf = perfbyclass(classe_Dobs,classe_Dmdl,nb_class);
    Tperf = np.round([i*100 for i in Tperf]).astype(int); 
    TTperf.append(Tperf); # !!! rem AFC est faite avec ca
    #
    # Perf globales 
    Perfglob             = Tperfglob4Sort[imodel];
    Tperfglob[imodel,0]  = Perfglob;
    Nperf_ = len(TypePerf)
    if Nperf_ > 1 :
        # On calcule les autres perf
        T_ = perfglobales(TypePerf[1:Nperf_], classe_Dobs, classe_Dmdl, nb_class)
        Tperfglob[imodel,1:Nperf_] = T_
    #
    Nmdl, Lmdl, Cmdl = np.shape(sst_mdl)
    NDmdl = len(classe_Dmdl); # needed for 108, ...? 
    #
    # l'AFC pourrait aussi etre faite avec ça
    if NIJ == 1 : # Nij = card|classes| ; Pourrait semblé inapproprié car les classes
                  # peuvent géographiquement être n'importe où, mais ...               
        Znij_ = [];
        for c in np.arange(nb_class) :
            imdlc = np.where(classe_Dmdl==c+1)[0]; # Indices des classes c du model
            Znij_.append(len(imdlc));
        TNIJ.append(Znij_);
    #
    #<:::------------------------------------------------------------------
    if OK104 : # Classification avec, "en transparance", les pixels mals
               # classés par rapport aux obs. (pour les modèles les Perf
               # par classe sont en colorbar)
        plt.figure(104); plt.subplot(nbsubl,nbsubc,isubplot);
        X_ = dto2d(classe_DD,Lobs,Cobs,isnumobs); #X_= classgeo(sst_obs, classe_DD);
        plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
        if FONDTRANS == "Obs" :
            plt.imshow(XC_ogeo, interpolation='none', cmap=ccmap, alpha=0.2,vmin=1,vmax=nb_class);
        elif FONDTRANS == "Mdl" :
            plt.imshow(XC_mgeo, interpolation='none', cmap=ccmap, alpha=0.2,vmin=1,vmax=nb_class);
        plt.imshow(X_, interpolation='none', cmap=ccmap,vmin=1,vmax=nb_class);
        del X_
        plt.axis('off'); #grid(); # for easier check
        plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob,'%'),fontsize=sztitle); 
        hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(Tperf);
        hcb.ax.tick_params(labelsize=8)
        #
    if OK105 : # Classification (pour les modèles les Perf par classe sont en colorbar)
        plt.figure(105); plt.subplot(nbsubl,nbsubc,isubplot);
        plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
        plt.imshow(XC_mgeo, interpolation='none',cmap=ccmap, vmin=1,vmax=nb_class);
        hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(Tperf);
        hcb.ax.tick_params(labelsize=8);
        plt.axis('off');
        #grid(); # for easier check
        plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob,'%'),fontsize=sztitle);
        #
    if OK106 : # Courbes des moyennes mensuelles par classe
        plt.figure(106); plt.subplot(nbsubl,nbsubc,isubplot);
        for i in np.arange(nb_class) :
            plt.plot(Tmoymensclass[imodel,:,i],'.-',color=pcmap[i]);
        plt.axis([0, 11, min_moymensclass, max_moymensclass]);
        plt.xticks([]); #plt.axis('off');     
        plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob,'%'),fontsize=sztitle);
        #       
    if OK107 : # Variance (not 'RED' compatible)
        Dmdl_TVar[imodel] = np.var(Dmdl, axis=1, ddof=0);
        #
    if MCUM>0 and (OK108 or OK109) : # Cumul et moyenne
        DMdl_Q  = DMdl_Q + Dmdl;     # Cumul Zone
        DMdl_Qm = DMdl_Q / (imodel+1);   # Moyenne Zone ##!!?? 
        #
    if MCUM>0 and OK108 : # Classification en Model Cumulé Moyen (Perf par classe en colorbar)
        plt.figure(108); plt.subplot(nbsubl,nbsubc,isubplot);
        bmusMdl_Qm = ctk.mbmus (sMapO, Data=DMdl_Qm);
        classe_DMdl_Qm= class_ref[bmusMdl_Qm].reshape(NDmdl);
                       # Ici classe_D* correspond à un résultats de classification
                       # (bon ou mauvais ; donc sans nan)
        XC_mgeo_Qm    = dto2d(classe_DMdl_Qm,Lobs,Cobs,isnumobs); # Classification géographique
                       # Mise sous forme 2D de classe_D*, en mettant nan pour les
                       # pixels mas classés
        classe_DD_Qm, Tperf_Qm = perfbyclass(classe_Dobs, classe_DMdl_Qm, nb_class);
        Perfglob_Qm = perfglobales([TypePerf[0]], classe_Dobs, classe_DMdl_Qm, nb_class)[0];  
                       # Ici pour classe_DD* : les pixels bien classés sont valorisés avec
                       # leur classe, et les mals classés ont nan
        Tperf_Qm = np.round([i*100 for i in Tperf_Qm]).astype(int);

        plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
        plt.imshow(XC_mgeo_Qm, interpolation='none',cmap=ccmap, vmin=1,vmax=nb_class);
        hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(Tperf_Qm);
        hcb.ax.tick_params(labelsize=8);
        plt.axis('off');
        #grid(); # for easier check
        plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob_Qm,'%'),fontsize=sztitle);
        #
        pgqm_ = np.round_(Perfglob_Qm*100)
        if pgqm_ >= MaxPerfglob_Qm :
            MaxPerfglob_Qm = pgqm_; # Utilisé pour savoir les quels premiers modèles
            IMaxPerfglob_Qm = imodel+1;   # prendre dans la stratégie du "meilleur cumul moyen"
            print("New best cumul perf for %dmodels : %d%c"%(imodel+1,pgqm_,'%'))
     #
    if MCUM>0 and OK109 : # Variance sur les Models Cumulés Moyens (not 'RED' compatible)
                          # Perf par classe en colorbar)
        Dmdl_TVm[imodel] = np.var(DMdl_Qm, axis=1, ddof=0);
#
# Fin de la DEUXIEME boucle sur les modèles
#__________________________________________
# Les Obs à la fin 
isubplot = 49; 
#isubplot = isubplot + 1; # Michel (ou pas ?)
if OK104 : # Obs for 104
    plt.figure(104); plt.subplot(nbsubl,nbsubc,isubplot);
    plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
    plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
    hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
    hcb.set_ticklabels(coches);
    hcb.ax.tick_params(labelsize=8)
    plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle);
    plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
    plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
    #grid(); # for easier check
    plt.suptitle(suptitle104)
    if SAVEFIG :
        plt.savefig("%s%s_%s%s%dMdlvsObstrans"%(fprefixe,SIZE_REDUCTION,fshortcode,method_cah,nb_class))
    #
if OK105 : # Obs for 105
    plt.figure(105); plt.subplot(nbsubl,nbsubc,isubplot);
    plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
    plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
    hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
    hcb.set_ticklabels(coches);
    hcb.ax.tick_params(labelsize=8)
    plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle);
    plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
    plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
    #grid(); # for easier check
    plt.suptitle(suptitle105)
    if SAVEFIG :
        plt.savefig("%s%s_%s%s%dMdl"%(fprefixe,SIZE_REDUCTION,fshortcode,method_cah,nb_class))
if OK106 : # Obs for 106
    plt.figure(106); plt.subplot(nbsubl,nbsubc,isubplot);
    TmoymensclassObs = moymensclass(sst_obs,isnumobs,classe_Dobs,nb_class);
    for i in np.arange(nb_class) :
        plt.plot(TmoymensclassObs[:,i],'.-',color=pcmap[i]);
    plt.axis([0, 11, min_moymensclass, max_moymensclass]);
    plt.xlabel('mois');
    plt.xticks(np.arange(12), np.arange(12)+1, fontsize=8)
    plt.legend(np.arange(nb_class)+1,loc=2,fontsize=6,numpoints=1,bbox_to_anchor=(1.1, 1.0));
    plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle);
    #
    # On repasse sur tous les supblots pour les mettre à la même echelle.
    print(min_moymensclass, max_moymensclass);
    plt.suptitle(suptitle106)
    if SAVEFIG :
        plt.savefig("%s%s_%s%s%dmoymensclass"%(fprefixe,SIZE_REDUCTION,fshortcode,method_cah,nb_class))
#
if OK107 or OK109 : # Calcul de la variance des obs par pixel de la climatologie
    Tlabs = np.copy(Tmdlname);  
    Tlabs = np.append(Tlabs,'');                # Pour le subplot vide
    Tlabs = np.append(Tlabs,'Observations');    # Pour les Obs
    varobs= np.ones(Lobs*Cobs)*np.nan;          # Variances des ...
    varobs[isnumobs] = np.var(Dobs, axis=1, ddof=0); # ... Obs par pixel
#
if OK107 : # Variance par pixels des modèles
    X_ = np.ones((Nmodels,Lobs*Cobs))*np.nan;
    X_[:,isnumobs] = Dmdl_TVar
    # Rajouter nan pour le subplot vide
    X_    = np.concatenate(( X_, np.ones((1,Lobs*Cobs))*np.nan))
    # Rajout de la variance des obs
    X_    = np.concatenate((X_, varobs.reshape(1,Lobs*Cobs)))
    #
    showimgdata(X_.reshape(Nmodels+2,1,Lobs,Cobs), Labels=Tlabs, n=Nmodels+2,fr=0,
                vmin=np.nanmin(Dmdl_TVar),vmax=np.nanmax(Dmdl_TVar),fignum=107);
    del X_
    plt.suptitle(suptitle107);
    if SAVEFIG :
        plt.savefig("%sVAR_%s_%sMdl"%(fprefixe,SIZE_REDUCTION,fshortcode))
#
if MCUM>0 and OK108 : # idem OK105, but ...
    plt.figure(108); plt.subplot(nbsubl,nbsubc,isubplot);
    plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
    plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
    hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
    hcb.set_ticklabels(coches);
    hcb.ax.tick_params(labelsize=8)
    plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle);
    plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
    plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
    #grid(); # for easier check
    plt.suptitle(suptitle108);
    if SAVEFIG :
        plt.savefig("%sMCUM_%s_%s%s%dMdl"%(fprefixe,SIZE_REDUCTION,fshortcode,method_cah,nb_class))
#
if MCUM>0 and OK109 : # Variance par pixels des moyenne des modèles cumulés
    X_ = np.ones((Nmodels,Lobs*Cobs))*np.nan;
    X_[:,isnumobs] = Dmdl_TVm
    # Rajouter nan pour le subplot vide
    X_ = np.concatenate(( X_, np.ones((1,Lobs*Cobs))*np.nan))
    # Rajout de la variance des obs
    X_ = np.concatenate((X_, varobs.reshape(1,Lobs*Cobs)))
    #
    showimgdata(X_.reshape(Nmodels+2,1,Lobs,Cobs), Labels=Tlabs, n=Nmodels+2,fr=0,
                vmin=np.nanmin(Dmdl_TVm),vmax=np.nanmax(Dmdl_TVm),fignum=109);
    del X_
    plt.suptitle(suptitle109);
    if SAVEFIG :
        plt.savefig("%sVCUM_%s_%sMdl"%(fprefixe,SIZE_REDUCTION,fshortcode))
##
##---------------------------------------------------------------------
# Redimensionnement de Tperfglob au nombre de modèles effectif
Tperfglob = Tperfglob[0:Nmodels]; 
#
# Edition des résultats
if 1 : # Tableau des performances en figure de courbes
    plt.figure(facecolor='w'); plt.plot(Tperfglob,'.-');
    plt.axis("tight"); plt.grid('on')
    plt.xticks(np.arange(Nmodels),Tmdlname, fontsize=8, rotation=45,
               horizontalalignment='right', verticalalignment='baseline');
    plt.legend(TypePerf,numpoints=1,loc=3)
    plt.title("%sSST(%s)) %s%d Indice(s) de classification of Completed Models (vs Obs)"\
                 %(fcodage,DATAMDL,method_cah,nb_class));
#
#___________________________________________
# Mettre les Tableaux-Liste en tableau Numpy
Tmdlname = np.array(Tmdlname);
TTperf   = np.array(TTperf);
if NIJ==1 :
    TNIJ = np.array(TNIJ);
TDmdl4CT = np.array(TDmdl4CT);
#
#======================================================================
#
if STOP_BEFORE_AFC :
    plt.show(); sys.exit(0)
#======================================================================
if NIJ > 0 : # A.F.C
    #Ajout de l'indice dans le nom du modèle
    Tm_ = np.empty(len(Tmdlname),dtype='<U32');
    for i in np.arange(Nmdlok) : 
        Tm_[i] = str(i+1) + '-' +Tmdlname[i];
    #
    # Harmonaiser la variable servant de tableau de contingence (->Tp_), selon le cas
    if NIJ==1 : # Nij = card|classes| ; Pourrait semblé inapproprié car les classes
                # peuvent géographiquement être n'importe où, mais ...
        Tp_ = TNIJ; # TNIJ dans ce cas a été préparé en amont
        #
    elif NIJ==2 or NIJ==3 :
        Tp_ = TTperf; # Pourcentages des biens classés par classe (transformé ci après
                      # pour NIJ==3 en effectif après éventuel ajout des obs dans l'afc)
    # On supprime les lignes dont la somme est 0 car l'afc n'aime pas ca.
    # (en espérant que la suite du code reste cohérente !!!???)
    som_   = np.sum(Tp_, axis=1);
    Iok_   = np.where(som_>0)[0]; # Indice des modèles valides pour l'AFC
    Tp_    = Tp_[Iok_];
    Nmdlok = len(Iok_); # !!! ATTENTION !!! redéfinition du nombre de modèles valides
    #  
    # Tableau (ou liste) des Noms des individus (Modèles valides et Obs)
    if AFCWITHOBS :
        AFCindnames   = np.concatenate((Tmdlname[Iok_],['Obs']));
        NoAFCindnames = np.concatenate((Tm_[Iok_],['Obs']));
    else : 
        AFCindnames   = Tmdlname[Iok_];
        NoAFCindnames = Tm_[Iok_];
    del som_;
    #
    if CAHWITHOBS :
        CAHindnames   = np.concatenate((Tmdlname[Iok_],['Obs'])); 
        NoCAHindnames = np.concatenate((Tm_[Iok_],['Obs'])); 
    else :
        CAHindnames   = Tmdlname[Iok_]; 
        NoCAHindnames = Tm_[Iok_]; 
    Nleaves_ = len(CAHindnames);   
    #
    if AFCWITHOBS : # On ajoute Obs (si required)
        if NIJ==1 :
            Tp_ = np.concatenate((Tp_, Nobsc[np.newaxis,:]), axis=0).astype(int);
        else :
            # Obs have 100% for any class par définition
            pobs_ = (np.ones(nb_class)*100).astype(int);
            Tp_   = np.concatenate((Tp_, pobs_[np.newaxis,:]), axis=0); # je mets les Obs A LA FIN #a
    #
    if NIJ == 3 : # On transforme les %tages en Nombre (i.e. en effectif)
        if 0 : # icicicicici
            Tp_ = Tp_ * Nobsc / 100;
        else : 
            Tp_ = np.round(Tp_ * Nobsc / 100).astype(int); ##$$
    #
    # _________________________
    # Faire l'AFC proprment dit
    if AFCWITHOBS :
        VAPT, F1U, CAi, CRi, F2V, CAj, F1sU = afaco(Tp_);
        XoU = F1U[Nmdlok,:]; # coord des Obs
    else : # Les obs en supplémentaires
        VAPT, F1U, CAi, CRi, F2V, CAj, F1sU = afaco(Tp_, Xs=[Nobsc]);
        XoU = F1sU; # coord des Obs (ok j'aurais pu mettre directement en retour de fonction...)
    #
    #-----------------------------------------
    # MODELE MOYEN (pondéré ou pas) PAR CLUSTER D'UNE CAH
    if 1 : # CAH on afc Models's coordinates (without obs !!!???)
        metho_ = 'ward'; #'complete' 'single' 'average' 'ward'; #<><><><><><<><><>
        dist_  = 'euclidean';
        coord2take = np.arange(NBCOORDAFC4CAH); # Coordonnées de l'AFC àprendre pour la CAH
        if AFCWITHOBS :
            if CAHWITHOBS : # Garder les Obs pour la CAH
                Z_ = linkage(F1U[:,coord2take], metho_, dist_);
            else : # Ne pas prendre les Obs dans la CAH (ne prendre que les modèles)
                Z_ = linkage(F1U[0:Nmdlok,coord2take], metho_, dist_);
            #
        else : # Cas AFC sans les Obs
            if CAHWITHOBS : # Alors rajouter les Obs en Supplémentaire
                F1U_ = np.concatenate((F1U, F1sU));
                Z_   = linkage(F1U_[:,coord2take], metho_, dist_);
            else : # Ne pas rajouter les obs pour la CAH
                Z_   = linkage(F1U[:,coord2take], metho_, dist_);

        if Visu_Dendro : # dendrogramme
            plt.figure();
            if CAHWITHOBS :
                R_ = dendrogram(Z_,Nmdlok+1,'lastp');
            else :
                R_ = dendrogram(Z_,Nmdlok,'lastp');           
            L_ = np.array(NoCAHindnames) # when AFCWITHOBS, "Obs" à déjà été rajouté à la fin
            plt.xticks((np.arange(Nleaves_)*10)+7,L_[R_['leaves']], fontsize=11,
                        rotation=45,horizontalalignment='right', verticalalignment='baseline')
            plt.title("AFC: Coord(%s), dendro. Métho=%s, dist=%s, nb_clust=%d"
                      %((coord2take+1).astype(str),metho_,dist_,nb_clust))
        #
        if nb_clust < 0 :
            Loop_nb_clust = np.arange(-nb_clust-1)+2;   MultiLevel = True;
        else :
            Loop_nb_clust = np.array([nb_clust]);       MultiLevel = False;
        if max(Loop_nb_clust) > Nmdlok :
            print("Warning : You should not require more clusters level than the number of (valid) models");
        subc_, subl_ = lcsub(max(Loop_nb_clust));
        if MultiLevel == True :
            bestglob_ = 0.0;
            bestloc_  = []; # meilleure perf localement (i.e pour un niveau de coupe)
            ninbest_  = []; # nombre de modèle concernés par bestloc_
        for nb_clust in Loop_nb_clust :
            best_ = 0.0;
            class_afc = fcluster(Z_,nb_clust,'maxclust'); 
            if CAHWITHOBS : # On enlève le dernier indice de classe qui correspond
                # aux obs (je crois !!!???) parce qu'on ne veut pas un modèle moyen
                # qui tienne compte des obs, ce serait tricher et donc faux
                class_afc = class_afc[0:Nleaves_-1];
            #
            if MultiLevel == False : # Si il n'y a qu'un seul niveau de découpe, on fera la figure 
                figclustmoy = plt.figure(); # pour les différents cluster induit par ce niveau.
            #
            for ii in np.arange(nb_clust) :
                iclust  = np.where(class_afc==ii+1)[0];
                # print(len(iclust),iclust)
                # Du coup, en ayant enlever l'indice de classe de l'obs dans class_afc,
                # iclust peut etre vide (ce qui devrait correspondre au cluster qui n'aurait
                # contenu que les Obs).
                if len(iclust) == 0 :
                    print("\nProbably Obs cluster (%d) empty\n"%(ii+1));
                    continue;
                #
                if  ii+1 in AFC_Visu_Classif_Mdl_Clust :
                    plt.figure(facecolor='w');
                    for jj in np.arange(len(iclust)) :
                        ijj      = iclust[jj];
                        bmusj_   = ctk.mbmus (sMapO, Data=TDmdl4CT[ijj]);
                        classej_ = class_ref[bmusj_].reshape(NDmdl);
                        XCM_     = dto2d(classej_,LObs,CObs,isnumObs); # Classification géographique
                        plt.subplot(8,6,jj+1); # plt.subplot(7,7,jj+1);
                        plt.imshow(XCM_, interpolation='none',cmap=ccmap, vmin=1,vmax=nb_class);
                        plt.axis('off');
                        plt.title("%s(%.0f%c)"%(CAHindnames[ijj],100*Tperfglob[ijj,0],'%'),fontsize=sztitle);
                            # même si y'a 'Obs' dans CAHindnames, ca devrait pas apparaitre
                            # car elles sont à la fin et ont été retirées de class_afc
                    plt.suptitle("Classification des modèles du cluster %d"%(ii+1));
                if MultiLevel == False :    
                    print("%d Modèles du cluster %d :\n"%(len(iclust),ii+1), CAHindnames[iclust]);
                #
                # Modèle Moyen d'un cluster (plus de gestion de pondération)
                CmdlMoy  = Dmdlmoy4CT(TDmdl4CT,iclust,pond=None);                
                #
                #if 1 : # Affichage Data cluster moyen for CT
                if  ii+1 in AFC_Visu_Clust_Mdl_Moy_4CT :
                    aff2D(CmdlMoy,Lobs,Cobs,isnumobs,isnanobs,wvmin=wvmin,wvmax=wvmax,
                          figsize=(12,9), varnames=varnames);
                    plt.suptitle("MdlMoy[%s]\nclust%d %s(%d-%d) for CT\nmin=%f, max=%f, moy=%f, std=%f"
                            %(Tmdlname[Iok_][iclust],ii+1,fcodage,andeb,anfin,np.min(CmdlMoy),
                              np.max(CmdlMoy),np.mean(CmdlMoy),np.std(CmdlMoy)))    
                #
                # Classification du, des modèles moyen d'un cluster
                if MultiLevel == False : # 1 seul niveau de découpe, on fait la figure
                    plt.figure(figclustmoy.number); plt.subplot(subl_,subc_,ii+1);
                    Perfglob_ = Dgeoclassif(sMapO,CmdlMoy,LObs,CObs,isnumObs,TypePerf[0]);
                    plt.title("cluster %d, perf=%.0f%c"%(ii+1,100*Perfglob_,'%'),fontsize=sztitle);
                else : # Plusieurs niveaux de découpe, c'est pas la peine de faire toutes ces
                       # figures, mais on a besoin de la perf
                    Perfglob_ = Dgeoclassif(sMapO,CmdlMoy,LObs,CObs,isnumObs,TypePerf[0],visu=False);
                #
                if MultiLevel == True :
                    if Perfglob_ > bestglob_ :
                        print("%d Modèles du cluster %d :\n"%(len(iclust),ii+1), Tmdlname[iclust]);
                        print("      >>>>>>>> clust %d-%d : new best perf = %f <<<<<<<<"
                              %(nb_clust, ii+1, Perfglob_));
                        bestglob_ = Perfglob_
                    if Perfglob_ > best_ :
                        best_ = Perfglob_
                        nbest_ = len(iclust)
            if MultiLevel == True :
                bestloc_.append(best_)
                ninbest_.append(nbest_)
            # FIN de la boucle sur le nombre de cluster
        # FIN de la boucle sur les différents niveau de cluster
        if MultiLevel == True :
            plt.figure();
            plt.subplot(2,1,1); plt.plot(bestloc_,'-*');
            plt.xticks(np.arange(len(Loop_nb_clust)),Loop_nb_clust);
            plt.title("Meilleures perf pour chaque cas de découpe");
            
            plt.subplot(2,1,2); plt.plot(ninbest_,'-*');
            plt.xticks(np.arange(len(Loop_nb_clust)),Loop_nb_clust);
            plt.xlabel("nombre de découpe (clusters)");
            plt.title("Nombre de modèles dans le meilleur cluster de la découpe");
            plt.suptitle("best cluster depending nb cluster");
        #
        del metho_, dist_, Z_
        if MultiLevel == True :
            plt.show(); sys.exit(0)
    #
    # FIN du if 1 : MODELE MOYEN (pondéré ou pas) PAR CLUSTER D'UNE CAH
    #-----------------------------------------
    #      
    # choisir K et son facteur de zoom
    #K=CAi; xoomK=3000; # Pour les contrib Abs (CAi)
    K=CRi; xoomK=1000;  # Pour les contrib Rel (CRi)
    if 1 : # ori afc avec tous les points
        afcnuage(F1U,cpa=pa,cpb=po,Xcol=class_afc,K=K,xoomK=xoomK,linewidths=2,indname=NoAFCindnames);
        if NIJ==1 :
            plt.title("%sSST(%s)). %s%d AFC on classes of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,method_cah,nb_class));
        elif NIJ==3 or NIJ==2 :
            plt.title("%sSST(%s)). %s%d AFC on good classes of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,method_cah,nb_class));
    #
    if AFCWITHOBS  : # Mettre en évidence Obs
        plt.plot(F1U[Nmdlok,pa-1],F1U[Nmdlok,po-1], 'oc', markersize=20,
                 markerfacecolor='none',markeredgecolor='m',markeredgewidth=2);    
    else : # Obs en supplémentaire
        plt.text(F1sU[0,0],F1sU[0,1], ".Obs")
        plt.plot(F1sU[0,0],F1sU[0,1], 'oc', markersize=20,
                     markerfacecolor='none',markeredgecolor='m',markeredgewidth=2);
    #
    if 1 : # AJOUT ou pas des colonnes (i.e. des classes)
        colnames = (np.arange(nb_class)+1).astype(str)
        afcnuage(F2V,cpa=pa,cpb=po,Xcol=np.arange(len(F2V)),K=CAj,xoomK=xoomK,
                 linewidths=2,indname=colnames,holdon=True) 
        plt.axis("tight"); #?
    #
    if Visu_afcnu_det : # plot afc etape par étape"
        # Que les points lignes (modèles)
        K=CRi; xoomK=1000
        afcnuage(F1U,cpa=pa,cpb=po,Xcol=class_afc,K=K,xoomK=xoomK,linewidths=2,indname=NoAFCindnames);
        plt.title("%sSST(%s)). \n%s%d AFC (nij=%d) of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,method_cah,nb_class,NIJ));
        # Que les points colonnes (Classe)
        plt.figure()
        afcnuage(F2V,cpa=pa,cpb=po,Xcol=np.arange(len(F2V)),K=CAj,xoomK=xoomK,
                 linewidths=2,indname=colnames,holdon=True)
        plt.title("%sSST(%s)). \n%s%d AFC (nij=%d) of Completed Models (vs Obs)" \
                 %(fcodage,DATAMDL,method_cah,nb_class,NIJ));          
    #
    if Visu_Inertie : # Inertie
        inertie, icum = acp.phinertie(VAPT); 
        if NIJ==1 :
            plt.title("%sSST(%s)). \n%s%d AFC on classes of Completed Models (vs Obs)" \
                     %(fcodage,DATAMDL,method_cah,nb_class));
        elif NIJ==3 :
            plt.title("%sSST(%s)). \n%s%d AFC on good classes of Completed Models (vs Obs)" \
                     %(fcodage,DATAMDL,method_cah,nb_class));
#
#======================================================================
#
if STOP_BEFORE_GENERAL :
    plt.show(); sys.exit(0)
#**********************************************************************
#............................. GENERALISATION .........................
def mixtgeneralisation (TMixtMdl) :
    ''' Ici, j'ai :
             - Tmdlname : une table de NOMS de N modèles valides ; ##!!??? 
             - TDmdl4ct : la table correspondante des modèles 4CT (N, v,12)
        D'autre part, je vais définir d'une manière ou d'une autre les
        modèles devant participer à la définition du modèles moyen.
        Ici, il y a plusieurs possibilités :
        
        1) Une seule ligne de modèles à utiliser sans dicernement de classe
           par exemple :
        TMixtMdl = ['ACCESS1-3','NorESM1-ME','bcc-csm1-1','NorESM1-M','CESM1-BGC']
        
        2) Plusieurs lignes de modèle : une par classe, par exemple
        TMixtMdl = [['ACCESS1-3','NorESM1-ME','bcc-csm1-1','NorESM1-M','CESM1-BGC'],
             ['IPSL-CM5B-LR','ACCESS1-3','MPI-ESM-P','CMCC-CMS','GISS-E2-R-CC'],
             ['bcc-csm1-1-m','MIROC-ESM-CHEM','MIROC-ESM','CSIRO-Mk3-6-0','CanESM2'],
             ['MPI-ESM-MR','CMCC-CM','IPSL-CM5A-MR','FGOALS-g2','MPI-ESM-LR'],
             ['IPSL-CM5A-MR','CNRM-CM5-2','MPI-ESM-MR','MRI-ESM1','MRI-CGCM3'],
             ['FGOALS-s2','CNRM-CM5','CNRM-CM5-2','GFDL-CM3','CMCC-CM'],
             ['GFDL-CM3','GFDL-CM2p1','GFDL-ESM2G','CNRM-CM5','GFDL-ESM2M']];

        Dans les 2 cas, il faut :
        - Prendre les modèles de TMixtMdl à condition qu'ils soient aussi
          dans Tmdlname
        - Envisager ou pas une 2ème phase ...  
    '''
    # Je commence par le plus simple : Une ligne de modèle sans classe en une phase
    # Je prend le cas : CAH effectuée sur les 6 coordonnées d’une AFC  nij=3 ... 
    # TMixtMdl = ['CMCC-CM',   'MRI-ESM1',    'HadGEM2-AO','MRI-CGCM3',   'HadGEM2-ES',
    #             'HadGEM2-CC','FGOALS-g2',   'CMCC-CMS',  'GISS-E2-R-CC','IPSL-CM5B-LR',
    #             'GISS-E2-R', 'IPSL-CM5A-LR','FGOALS-s2', 'bcc-csm1-1'];
    #
    # déterminer l'indice des modèles de TMixtMdl dans Tmdlname
    IMixtMdl = [];
    for mname in TMixtMdl :
        im = np.where(Tmdlname == mname)[0];
        if len(im) == 1 :
            IMixtMdl.append(im[0])
    #
    if len(IMixtMdl) == 0 :
        print("\nGENERALISATION IMPOSSIBLE : AUCUN MODELE DISPONIBLE (sur %d)"%(len(TMixtMdl)))
        return
    else :
        print("\n%d modèles disponibles (sur %d) pour la generalisation : %s"
              %(len(IMixtMdl),len(TMixtMdl),Tmdlname[IMixtMdl]));
    #
    # Modèle moyen
    MdlMoy = Dmdlmoy4CT(TDmdl4CT,IMixtMdl);
    if 1 : # Affichage du moyen for CT
        aff2D(MdlMoy,Lobs,Cobs,isnumobs,isnanobs,wvmin=wvmin,wvmax=wvmax,figsize=(12,9));
        plt.suptitle("MdlMoy (%s) \n%s(%d-%d) for CT\nmin=%f, max=%f, moy=%f, std=%f"
                    %(Tmdlname[IMixtMdl], fcodage,andeb,anfin,np.min(MdlMoy),
                     np.max(MdlMoy),np.mean(MdlMoy),np.std(MdlMoy)))
    #
    # Classification du modèles moyen
    plt.figure();
    Perfglob_ = Dgeoclassif(sMapO,MdlMoy,LObs,CObs,isnumObs,TypePerf[0]); #use perfbyclass
    plt.title("MdlMoy(%s), perf=%.0f%c"%(Tmdlname[IMixtMdl],100*Perfglob_,'%'),fontsize=sztitle);
    #tls.klavier();
#-----------------------------------------------------------
# Modèles Optimaux (Sopt) ;  Avec "NEW Obs - v3b " 1975-2005
# Je commence par le plus simple : Une ligne de modèle sans classe en une phase
# et une seule codification à la fois
# Sopt-1975-2005 : Les meilleurs modèles de la période "de référence" 1975-2005
#TMixtMdl= [];
#TMixtMdl =['CNRM-CM5', 'CMCC-CMS', 'CNRM-CM5-2', 'GFDL-CM3', 'FGOALS-s2']; 
TMixtMdl = Sfiltre; 
#
if TMixtMdl == [] :
    print("\nSopt non renseigné ; Ce Cas n'a pa encore été prévu")
else :
    print("\n%d modele(s) de generalisation : %s "%(len(TMixtMdl),TMixtMdl))
    mixtgeneralisation (TMixtMdl);
#
#**********************************************************************
plt.show();
#___________
print("WITHANO,UISST,climato,NIJ :\n", WITHANO, UISST,climato,NIJ)
import os
print("whole time code %s: %f" %(os.path.basename(sys.argv[0]), time()-tpgm0));
#
#======================================================================
#
