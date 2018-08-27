#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:38:49 2018

@author: carlos
"""
    
import numpy as np
import os
import matplotlib.pyplot as plt
from   matplotlib import cm
import pickle
from   scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from models_def_tb import *

import ctLoopTools  as ctloop
import ctObsMdldef  as ctobs
import UW3_triedctk as ctk
import localdef     as ldef
##from   ctObsMdldef import *

# PARAMETRAGE (#1) DU CAS
from ParamCas import *
#import ParamCas


obs_data_path = '../Datas'

#======================================================================
# Pour initialiser le generateur de nombres aleatoires utilise 
# Reset effectué juste avant l'initialisation de la Carte Topologique:
# (si tseed est diff de 0 alors il est ajouté dans le nom du cas)
tseed = 0;
#tseed = 9;
#tseed = np.long(time());
#tseed = np.long(np.mod(time()*1e6,1e3)); # un chiffre aleatoire entre 0 et 999

### INITIALISATION ############################################################
ctloop.printwarning(["","INITIALISATION".center(75),""],
                    "   Nom du Cas de Base ....... '{:s}'".format(case_label_base))
case_name_base,casetime,casetimelabel,casetimeTlabel,varnames,list_of_plot_colors = \
    ctloop.initialisation(case_label_base,tseed=tseed)
#
### ACQUISITION DES DONNEES D'OBSERVATION #####################################
ctloop.printwarning([ "","ACQUISITION DES DONNEES D'OBSERVATION: '{:s}'".format(DATAOBS).center(75),""])
data_label_base,sst_obs,lon,lat = ctloop.read_obs(obs_data_path,DATAOBS)

# -----------------------------------------------------------------------------
# Complete le Nom du Cas
case_label = case_name_base+"_"+data_label_base
print("\n{:*>86s}\nCase label with data version: {}\n".format('',case_label))
#
### ACQUISITION DES DONNEES D'OBSERVATION #####################################
sst_obs,lon,lat,ilat,ilon = ctloop.get_zone_obs(sst_obs,lon,lat,
                                                size_reduction=SIZE_REDUCTION,
                                                frlon=frlon,tolon=tolon,
                                                frlat=frlat,tolat=tolat)
#
print(" - Nouvelles dimensions of SST Obs : {}".format(sst_obs.shape))
print(" - Lat : {} values from {} to {}".format(len(lat),lat[0],lat[-1]))
print(" - Lon : {} values from {} to {}".format(len(lon),lon[0],lon[-1]))

Nobs,Lobs,Cobs = np.shape(sst_obs);
print("obs.shape : ", Nobs,Lobs,Cobs);
#
# Calcule la position des NaN et des non NaN 
# (cherche uniquement dans la premiere image de la serie)
isnanobs = np.where(np.isnan(sst_obs[0].reshape(Lobs*Cobs)))[0];
isnumobs = np.where(~np.isnan(sst_obs[0].reshape(Lobs*Cobs)))[0];
#
# Codification des Obs 4CT 
sst_obs_coded, Dobs, NDobs = ctobs.datacodification4CT(sst_obs,
            TRENDLESS=TRENDLESS, WITHANO=WITHANO, climato=climato, UISST=UISST,
            NORMMAX=NORMMAX, CENTRED=CENTRED);
#
# -----------------------------------------------------------------------------
# Repertoire principal des maps (les objets des SOM) et sous-repertoire por le cas en cours 
if SAVEMAP :
    if not os.path.exists(MAPSDIR) :
        os.makedirs(MAPSDIR)
    case_maps_dir = os.path.join(MAPSDIR,case_label)
    if not os.path.exists(case_maps_dir) :
        os.makedirs(case_maps_dir)
# -----------------------------------------------------------------------------
# Repertoire principal des figures et sous-repertoire por le cas en cours 
if SAVEFIG :
    if not os.path.exists(FIGSDIR) :
        os.makedirs(FIGSDIR)
    case_figs_dir = os.path.join(FIGSDIR,case_label)
    if not os.path.exists(case_figs_dir) :
        os.makedirs(case_figs_dir)
#
#%%
if WITHANO :
    #wvmin=-3.9; wvmax = 4.9; # ok pour obs 1975-2005 : ANO 4CT: min=-3.8183; max=4.2445 (4.9 because ...)
    #wvmin=-4.3; wvmax = 4.9; # ok pour obs 2006-2017 : ANO 4CT: min=-4.2712; max=4.3706
    #wvmin = -4.9; wvmax = 4.9; # pour mettre tout le monde d'accord ?
    #wvmin = -4.0; wvmax = 4.0; # pour mettre tout le monde d'accord ?
    #wvmin = -3.0; wvmax = 3.0; # pour mettre tout le monde d'accord ?
    wvmin = -2.5; wvmax = 2.5; # pour mettre tout le monde d'accord ?
else : # On suppose qu'il s'agit du brute ...
    wvmin =16.0; wvmax =30.0; # ok pour obs 1975-2005 : SST 4CT: min=16.8666; max=29.029
#
if Visu_ObsStuff : # Visu (et sauvegarde éventuelle de la figure) des données
    if SIZE_REDUCTION == 'All' :
        lolast = 4
    else :
        lolast = 2
    if SIZE_REDUCTION == 'All' :
        figsize = (12,7)
        wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.97;
    elif SIZE_REDUCTION == 'sel' :
        figsize=(10,8.5)
        wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.965;
    #
    localcmap = eqcmap
    stitre = "Observed SST {:s} MEAN ({:d}-{:d})".format(fcodage,andeb,anfin)
    if Show_ObsSTD :
        stitre += "(monthly {:d} years STD in contours)".format(Nda)
    #
    ctloop.plot_obs(sst_obs_coded,Dobs,lon,lat,varnames=varnames,cmap=localcmap,
                    isnanobs=isnanobs,isnumobs=isnumobs,
                    title=stitre, wvmin=wvmin, wvmax=wvmax,
                    lolast=lolast,figsize=figsize,
                    wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                    climato=climato,NORMMAX=NORMMAX,CENTRED=CENTRED,Show_ObsSTD=Show_ObsSTD
                    )
    #
    if SAVEFIG : # sauvegarde de la figure
        figfile = "Fig_Obs4CT"
        if Show_ObsSTD :
            figfile += "+{:d}ySTD".format(Nda)
        figfile += "_Lim{:+.1f}-{:+.1f}_{:s}_{:s}{:s}Clim-{:d}-{:d}_{:s}".format(wvmin,wvmax,localcmap.name,fprefixe,fshortcode,andeb,anfin,data_label_base)
        # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
        # eventuellement en PDF, si SAVEPDF active. 
        ctloop.do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    del localcmap
#%%
if 0 :
    if Visu_ObsStuff : # Visu (et sauvegarde éventuelle de la figure) des données
        if SIZE_REDUCTION == 'All' :
            lolast = 4
        else :
            lolast = 2
        if SIZE_REDUCTION == 'All' :
            figsize = (12,7)
            wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.97;
        elif SIZE_REDUCTION == 'sel' :
            figsize=(10,8.5)
            wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.965;
        #
        localcmap = eqcmap
        stitre = "Observed SST {:s} MEAN ({:d}-{:d}) - FREE LIMITS".format(fcodage,andeb,anfin)
        #
        ctloop.plot_obsbis(sst_obs_coded,Dobs,varnames=varnames,cmap=localcmap,
                        title=stitre,
                        figsize=figsize,
                        wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                        )
        #
        if SAVEFIG : # sauvegarde de la figure
            figfile = "Fig_Obs4CT_FREELIMITS_{:s}_{:s}{:s}Clim-{:d}-{:d}_{:s}".format(localcmap.name,
                                             fprefixe,fshortcode,andeb,anfin,data_label_base)
            # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
            # eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT)
#
if STOP_BEFORE_CT :
    plt.show(); sys.exit(0);
#
#%%
#%% ###################################################################
#                       Carte Topologique
#======================================================================
ctloop.printwarning([ "","CARTE TOPOLOGIQUE".center(75),""])
DO_NEXT = True
if SAVEMAP : # SI sauvegarde de la Map de SOM est ACTIVE
    mapfile = "Map_{:s}{:s}Clim-{:d}-{:d}_{:s}_ts-{}{}".format(fprefixe,fshortcode,
                   andeb,anfin,data_label_base,tseed,mapfileext)
    mapPathAndFile = case_maps_dir+os.sep+mapfile
    if  os.path.exists(mapPathAndFile):
        if REWRITEMAP :
            msg2 = u"REWRITEMAP active, le fichier MAP sera reecrit"
            ctloop.printwarning([ u"Attention, le fichier MAP existe déjà, ",
                           "    {}/".format(os.path.dirname(mapPathAndFile)),
                           "         {}".format(os.path.basename(mapPathAndFile)),
                           "",
                           u"mais le ReRun est active, on re-apprend." ],
                    msg2)
            DO_NEXT = True
        #
        else :
            msg2 = u"REWRITEMAP inactive"
            ctloop.printwarning([ u"Attention, le fichier MAP existe déjà et sera chargé, ",
                           "    {}/".format(os.path.dirname(mapPathAndFile)),
                           "         {}".format(os.path.basename(mapPathAndFile)),
                           "",
                           u"on saute le processus d'entrainemant de la MAP." ],
                    msg2)
            DO_NEXT = False
    #
    elif not REWRITEMAP and os.path.exists(mapPathAndFile):
        msg2 = u"REWRITEMAP inactive"
        ctloop.printwarning([ u"Attention, le fichier MAP existe déjà, ",
                       "    {}/".format(os.path.dirname(mapPathAndFile)),
                       "         {}".format(os.path.basename(mapPathAndFile)),
                       "",
                       u"on saute le processus d'entrainemant de la MAP." ],
                msg2)
        DO_NEXT = False
#
if DO_NEXT :
    #    sMapO = SOM.SOM('sMapObs', Dobs, mapsize=[nbl, nbc], norm_method=norm_method, \
    #                    initmethod='random', varname=varnames)
    sMapO,eqO,etO = ctloop.do_ct_map_process(Dobs,name='sMapObs',mapsize=[nbl, nbc],
                                             tseed=0,varname=varnames,
                                             norm_method='data',
                                             initmethod='random',
                                             etape1=[epoch1,radini1,radfin1],
                                             etape2=[epoch2,radini2,radfin2],
                                             verbose='on', retqerrflg=True)
    #
    print("Obs case: {}\n          date ... {}]\n          tseed={}\n          Qerr={:8.6f} ... Terr={:.6f}".format(
        case_label,casetimelabel,tseed,eqO,etO))
    somtime = casetime
    #
    if SAVEMAP : # sauvegarde de la Map de SOM
        ctloop.printwarning([ "==> Saving MAP in file :",
                       "    {}/".format(os.path.dirname(mapPathAndFile)),
                       "         {}".format(os.path.basename(mapPathAndFile)) ])
        map_d ={ "map" : sMapO, "tseed" : tseed, "somtime" : somtime }
        map_f = open(mapPathAndFile, 'wb')
        pickle.dump(map_d, map_f)
        map_f.close()

elif os.path.exists(mapPathAndFile) and RELOADMAP :
        #reload object from file
        ctloop.printwarning([ "==> Loading MAP from file :",
                       "    {}/".format(os.path.dirname(mapPathAndFile)),
                       "         {}".format(os.path.basename(mapPathAndFile)) ])
        map_f = open(mapPathAndFile, 'rb')
        map_d = pickle.load(map_f)
        map_f.close()
        sMapO   = map_d['map']
        tseed   = map_d['tseed'] # seed used whent initializing sMapO, originally
        somtime = map_d['somtime'] # seed used whent initializing sMapO, originally
        somtimelabel = somtime.strftime("%d %b %Y @ %H:%M:%S")
        somtimeTlabel = somtime.strftime("%Y%m%dT%H%M%S")
        #            err = np.mean(getattr(self, 'bmu')[1])
        qerr = np.mean(sMapO.bmu[1])
        # + err topo maison
        bmus2O = ctk.mbmus (sMapO, Data=None, narg=2);
        etO    = ctk.errtopo(sMapO, bmus2O); # dans le cas 'rect' uniquement
        #print("Obs, erreur topologique = %.4f" %etO)
        print("Obs case: {}\n          loaded sMap date ... {}]\n          used tseed={} ... Qerr={:8.6f} ... Terr={:.6f}".format(case_label,
              somtimelabel,tseed,qerr,etO))
else :
    try :
        # un print utilisant sMapO, s'il nexiste pas declanche une exeption !
        print("Current sMap name {}".format(sMapO.name))
        
    except Exception as exc :
        print("\n*** {:*<66s} ***".format("*"))
        print("*** {:66s} ***".format(" "))
        print("*** {:^66s} ***".format("Can't continue, no sMap in memory"))
        if not REWRITEMAP :
            print("*** {:^66s} ***".format("Think to activate REWRITEMAP variable and rerun"))
        print("*** {:66s} ***".format(" "))
        print("*** {:*<66s} ***\n***".format("*"))
        print("*** exception type .. {} ".format(exc.__class__))
        # affiche exception de type  exceptions.ZeroDivisionError
        print("*** message ......... {}".format(exc))
        print("***\n*** {:*<66s} ***\n***".format("*"))
        # Force l'arret (ou l'activation d'un try a plus haut niveau, s'il existe)
        raise Exception("*** no sMap in memory, stop running ***")
        #sys.exit()

    else :
        print("Using current sMap of size {}".format(sMapO.mapsize))
# #########################################################################
# C.T. Visualisation ______________________________________________________
# #########################################################################
if Visu_CTStuff : #==>> la U_matrix
    fig = plt.figure(figsize=(6,8));
    fignum = fig.number
    #wspace=0.02; hspace=0.14; top=0.80; bottom=0.02; left=0.01; right=0.95;
    #fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
    a=sMapO.view_U_matrix(fignum=fignum, distance2=2, row_normalized='No', show_data='Yes', \
                      contooor='Yes', blob='No', save='No', save_dir='');
    plt.suptitle("Obs, The U-MATRIX", fontsize=16,y=1.0);
#
if Visu_CTStuff : #==>> La carte
    fig = plt.figure(figsize=(8,8));
    fignum = fig.number
    wspace=0.02; hspace=0.14; top=0.925; bottom=0.02; left=0.01; right=0.95;
    fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
    #ctk.showmap(sMapO,sztext=11,colbar=1,cmap=cm.rainbow,interp=None);
    ctk.showmap(sMapO,sztext=11,cbsztext=8,fignum=fignum,
                colbar=1,cmap=cm.rainbow,interp=None,noaxes=False,
                noticks=False,xticks=np.arange(nbc),yticks=np.arange(nbl),
                nolabels=True,
                );
    plt.suptitle("Obs, Les Composantes de la carte", fontsize=16);
#
# Other stuffs ______________________________________
# C.T. Dendrogram _________________________________________________________
bmusO     = ctk.mbmus (sMapO, Data=Dobs); # déjà vu ? conditionnellement ?

# Performs hierarchical/agglomerative clustering on the condensed distance matrix data
ctZ_ = linkage(sMapO.codebook, method=method_cah, metric=dist_cah);
#
# Forms flat clusters from the hierarchical clustering defined by the linkage matrix Z.
class_ref = fcluster(ctZ_,nb_class,criterion='maxclust'); # Classes des referents

if Visu_Dendro :
    figsize=(14,6);
    wspace=0.0; hspace=0.2; top=0.92; bottom=0.12; left=0.05; right=0.99;
    stitre = "SOM Codebook Dendrogram for HAC (map size={:d}x{:d})".format(nbl,nbc);
    ctloop.plot_ct_dendro(sMapO.codebook, ctZ_, nb_class,
                          title=stitre,
                          figsize=figsize,
                          wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right
                          )
#
# Transcodage des indices des classes
if TRANSCOCLASSE is not '' :
    class_ref = ctobs.transco_class(class_ref,sMapO.codebook,crit=TRANSCOCLASSE);
#
classe_Dobs = class_ref[bmusO].reshape(NDobs); #(sMapO.dlen)
XC_Ogeo     = ctobs.dto2d(classe_Dobs,Lobs,Cobs,isnumobs); # Classification géographique
#
fond_C = np.ones(len(classe_Dobs))
fond_C = ctobs.dto2d(fond_C,Lobs,Cobs,isnumobs,missval=0.5)
#
# Nombre de pixels par classe (pour les obs)
Nobsc = np.zeros(nb_class)
for c in np.arange(nb_class)+1 :
    iobsc = np.where(classe_Dobs==c)[0]; # Indices des classes c des obs
    Nobsc[c-1] = len(iobsc);

#
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Figure 1 pour Article 
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
if Visu_ObsStuff or Visu_UpwellArt : # Visualisation de truc liés au Obs
    if SIZE_REDUCTION == 'All' :
        figsize = (9,6)
        wspace=0.0; hspace=0.0; top=0.94; bottom=0.08; left=0.08; right=0.98;
        nticks = 5; # 4
    elif SIZE_REDUCTION == 'sel' :
        figsize = (9,9)
        wspace=0.0; hspace=0.0; top=0.94; bottom=0.08; left=0.05; right=0.96;
        nticks = 2; # 4
    #
    stitre = "Observations ({:d}-{:d}), {} Class Geographical Representation".format(andeb,anfin,nb_class)
    #
    ctloop.plot_fig01_article(XC_Ogeo,lon,lat,nb_class,classe_Dobs,title=stitre,
                   cmap=ccmap,
                   figsize=figsize,
                   wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                   nticks=nticks,
                   )
    #
    if SAVEFIG : # sauvegarde de la figure
        if Visu_UpwellArt :
            figfile = "FigArt_"
        else :
            figfile = "Fig_"
        figfile += "ObsGeo{:d}Class_{:s}{:s}Clim-{:d}-{:d}_{:s}".format(nb_class,fprefixe,fshortcode,andeb,anfin,data_label_base)
        ctloop.do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT,fig2ok=SAVEPDF,ext2=VFIGEXT)
#
if Visu_ObsStuff or Visu_UpwellArt : # Visualisation de truc liés au Obs
    stitre = "Observations, Monthly Mean by Class (method: {:s})".format(method_cah)
    ctloop.plot_mean_curve_by_class(sst_obs_coded,nb_class,classe_Dobs,varnames=varnames,
                                    title=stitre,
                                    pcmap=pcmap,
                                    figsize=figsize,
                                    wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                                    )
    if SAVEFIG : # sauvegarde de la figure
        if Visu_UpwellArt :
            figfile = "FigArt_"
        else :
            figfile = "Fig_"
        figfile += "MeanByClass_{:d}Class_{:s}{:s}Clim-{:d}-{:d}_{:s}".format(nb_class,fprefixe,fshortcode,andeb,anfin,data_label_base)
        ctloop.do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT,fig2ok=SAVEPDF,ext2=VFIGEXT)
#
if Visu_CTStuff : # Visu des profils des référents de la carte
    if SIZE_REDUCTION == 'All' :
        figsize = (7.5,12)
        wspace=0.01; hspace=0.05; top=0.945; bottom=0.04; left=0.15; right=0.86;
    elif SIZE_REDUCTION == 'sel' :
        figsize=(8,8)
        wspace=0.01; hspace=0.04; top=0.945; bottom=0.04; left=0.04; right=0.97;
    ctloop.plot_ct_profils(sMapO,Dobs,class_ref,varnames=varnames,
                           pcmap=pcmap,
                           figsize=figsize,
                           wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                           )

if STOP_BEFORE_MDLSTUFF :
    plt.show(); sys.exit(0)
#
#%%
if 1 :
    ctloop.printwarning([ "","TRAITEMENTS DES DONNEES DES MODELES OCEANOGRAPHIQUES".center(75),""])
    #######################################################################
    #                        MODELS STUFFS START HERE
    #======================================================================
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
    #Sfiltre = ['CMCC-CM', 'HadGEM2-ES', 'HadGEM2-AO', 'HadGEM2-CC', 'CMCC-CMS',
    #           'CNRM-CM5-2', 'CanESM2', 'CanCM4', 'GFDL-CM3', 'CNRM-CM5',
    #           'FGOALS-s2', 'CSIRO-Mk3-6-0', 'CMCC-CESM'];
    # best cum Methode des Regroupements Icrementaux
    #Sfiltre = ['CMCC-CM'];
    Sfiltre = None
    #Sfiltre = ['CanCM4', 'CNRM-CM5', 'CMCC-CMS',]
           #<<<<<<<<
    #                        MODELS STUFFS START HERE
    #======================================================================
    ctloop.printwarning([ "    MODELE: INITIALISATION AND FIRST LOOP" ])
    TDmdl4CT0,Tmdlname0,Tmdlnamewnb0,Tmdlonlynb0,Tperfglob4Sort0,Tclasse_DMdl0,\
        Tmoymensclass0,Dmdl,NDmdl,Nmdlok,Smoy_101,Tsst_102 = ctloop.do_models_startnloop(sMapO,
                                Tmodels,Tinstit,ilat,ilon,
                                isnanobs,isnumobs,nb_class,class_ref,classe_Dobs,
                                Tnmodel=Tnmodel,
                                Sfiltre=Sfiltre,
                                TypePerf=TypePerf,
                                obs_data_path=obs_data_path,
                                DATAMDL=DATAMDL,
                                MDLCOMPLETION=MDLCOMPLETION,
                                SIZE_REDUCTION=SIZE_REDUCTION,
                                NIJ=NIJ,
                                OK101=OK101,
                                OK102=OK102,
                                OK106=OK106,
                                )
    #
    # -------------------------------------------------------------------------
    ctloop.printwarning([ "    MODELS: PLOT 101 AND 102 AND PAST FIRST LOOP" ])
    if mdlnamewnumber_ok :
        Tmdlname10X = Tmdlnamewnb;
    else :
        Tmdlname10X = Tmdlname;
    #
    if len(Tmdlname10X) > 6 :            # Sous forme de liste, la liste des noms de modèles
        Tnames_ = Tmdlname10X;           # n'est pas coupé dans l'affichage du titre de la figure
    else :                               # par contre il l'est sous forme d'array; selon le cas, ou 
        Tnames_ = np.array(Tmdlname10X); # le nombre de modèles, il faut adapter comme on peut
    stitre = "Mdl_MOY{:}\n({:d} mod.) {:s}SST({:s})".format(Tnames_,len(Tnames_),fcodage,DATAMDL)
    ctloop.do_models_plot101et102_past_fl(Nmdlok,Lobs,Cobs,
                                          isnumobs,isnanobs,
                                          cmap=eqcmap,varnames=varnames,
                                          title101=stitre,
                                          title102=stitre,
                                          Smoy_101=Smoy_101,
                                          Tsst_102=Tsst_102,
                                          OK101=OK101,
                                          OK102=OK102,
                                          )
    #
    if OK101 :
        if SAVEFIG : # sauvegarde de la figure
            plt.figure(101);
            figfile = "Fig-101_{:s}{:s}_{:s}{:s}{:d}Mdl_MOY_{:d}-mod".format(fprefixe,
                               SIZE_REDUCTION,fshortcode,method_cah,nb_class,Nmodels)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    if OK102 : 
        if SAVEFIG :
            plt.figure(102);
            figfile = "Fig-102_{:s}{:s}_{:s}{:s}{:d}Mdl_ECRTYPE_{:d}-mod".format(fprefixe,
                               SIZE_REDUCTION,fshortcode,method_cah,nb_class,Nmodels)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    # -------------------------------------------------------------------------
    ctloop.printwarning([ "    MODELES: PRIOR SECOND LOOP" ])
    # re-initialise des variables qui seront alterees par la fonction "do_models_pior_second_loop"
    TDmdl4CT       = np.copy(TDmdl4CT0)
    Tmdlname       = np.copy(Tmdlname0)
    Tmdlnamewnb    = np.copy(Tmdlnamewnb0)
    Tmdlonlynb     = np.copy(Tmdlonlynb0)
    Tperfglob4Sort = np.copy(Tperfglob4Sort0)
    Tclasse_DMdl   = np.copy(Tclasse_DMdl0)
    Tmoymensclass  = np.copy(Tmoymensclass0)
    #
    TDmdl4CT,Tmdlname,Tmdlnamewnb,Tmdlonlynb,Tperfglob4Sort,Tclasse_DMdl,Tmoymensclass,\
        min_moymensclass,max_moymensclass,\
        MaxPerfglob_Qm,IMaxPerfglob_Qm = ctloop.do_models_pior_second_loop(
                                TDmdl4CT,Tmdlname,Tmdlnamewnb,Tmdlonlynb,Tperfglob4Sort,
                                Tclasse_DMdl,Tmoymensclass,
                                MCUM,
                                same_minmax_ok=same_minmax_ok,
                                OK106=OK106,
                                )
    #
    # -------------------------------------------------------------------------
    ctloop.printwarning([ "    MODELS: SECOND LOOP" ])
    if OK104 :
        suptitle104="%sSST(%s)).  Classification of Completed Models (vs Obs) (%d models)" \
                     %(fcodage,DATAMDL,Nmodels);
    if OK105 : 
        suptitle105="%sSST(%s)).  Classification of Completed Models (vs Obs) (%d models)" \
                     %(fcodage,DATAMDL,Nmodels);
    if OK106 :
        suptitle106="MOY - MoyMensClass(%sSST(%s)).)  Classification of Completed Models (vs Obs) (%d models)" \
                     %(fcodage,DATAMDL,Nmodels);
    if OK107 :
        suptitle107="VARiance(%sSST(%s)).) Variance (by pixel) on Completed Models (%d models)" \
                     %(fcodage,DATAMDL,Nmodels);
    if OK108 :
        suptitle108="MCUM - %sSST(%s)).  Classification of Completed Models (vs Obs) (%d models)" \
                     %(fcodage,DATAMDL,Nmodels);
    #
    if OK109 :
        suptitle109="VCUM - %sSST(%s)). Variance sur la Moyenne Cumulée de Modeles complétés (%d models)" \
                     %(fcodage,DATAMDL,Nmodels);
    if SIZE_REDUCTION == 'All' :
        figsize=(18,11);
        wspace=0.01; hspace=0.15; top=0.94; bottom=0.05; left=0.01; right=0.99;
    elif SIZE_REDUCTION == 'sel' :
        figsize=(18,12);
        wspace=0.01; hspace=0.14; top=0.94; bottom=0.04; left=0.01; right=0.99;
    #
    sztitle = 10;
    suptitlefs = 16
    ysuptitre = 0.99
    Tperfglob,Tperfglob_Qm,Tmdlname,Tmdlnamewnb,Tmdlonlynb,TTperf,\
        TDmdl4CT = ctloop.do_models_second_loop(sst_obs_coded,Dobs,lon,lat,sMapO,XC_Ogeo,TDmdl4CT,
                                Tmdlname,Tmdlnamewnb,Tmdlonlynb,
                                Tperfglob4Sort,Tclasse_DMdl,Tmoymensclass,
                                MaxPerfglob_Qm,IMaxPerfglob_Qm,
                                min_moymensclass,max_moymensclass,
                                MCUM,Lobs,Cobs,NDobs,NDmdl,
                                isnumobs,nb_class,class_ref,classe_Dobs,fond_C,
                                ccmap=ccmap,pcmap=pcmap,sztitle=sztitle,ysstitre=ysstitre,
                                ysuptitre=ysuptitre,suptitlefs=suptitlefs,
                                NIJ=NIJ,
                                FONDTRANS=FONDTRANS,
                                TypePerf=TypePerf,
                                mdlnamewnumber_ok=mdlnamewnumber_ok,
                                OK104=OK104, OK105=OK105, OK106=OK106, OK107=OK107, OK108=OK108, OK109=OK109,
                                figsize=figsize,
                                wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                                suptitle104=suptitle104,  suptitle105=suptitle105,
                                suptitle106=suptitle106,  suptitle107=suptitle107,
                                suptitle108=suptitle108,  suptitle109=suptitle109,
                                )
    # Dmdl_TVar,DMdl_Q,DMdl_Qm,Dmdl_TVm
    if OK104 :
        if SAVEFIG : # sauvegarde de la figure
            plt.figure(104);
            figfile = "Fig-104_{:s}{:s}_{:s}{:s}{:d}MdlvsObstrans_{:d}-mod".format(fprefixe,
                               SIZE_REDUCTION,fshortcode,method_cah,nb_class,Nmodels)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    if OK105 : 
        if SAVEFIG :
            plt.figure(105);
            figfile = "Fig-105_{:s}{:s}_{:s}{:s}{:d}Mdl_{:d}-mod".format(fprefixe,
                               SIZE_REDUCTION,fshortcode,method_cah,nb_class,Nmodels)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    if OK106 :
        if SAVEFIG :
            plt.figure(106);
            figfile = "Fig-106_{:s}{:s}_{:s}{:s}{:d}moymensclass_{:d}-mod".format(fprefixe,
                               SIZE_REDUCTION,fshortcode,method_cah,nb_class,Nmodels)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    if OK107 :
        if SAVEFIG :
            plt.figure(107);
            figfile = "Fig-107_{:s}VAR_{:s}_{:s}Mdl_{:d}-mod".format(fprefixe,
                               SIZE_REDUCTION,fshortcode,Nmodels)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    if MCUM>0 and OK108 :
        if SAVEFIG :
            plt.figure(108);
            figfile = "Fig-108_{:s}MCUM_{:s}_{:s}{:s}{:d}Mdl_{:d}-mod".format(fprefixe,
                               SIZE_REDUCTION,fshortcode,method_cah,nb_class,Nmodels)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    if MCUM>0 and OK109 :
        if SAVEFIG :
            plt.figure(109);
            figfile = "Fig-109_{:s}VCUM_{:s}_{:s}Mdl_{:d}-mod".format(fprefixe,
                               SIZE_REDUCTION,fshortcode,Nmodels)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    # -------------------------------------------------------------------------
    ctloop.printwarning([ "    MODELS: APRES SECOND LOOP" ])
    # Figure a ploter apres le deuxieme loop
    if Visu_preACFperf : # Tableau des performances en figure de courbes
        stitre = "SST {:s} ({:d}-{:d}) - {:d} Classes -  Classification Indices of Completed Models (vs Obs) ({:d} models)".format(fcodage,andeb,anfin,nb_class,Nmodels)
        ctloop.do_models_after_second_loop(Tperfglob,Tperfglob_Qm,Tmdlname,list_of_plot_colors,
                                           title=stitre,
                                           TypePerf=TypePerf,fcodage=fcodage)
        #
        if SAVEFIG :
            figfile = "Fig_{:s}perf-by_model_{:s}_{:s}{:s}{:d}Mdl_{:d}-mod".format(fprefixe,
                           SIZE_REDUCTION,fshortcode,method_cah,nb_class,Nmodels)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
#
if STOP_BEFORE_AFC :
    plt.show(); sys.exit(0)
