#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:38:49 2018

Version sur 'Master'

Exemple d'appel depuis python:
      runfile('/Users/carlos/Labo/NN_divers/DeepLearning/Projets/Projet_Upwelling/Charles/PourCarlos2_pour_Article/code/ctLoopMain.py',
              wdir='/Users/carlos/Labo/NN_divers/DeepLearning/Projets/Projet_Upwelling/Charles/PourCarlos2_pour_Article/code',
              args="--case=All -v")         

      runfile('/Users/carlos/Labo/NN_divers/DeepLearning/Projets/Projet_Upwelling/Charles/PourCarlos2_pour_Article/code/ctLoopMain.py',
              wdir='/Users/carlos/Labo/NN_divers/DeepLearning/Projets/Projet_Upwelling/Charles/PourCarlos2_pour_Article/code',
              args="--case=sel -v")         

ou bien en mode debug:
      debugfile('/Users/carlos/Labo/NN_divers/DeepLearning/Projets/Projet_Upwelling/Charles/PourCarlos2_pour_Article/code/ctLoopMain.py',
                wdir='/Users/carlos/Labo/NN_divers/DeepLearning/Projets/Projet_Upwelling/Charles/PourCarlos2_pour_Article/code',
                args="--case=All -v")
      
      debugfile('/Users/carlos/Labo/NN_divers/DeepLearning/Projets/Projet_Upwelling/Charles/PourCarlos2_pour_Article/code/ctLoopMain.py',
                wdir='/Users/carlos/Labo/NN_divers/DeepLearning/Projets/Projet_Upwelling/Charles/PourCarlos2_pour_Article/code',
                args="--case=sel -v")
@author: carlos
"""
    
import numpy as np
import sys, getopt, os
from   time  import time
import matplotlib.pyplot as plt
from   matplotlib import colors
from   matplotlib import cm
from   mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pickle
from   scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import ctLoopTools  as ctloop
import ctObsMdldef  as ctobs
import UW3_triedctk as ctk
import localdef     as ldef

# PARAMETRAGE (#1) DU CAS
from ParamCas import *

#%%
# Debugging: unable to see plots made with matplotlib ...
# try this ...
import matplotlib
#matplotlib.use('Qt4Agg', warn=False)
#matplotlib.use('Qt5Agg', warn=False)
matplotlib.rcParams['figure.dpi'] = 66    # to avoid figure windows to be reduced in size because of a tiny screen
matplotlib.rcParams['savefig.dpi'] = 180

#%%
def ctloop_init(case='None',verbose=False):
    print(case)
    print(verbose)
    
    if case is None or not isinstance(case, str) or case.upper() not in ('ALL', 'SEL') :
        if case is None or not isinstance(case, str) :
            ctloop.printwarning(["","INITIALISATION ERROR".center(75),""],
                    "   Undefined or invalid parameter value {}={}".format('case',case).center(75),
                    "You should give one in {} set.".format(('All','sel')).center(75))
        elif case.upper() not in ('ALL', 'SEL')  :
            ctloop.printwarning(["","INITIALISATION ERROR".center(75),""],
                    "   Undefined or invalid parameter value {}='{}''".format('case',case).center(75),
                    "You should give one in {} set.".format(('All','sel')).center(75))
        else :
            ctloop.printwarning(["","INITIALISATION ERROR".center(75),""],
                    "   bizarre, not expected error !  Testing parameter value {}={}".format('case',case).center(75),
                    "You should give one in {} set.".format(('All','sel')).center(75))
        raise
    #
    # Variables declarees dans ParamCas traitees comme globales !
    global SAVEFIG, SAVEPDF, SAVEMAP, REWRITEMAP, RELOADMAP, DATAOBS, DATAMDL, scenar, \
           Tinstit, Tmodels, Nmodels, Tnmodel
    global SIZE_REDUCTION, frlat, tolat, frlon, tolon, nbl, nbc, Parm_app, nb_class, NIJ, \
           nb_clust, NBCOORDAFC4CAH
    global MDLCOMPLETION, TRANSCOCLASSE, MCUM, ecvmin, ecvmax, OK101, OK102, OK104, \
           OK105, OK106, OK107, OK108, OK109
    global FONDTRANS, FIGSDIR, FIGEXT, FIGDPI, VFIGEXT, MAPSDIR, mapfileext, Nda
    global INDSC, TRENDLESS, WITHANO, climato, UISST, NORMMAX, CENTRED
    global method_cah, dist_can, method_afc, dist_afc, ccmap, dcmap, eqcmap
    global TypePerf, AFCWITHOBS, pa, po, CAHWITHOBS, STOP_BEFORE_CT, STOP_BEFORE_MDLSTUFF, \
           STOP_BEFORE_AFC, STOP_BEFORE_GENERAL
    global Show_ObsSTD, Show_ModSTD, Visu_ObsStuff, Visu_CTStuff, Visu_Dendro, Visu_preACFperf,\
           Visu_AFC_in_one, Visu_afcnu_det, Visu_Inertie
    global Visu_UpwellArt, FIGARTDPI, ysstitre, same_minmax_ok, mdlnamewnumber_ok, onlymdlumberAFC_ok, onlymdlumberAFCdendro_ok
    #
    # Variables globales declarees localement (les autres variables declarrees ici
    # seront retounees en argument de sortie (fonction return))
    global fprefixe, tpgm0, blockshow, fcodage, fshortcode

    #import ParamCas
    if case.upper() == 'ALL' :
        SIZE_REDUCTION = 'All';
        # A - Grande zone de l’upwelling (25x36) :
        #    Longitude : 45W à 10W (-44.5 à -9.5)
        #    Latitude :  30N à 5N ( 29.5 à  4.5)
        frlat =  29.5;  tolat =  4.5; #(excluded)
        #frlon = -44.5;  tolon = -8.5; #(excluded)   #(§:25x35)
        frlon = -44.5;  tolon = -8.5; #(excluded)   #(§:25x35)
        #   * Carte topologique et CAH : 30x4 (5, 5, 1, - 16, 1, 0.1) : TE=0.6824 ; QE=0.153757
        #   Nb_classe = 7
        nbl            = 30;  nbc =  4;  # Taille de la carte
        Parm_app       = ( 5, 5., 1.,  16, 1., 0.1); # Température ini, fin, nb_it
        #Parm_app       = ( 50, 5., 1.,  160, 1., 0.1); # Température ini, fin, nb_it
        nb_class       = 7; #6, 7, 8  # Nombre de classes retenu
        # et CAH for cluster with AFC
        NIJ            = 2;
        nb_clust       = 4; # Nombre de cluster
        #nb_clust       = 6; # Nombre de cluster
        NBCOORDAFC4CAH = nb_class - 1 - 1; # n premières coordonnées de l'afc à
        #NBCOORDAFC4CAH = nb_class - 1; # n premières coordonnées de l'afc à
        #NBCOORDAFC4CAH = nb_class; # n premières coordonnées de l'afc à
                        # utiliser pour faire la CAH (limité à nb_class-1).
    elif case.upper() == 'SEL' :
        SIZE_REDUCTION = 'sel';
        # B - Sous-zone ciblant l’upwelling (13x12) :
        #    LON:  28W à 16W (-27.5 to -15.5)
        #    LAT : 23N à 10N ( 22.5 to  9.5)
        frlat =  22.5;  tolat =   9.5; #(excluded)
        frlon = -27.5;  tolon = -15.5; #(excluded)   #(§:13x12)
        #   * Carte topologique et CAH : 17x6 (4, 4, 1, - 16, 1, 0.1) : TE=0.6067 ; QE=0.082044
        #   Nb_classe = 4
        nbl            = 17;  nbc =  6;  # Taille de la carte
        Parm_app       = ( 4, 4., 1.,  16, 1., 0.1); # Température ini, fin, nb_it
        nb_class       = 4; #6, 7, 8  # Nombre de classes retenu
        # et CAH for cluster with AFC
        NIJ            = 2;
        nb_clust       = 5; # Nombre de cluster
        NBCOORDAFC4CAH = nb_class - 1; # n premières coordonnées de l'afc à
    else :
        ctloop.printwarning(["","INITIALISATION ERROR".center(75),""],
                        "   Invalid parameter value {}='{}''".format('case',case).center(75),
                        "You should give one in {} set.".format(('All','sel')).center(75))
        raise
    #
    print(("\n{} '{}',\n{} {} to {},\n{} {} to {},\n{} {}x{},\n{} {}\n").format(
               "SIZE_REDUCTION ".ljust(18,'.'),SIZE_REDUCTION,
               "Lat ".ljust(18,'.'),frlat,tolat,
               "Lon ".ljust(18,'.'),frlon,tolon,
               "Map ".ljust(18,'.'),nbl, nbc,
               "Parm_app ".ljust(18,'.'),Parm_app,
               "nb_class ".ljust(18,'.'),nb_class,
               "nb_clust ".ljust(18,'.'),nb_clust,
               "NIJ ".ljust(18,'.'),NIJ))
    #
    ###########################################################################
    # For the Carte Topo (see also ctObsMdl)
    epoch1,radini1,radfin1,epoch2,radini2,radfin2 = Parm_app
    #--------------------------------------------------------------------------
    if SIZE_REDUCTION == 'All' :
        fprefixe  = 'Zall_'
    elif SIZE_REDUCTION == 'sel' :
        fprefixe  = 'Zsel_'
    elif SIZE_REDUCTION == 'RED' :
        fprefixe  = 'Zred_'
    else :
        print(" *** unknown SIZE_REDUCTION <{}> ***".format(SIZE_REDUCTION))
        raise
    #--------------------------------------------------------------------------
    #pcmap = ctloop.build_pcmap(nb_class,ccmap,factor=0.95)
    pcmap = ctloop.build_pcmap(nb_class,ccmap,factor=1.0)
    # si la colormap est 'jet' et 4 couleurs alors on la change la derniere
    # couleur a du marron
    if ccmap.name == 'jet' and pcmap.shape[0] == 4 :
        print('** Changing manually the higher ({}-th) color of PCMAP palette\n   from [{}] ...', 
              pcmap.shape[0],pcmap[3,0:3])
        #pcmap[3,0] *= (88/127);
        #pcmap[3,1] += pcmap[3,0]*(41/88);
        #pcmap[3,0:3] = [x / 256. for x in [88, 41, 0]]; # marron
        pcmap[3,0:3] = [x / 256. for x in [102, 51, 0]]; # plus proche couleur Web du marron[88, 41, 0]
        #pcmap[3,0:3] = [x / 256. for x in [90, 58, 34]]; # chocolat
        print('   to [{}]', pcmap[3,0:3])
    #
    cpcmap = colors.ListedColormap(pcmap)
    #
    #--------------------------------------------------------------------------
    fcodage,fshortcode = ctloop.build_fcode_and_short(climato=climato, 
                                  INDSC=INDSC, TRENDLESS=TRENDLESS, WITHANO=WITHANO,
                                  UISST=UISST, NORMMAX=NORMMAX, CENTRED=CENTRED,
                                  )
    #--------------------------------------------------------------------------
    #Flag visu classif des modèles des cluster
    AFC_Visu_Classif_Mdl_Clust  = []; # liste des cluster a afficher (à partir de 1)
    #AFC_Visu_Classif_Mdl_Clust = [1,2,3,4,5,6,7]; 
    #Flag visu Modèles Moyen 4CT des cluster
    AFC_Visu_Clust_Mdl_Moy_4CT  = []; # liste des cluster a afficher (à partir de 1)
    #AFC_Visu_Clust_Mdl_Moy_4CT = [1,2,3,4,5,6,7];
    #--------------------------------------------------------------------------
    TM_label_base = "TM{}x{}_Ep1-{}_Ep2-{}".format(nbl, nbc, epoch1, epoch2)
    case_label_base="Case_{}{}_NIJ{:d}".format(fprefixe,TM_label_base,NIJ)
    #######################################################################

    obs_data_path = '../Datas'
    
    # flag pour bloquer ou pas apres chaque figure creee (apres un plt.show()
    blockshow = False
    #blockshow = True
    
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
    case_name_base,casetime,casetimelabel,casetimeTlabel,varnames,list_of_plot_colors,\
        tpgm0 = ctloop.initialisation(case_label_base,tseed=tseed)
    #
    return  pcmap,cpcmap,AFC_Visu_Classif_Mdl_Clust, AFC_Visu_Clust_Mdl_Moy_4CT,\
            TM_label_base, case_label_base, obs_data_path, \
            tseed, case_name_base, casetime, casetimelabel, casetimeTlabel, varnames,\
            list_of_plot_colors 
#
#%%
def ctloop_load_obs(DATAOBS, path=".", case_name="case") :
    #
    # Variables declarees dans ParamCas traitees comme globales !
    global SIZE_REDUCTION, frlat, tolat, frlon, tolon, TRENDLESS, WITHANO, climato, UISST, NORMMAX, CENTRED

    ### ACQUISITION DES DONNEES D'OBSERVATION #####################################
    ctloop.printwarning([ "","ACQUISITION DES DONNEES D'OBSERVATION: '{:s}'".format(DATAOBS).center(75),""])
    data_label_base,sst_obs,lon,lat = ctloop.read_obs(path,DATAOBS)
    
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
    #
    # -----------------------------------------------------------------------------
    # Complete le Nom du Cas
    case_label = "{}_ZG{:d}x{:d}px_{}".format(case_name,len(lat),len(lon),data_label_base)
    if AFCWITHOBS:
        case_label += '_AFC-Mod+Obs';
    else:
        case_label += '_AFC-ModOnly';
    print("\n{:*>86s}\nCase label with data version: {}\n".format('',case_label))
    #
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
    return sst_obs, sst_obs_coded, Dobs, NDobs, lon, lat, ilat, ilon, isnanobs, isnumobs,\
        case_label, data_label_base, Nobs, Lobs, Cobs
#
#%%
def plot_obs4ct(sst_obs,Dobs,lon,lat,isnanobs=None,isnumobs=None,varnames=None,wvmin=None,wvmax=None,eqcmap=None,
                Show_ObsSTD=False,fileext="_Obs4CTFig",filecomp="",figpath=".",fcodage="",freelimststoo=False) :
    #
    # Variables declarees dans ParamCas traitees comme globales !
    global SIZE_REDUCTION, frlat, tolat, frlon, tolon, Nda, SAVEFIG, FIGDPI
    #
    global fprefixe, blockshow
    #
    if eqcmap is None :
        eqcmap = cm.get_cmap('RdYlBu_r')  # Palette RdYlBu inversée
    #
    if type(eqcmap) is list :
        iter_cmap = eqcmap
    else :
        iter_cmap = [ eqcmap ]
    #
    dataystartend = datemdl2dateinreval(DATAMDL)
    #
    if SIZE_REDUCTION == 'All' :
        lolast = 4
        figsize = (12,7)
        wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.97;
    elif SIZE_REDUCTION == 'sel' :
        lolast = 2
        figsize=(10,8.5)
        wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.965;
    #
    stitre = "Observed SST {:s} MEAN ({:s})".format(fcodage,dataystartend)
    if Show_ObsSTD :
        stitre += "(monthly {:d} years STD in contours)".format(Nda)
    #
    for ii,v_eqcmap in enumerate(iter_cmap) :
        ctloop.plot_obs(sst_obs,Dobs,lon,lat,varnames=varnames,cmap=v_eqcmap,
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
            figfile += "{:s}{:s}".format(filecomp,fileext)
            figfile += "_{:s}".format(v_eqcmap.name)
            # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
            # eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figpath,ext=FIGEXT,fig2ok=SAVEPDF,ext2=VFIGEXT)
        #
        plt.show(block=blockshow)
    #
    if freelimststoo :
        if SIZE_REDUCTION == 'All' :
            lolast = 4
            figsize = (12,7)
            wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.97;
        elif SIZE_REDUCTION == 'sel' :
            lolast = 2
            figsize=(10,8.5)
            wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.965;
        #
        stitre = "Observed SST {:s} MEAN ({:s}) - FREE LIMITS".format(fcodage,dataystartend)
        #
        for ii,v_eqcmap in enumerate(iter_cmap) :
            ctloop.plot_obsbis(sst_obs,Dobs,varnames=varnames,cmap=v_eqcmap,
                            title=stitre,
                            figsize=figsize,
                            wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                            )
        #
        if SAVEFIG : # sauvegarde de la figure
            figfile = "Fig_Obs4CT_FREELIMITS"
            figfile += fileext
            figfile += "_{}".format(eqcmap.name)
            # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
            # eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figpath,ext=FIGEXT)
        #
        plt.show(block=blockshow)
    #
    return
#%%
def ctloop_topol_map_traitement (Dobs,Parm_app=(5,5.,1.,10,1.,0.1),mapsize=[7, 3],tseed=0,
                                 mapfile="Map",mappath=".",
                                 varnames=None,case_label="case",
                                 casetime=None,casetimelabel=None) :
    #
    # Variables declarees dans ParamCas traitees comme globales !
    global SAVEMAP, REWRITEMAP
    #
    #######################################################################
    #                       Carte Topologique
    #======================================================================
    ctloop.printwarning([ "","CARTE TOPOLOGIQUE".center(75),""])
    DO_NEXT = True
    if SAVEMAP : # SI sauvegarde de la Map de SOM est ACTIVE
        #mapfile = "Map_{:s}{:s}Clim-{:d}-{:d}_{:s}_ts-{}{}".format(fprefixe,fshortcode,
        #               andeb,anfin,data_label_base,tseed,mapfileext)
        mapPathAndFile = mappath+os.sep+mapfile
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
        ###########################################################################
        # For the Carte Topo (see also ctObsMdl)
        epoch1,radini1,radfin1,epoch2,radini2,radfin2 = Parm_app
        nbl, nbc = mapsize
        #    sMapO = SOM.SOM('sMapObs', Dobs, mapsize=[nbl, nbc], norm_method=norm_method, \
        #                    initmethod='random', varname=varnames)
        sMapO,q_err,t_err = ctloop.do_ct_map_process(Dobs,name='sMapObs',mapsize=[nbl, nbc],
                                                 tseed=tseed,varname=varnames,
                                                 norm_method='data',
                                                 initmethod='random',
                                                 etape1=[epoch1,radini1,radfin1],
                                                 etape2=[epoch2,radini2,radfin2],
                                                 verbose='on', retqerrflg=True)
        #
        print("Obs case: {}\n          date ... {}]\n          tseed={}\n          Qerr={:8.6f} ... Terr={:.6f}".format(
            case_label,casetimelabel,tseed,q_err,t_err))
        somtime = casetime
        #
        if SAVEMAP : # sauvegarde de la Map de SOM
            if  os.path.exists(mapPathAndFile):
                ctloop.printwarning([ "==> Saving MAP, file exists, deleted before saving it :",
                               "    {}/".format(os.path.dirname(mapPathAndFile)),
                               "         {}".format(os.path.basename(mapPathAndFile)) ])
                os.remove(mapPathAndFile)
            else:
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
            q_err = np.mean(sMapO.bmu[1])
            # + err topo maison
            bmus2O = ctk.mbmus (sMapO, Data=None, narg=2);
            t_err    = ctk.errtopo(sMapO, bmus2O); # dans le cas 'rect' uniquement
            #print("Obs, erreur topologique = %.4f" %t_err)
            print("Obs case: {}\n          loaded sMap date ... {}]\n          used tseed={} ... Qerr={:8.6f} ... Terr={:.6f}".format(case_label,
                  somtimelabel,tseed,q_err,t_err))
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
    #
    return sMapO,q_err,t_err
#%%
def plot_ct_Umatrix(sMapO, figsize=(4,8)) :
    global blockshow
    # #########################################################################
    # C.T. Visualisation ______________________________________________________
    # #########################################################################
    #==>> la U_matrix
    fig = plt.figure(figsize=figsize);
    fignum = fig.number
    #wspace=0.02; hspace=0.14; top=0.80; bottom=0.02; left=0.01; right=0.95;
    #fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
    a=sMapO.view_U_matrix(fignum=fignum, distance2=2, row_normalized='No', show_data='Yes', \
                      contooor='Yes', blob='No', save='No', save_dir='');
    plt.suptitle("Obs, The U-MATRIX", fontsize=16,y=1.0);
    #
    plt.show(block=blockshow)
    #
    return
#%%
def plot_ct_map_wei(sMapO, figsize=(6,8)) :
    #
    global blockshow
    #
    fig = plt.figure(figsize=figsize);
    fignum = fig.number
    wspace=0.01; hspace=0.10; top=0.935; bottom=0.02; left=0.01; right=0.94;
    fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
    #ctk.showmap(sMapO,sztext=11,colbar=1,cmap=cm.rainbow,interp=None);
    ctk.showmap(sMapO,sztext=11,cbsztext=8,fignum=fignum,
                colbar=1,cmap=cm.rainbow,interp=None,noaxes=False,
                noticks=False,xticks=np.arange(nbc),yticks=np.arange(nbl),
                nolabels=True,
                );
    plt.suptitle("Obs, Les Composantes de la carte", fontsize=16);
    #
    plt.show(block=blockshow)
    #
    return
#%%
def plot_ct_dendro(sMapO, nb_class, datalinkg=None, title="SOM Map Dendrogram",
                   fileext="",figdir=".",
                   xlabel="codebook number",
                   ylabel="inter class distance",
                   ) :
    #
    global FIGDPI, FIGEXT, blockshow
    #
    figsize=(14,6);
    wspace=0.0; hspace=0.2; top=0.92; bottom=0.12; left=0.05; right=0.99;
    if sMapO.codebook.shape[0] > 200 :
        labelsize = 4
    elif sMapO.codebook.shape[0] > 120 :
        labelsize = 6
    elif sMapO.codebook.shape[0] > 80 :
        labelsize = 8
    else:
        labelsize = 10
    ctloop.do_plot_ct_dendrogram(sMapO, nb_class,
                          datalinkg=datalinkg,
                          title=title, ytitle=1.02,
                          xlabel=xlabel,
                          ylabel=ylabel,
                          titlefnsize=18, labelfnsize=10,
                          labelrotation=-90, labelsize=labelsize,
                          figsize=figsize,
                          wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right
                          )
    #
    if SAVEFIG : # sauvegarde de la figure
        figfile = "Fig_"
        dpi = FIGDPI*2
        figfile += "SOMCodebookDendro-{:d}Class{:s}".format(nb_class,fileext)
        ctloop.do_save_figure(figfile,dpi=dpi,path=figdir,ext=FIGEXT)
    #
    plt.show(block=blockshow)
    #
    return

#%%
def plot_geo_classes(lon,lat,XC_ogeo,fond_C,nb_class,
                     nticks=2,
                     title="Obs Class Geographical Representation", fileext="", figdir=".",
                     figfile=None, dpi=None, figpdf=False,
                     ccmap=cm.jet,
                     bgmap='gray',
                     figsize=(9,6),
                     top=0.94, bottom=0.08, left=0.06, right=0.925,
                     ticks_fontsize=10,labels_fontsize=12,title_fontsize=16,
                     cblabel='Class',cbticks_fontsize=12,cblabel_fontsize=14,
                     title_y=1.015,
                     notitle=False
                     ) :
    #
    global SAVEFIG, FIGDPI, FIGEXT, FIGARTDPI, SAVEPDF, VFIGEXT, blockshow
    #
    coches = np.arange(nb_class)+1;   # ex 6 classes : [1,2,3,4,5,6]
    ticks  = coches + 0.5;            # [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    bounds = np.arange(nb_class+1)+1; # pour bounds faut une frontière de plus [1, 2, 3, 4, 5, 6, 7]

    fig = plt.figure(figsize=figsize)
    fignum = fig.number # numero de figure en cours ...
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)
    if lat[0] < lat[1] :
        origin = 'lower'
    else :
        origin = 'upper'
    fig, ax = plt.subplots(nrows=1, ncols=1, num=fignum,facecolor='w')
    ax.imshow(fond_C, interpolation=None,cmap=bgmap,vmin=0,vmax=1)
    ims = ax.imshow(XC_ogeo, interpolation=None,cmap=ccmap,vmin=1,vmax=nb_class,origin=origin);
    if 0:
        plt.xticks(np.arange(0,Cobs,nticks), lon[np.arange(0,Cobs,nticks)], rotation=45, fontsize=ticks_fontsize)
        plt.yticks(np.arange(0,Lobs,nticks), lat[np.arange(0,Lobs,nticks)], fontsize=ticks_fontsize)
    else :
        #plt.xticks(np.arange(-0.5,Cobs,lolast), np.round(lon[np.arange(0,Cobs,lolast)]).astype(int), fontsize=12)
        #plt.yticks(np.arange(0.5,Lobs,lolast), np.round(lat[np.arange(0,Lobs,lolast)]).astype(int), fontsize=12)
        ctloop.set_lonlat_ticks(lon,lat,step=nticks,fontsize=ticks_fontsize,verbose=False,lengthen=True)
        #set_lonlat_ticks(lon,lat,fontsize=10,londecal=0,latdecal=0,roundlabelok=False,lengthen=False)
    #plt.axis('tight')
    plt.xlabel('Longitude', fontsize=labels_fontsize); plt.ylabel('Latitude', fontsize=labels_fontsize)
    if not notitle :
        plt.title(title,fontsize=title_fontsize,y=title_y); 
    #grid(); # for easier check
    # Colorbar
    #cbar_ax,kw = cb.make_axes(ax,orientation="vertical",fraction=0.04,pad=0.03,aspect=20)
    #fig.colorbar(ims, cax=cbar_ax, ticks=ticks,boundaries=bounds,values=bounds, **kw);
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="4%", pad="3%")
    hcb = plt.colorbar(ims,cax=cax,ax=ax,ticks=ticks,boundaries=bounds,values=bounds);
    cax.set_yticklabels(coches);
    cax.tick_params(labelsize=cbticks_fontsize)
    cax.set_ylabel(cblabel,size=cblabel_fontsize)

    #hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
    #hcb.set_ticklabels(coches);
    #hcb.ax.tick_params(labelsize=12)
    #hcb.set_label('Class',size=14)
    if SAVEFIG : # sauvegarde de la figure
        if figfile is None :
            figfile = "Fig_"
        if dpi is None :
            dpi = FIGDPI
        figfile += "ObsGeo{:d}Class_{:s}".format(nb_class,fileext)
        #
        ctloop.do_save_figure(figfile,dpi=dpi,path=figdir,ext=FIGEXT,figpdf=figpdf)
    #
    plt.show(block=blockshow)
    #
    return
#%%
def plot_mean_profil_by_class(sst_obs,nb_class,classe_Dobs,varnames=None,
                              title="figart1", fileext="", figdir=".",
                              figfile=None, dpi=None, figpdf=False,
                              getstd=False,
                              pcmap=None,
                              figsize=(12,6),
                              wspace=0.0, hspace=0.0, top=0.96, bottom=0.08, left=0.06, right=0.92,
                              linewidth=1,
                              ticks_fontsize=10,labels_fontsize=12,title_fontsize=16,
                              ylabel_fontsize=None,
                              lgtitle='Class',
                              lgticks_fontsize=12,lglabel_fontsize=14,
                              title_y=1.015,
                              notitle=False,
                              errorcaps=False,
                              capslength=3,
                              plot_back_black=False,
                              back_black_color=[0.25, 0.25, 0.25, 1],
                              back_black_diffsize=0.75,
                              ) :
    #
    global SAVEFIG, FIGDPI, FIGEXT, FIGARTDPI, SAVEPDF, VFIGEXT, blockshow
    #
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Figure 1 pour Article 
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    stitre = title
    ctloop.plot_mean_curve_by_class(sst_obs,nb_class,classe_Dobs,varnames=varnames,
                                    title=stitre,
                                    getstd=getstd,
                                    pcmap=pcmap, darkcmapfactor=1.0,
                                    figsize=figsize,
                                    wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                                    linewidth=linewidth,
                                    ticks_fontsize=ticks_fontsize,labels_fontsize=labels_fontsize,
                                    ylabel_fontsize=ylabel_fontsize,
                                    title_fontsize=title_fontsize,
                                    lgtitle=lgtitle,
                                    lgticks_fontsize=lgticks_fontsize,lglabel_fontsize=lglabel_fontsize,
                                    title_y=title_y,
                                    notitle=notitle,
                                    errorcaps=errorcaps,
                                    capslength=capslength,
                                    plot_back_black=plot_back_black,
                                    back_black_color=back_black_color,
                                    back_black_diffsize=back_black_diffsize,
                                    )
    if SAVEFIG : # sauvegarde de la figure
        if figfile is None :
            figfile = "Fig_"
        if dpi is None :
            dpi = FIGDPI
        figfile += "MeanByClass_{:d}Class{:s}".format(nb_class,fileext)
        if getstd :
            figfile += "_Errbar"
            if errorcaps:
                figfile += "_wCaps{}".format(capslength)
            if plot_back_black:
                figfile += "_Blkbord"
        #
        ctloop.do_save_figure(figfile,dpi=dpi,path=figdir,ext=FIGEXT,figpdf=figpdf)
    #
    plt.show(block=blockshow)
    #
    return
#%%
def plot_ct_profils(sMapO,Dobs,class_ref,varnames=None, fileext="", figdir=".",
                    pcmap=None,
                    title="SOM Map Profils by Cell (background color represents classes)",
                    titlefntsize=16,
                    ytitle=0.98,
                    ) :
    #
    global SIZE_REDUCTION, SAVEFIG, FIGDPI, FIGEXT, blockshow
    #
    if SIZE_REDUCTION == 'All' :
        figsize = (5,12)
        wspace=0.01; hspace=0.05; top=0.945; bottom=0.04; left=0.15; right=0.86;
    elif SIZE_REDUCTION == 'sel' :
        figsize=(8,8)
        wspace=0.01; hspace=0.04; top=0.945; bottom=0.04; left=0.04; right=0.97;
    #
    #stitre="SOM Map Profils by Cell ({:s})\n(background color represents classes)",

    ctloop.do_plot_ct_profils(sMapO,Dobs,class_ref,varnames=varnames,
                           pcmap=pcmap,
                           figsize=figsize,
                           title=title,
                           titlefntsize=titlefntsize,
                           wspace=wspace, hspace=hspace,
                           top=top, bottom=bottom,left=left, right=right,
                           ysuptitle=ytitle,
                           )
    if SAVEFIG : # sauvegarde de la figure
        figfile = "Fig_"
        dpi = FIGDPI
        figfile += "Fig_SOM-Map-Profils_{:d}Class{:s}".format(nb_class,fileext)
        ctloop.do_save_figure(figfile,dpi=dpi,path=figdir,ext=FIGEXT)
    #
    plt.show(block=blockshow)
#%%
def ctloop_model_traitement(sst_obs,Dobs,XC_Ogeo,sMapO,lon,lat,ilat,ilon,
                            nb_class,class_ref,classe_Dobs,NDobs,fond_C,
                            isnanobs,isnumobs,Lobs,Cobs,list_of_plot_colors,
                            varnames=None, figdir=".", commonfileext="", commonfileext79="", 
                            data_period_ident="raverage_1975_2005",
                            pair_nsublc=None,
                            subtitle=None,
                            Sfiltre=None, eqcmap=cm.jet,ccmap=cm.jet,pcmap=None,
                            bgmap='gray', bgval=0.5,
                            cbticksz=8, cbtickszobs=8,
                            obs_data_path=".",
                            OK101=False,
                            OK102=False,
                            OK104=False,
                            OK105=False,
                            OK106=False,
                            OK107=False,
                            OK108=False,
                            OK109=False,
                            OK105Art=False,
                            ) :
    #
    global SIZE_REDUCTION, MDLCOMPLETION, NIJ, FONDTRANS 
    #global OK101, OK102, OK104, OK105, OK106, OK107, OK108, OK109
    global Tinstit, Tmodels, Tnmodel, TypePerf, mdlnamewnumber_ok
    #
    global fshortcode, fprefixe, fcodage, blockshow
    global SAVEFIG, FIGDPI, FIGEXT
    #
    if OK105Art :
        OK105 = True;
        cbticksz=12
        cbtickszobs=14
    #
    ctloop.printwarning([ "","TRAITEMENTS DES DONNEES DES MODELES OCEANOGRAPHIQUES".center(75),""])
    print(" OK101: {}, OK102: {}, OK104: {}, OK105: {}, OK106: {}, OK107: {}, OK108: {}, OK109: {}".format(OK101, OK102, OK104, OK105, OK106, OK107, OK108, OK109))
    #######################################################################
    #                        MODELS STUFFS START HERE
    #======================================================================
           #<<<<<<<<
    #                        MODELS STUFFS START HERE
    #======================================================================
    ctloop.printwarning([ "    MODEL: INITIALIZATION AND FIRST LOOP" ])
    #
    TDmdl4CT0,Tmdlname0,Tmdlnamewnb0,Tmdlonlynb0,Tperfglob4Sort0,Tclasse_DMdl0,\
        Tmoymensclass0,NDmdl,Nmdlok,Smoy_101,Tsst_102 = ctloop.do_models_startnloop(sMapO,
                                Tmodels,Tinstit,ilat,ilon,
                                isnanobs,isnumobs,nb_class,class_ref,classe_Dobs,
                                Tnmodel=Tnmodel,
                                Sfiltre=Sfiltre,
                                TypePerf=TypePerf,
                                obs_data_path=obs_data_path,
                                data_period_ident=data_period_ident,
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
    dataystartend = datemdl2dateinreval(data_period_ident)
    if mdlnamewnumber_ok :
        Tmdlname10X = Tmdlnamewnb0;
    else :
        Tmdlname10X = Tmdlname0;
    #
    if len(Tmdlname10X) > 4 :            # Sous forme de liste, la liste des noms de modèles
        Tnames_ = Tmdlonlynb0;           # n'est pas coupé dans l'affichage du titre de la figure
    else :                               # par contre il l'est sous forme d'array; selon le cas, ou 
        Tnames_ = np.array(Tmdlname10X); # le nombre de modèles, il faut adapter comme on peut
    stitre = "Mdl_MOY{:} ({:d} mod.)".format(Tnames_,len(Tnames_))
    if subtitle is not None :
        stitre += " -"+subtitle+"- "
    stitre += " {:s}SST({:s})".format(fcodage,dataystartend)
    #
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
            figfile = "Fig-101{:s}{:d}Mdl_MOY-{:s}_{:d}-mod".format(commonfileext,nb_class,dataystartend,Nmdlok)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figdir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    if OK102 : 
        if SAVEFIG :
            plt.figure(102);
            figfile = "Fig-102{:s}{:d}Mdl_ECRTYPE-{:s}_{:d}-mod".format(commonfileext,nb_class,dataystartend,Nmdlok)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figdir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
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
    suptitle104=None;  suptitle105=None
    suptitle106=None;  suptitle107=None
    suptitle108=None;  suptitle109=None
    if OK104 :
        suptitle104 = "F104:"
        if subtitle is not None :
            suptitle104 += " -"+subtitle+"- "
        suptitle104 += " (%sSST(%s)).  Classification of Completed Models (vs Obs) (%d models)" \
                        %(fcodage,dataystartend,Nmdlok);
    if OK105 : 
        suptitle105 = "F105:"
        if subtitle is not None :
            suptitle105 += " -"+subtitle+"- "
        suptitle105 += " (%sSST(%s)).  Classification of Completed Models (vs Obs) (%d models)" \
                        %(fcodage,dataystartend,Nmdlok);
    if OK106 :
        suptitle106 = "F106: MOY - MoyMensClass"
        if subtitle is not None :
            suptitle106 += " -"+subtitle+"- "
        suptitle106 += " (%sSST(%s)).  Classification of Completed Models (vs Obs) (%d models)" \
                        %(fcodage,dataystartend,Nmdlok);
    if OK107 :
        suptitle107 = "F107: VARiance"
        if subtitle is not None :
            suptitle107 += " -"+subtitle+"- "
        suptitle107 += "VARiance(%sSST(%s)). Variance (by pixel) on Completed Models (%d models)" \
                        %(fcodage,dataystartend,Nmdlok);
    if OK108 :
        suptitle108 = "F108: MCUM"
        if subtitle is not None :
            suptitle108 += " -"+subtitle+"- "
        suptitle108 += " (%sSST(%s)).  Classification of Completed Models (vs Obs) (%d models)" \
                        %(fcodage,dataystartend,Nmdlok);
    #
    if OK109 :
        suptitle109 = "F109: VCUM"
        if subtitle is not None :
            suptitle109 += " -"+subtitle+"- "
        suptitle109 += " (%sSST(%s)). Variance sur la Moyenne Cumulée de Modeles complétés (%d models)" \
                        %(fcodage,dataystartend,Nmdlok);
    #
    if SIZE_REDUCTION == 'All' :
        figsize=(18,11);
        wspace=0.01; hspace=0.15; top=0.94; bottom=0.05; left=0.01; right=0.99;
    elif SIZE_REDUCTION == 'sel' :
        figsize=(16,12);
        wspace=0.01; hspace=0.14; top=0.94; bottom=0.04; left=0.01; right=0.99;
    #
    if pair_nsublc is None :
        nsub   = Nmdlok + 1; # actuellement au plus 48 modèles + 1 pour les obs. 
        nbsubl, nbsubc = ldef.nsublc(nsub);
    else :
        if np.prod(pair_nsublc) < (Nmdlok + 1):
            ctloop.printwarning(["** TOO SMALL Combination between number of lines and number of columns **".upper().center(75),
                    "** specified in  pair_nsublc=[{0[0]},{0[1]}]  argument **".upper().format(pair_nsublc).center(75) ],
                    "** You have {:d} models + 1 for Obs **".format((Nmdlok)).center(75),
                    "** try another pair_nsublc=[nbsubl, nbsubc] combination. **".center(75))
            raise
        nbsubl, nbsubc = pair_nsublc;
        fsizex,fsizey = figsize
        szlin = fsizey * (top - bottom) / nbsubl
        szcol = fsizex * (right - left) / nbsubc
        nblinmod = np.int(np.ceil((Nmdlok + 1) / nbsubc))
        # correction de la teille de figure
        print("--figsize correction:")
        print("  We had figsize={}, nbsubl, nbsubc={} for {} models + Obs...".format(figsize, (nbsubl, nbsubc), Nmdlok))
        figsize=(fsizex,nblinmod*szlin + fsizey*(1-top + bottom));
        bottom *= 0.5*nbsubl/nblinmod
        top1 = 1 - top
        top1 *= 0.5*nbsubl/nblinmod
        top = 1 - top1
        wspace=0.10
        nbsubl = nblinmod
        print("  and we have now figsize={}, nbsubl, nbsubc={} ...".format(figsize, (nbsubl, nbsubc)))
    nsubmax = nbsubl * nbsubc; # derniere casse subplot, pour les OBS
    pair_nsublc = nbsubl, nbsubc
    #
    sztitle = 10;
    suptitlefs = 16
    ysuptitre = 0.99
    
    Tperfglob,Tperfglob_Qm,Tmdlname,Tmdlnamewnb,Tmdlonlynb,TTperf,TTperf_Qm,\
        TDmdl4CT = ctloop.do_models_second_loop(sst_obs,Dobs,lon,lat,sMapO,XC_Ogeo,TDmdl4CT,
                                Tmdlname,Tmdlnamewnb,Tmdlonlynb,
                                Tperfglob4Sort,Tclasse_DMdl,Tmoymensclass,
                                MaxPerfglob_Qm,IMaxPerfglob_Qm,
                                min_moymensclass,max_moymensclass,
                                MCUM,Lobs,Cobs,NDobs,NDmdl,
                                isnumobs,nb_class,class_ref,classe_Dobs,fond_C,
                                ccmap=ccmap,pcmap=pcmap,
                                bgmap=bgmap, bgval=bgval,
                                sztitle=sztitle,ysstitre=ysstitre,
                                ysuptitre=ysuptitre,suptitlefs=suptitlefs,
                                cbticksz=cbticksz, cbtickszobs=cbticksz, 
                                NIJ=NIJ,
                                pair_nsublc=pair_nsublc,
                                FONDTRANS=FONDTRANS,
                                TypePerf=TypePerf,
                                mdlnamewnumber_ok=mdlnamewnumber_ok,
                                OK104=OK104, OK105=OK105, OK106=OK106, OK107=OK107, OK108=OK108, OK109=OK109,
                                OK105Art=OK105Art,
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
            figfile = "Fig-104{:s}{:d}MdlvsObstrans-{:s}_{:d}-mod".format(commonfileext,nb_class,dataystartend,Nmdlok)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figdir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    if OK105 : 
        if SAVEFIG :
            plt.figure(105);
            if OK105Art:
                nFigArt = 3;
                figfile = "FigArt{:02d}-105_".format(nFigArt)
                dpi     = FIGARTDPI
                figpdf  = True
            else:
                figfile = "Fig-105_"
                dpi     = FIGDPI
                figpdf  = False
            #
            figfile += "{:s}{:d}Mdl-{:s}_{:d}-mod".format(commonfileext,nb_class,dataystartend,Nmdlok)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=dpi,path=figdir,ext=FIGEXT,figpdf=figpdf) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    if OK106 :
        if SAVEFIG :
            plt.figure(106);
            figfile = "Fig-106{:s}{:d}moymensclass-{:s}_{:d}-mod".format(commonfileext,nb_class,dataystartend,Nmdlok)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figdir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    if OK107 :
        if SAVEFIG :
            plt.figure(107);
            figfile = "Fig-107{:s}VAR_Mdl-{:s}_{:d}-mod".format(commonfileext,dataystartend,Nmdlok)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figdir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    if MCUM>0 and OK108 :
        if SAVEFIG :
            plt.figure(108);
            figfile = "Fig-108{:s}MCUM_{:d}Mdl-{:s}_{:d}-mod".format(commonfileext,nb_class,dataystartend,Nmdlok)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figdir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    if MCUM>0 and OK109 :
        if SAVEFIG :
            plt.figure(109);
            figfile = "Fig-109{:s}VCUM_Mdl-{:s}_{:d}-mod".format(commonfileext,dataystartend,Nmdlok)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figdir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    #
    return Tperfglob, Tperfglob_Qm, TDmdl4CT, Tmdlname, Tmdlnamewnb, Tmdlonlynb,\
           TTperf, TTperf_Qm, Nmdlok, NDmdl
#
#%%
def datemdl2dateinreval(datamdl):
    datamdllist = datamdl.split('_')
    if len(datamdllist) == 3 :
        dataystartend = "{0[1]}-{0[2]}".format(datamdllist)
    return dataystartend
#
#%%
def ctloop_compute_afc(sMapO, lon, lat, TDmdl4CT, Tmdlname, Tmdlnamewnb, Tmdlonlynb, 
                       nb_class, nb_clust, isnumobs, isnanobs, class_ref, classe_Dobs,
                       fond_C, XC_Ogeo,
                       TTperf, Nmdlok, Lobs, Cobs, NDmdl, Nobsc, data_label_base,
                       AFC_Visu_Classif_Mdl_Clust=[],
                       AFC_Visu_Clust_Mdl_Moy_4CT=[],
                       ccmap=cm.jet,
                       bgmap='gray', bgval=0.5,
                       sztitle=10,
                       figdir=".",
                       figfile=None, dpi=None, figpdf=False,
                       clustfigsublc=None, clustfigsize=None,
                       plotobs=False, notitle=False,
                       ) :
    '''
    Analyse Factorielle des Correspondances (Correspondence Analysis)
    
    Correspondence analysis (CA) or reciprocal averaging is a multivariate
    statistical technique proposed[1] by Herman Otto Hartley (Hirschfeld)
    and later developed by Jean-Paul Benzécri. It is conceptually similar
    to principal component analysis, but applies to categorical rather than
    continuous data. In a similar manner to principal component analysis,
    it provides a means of displaying or summarising a set of data in
    two-dimensional graphical form.
    
    All data should be nonnegative and on the same scale for CA to be
    applicable, keeping in mind that the method treats rows and columns
    equivalently. It is traditionally applied to contingency tables — CA
    decomposes the chi-squared statistic associated with this table into
    orthogonal factors. Because CA is a descriptive technique, it can be
    applied to tables whether or not the chi-2 statistic is appropriate.
    
    '''
    #
    global SIZE_REDUCTION, DATAMDL, NIJ, NBCOORDAFC4CAH, AFCWITHOBS, CAHWITHOBS
    global TypePerf, ysstitre, mdlnamewnumber_ok, onlymdlumberAFC_ok, onlymdlumberAFCdendro_ok
    #
    global fshortcode, fprefixe, fcodage, blockshow
    global SAVEFIG, FIGDPI, FIGEXT
    #
    Nmdlafc = Tmdlname.shape[0]
    dataystartend = datemdl2dateinreval(DATAMDL)
    ctloop.printwarning([ "","CALCUL DE L'A.F.C".center(75),""])
    #
#
    VAPT,F1U,F1sU,F2V,CRi,CAj,TTperf4afc,\
        CAHindnames,CAHindnameswnb,NoCAHindnames,\
        figclustmoynum,class_afc,AFCindnames,AFCindnameswnb,\
        NoAFCindnames, \
        allclusrPerfG, allclustTperf = ctloop.do_afc(NIJ,
                          sMapO, TDmdl4CT, lon, lat,
                          Tmdlname, Tmdlnamewnb, Tmdlonlynb, TTperf,
                          Nmdlok, Lobs, Cobs, NDmdl, Nobsc,
                          NBCOORDAFC4CAH, nb_clust,
                          isnumobs, isnanobs, nb_class, class_ref, classe_Dobs,
                          ccmap=ccmap,
                          bgmap=bgmap, bgval=bgval,
                          sztitle=sztitle, ysstitre=ysstitre,
                          AFC_Visu_Classif_Mdl_Clust=AFC_Visu_Classif_Mdl_Clust,
                          AFC_Visu_Clust_Mdl_Moy_4CT=AFC_Visu_Clust_Mdl_Moy_4CT,
                          TypePerf=TypePerf,
                          AFCWITHOBS = AFCWITHOBS, CAHWITHOBS=CAHWITHOBS,
                          SIZE_REDUCTION=SIZE_REDUCTION,
                          mdlnamewnumber_ok=mdlnamewnumber_ok,
                          onlymdlumberAFC_ok=onlymdlumberAFC_ok,
                          figsublc=clustfigsublc, figsize=clustfigsize,
                          notitle=notitle,
                          )
        
#    if plotobs : # Obs --------
#        nbsubl,nbsubc=clustfigsublc; isubplot = nbsubl * nbsubc
#        bounds = np.arange(nb_class+1)+1;
#        coches = np.arange(nb_class)+1;   # ex 6 classes : [1,2,3,4,5,6]
#        ticks  = coches + 0.5;            # [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
#        if SIZE_REDUCTION == 'All' :
#            nticks = 5; # 4
#        elif SIZE_REDUCTION == 'sel' :
#            nticks = 2; # 4
#        plt.figure(figclustmoynum);
#        ax = plt.subplot(nbsubl,nbsubc,isubplot);
#        ax.imshow(fond_C, interpolation=None,cmap=cm.gray,vmin=0,vmax=1)
#        ims = ax.imshow(XC_Ogeo, interpolation=None,cmap=ccmap,vmin=1,vmax=nb_class);
#        ax.set_title("Obs, %d classes "%(nb_class),fontsize=12);
#        if 0 :
#            plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
#            plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
#        else :
#            ctloop.set_lonlat_ticks(lon,lat,step=nticks,fontsize=10,verbose=False,lengthen=True)
#        if True :
#            #cbar_ax,kw = cb.make_axes(ax,orientation="vertical",fraction=0.04,pad=0.03,aspect=20)
#            #fig.colorbar(ims, cax=cbar_ax, ticks=ticks,boundaries=bounds,values=bounds, **kw);
#            ax_divider = make_axes_locatable(ax)
#            cax = ax_divider.append_axes("right", size="4%", pad="3%")
#            hcb = plt.colorbar(ims,cax=cax,ax=ax,ticks=ticks,boundaries=bounds,values=bounds);
#        else :
#            hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
#        hcb.set_ticklabels(coches);
#        hcb.ax.tick_params(labelsize=10)
#        hcb.set_label('Class',size=10)
#        #grid(); # for easier check
#
#    #print("--CAHindnames: {}".format(CAHindnames))
#    #print("--NoCAHindnames: {}".format(NoCAHindnames))
#    #print("--Tmdlname: {}".format(Tmdlname))
#    print("\n--Tmdlnamewnb: {}".format(Tmdlnamewnb))
#    # reprend la figure de performanes par cluster
#    plt.figure(figclustmoynum)
#    plt.suptitle("AFC Clusters Class Performance ({}) ({} models)".format(dataystartend,Nmdlafc),fontsize=18);
#    #
#    if SAVEFIG : # sauvegarde de la figure
#        plt.figure(figclustmoynum)
#        if figfile is None :
#            figfile = "Fig_"
#        if dpi is None :
#            dpi = FIGDPI
#        figfile += "AFCClustersPerf-{:d}Clust-{:d}Classes_{:s}{:s}Clim-{:s}_{:d}-mod".format(nb_clust,
#                    nb_class,fprefixe,fshortcode,dataystartend,Nmdlafc)
#        #
#        ctloop.do_save_figure(figfile,dpi=dpi,path=figdir,ext=FIGEXT,figpdf=figpdf)
        
        
    #
    #            if Visu_Dendro :
    #        figsize=(14,6);
    #        wspace=0.0; hspace=0.2; top=0.92; bottom=0.12; left=0.05; right=0.99;
    #        stitre = ("SOM Codebook Dendrogram for HAC (map size={:d}x{:d})"+\
    #                  " - {:d} classes").format(nbl,nbc,nb_class);
    #        ctloop.do_plot_ct_dendrogram(sMapO, nb_class,
    #                              datalinkg=ctZ_,
    #                              title=stitre,ytitle=1.02,
    #                              xlabel="codebook number",
    #                              ylabel="inter class distance ({})".format(method_cah),
    #                              titlefnsize=18, labelfnsize=10,
    #                              figsize=figsize,
    #                              wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right
    #                              )
    #        #
    #        if SAVEFIG : # sauvegarde de la figure
    #            figfile = "Fig_"
    #            dpi = FIGDPI
    #            figfile += "SOMCodebookDendro-{:d}Class_{:s}{:s}Clim-{:d}-{:d}_{:s}".format(nb_class,fprefixe,fshortcode,andeb,anfin,data_label_base)
    #            ctloop.do_save_figure(figfile,dpi=dpi,path=figdir,ext=FIGEXT)
    #        #
    #        plt.show(block=blockshow)
    #

    if Visu_Dendro :
        axeshiftfactor=150 # decalage des axes deu dendrogramme pour poivoir voir les modeles qui sont identiques et qui se melangent a l'axe sinon
        wspace=0.0; hspace=0.2; top=0.92; bottom=0.12; left=0.05; right=0.99;
        mdlnameok = False
        if mdlnameok :
            prtlbl="MldName"
            figsize=(14,7);
            indnames = CAHindnames;
            bottom=0.27; labelsize=10
            axeshiftfactor=100 # decalage des axes deu dendrogramme pour poivoir voir les modeles qui sont identiques et qui se melangent a l'axe sinon
        else:
            if onlymdlumberAFCdendro_ok :
                prtlbl="MdlNo"
                figsize=(14,6);
                indnames = NoCAHindnames;
                bottom=0.13; labelsize=12
                axeshiftfactor=150 # decalage des axes deu dendrogramme pour poivoir voir les modeles qui sont identiques et qui se melangent a l'axe sinon
            elif mdlnamewnumber_ok :
                prtlbl="MdlNoAndName"
                figsize=(14,7);
                indnames = CAHindnameswnb;
                bottom=0.31; labelsize=10
                axeshiftfactor=100 # decalage des axes deu dendrogramme pour poivoir voir les modeles qui sont identiques et qui se melangent a l'axe sinon
            else :
                prtlbl="MdlName"
                figsize=(14,7);
                indnames = CAHindnames;
                bottom=0.25; labelsize=10
                axeshiftfactor=100 # decalage des axes deu dendrogramme pour poivoir voir les modeles qui sont identiques et qui se melangent a l'axe sinon
        coord2take = np.arange(NBCOORDAFC4CAH); # Coordonnées de l'AFC àprendre pour la CAH
        stitre = "AFC Dendrogram : Coord(%s). Method=%s, Metric=%s, nb_clust=%d"%\
                    ((coord2take+1).astype(str),method_afc,dist_afc,nb_clust)
        if AFCWITHOBS :
            stitre += " (Obs IN)"
        else:
            stitre += " (Obs OUT)"
        #
        ctloop.do_plot_afc_dendro(F1U,F1sU,nb_clust,Nmdlok,
                      afccoords=coord2take,
                      indnames=indnames,
                      AFCWITHOBS = AFCWITHOBS,
                      CAHWITHOBS = CAHWITHOBS,
                      afc_method=method_afc, afc_metric=dist_afc,
                      truncate_mode=None,
                      xlabel="model",
                      ylabel="AFC inter cluster distance ({}/{})".format(method_afc,dist_afc),
                      title=stitre,
                      titlefnsize=18, ytitle=1.02, labelfnsize=12,
                      labelrotation=-90, labelsize=labelsize,
                      axeshiftfactor=axeshiftfactor,
                      figsize=figsize,
                      wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right
                      )
        #
        if SAVEFIG : # sauvegarde de la figure
            figfile = "Fig_"
            dpi = FIGDPI
            figfile += "AFCDendro-{:s}-{:d}Clust_{:d}Class_{:s}".format(prtlbl,
                    nb_clust,nb_class,fprefixe)
            if AFCWITHOBS :
                figfile += "ObsInAFC_"
            else:
                figfile += "ObsOutAFC_"
            figfile += "{:s}Clim-{:s}_{:s}".format(fshortcode,dataystartend,data_label_base)
            #
            ctloop.do_save_figure(figfile,dpi=dpi,path=figdir,ext=FIGEXT)
    #
    plt.show(block=blockshow)

    if Visu_Inertie : # Inertie
        dataystartend = datemdl2dateinreval(DATAMDL)
        ctloop.printwarning([ "    AFC: INERTIE" ])
        if NIJ==1 :
            stitre = "{:s}SST({:s})) [{:s}]\n{:s}{:d} AFC on classes of Completed Models (vs Obs)".format(
                     fcodage,dataystartend,case_label,method_cah,nb_class)
        elif NIJ==2 :
            stitre = "{:s}SST({:s}))\n{:s}{:d} AFC on good classes of Completed Models (vs Obs)".format(
                     fcodage,dataystartend,method_cah,nb_class)
        elif NIJ==3 :
            stitre = "{:s}SST({:s})) [{:s}]\n{:s}{:d} AFC on good classes of Completed Models (vs Obs)".format(
                     fcodage,dataystartend,case_label,method_cah,nb_class)
        if AFCWITHOBS :
            stitre += " (Obs IN)"
        else:
            stitre += " (Obs OUT)"
        #
        ctloop.do_plot_afc_inertie(VAPT,
                     title=stitre,
                     figsize=(8,6),
                     top=0.93, bottom=0.08, left=0.08, right=0.98,
        )
        if SAVEFIG : # sauvegarde de la figure de performanes par cluster
            figfile = "Fig_"
            figfile += "Inertia-{:d}Clust-{:d}Classes_{:s}".format(nb_clust,nb_class,fprefixe)
            if AFCWITHOBS :
                figfile += "ObsInAFC_"
            else:
                figfile += "ObsOutAFC_"
            figfile += "{:s}Clim-{:s}".format(fshortcode,dataystartend)
            # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
            # eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figdir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    return VAPT,F1U,F1sU,F2V,CRi,CAj,TTperf4afc,\
           CAHindnames,CAHindnameswnb,NoCAHindnames,\
           class_afc,AFCindnames,AFCindnameswnb,NoAFCindnames,figclustmoynum,\
           allclusrPerfG, allclustTperf
#
#%%
def plot_afc_proj(F1U,F2V,CRi,CAj,F1sU,pa,po,class_afc,nb_class,NIJ,Nmdlok,indnames=None,
                  figdir=".",
                  figfile=None, dpi=None, figpdf=False,
                  mdlmarker='o',obsmarker='o',xextramarker='*',clsmarker='s',
                  xextraF1=None,xextraLbl=" Extra",xextracolor=[ 1.0, 0.0, 0.0, 1.],
                  obsLbl=" Obs",obscolor=[0.90, 0.90, 0.90, 1.],
                  xtracumbygrF1=None,xtracumbygrmarker=None,
                  ) :
    global SIZE_REDUCTION, AFCWITHOBS
    global FIGDPI, FIGEXT, Visu_UpwellArt
    #
    Nmdlafc = CRi.shape[0]
    Nclustafc = len(np.unique(class_afc))
    dataystartend = datemdl2dateinreval(DATAMDL)
    ctloop.printwarning([ "    AFC: 2-D PROJECTION" ])
    #
    if Visu_UpwellArt :
        lblfontsize       = 14; mdlmarkersize    = 250;
        lblfontsizeobs    = 16; obsmarkersize    = 250;
        lblfontsizexextra = 16; xextramarkersize = 450;
        lblfontsizecls    = 16; clsmarkersize    = 280;
        #
        if SIZE_REDUCTION == 'All' :
            zone_stitre = "Large"
            figsize=(15,12)
            top = 0.93; bottom = 0.05; left = 0.05; right = 0.95
            xdeltapos       = 0.025; ydeltapos       =-0.002; linewidths       = 3
            xdeltaposobs    = 0.030; ydeltaposobs    =-0.003; linewidthsobs    = 4
            xdeltaposxextra = 0.030; ydeltaposxextra =-0.003; linewidthsxextra = 2
            xdeltaposcls    = 0.001; ydeltaposcls    =-0.003; linewidthscls    = 3
            xdeltaposlgnd   = 0.03;  ydeltaposlgnd   =-0.002
            if AFCWITHOBS :
                if Nmdlok == 47 :
                    #legendXstart = 0.935;
                    legendXstart =-1.22;  legendYstart =-0.50; legendYstep = 0.058
                else :
                    legendXstart =-1.22;  legendYstart = 0.88; legendYstep = 0.06
            else :
                if Nmdlok == 47 :
                    legendXstart =-1.02; legendYstart = 0.84; legendYstep = 0.058
                else :
                    legendXstart =-1.22;  legendYstart = 0.88; legendYstep = 0.06
            legendXstartcls = legendXstart;
            legendYstartcls = legendYstart - legendYstep * nb_clust
        elif SIZE_REDUCTION == 'sel' :
            zone_stitre = "Selected"
            figsize=(15,12)
            top = 0.93; bottom = 0.05; left = 0.05; right = 0.95
            xdeltapos       = 0.035; ydeltapos       =-0.002; linewidths       = 3
            xdeltaposobs    = 0.040; ydeltaposobs    =-0.003; linewidthsobs    = 4
            xdeltaposxextra = 0.040; ydeltaposxextra =-0.003; linewidthsxextra = 2
            xdeltaposcls    = 0.001; ydeltaposcls    =-0.005; linewidthscls    = 3
            xdeltaposlgnd   = 0.040; ydeltaposlgnd   =-0.002
            if AFCWITHOBS :
                legendXstart    =-0.76;  legendYstart    =-0.77;  legendYstep      = 0.080
            else :
                legendXstart    =-0.76;  legendYstart    =-1.03;  legendYstep      = 0.080
            legendXstartcls = legendXstart;
            legendYstartcls = legendYstart - legendYstep * nb_clust
        #
        #if res_for_afc_clust:
        #    for ii in np.arange(nb_clust) :
        #        iclust  = np.where(class_afc==ii+1)[0];
        #            Perfglob_, _, _, Tperf_ = Dgeoclassif(sMapO,CmdlMoy,lon,lat,class_ref,
        #                                             classe_Dobs,nb_class,
        #                                             LObs,CObs,isnumObs,TypePerf[0],
        #                                             bgval=bgval,
        #                                             visu=False);

        #
        if AFCWITHOBS :
            stitre = ("SST {:s} -on zone \"{:s}\"- A.F.C Models+Obs -").format(fcodage,zone_stitre)
        else:
            stitre = ("SST {:s} -on zone \"{:s}\"- A.F.C Models only -").format(fcodage,zone_stitre)
        #
        stitre += (" Projection with Models, Observations and Classes ({:s})"+\
                   "\n- {:s}, AFC on {:d} CAH Classes for {} models (in {}"+\
                   " AFC clusters) + Obs -").format(dataystartend,method_cah,nb_class,Nmdlok,nb_clust)
        #
        ctloop.do_plotart_afc_projection(F1U,F2V,CRi,CAj,F1sU,pa,po,class_afc,nb_class,NIJ,Nmdlok,
                    indnames=indnames,
                    title=stitre,
                    Visu4Art=Visu_UpwellArt,
                    AFCWITHOBS = AFCWITHOBS,
                    xextraF1=xextraF1,
                    xextraLbl=xextraLbl,xextracolor=xextracolor,
                    obsLbl=obsLbl, obscolor=obscolor,
                    figsize=figsize,
                    top=top, bottom=bottom, left=left, right=right,
                    lblfontsize=lblfontsize,             mdlmarkersize=mdlmarkersize,
                    xdeltapos=xdeltapos,                 ydeltapos=ydeltapos,
                    linewidths=linewidths,               mdlmarker=mdlmarker,
                    lblfontsizeobs=lblfontsizeobs,       obsmarkersize=obsmarkersize,
                    xdeltaposobs=xdeltaposobs,           ydeltaposobs=ydeltaposobs,
                    linewidthsobs=linewidthsobs,         obsmarker=obsmarker,
                    lblfontsizexextra=lblfontsizexextra, xextramarkersize=xextramarkersize,
                    xdeltaposxextra=xdeltaposxextra,     ydeltaposxextra=ydeltaposxextra,
                    linewidthsxextra=linewidthsxextra,   xextramarker=xextramarker,
                    lblfontsizecls=lblfontsizecls,       clsmarkersize=clsmarkersize,
                    xdeltaposcls=xdeltaposcls,           ydeltaposcls=ydeltaposcls,
                    linewidthscls=linewidthscls,         clsmarker=clsmarker,
                    xtracumbygrF1=xtracumbygrF1,         xtracumbygrmarker=xtracumbygrmarker,
                    legendok=True,
                    xdeltaposlgnd=xdeltaposlgnd,ydeltaposlgnd=ydeltaposlgnd,
                    legendXstart=legendXstart,legendYstart=legendYstart,legendYstep=legendYstep,
                    legendprefixlbl="AFC Cluster",
                    legendprefixlblobs="Observations",
                    legendprefixlblxextra="Model-all",
                    legendokcls=True,
                    legendXstartcls=legendXstartcls,legendYstartcls=legendYstartcls,
                    legendprefixlblcls="CAH Classes",
                    )
    else :
        stitre = ("AFC Projection - {:s} SST ({:s}). {:s}".format(fcodage,
                  dataystartend,method_cah));
        lblfontsize=14; linewidths = 2.0
        #
        ctloop.do_plot_afc_projection(F1U,F2V,CRi,CAj,F1sU,pa,po,class_afc,nb_class,NIJ,Nmdlok,
                    indnames=indnames,
                    title=stitre,
                    AFCWITHOBS = AFCWITHOBS,
                    figsize=(16,12),
                    top=0.93, bottom=0.05, left=0.05, right=0.95,
                    lblfontsize=lblfontsize, linewidths=linewidths,
                    )
    #
    if SAVEFIG : # sauvegarde de la figure
        if figfile is None :
            figfile = "Fig_"
        if dpi is None :
            dpi = FIGDPI
        figfile += "AFC2DProj-{:d}-{:d}_{:d}Clust-{:d}Classes_{:s}".format(
                pa,po,nb_clust,nb_class,fprefixe,fshortcode,dataystartend,Nmdlok)
        if AFCWITHOBS :
            figfile += "ObsInAFC_"
        else:
            figfile += "ObsOutAFC_"
        if xtracumbygrF1 is not None:
            figfile += "wGroupProj_"
        figfile += "{:s}Clim-{:s}_{:d}-mod".format(fshortcode,dataystartend,Nmdlok)
        #
        ctloop.do_save_figure(figfile,dpi=dpi,path=figdir,ext=FIGEXT,figpdf=figpdf)
#
#%%
def ctloop_generalisation(sMapO, lon, lat, TMixtMdl, TMixtMdlLabel, TDmdl4CT, Tmdlname,
                          nb_class, isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs,
                          varnames=None,
                          modstdflg=False,
                          subtitle=None,
                          figdir=".",
                          figfile1=None, figfile2=None, figfileext1="", figfileext2="", dpi=None, figpdf=False,
                          wvmin=None,wvmax=None,
                          eqcmap=None,
                          ccmap="jet",
                          bgmap='gray', bgval=0.5,
                          plotmeanmodel=True,
                          ) :
    #
    global SIZE_REDUCTION
    global SAVEFIG, FIGDPI, FIGEXT, FIGARTDPI, SAVEPDF, VFIGEXT, blockshow
    global TypePerf
    #
    if eqcmap is None :
        eqcmap = cm.get_cmap('RdYlBu_r')  # Palette RdYlBu inversée
    #
    ctloop.printwarning([ "","GENERALISATION".center(75),""])
    #==========================================================================
    #
    #------------------------------------------------------------------------
    if SIZE_REDUCTION == 'All' :
        misttitlelabel = TMixtMdlLabel+" (Big Zone)"
    elif SIZE_REDUCTION == 'sel' :
        misttitlelabel = TMixtMdlLabel+" (Small Zone)"
    #print("misttitlelabel AVANT> {}".format(misttitlelabel))
    mistfilelabel = misttitlelabel.replace(' ','').replace(':','-').replace('(','_').replace(')','')
    #print("mistfilelabel APRES>  {}".format(mistfilelabel))
    if subtitle is not None :
        misttitlelabel += " - {:s}".format(subtitle)
    #
    if type(Tmdlname) is list :
        Tmdlname = np.array(Tmdlname)
    if type(TDmdl4CT) is list :
        TDmdl4CT = np.array(TDmdl4CT)
    #
    #------------------------------------------------------------------------
    # PZ: BEST AFC CLUSTER:
    #TMixtMdl = []
    #
    
    if TMixtMdl == [] :
        print("\nSopt non renseigné ; Ce Cas n'a pas encore été prévu")
        return
    #
    if plotmeanmodel :
        if SIZE_REDUCTION == 'All' :
            figsize = (10.5,7)
            top=0.93; bottom=0.095; left=0.07; right=0.915;
            wspace=0.0; hspace=0.0;
            nticks = 5; # 4
        elif SIZE_REDUCTION == 'sel' :
            figsize=(8.5,7)
            top=0.92; bottom=0.10; left=0.06; right=0.96;
            wspace=0.0; hspace=0.0;
            nticks = 2; # 4
        #
        fig = plt.figure(figsize=figsize)
        fignum = fig.number # numero de figure en cours ...
        fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
        #
        print("\n{:d}-model(s)' generalization: {} ".format(len(TMixtMdl),TMixtMdl))
        MdlMoy, IMixtMdl, MGPerfglob, XMgeo, \
            Tperf = ctloop.mixtgeneralisation (sMapO, TMixtMdl, Tmdlname, TDmdl4CT, 
                                               class_ref, classe_Dobs, nb_class, Lobs, Cobs, isnumobs,
                                               lon=lon, lat=lat,
                                               TypePerf=TypePerf,
                                               label=misttitlelabel,
                                               fignum=fignum,
                                               title_fontsize=16,labels_fontsize=14,
                                               tickfontsize=14,
                                               cbticklabelsize=16,cblabelsize=18,
                                               ytitre=1.015, nticks=nticks,
                                               visu=True,
                                               ccmap=ccmap,
                                               bgmap=bgmap, bgval=bgval,
                                               );
        #
        if len(IMixtMdl) != 0 :
            if SAVEFIG : # sauvegarde de la figure
                if figfile1 is None :
                    figfile1 = "Fig_"
                if dpi is None :
                    dpi = FIGDPI
                figfile1 += "MeanModel_{:s}-{:d}-mod_Mean".format(mistfilelabel,len(Tmdlname[IMixtMdl]))
                figfile1 += figfileext1
                #
                ctloop.do_save_figure(figfile1,dpi=dpi,path=figdir,ext=FIGEXT,figpdf=figpdf)
    else :
        MdlMoy, IMixtMdl, MGPerfglob, XMgeo, \
            Tperf = ctloop.mixtgeneralisation (sMapO, TMixtMdl, Tmdlname, TDmdl4CT, 
                                               class_ref, classe_Dobs, nb_class, Lobs, Cobs, isnumobs,
                                               lon=lon, lat=lat,
                                               TypePerf=TypePerf,
                                               visu=False,
                                               bgmap=grcmap, bgval=bggray,
                                               );
    if len(IMixtMdl) == 0 :
        print("\n *** PAS DE MODELES POUR GENERALISATION !!! ***\n" )
        return
    #
    # Affichage du moyen for CT
    if SIZE_REDUCTION == 'All' :
        lolast = 4
        figsize = (12,7)
        wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.97;
        nticks = 5; # 4
    elif SIZE_REDUCTION == 'sel' :
        lolast = 2
        figsize=(11.5,8.5)
        wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.965;
        nticks = 2; # 4
    fig = plt.figure(figsize=figsize)
    fignum = fig.number # numero de figure en cours ...
    if modstdflg and len(Tmdlname[IMixtMdl]) > 2:
        std_ = np.std(TDmdl4CT[IMixtMdl],axis=0);
        print(np.min(std_),np.max(std_),np.mean(std_),np.std(std_))
        ctobs.aff2D(MdlMoy,Lobs,Cobs,isnumobs,isnanobs,
              wvmin=wvmin,wvmax=wvmax,
              fignum=fignum,varnames=varnames,cmap=eqcmap,
              wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
              noaxes=False,noticks=False,nolabels=False,y=0.98,cblabel="SST Anomaly [°C]",
              lolast=lolast,lonlat=(lon,lat),
              vcontour=std_, ncontour=np.arange(0,2,1/10), ccontour='k', lblcontourok=True,
              ); #...
    else :
        ctobs.aff2D(MdlMoy,Lobs,Cobs,isnumobs,isnanobs,
              wvmin=wvmin,wvmax=wvmax,
              fignum=fignum,varnames=varnames,cmap=eqcmap,
              wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
              noaxes=False,noticks=False,nolabels=False,y=0.98,cblabel="SST Anomaly [°C]",
              lolast=lolast,lonlat=(lon,lat),
              ); #...
    sptitre = "{}".format(misttitlelabel)
    if modstdflg :
        if len(Tmdlname[IMixtMdl]) > 2 :
            sptitre += " and inter-model STD in contours"
        else:
            sptitre += " (not enouth models STD)"
    sptitre += " ({} models)".format(len(Tmdlname[IMixtMdl]))
    sptitre += ", mean perf={:.0f}%".format(100*MGPerfglob)
    sptitre += "\nmin=%f, max=%f, moy=%f, std=%f"%(np.min(MdlMoy),np.max(MdlMoy),
                                                   np.mean(MdlMoy),np.std(MdlMoy))
    plt.suptitle(sptitre,fontsize=14,y=0.995);
    #
    if SAVEFIG : # sauvegarde de la figure
        if figfile2 is None :
            figfile2 = "Fig_"
        if dpi is None :
            dpi = FIGDPI
        figfile2 += "{:s}-{:d}-mod_Mean".format(mistfilelabel,len(Tmdlname[IMixtMdl]))
        figfile2 += figfileext2
        figfile2 += "_{}".format(eqcmap.name)
        # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
        # eventuellement en PDF, si SAVEPDF active. 
        ctloop.do_save_figure(figfile2,dpi=dpi,path=figdir,ext=FIGEXT,figpdf=figpdf)
    #
    #
    #**********************************************************************
    return
#%%
def generalisation_proc(sMapO, lon, lat, varnames, wvmin, wvmax, nb_class,
                        isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, 
                        TDmdl4CT, Tmdlname,
                        data_period_ident=None,
                        eqcmap=None,
                        bgmap='gray', bgval=0.5,
                        figdir=None,
                        generalisation_type=None,  # 'bestclust', 'bestcum'
                        scenario=None,
                        nFigArt=None,
                        ) :
    ##########################################################################
    #    global , eqcmap, ccmap, nFigArt
    #    global , Dobs, XC_Ogeo, NDobs, fond_C, pcmap, obs_data_path
    #    global , ilon, ilat, 
    #    global AFCindnames, NoAFCindnames, 
    #    global 
    #    global class_afc, list_of_plot_colors
    #
    if eqcmap is None :
        eqcmap = cm.get_cmap('RdYlBu_r')  # Palette RdYlBu inversée
    #
    global SAVEFIG
    #
    #**************************************************************************
    #.............................. GENERALISATION ............................
    #
    #--------------------------------------------------------------------------
    # Modèles Optimaux (Sopt) ;  Avec "NEW Obs - v3b " 1975-2005
    # Je commence par le plus simple : Une ligne de modèle sans classe en une phase
    # et une seule codification à la fois
    # Sopt-1975-2005 : Les meilleurs modèles de la période "de référence" 1975-2005
    #TMixtMdl= [];
    #TMixtMdl =['CNRM-CM5', 'CMCC-CMS', 'CNRM-CM5-2', 'GFDL-CM3', 'FGOALS-s2']; 
    #TMixtMdl = Sfiltre;
    all_possible_values = ["bestclust", "bestcum", "all" ]
    if generalisation_type is None:
        generalisation_type = "bestclust"
    elif generalisation_type not in all_possible_values : # SI ni l'un ni l'autre ...
        ctloop.printwarning(["","generalisation_proc ERROR".upper().center(75),""],
                ["Bad generalisation_type: '{}'".format(generalisation_type).center(75),
                 "chose one from {}".format(generalisation_type,all_possible_values).center(75)])
        raise
    #
    ctloop.printwarning(["","current generalisation '{}'".upper().format(generalisation_type).center(75),""])
    if generalisation_type == "bestclust" : # Best AFC Clusters
        if SIZE_REDUCTION == 'All' :
            # Grande Zone (All): BEST AFC CLUSTER: -> Cluster 4, 13 Models, performance: 69.3 :
            TMixtMdlLabel = 'Best AFC Cluster'
            TMixtMdl = ['CMCC-CM', 'HadGEM2-ES', 'HadGEM2-AO', 'HadGEM2-CC', 'CMCC-CMS',
                        'CNRM-CM5-2', 'CanESM2', 'CanCM4', 'GFDL-CM3', 'CNRM-CM5', 'FGOALS-s2', 
                        'CSIRO-Mk3-6-0', 'CMCC-CESM']
        elif SIZE_REDUCTION == 'sel' :
            # Petite Zone (sel): BEST AFC CLUSTER: -> Cluster 2, 5 Models, performance: 66.0 :
            TMixtMdlLabel = 'Best AFC Cluster'
            TMixtMdl = ['CNRM-CM5', 'CMCC-CMS', 'CNRM-CM5-2', 'GFDL-CM3', 'FGOALS-s2']
        #
    elif generalisation_type == "bestcum" : # Best Cum Clusters Mopr
        if SIZE_REDUCTION == 'All' :
            # Grande Zone (All): BEST CUM GROUP OF MODELS:
            TMixtMdlLabel = 'Best Cumulated Models Group'
            TMixtMdl = ['CMCC-CM']
        elif SIZE_REDUCTION == 'sel' :
            # Petite Zone (sel): BEST CUM GROUP OF MODELS:
            TMixtMdlLabel = 'Best Cumulated Models Group'
            TMixtMdl = ['CanCM4', 'CNRM-CM5', 'CMCC-CMS', 'CNRM-CM5-2', 'GFDL-CM3', 'CanESM2', 'NorESM1-ME']
        #
    elif generalisation_type == "all" : # Best Cum Clusters Mopr
        TMixtMdlLabel = 'All Models Cumulated'
        TMixtMdl = Tmdlname
        #
    else :
        # ALL MODELS (but 'FGOALS-s2', there is no 1975-2005 data for it):
        TMixtMdlLabel = 'All Models'
        TMixtMdl = Tmdlname
        #TMixtMdl = ['CMCC-CM', 'HadGEM2-ES', 'HadGEM2-AO', 'HadGEM2-CC', 'CMCC-CMS',
        #            'FGOALS-g2', 'IPSL-CM5B-LR', 'CNRM-CM5-2', 'CanESM2', 'CanCM4',
        #            'GFDL-CM3', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'MPI-ESM-MR', 'MRI-CGCM3',
        #            'MRI-ESM1', 'CMCC-CESM', 'inmcm4', 'bcc-csm1-1', 'MPI-ESM-LR',
        #            'CESM1-BGC', 'MPI-ESM-P', 'IPSL-CM5A-LR', 'GISS-E2-R',
        #            'GISS-E2-R-CC', 'NorESM1-M', 'CCSM4', 'NorESM1-ME', 'bcc-csm1-1-m',
        #            'GFDL-CM2p1', 'GISS-E2-H', 'ACCESS1-3', 'MIROC5', 'GFDL-ESM2G',
        #            'MIROC-ESM-CHEM', 'GFDL-ESM2M', 'GISS-E2-H-CC', 'MIROC-ESM',
        #            'IPSL-CM5A-MR', 'HadCM3', 'CESM1-CAM5', 'CESM1-CAM5-1-FV2',
        #            'ACCESS1-0']
    #
    # -------------------------------------------------------------------------
    # Figure 5 pour Article : Classes par cluster sur projectrion geo best cluster AFC
    # -------------------------------------------------------------------------
    if nFigArt is not None :
        figfile1 = "FigArt{:02d}_".format(nFigArt)
        nFigArt += 1;
        figfile2 = "FigArt{:02d}_".format(nFigArt)
        dpi     = FIGARTDPI
        figpdf  = True
    else :
        figfile1 = figfile2 = "Fig_"
        dpi     = FIGDPI
        figpdf  = False
    #
    if type(eqcmap) is list :
        iter_cmap = eqcmap
    else :
        iter_cmap = [ eqcmap ]
    #
    if data_period_ident is not None :
        dataystartend = datemdl2dateinreval(data_period_ident)
        stitre = "({:s})".format(dataystartend)
        datafileext = "-{:s}".format(dataystartend)
    else :
        stitre = None
        datafileext = "-{:s}".format(dataystartend)
    #
    if scenario is not None :
        if stitre is None :
            stitre = ""
        stitre += " {:s}".format(scenario.upper())
        datafileext += "-{:s}".format(scenario.upper())
    #
    current_figs_dir = None
    if SAVEFIG :
        if figdir is None :
            global FIGSDIR
            figdir = FIGSDIR
        #
        if generalisation_type is None :
            local_figs_subdir = "generaliz"
        else :
            local_figs_subdir = "gen-{}".format(generalisation_type)
        local_figs_subdir += "{}".format(datafileext)
            #
        current_figs_dir = ctloop.do_check_and_create_dirname(figdir, subdir=local_figs_subdir)
    # -------------------------------------------------------------------------
    figfileext1 = "_{:s}{:s}_{:d}Class{:s}".format(fprefixe,fshortcode,nb_class,datafileext)
    figfileext2 = ""
    #if Show_ModSTD :
    #    figfileext2 += "+{:d}ySTD".format(Nda)
    figfileext2 += "_Lim{:+.1f}-{:+.1f}_{:s}{:s}Clim{:s}".format(wvmin,wvmax,
                        fprefixe,fshortcode,datafileext)
    #
    for ii,v_eqcmap in enumerate(iter_cmap) :
        plotmeanmodel = (ii==0) # seulemen une fois (pour le premier)
        #plotmeanmodel = True # a chaqu fois, pour l'instant ...
        ctloop_generalisation(sMapO, lon, lat, TMixtMdl, TMixtMdlLabel, TDmdl4CT, Tmdlname,
                          nb_class, isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs,
                          varnames=varnames,
                          modstdflg=Show_ModSTD,
                          figdir=current_figs_dir,
                          figfile1=figfile1, figfile2=figfile2, figfileext1=figfileext1, figfileext2=figfileext2,
                          dpi=dpi, figpdf=figpdf,
                          wvmin=wvmin,wvmax=wvmax,
                          subtitle=stitre,
                          eqcmap=v_eqcmap,
                          bgmap=bgmap, bgval=bgval,
                          plotmeanmodel=plotmeanmodel,
                          )
    #
    return
#
#%%
def generalisafcclust_proc(sst_obs_coded,Dobs,XC_Ogeo,sMapO,lon,lat,ilat,ilon, varnames, wvmin, wvmax, nb_class,
                        isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, 
                        TDmdl4CT, Tmdlname,
                        data_period_ident=None,
                        eqcmap=None,
                        bgmap='gray', bgval=0.5,
                        figdir=None,
                        nFigArt=None,
                        ) :
    # #########################################################################
    #    global , eqcmap, ccmap, nFigArt
    #    global , Dobs, XC_Ogeo, NDobs, fond_C, pcmap, obs_data_path
    #    global , ilon, ilat, 
    #    global AFCindnames, NoAFCindnames, 
    #    global 
    #    global class_afc, list_of_plot_colors
    #
    if eqcmap is None :
        eqcmap = cm.get_cmap('RdYlBu_r')  # Palette RdYlBu inversée
    #
    if data_period_ident is not None :
        dataystartend = datemdl2dateinreval(data_period_ident)
        datastitre = " ({:s})".format(dataystartend)
        datafileext = "-{:s}".format(dataystartend)
    else :
        datastitre = ""
        datafileext = ""
    #
    current_figs_dir = None
    if SAVEFIG :
        if figdir is None :
            global FIGSDIR
            figdir = FIGSDIR
    #
    if type(eqcmap) is list :
        iter_cmap = eqcmap
    else :
        iter_cmap = [ eqcmap ]
    #
    #**************************************************************************
    #.............................. GENERALISATION ............................
    #
    #
    #  Generalisation d'une ensemble ou cluster precis
    for kclust in np.arange(1,nb_clust + 1) :
        #kclust = 1
        if SAVEFIG :
            local_figs_subdir = "gen-afcclust{:02d}".format(kclust)
            local_figs_subdir += "{}".format(datafileext)
            current_figs_dir = ctloop.do_check_and_create_dirname(figdir, subdir=local_figs_subdir)
        #
        SfiltredMod = AFCindnames[np.where(class_afc==kclust)[0]]
        print("--Generalizing for AFC Cluster{:s}: {:d} with models:\n  {}".format(
                datastitre,kclust,SfiltredMod))
        stitre = "AFC Cluster: {:d}{:s}".format(kclust,datastitre)
        fileextIV = "_AFCclust{:d}{:s}-{:s}{:s}_{:s}{:s}".format(kclust,datafileext,fprefixe,
                           SIZE_REDUCTION,fshortcode,method_cah)
        fileextIV79 = "_AFCclust{:d}{:s}-{:s}{:s}_{:s}".format(kclust,datafileext,fprefixe,
                             SIZE_REDUCTION,fshortcode)
        xTperfglob, xTperfglob_Qm, xTDmdl4CT, xTmdlname, xTmdlnamewnb, \
            xTmdlonlynb, xTTperf, xTTperf_Qm, xNmdlok, \
            xNDmdl = ctloop_model_traitement(
                        sst_obs_coded,Dobs,XC_Ogeo,sMapO,lon,lat,ilat,ilon,
                        nb_class,class_ref,classe_Dobs,NDobs,fond_C,
                        isnanobs,isnumobs,Lobs,Cobs,list_of_plot_colors,
                        Sfiltre=SfiltredMod,
                        varnames=varnames, figdir=current_figs_dir,
                        commonfileext=fileextIV, commonfileext79=fileextIV79,
                        pair_nsublc=[7,7],
                        subtitle=stitre,
                        eqcmap=eqcmap, ccmap=ccmap,pcmap=pcmap,
                        obs_data_path=obs_data_path,
                        data_period_ident=data_period_ident,
                        OK101=False,
                        OK102=False,
                        OK104=OK104,
                        OK105=OK105,
                        OK106=False,
                        OK107=False,
                        OK108=OK108,
                        OK109=False,
                        )
        #
        TMixtMdlLabel = "Cumulated for AFC Cluster {:d} {:s}".format(kclust,datastitre)
        TMixtMdl = AFCindnames[np.where(class_afc==kclust)[0]]
        for ii,v_eqcmap in enumerate(iter_cmap) :
            plotmeanmodel = (ii==0) # seulemen une fois (pour le premier)
            plotmeanmodel = True # a chaqu fois, pour l'instant ...
            ctloop_generalisation(sMapO, lon, lat, TMixtMdl, TMixtMdlLabel,
                              TDmdl4CT, Tmdlname,
                              nb_class, isnumobs, isnanobs, Lobs, Cobs,
                              class_ref, classe_Dobs,
                              varnames=varnames,
                              wvmin=wvmin,wvmax=wvmax,
                              eqcmap=v_eqcmap,
                              bgmap=bgmap, bgval=bgval,
                              figdir=current_figs_dir,
                              plotmeanmodel=plotmeanmodel,
                              )
    return
#%%
# #############################################################################
#
# Fin de declaration des fonctions locales 
#
# #############################################################################
#%% 
# #############################################################################
#
#  MAIN 
#
# #############################################################################
def main(argv):
    #%%
    global SAVEMAP, MAPSDIR, FIGSDIR, WITHANO
    #
    global fprefixe, tpgm0, blockshow, fcodage, fshortcode
    #
    # Globals necessaires a la generalisation manuelle ...
    global nb_class, eqcmap, ccmap, grcmap, bggray

    global sst_obs_coded, Dobs, XC_Ogeo, NDobs, fond_C, pcmap, obs_data_path
    global sMapO, lon, lat, ilon, ilat, varnames, wvmin, wvmax
    global AFCindnames, NoAFCindnames, TDmdl4CT, Tmdlname, Tmdlnamewnb, Tmdlonlynb
    global isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, case_figs_dir
    global class_afc, list_of_plot_colors
    global Visu_UpwellArt
    #
    caseconfig = ''
    caseconfig_valid_set = ( 'All', 'sel' )   # toutes en minuscules svp !
    verbose = False
    manualmode = True
    #
    generalisation_ok = False
    #generalisation_ok = True
    generalisa_bestafc_clust_ok = True
    generalisa_bestcum_ok = True
    generalisa_allcum_ok = True
    generalisa_allafc_clust_ok = True
    generalisa_other_periods_ok = False
    #
    if Visu_UpwellArt :
        FIGSDIR = 'FigArt'
    #%% NE PAS EXECUTER CE BLOCK EN MODE MANUEL
    manualmode = False
    try:
        opts, args = getopt.getopt(argv,"hvc:g",["case=","verbose","generalization"])
        #
        for opt, arg in opts:
            if opt == '-h':
                print('ctLoopMain.py -c all -v -n | --case=all --verbose --no-generalization /* cases are all or sel */')
                sys.exit()
            elif opt in ("-v", "--verbose"):
                verbose = True
            elif opt in ("-c", "--case"):
                caseconfig = arg
            elif opt in ("-g", "--generalization"):
                generalisation_ok = True
        if caseconfig.lower() not in [ x.lower() for x in caseconfig_valid_set ] :
            ctloop.printwarning(["","Start error".center(75),""],
                    "   Not CASE value '{}'".format(caseconfig).center(75),
                    "You should give one in {} set.".format(caseconfig_valid_set).center(75))
            raise

    except getopt.GetoptError:
        print(('\n {:s}\n'.format("".center(66,'*'))+\
               ' * {:s} *\n'.format("Invalid Call:".center(62))+\
               ' **{:s}**\n'.format("".center(62,'*'))+\
               ' * {:s} *\n'.format("Call:".ljust(62))+\
               ' * {:s} *\n'.format("  ctLoopMain.py <OPTIONS> -c CASE".ljust(62))+\
               ' * {:s} *\n'.format("".ljust(62))+\
               ' * {:s} *\n'.format("  where CASE is one in < {} >".format(caseconfig_valid_set).ljust(62))+\
               ' * {:s} *\n'.format("      <OPTIONS> are -h, -v, -n".ljust(62))+\
               ' {:s}\n'.format("".center(66,'*'))))
        sys.exit(2)
    #print("Case config is '{:s}'".format(caseconfig))
    #print("Verbose is '{}'".format(verbose))
    #
    # #########################################################################
    #
    #                             INITIALISATION
    #
    #%% DECOMPRESSER / COMPRESSER la ligne suivante selon si executione MANUELLE UN A UN des bloques du MAIN ou complete avec appel externe ... 
    if manualmode :
        caseconfig='sel' # 'all' ou 'sel'
        #caseconfig='all' # 'all' ou 'sel'
    #
    print("Case config is '{:s}'".format(caseconfig))
    pcmap,cpcmap,AFC_Visu_Classif_Mdl_Clust, AFC_Visu_Clust_Mdl_Moy_4CT,\
        TM_label_base, case_label_base, obs_data_path, \
        tseed, case_name_base, casetime, casetimelabel, casetimeTlabel, varnames,\
        list_of_plot_colors  =  ctloop_init(case=caseconfig,verbose=verbose)
    #
    print("fcodage,fshortcode ... {}".format((fcodage,fshortcode)))
    #
    #%% #######################################################################
    #
    #      LECTURE DES DONNEES D'OBSERVATION: 'raverage_1975_2005' ou autre
    #
    sst_obs, sst_obs_coded, Dobs, NDobs, lon, lat, ilat, ilon, isnanobs, isnumobs, case_label,\
        data_label_base, Nobs, Lobs, Cobs = ctloop_load_obs(DATAOBS, path=obs_data_path,
                                                            case_name=case_name_base)
    #
    dataobsystartend = datemdl2dateinreval(DATAOBS)
    #
    # -----------------------------------------------------------------------------
    # Repertoire principal des maps (les objets des SOM) et sous-repertoire por le cas en cours 
    case_maps_dir = None
    if SAVEMAP :
        case_maps_dir = ctloop.do_check_and_create_dirname(MAPSDIR, subdir=case_label)
    # -------------------------------------------------------------------------
    # Repertoire principal des figures et sous-repertoire por le cas en cours 
    case_figs_dir = None
    if SAVEFIG :
        case_figs_dir = ctloop.do_check_and_create_dirname(FIGSDIR, subdir=case_label)
    #
    # Limites (min, max) manuelles de l'annomalie de la SST
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
    #%% -----------------------------------------------------------------------
    # Figure Obs4CT
    if Visu_ObsStuff : # Visu (et sauvegarde éventuelle de la figure) des données
        #list_of_eqcmap = [eqcmap, cm.jet]
        list_of_eqcmap = eqcmap
        #list_of_eqcmap = cm.jet
        if type(list_of_eqcmap) is list :
            iter_cmap = list_of_eqcmap
        else :
            iter_cmap = [ eqcmap ]
        #  
        for ii,v_eqcmap in enumerate(iter_cmap) :
            fileext1 = "_Lim{:+.1f}-{:+.1f}".format(wvmin,wvmax)
            fileext2 = "_{:s}{:s}Clim-{:s}_{:s}".format(fprefixe,
                          fshortcode,dataobsystartend,data_label_base)
            
            plot_obs4ct(sst_obs_coded,Dobs,lon,lat,isnanobs=isnanobs,isnumobs=isnumobs,
                        varnames=varnames,wvmin=wvmin,wvmax=wvmax,
                        eqcmap=v_eqcmap,Show_ObsSTD=Show_ObsSTD,
                        filecomp=fileext1, fileext=fileext2,
                        figpath=case_figs_dir,fcodage=fcodage,
                        freelimststoo=False)
    #
    if STOP_BEFORE_CT :
        plt.show(); sys.exit(0);
    #
    #%% #######################################################################
    #
    #                             CARTE TOPOLOGIQUE
    #
    mapfile = "Map_{:s}{:s}Clim-{:s}_{:s}_ts-{}{}".format(fprefixe,fshortcode,
               dataobsystartend,data_label_base,tseed,mapfileext)
    sMapO,q_err,t_err = ctloop_topol_map_traitement (Dobs, Parm_app=Parm_app,
                              mapsize=[nbl, nbc], tseed=tseed,
                              mapfile=mapfile,
                              mappath=case_maps_dir, varnames=varnames,
                              case_label=case_label,
                              casetime=casetime,casetimelabel=casetimelabel)
    #
    # -------------------------------------------------------------------------
    # Figure dec CT
    if Visu_CTStuff : # Visu (et sauvegarde éventuelle de la figure) des données
        plot_ct_Umatrix(sMapO,figsize=(4,10))
        plot_ct_map_wei(sMapO,figsize=(6,11))
    #
    #%% -----------------------------------------------------------------------
    # Computing C.T. Linkage for Dendrogram ___________________________________
    bmusO     = ctk.mbmus (sMapO, Data=Dobs); # déjà vu ? conditionnellement ?
    #
    # Performs hierarchical/agglomerative clustering on the condensed distance matrix data
    cblnkg = linkage(sMapO.codebook, method=method_cah, metric=dist_cah);
    #
    # Forms flat clusters from the hierarchical clustering defined by the linkage matrix Z.
    class_ref = fcluster(cblnkg,nb_class,criterion='maxclust'); # Classes des referents
    #
    #%% -----------------------------------------------------------------------
    if Visu_Dendro :
        stitre = ("SOM Codebook Dendrogram for HAC ({:s}) (map size={:d}x{:d})"+\
                  " - {:d} classes").format(dataobsystartend,nbl,nbc,nb_class);
        xlabel="codebook number"
        ylabel="inter class distance ({}/{})".format(method_cah,dist_cah)
        fileextdendro = "_{:s}{:s}Clim-{:s}_{:s}".format(fprefixe,
                      fshortcode,dataobsystartend,data_label_base)
        plot_ct_dendro(sMapO, nb_class, datalinkg=cblnkg, title=stitre, xlabel=xlabel, ylabel=ylabel,
                       figdir=case_figs_dir, fileext=fileextdendro)
    #
    #%% #######################################################################
    # Transcodage des indices des classes
    # (trie les classes pour avoir un certain ordre a l'affichage ...)
    if TRANSCOCLASSE is not '' :
        class_ref = ctobs.transco_class(class_ref,sMapO.codebook,crit=TRANSCOCLASSE);
    #
    classe_Dobs = class_ref[bmusO].reshape(NDobs); #(sMapO.dlen)
    XC_Ogeo     = ctobs.dto2d(classe_Dobs,Lobs,Cobs,isnumobs); # Classification géographique
    #
    fond_C = np.ones(len(classe_Dobs))
    fond_C = ctobs.dto2d(fond_C,Lobs,Cobs,isnumobs,missval=bggray)

    # Nombre de pixels par classe (pour les obs)
    Nobsc = np.zeros(nb_class)
    for c in np.arange(nb_class)+1 :
        iobsc = np.where(classe_Dobs==c)[0]; # Indices des classes c des obs
        Nobsc[c-1] = len(iobsc);
    #
    #%% -----------------------------------------------------------------------
    # Figure 1 pour Article : profils moyens par classe
    # -------------------------------------------------------------------------
    if Visu_ObsStuff or Visu_UpwellArt :
        stitre = "Observations ({:s}), {} Class Geographical Representation".format(dataobsystartend,nb_class)
        fileextbis = "_{:s}{:s}Clim-{:s}_{:s}".format(fprefixe,
                      fshortcode,dataobsystartend,data_label_base)
        if SIZE_REDUCTION == 'All' :
            figsize = (10.5,7)
            #top=0.94; bottom=0.08; left=0.06; right=0.925;
            top=0.93; bottom=0.095; left=0.07; right=0.915;
            nticks = 5; # 4
        elif SIZE_REDUCTION == 'sel' :
            figsize=(8.5,7)
            top=0.93; bottom=0.095; left=0.06; right=0.96;
            nticks = 2; # 4
        #
        if Visu_UpwellArt :
            if SIZE_REDUCTION == 'All' :
                nFigArt = 2;
            elif SIZE_REDUCTION == 'sel' :
                nFigArt = 6;
            else:
                nFigArt = 99;
            FigArtId = 'a';
            figfile = "FigArt{:02d}{:s}_".format(nFigArt,FigArtId);
            dpi     = FIGARTDPI
            figpdf  = True
            notitle = False
            #top = 0.98
        else :
            figfile = "Fig_"
            dpi     = FIGDPI
            figpdf  = False
            notitle = False
        #
        if Visu_UpwellArt :
            if SIZE_REDUCTION == 'All' :
                cb_label = 'Region cluster';
            elif SIZE_REDUCTION == 'sel' :
                cb_label = 'ZRegion cluster';
            else:
                cb_label = 'Region cluster';
        else:
            cb_label = 'Class';

        plot_geo_classes(lon,lat,XC_Ogeo,fond_C,nb_class,
                         nticks=nticks,
                         title=stitre,
                         fileext=fileextbis, figdir=case_figs_dir,
                         figfile=figfile, dpi=dpi, figpdf=figpdf,
                         ccmap=cpcmap,
                         bgmap=grcmap,
                         figsize=figsize,
                         top=top, bottom=bottom, left=left, right=right,
                         ticks_fontsize=14,labels_fontsize=16,title_fontsize=18,
                         cblabel=cb_label,
                         cbticks_fontsize=16,cblabel_fontsize=18,
                         title_y=1.015,
                         notitle=notitle,
                         )
                         #ticks_fontsize=10,labels_fontsize=12,title_fontsize=16,
                         #cbticks_fontsize=12,cblabel_fontsize=14,
    #%% -----------------------------------------------------------------------
    # Figure 2 pour Article : profils moyens par classe
    # -------------------------------------------------------------------------
    if Visu_ObsStuff or Visu_UpwellArt :
        stitre = "Observations ({:s}), Monthly Mean by Class (method: {:s})".format(dataobsystartend,method_cah)
        fileextbis = "_{:s}{:s}Clim-{:s}_{:s}".format(fprefixe,
                      fshortcode,dataobsystartend,data_label_base)
        #top=0.95; bottom=0.08; left=0.06; right=0.92;
        top=0.93; bottom=0.095; left=0.07; right=0.98;
        #
        if Visu_UpwellArt :
            if SIZE_REDUCTION == 'All' :
                nFigArt = 2;
                lgtitle = 'Region-\n cluster'
                lglabel_fontsize = 18
                errorcaps  = True
                capslength = 3
                tmppcmap = pcmap   
                plot_back_black     = True;  # autorise un plot identique mais foncé au fond
                back_black_color    = [0.5, 0.5, 0.5, 1];
                back_black_diffsize = 0.5;
            elif SIZE_REDUCTION == 'sel' :
                nFigArt = 6;
                lgtitle = 'ZRg-clst'
                lgtitle = 'ZRegion-\n cluster'
                lglabel_fontsize = 18
                errorcaps  = True
                capslength = 3
                tmppcmap = pcmap
                plot_back_black     = True;  # autorise un plot identique mais foncé au fond
                back_black_color    = [0.5, 0.5, 0.5, 1];
                back_black_diffsize = 0.5;
            else:
                nFigArt = 99;
                lgtitle = 'Region-\n cluster'
                lglabel_fontsize = 16
                errorcaps  = True
                capslength = 3
                tmppcmap = pcmap
                plot_back_black     = False;  # autorise un plot identique mais foncé au fond
                back_black_color    = [0.5, 0.5, 0.5, 1];
                back_black_diffsize = 0.5;
            FigArtId = 'b';
            figfile = "FigArt{:02d}{:s}_".format(nFigArt,FigArtId);
            dpi     = FIGARTDPI
            figpdf  = True
            notitle = False
            #top = 0.98
        else :
            figfile = "Fig_"
            dpi     = FIGDPI
            figpdf  = False
            notitle = False
            lgtitle = 'Class'
            lglabel_fontsize=16
            errorcaps  = False
            capslength = 3
        #
        plot_mean_profil_by_class(sst_obs_coded,nb_class,classe_Dobs,varnames=varnames,
                                  title=stitre,
                                  fileext=fileextbis, figdir=case_figs_dir,
                                  figfile=figfile, dpi=dpi, figpdf=figpdf,
                                  getstd=plotctprofilsstd,
                                  pcmap=tmppcmap,
                                  figsize=(14,7),
                                  top=top, bottom=bottom, left=left, right=right,
                                  linewidth=2.5,
                                  ticks_fontsize=14,labels_fontsize=16,title_fontsize=18,
                                  ylabel_fontsize=18,
                                  lgtitle=lgtitle,
                                  lgticks_fontsize=16,lglabel_fontsize=lglabel_fontsize,
                                  title_y=1.015,
                                  notitle=notitle,
                                  errorcaps=errorcaps,
                                  capslength=capslength,
                                  plot_back_black=plot_back_black,
                                  back_black_color=back_black_color,
                                  back_black_diffsize=back_black_diffsize,
                                  )
    #
    #%% -----------------------------------------------------------------------
    if Visu_CTStuff : # Visu des profils des référents de la carte SOM
        stitre="SOM Map Profils by Cell ({:s})\n(background color represents classes)".format(dataobsystartend)
        fileextter = "_{:s}{:s}Clim-{:s}_{:s}".format(fprefixe,
                      fshortcode,dataobsystartend,data_label_base)
        plot_ct_profils(sMapO,Dobs,class_ref,varnames=varnames,
                        fileext=fileextter, figdir=case_figs_dir,
                        pcmap=pcmap*0.7+0.299,
                        title=stitre,
                        titlefntsize=12,
                        ytitle=0.995,
                        )
    #
    if STOP_BEFORE_MDLSTUFF :
        plt.show(); sys.exit(0)
    #
    #%% #######################################################################
    #
    #                           TRAITEMENT DES MODELES
    #
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

    fileextIV = "_{:s}{:s}_{:s}{:s}".format(fprefixe,SIZE_REDUCTION,fshortcode,method_cah)
    fileextIV79 = "_{:s}{:s}_{:s}".format(fprefixe,SIZE_REDUCTION,fshortcode)
    #
    Tperfglob, Tperfglob_Qm, TDmdl4CT, Tmdlname, Tmdlnamewnb, Tmdlonlynb, TTperf, \
        TTperf_Qm, Nmdlok, \
        NDmdl = ctloop_model_traitement(sst_obs_coded,Dobs,XC_Ogeo,sMapO,
                    lon,lat,ilat,ilon,
                    nb_class,class_ref,classe_Dobs,NDobs,fond_C,
                    isnanobs,isnumobs,Lobs,Cobs,list_of_plot_colors,
                    Sfiltre=Sfiltre,
                    data_period_ident=DATAMDL,
                    varnames=varnames, figdir=case_figs_dir,
                    commonfileext=fileextIV, commonfileext79=fileextIV79,
                    eqcmap=eqcmap, ccmap=cpcmap,pcmap=pcmap,
                    bgmap=grcmap, bgval=bggray,
                    obs_data_path=obs_data_path,
                    OK101=OK101,
                    OK102=OK102,
                    OK104=OK104,
                    OK105Art=OK105,
                    OK106=OK106,
                    OK107=OK107,
                    OK108=OK108,
                    OK109=OK109,
                    )
    indexlastcummod = np.where(Tperfglob_Qm == max(Tperfglob_Qm))[0]
    if len(indexlastcummod) > 1 : # si plussieurs maximuns on prend le dernier
        indexlastcummod = indexlastcummod[-1]
    else :
        indexlastcummod = indexlastcummod[0]
    nbestcummod = indexlastcummod + 1
    print("\n-- {:d} cum. models for best cumulated performances --> {:.1f}%\n   {}".format(
            nbestcummod,100*Tperfglob_Qm[indexlastcummod],Tmdlnamewnb[np.arange(nbestcummod)]))
    #%% 
    # -------------------------------------------------------------------------
    ctloop.printwarning([ "    MODELS: APRES SECOND LOOP" ])
    # Figure a ploter apres le deuxieme loop
    if Visu_preACFperf : # Tableau des performances en figure de courbes
        stitre = "Performance Curves"
        stitre += (" - SST {:s} ({:s}) - {:d} Classes -  Classification Indices"+\
                   " of Completed Models (vs Obs) ({:d} models)").format(fcodage,
                                         dataobsystartend,nb_class,Nmdlok)
        if mdlnamewnumber_ok :
            TmdlnameX = Tmdlnamewnb
            figsize=(12,6)
            top=0.93; bottom=0.24; left=0.06; right=0.98

        else :
            TmdlnameX = Tmdlname
            figsize=(12,6)
            top=0.93; bottom=0.22; left=0.06; right=0.98
        #
        xticklab_rot=-90
        xticklab_ha='center'    # [ ‘center’ | ‘right’ | ‘left’ ]
        xticklab_va='top' # [ ‘center’ | ‘top’ | ‘bottom’ | ‘baseline’ ]
        #
        fignum = ctloop.do_models_after_second_loop(Tperfglob,Tperfglob_Qm,TmdlnameX,
                                           list_of_plot_colors,
                                           title=stitre,
                                           TypePerf=TypePerf,fcodage=fcodage,
                                           figsize=figsize,
                                           top=top, bottom=bottom, left=left, right=right,
                                           xticklabels_rot=xticklab_rot,
                                           xticklabels_ha=xticklab_ha,
                                           xticklabels_va=xticklab_va,
                                           )
        #
        if SAVEFIG :
            plt.figure(fignum)
            figfile = "Fig{:s}_perf-curves-by_model_{:s}_{:d}Mdl_{:d}-mod".format(
                    fileextIV,dataobsystartend,nb_class,Nmdlok)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    
    if STOP_BEFORE_AFC :
        plt.show(); sys.exit(0)
    
    #%% #######################################################################
    #
    #     ANALYSE FACTORIELLE DES CORRESPONDANCES (CORRESPONDENCE ANALYSIS)
    #
    # -------------------------------------------------------------------------
    # Figure 3 pour Article : Performances selon les clusters de l'AFC
    # Figure du dendrogramme 
    # Figure de l'inercie
    # -------------------------------------------------------------------------
    savedata = True
    if Visu_UpwellArt :
        if SIZE_REDUCTION == 'All' :
            nFigArt = 5;
        elif SIZE_REDUCTION == 'sel' :
            nFigArt = 7;
        else:
            nFigArt = 99;
        figfile = "FigArt{:02d}_".format(nFigArt)
        dpi     = FIGARTDPI
        figpdf  = True
        notitle = True
        #top = 0.98
    else :
        figfile = "Fig_"
        dpi     = FIGDPI
        figpdf  = False
        notitle = False
    #
    if SIZE_REDUCTION == 'All' :
        clustfigsublc=(2,3)
        clustfigsize=(13,6)
    elif SIZE_REDUCTION == 'sel' :
        clustfigsublc=(2,4)
        clustfigsize=(15,7)
    else:
        clustfigsublc=(2,3)
        clustfigsize=(10,5)
    #
    plotobs = True
    plotmodall = True
    plotbestmodXZinRZ = False
    #
    VAPT, F1U, F1sU, F2V, CRi, CAj, TTperf4afc,\
        CAHindnames, CAHindnameswnb, NoCAHindnames,\
        class_afc,AFCindnames,AFCindnameswnb,NoAFCindnames,\
        figclustmoynum, \
        allclusrPerfG, allclustTperf = ctloop_compute_afc(sMapO, lon, lat, TDmdl4CT,
                           Tmdlname, Tmdlnamewnb, Tmdlonlynb,
                           nb_class, nb_clust, isnumobs, isnanobs, class_ref, classe_Dobs,
                           fond_C, XC_Ogeo,
                           TTperf, Nmdlok, Lobs, Cobs, NDmdl, Nobsc, data_label_base,
                           AFC_Visu_Classif_Mdl_Clust=AFC_Visu_Classif_Mdl_Clust,
                           AFC_Visu_Clust_Mdl_Moy_4CT=AFC_Visu_Clust_Mdl_Moy_4CT,
                           sztitle=6,
                           ccmap=cpcmap,
                           bgmap=grcmap, bgval=bggray,
                           figdir=case_figs_dir,
                           figfile=figfile, dpi=dpi, figpdf=figpdf,
                           clustfigsublc=clustfigsublc, clustfigsize=clustfigsize,
                           notitle=notitle,
                           )
    #
    Nmdlafc = Tmdlname.shape[0]
    #
    if savedata:
        import scipy.io as sio
        dataystartend = datemdl2dateinreval(DATAMDL)
        datafile = "Data_AFC2DProj-{:d}-{:d}_{:d}Clust-{:d}Classes_{:s}".format(
                pa,po,nb_clust,nb_class,fprefixe)
        if AFCWITHOBS :
            datafile += "ObsInAFC_"
        else:
            datafile += "ObsOutAFC_"
        datafile += "{:s}Clim-{:s}_{:d}-mod".format(fshortcode,dataystartend,Nmdlafc)
        print('-- savinf AFC data in file {}.mat...'.format(datafile))
        if F1sU is not None :
            dic_to_save = {'perf':TTperf4afc, 'VAPT':VAPT, 'F1U':F1U,
                           'F1sU':F1sU, 'F2V':F2V, 'CRi':CRi, 'CAj':CAj,
                           'mdlname':AFCindnames, 'mdlnb':NoAFCindnames};
        else:
            # no F1sU
            dic_to_save = {'perf':TTperf4afc, 'VAPT':VAPT, 'F1U':F1U, 
                           'F2V':F2V, 'CRi':CRi, 'CAj':CAj,
                           'mdlname':AFCindnames, 'mdlnb':NoAFCindnames};
        #
        sio.savemat(case_figs_dir+os.sep+datafile+'.mat',dic_to_save)
    #        
    if plotmodall : # Model-all --------
        Nmdlafc = Tmdlname.shape[0]
        dataystartend = datemdl2dateinreval(DATAMDL)

        if Visu_UpwellArt :
            #TMixtMdlLabel = '47multi-model'
            if SIZE_REDUCTION == 'All' :
                TMixtMdlLabel = 'Model-all'
            elif SIZE_REDUCTION == 'sel' :
                TMixtMdlLabel = 'ZModel-all'
            else:
                TMixtMdlLabel = 'Model-all'
            titlefnsz = 14
        else:
            TMixtMdlLabel = 'Mod Cum.'
            titlefnsz = 12
        #print("misttitlelabel AVANT> {}".format(misttitlelabel))
        #mistfilelabel = misttitlelabel.replace(' ','').replace(':','-').replace('(','_').replace(')','')
        #print("mistfilelabel APRES>  {}".format(mistfilelabel))
        #
        nbsubl,nbsubc=clustfigsublc; isubplot = nbsubl * nbsubc - 1
        plt.figure(figclustmoynum);
        ax = plt.subplot(nbsubl,nbsubc,isubplot);
        print("\n{:d}-model(s)' generalization: {} ".format(len(Tmdlname),Tmdlname))
        MdlMoy, IMixtMdl, MGPerfglob, XMgeo, \
            Tperf = ctloop.mixtgeneralisation (sMapO, Tmdlname, Tmdlname, TDmdl4CT, 
                                               class_ref, classe_Dobs, nb_class, Lobs, Cobs, isnumobs,
                                               lon=lon, lat=lat,
                                               TypePerf=TypePerf,
                                               label=TMixtMdlLabel, shorttitle=True,
                                               fignum=figclustmoynum,ax=ax,
                                               title_fontsize=titlefnsz, ytitre=1.00, nticks=nticks,
                                               tickfontsize=10,
                                               cbticklabelsize=12,cblabelsize=12,
                                               show_xylabels=False,
                                               visu=True,
                                               ccmap=cpcmap,
                                               bgmap=grcmap, bgval=bggray,
                                               );
    #
    if plotobs : # Obs --------
        nbsubl,nbsubc=clustfigsublc; isubplot = nbsubl * nbsubc
        bounds = np.arange(nb_class+1)+1;
        coches = np.arange(nb_class)+1;   # ex 6 classes : [1,2,3,4,5,6]
        ticks  = coches + 0.5;            # [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        if SIZE_REDUCTION == 'All' :
            nticks = 5; # 4
        elif SIZE_REDUCTION == 'sel' :
            nticks = 2; # 4
        plt.figure(figclustmoynum);
        ax = plt.subplot(nbsubl,nbsubc,isubplot);
        ax.imshow(fond_C, interpolation=None,cmap=grcmap,vmin=0,vmax=1) # fond gris pour NaN
        ims = ax.imshow(XC_Ogeo, interpolation=None,cmap=cpcmap,vmin=1,vmax=nb_class);
        if Visu_UpwellArt :
            #if SIZE_REDUCTION == 'All' :
            #    cluster_tlabel = 'region-clusters';
            #elif SIZE_REDUCTION == 'sel' :
            #    cluster_tlabel = 'ZRegion-clusters';
            #else:
            #    cluster_tlabel = 'region-clusters';
            #ax.set_title("Obs, %d %s"%(nb_class,cluster_tlabel),fontsize=14);
            ax.set_title("Observations",fontsize=14);
        else:
            ax.set_title("Obs, %d classes "%(nb_class),fontsize=12);
        if 0 :
            plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
            plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
        else :
            ctloop.set_lonlat_ticks(lon,lat,step=nticks,fontsize=10,verbose=False,lengthen=True)
        if True :
            #cbar_ax,kw = cb.make_axes(ax,orientation="vertical",fraction=0.04,pad=0.03,aspect=20)
            #fig.colorbar(ims, cax=cbar_ax, ticks=ticks,boundaries=bounds,values=bounds, **kw);
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="4%", pad="3%")
            hcb = plt.colorbar(ims,cax=cax,ax=ax,ticks=ticks,boundaries=bounds,values=bounds);
        else :
            hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(coches);
        hcb.ax.tick_params(labelsize=12)
        if Visu_UpwellArt :
            if SIZE_REDUCTION == 'All' :
                cluster_label = 'Region cluster';
            elif SIZE_REDUCTION == 'sel' :
                cluster_label = 'ZRegion cluster';
            else:
                cluster_label = 'Region cluster';

            hcb.set_label(cluster_label,size=12)
        else:
            hcb.set_label('Class',size=10)
        #grid(); # for easier check

    #print("--CAHindnames: {}".format(CAHindnames))
    #print("--NoCAHindnames: {}".format(NoCAHindnames))
    #print("--Tmdlname: {}".format(Tmdlname))
    print("\n--Tmdlnamewnb: {}".format(Tmdlnamewnb))
    # reprend la figure de performanes par cluster
    plt.figure(figclustmoynum)
    if not notitle:
        plt.suptitle("AFC Clusters Class Performance ({}) ({} models)".format(dataystartend,Nmdlafc),fontsize=18);
    #
    if SAVEFIG : # sauvegarde de la figure
        plt.figure(figclustmoynum)
        if figfile is None :
            figfile = "Fig_"
        if dpi is None :
            dpi = FIGDPI
        figfile += "AFCClustersPerf-{:d}Clust-{:d}Classes_{:s}".format(nb_clust,
                    nb_class,fprefixe)
        if AFCWITHOBS :
            figfile += "ObsInAFC_"
        else:
            figfile += "ObsOutAFC_"
        figfile += "{:s}Clim-{:s}_{:d}-mod".format(fshortcode,dataystartend,Nmdlafc)
        #
        ctloop.do_save_figure(figfile,dpi=dpi,path=case_figs_dir,ext=FIGEXT,figpdf=figpdf)
    #
    #%%
    # ONY FOR REDUCED ZONE:
    #   Adds the results for best model in eXtended Zone, for comparison
    if plotbestmodXZinRZ and SIZE_REDUCTION == 'sel' : # plot in Petite-zone the Best model BigZone --------
        XZBestModName =  'CMCC-CM'
        iBMXZ = np.where(XZBestModName == Tmdlname)[0]
        BMXZTmdlname = Tmdlname[iBMXZ]
        BMXZTDmdl4CT = TDmdl4CT[iBMXZ,:]

        if Visu_UpwellArt :
            #TMixtMdlLabel = '47multi-model'
            TMixtMdlLabel = 'Best-mod.ext.reg.'
            titlefnsz = 14
        else:
            TMixtMdlLabel = 'Mod Cum.'
            titlefnsz = 12
        #print("misttitlelabel AVANT> {}".format(misttitlelabel))
        #mistfilelabel = misttitlelabel.replace(' ','').replace(':','-').replace('(','_').replace(')','')
        #print("mistfilelabel APRES>  {}".format(mistfilelabel))
        #
        nbsubl,nbsubc=clustfigsublc; isubplot = nbsubl * nbsubc - 2
        plt.figure(figclustmoynum);
        ax = plt.subplot(nbsubl,nbsubc,isubplot);
        print("\n{:d}-model(s)' generalization: {} ".format(len(BMXZTmdlname),BMXZTmdlname))
        MdlMoy, IMixtMdl, MGPerfglob, XMgeo, \
            Tperf = ctloop.mixtgeneralisation (sMapO, BMXZTmdlname, BMXZTmdlname, BMXZTDmdl4CT, 
                                               class_ref, classe_Dobs, nb_class, Lobs, Cobs, isnumobs,
                                               lon=lon, lat=lat,
                                               TypePerf=TypePerf,
                                               label=TMixtMdlLabel, shorttitle=True,
                                               fignum=figclustmoynum,ax=ax,
                                               title_fontsize=titlefnsz, ytitre=1.00, nticks=nticks,
                                               tickfontsize=10,
                                               cbticklabelsize=12,cblabelsize=12,
                                               show_xylabels=False,
                                               visu=True,
                                               ccmap=cpcmap,
                                               bgmap=grcmap, bgval=bggray,
                                               );
        #
        if SAVEFIG : # sauvegarde de la figure
            plt.figure(figclustmoynum)
            #if figfile is None :
            #    figfile = "Fig_"
            #if dpi is None :
            #    dpi = FIGDPI
            #figfile += "AFCClustersPerf-{:d}Clust-{:d}Classes_{:s}{:s}Clim-{:s}_{:d}-mod_andBMXZ".format(nb_clust,
            #            nb_class,fprefixe,fshortcode,dataystartend,Nmdlafc)
            #
            figfile += "_andBMXZ"
            ctloop.do_save_figure(figfile,dpi=dpi,path=case_figs_dir,ext=FIGEXT,figpdf=figpdf)
    #
    #%%
    tmp_max_cluster_afc = class_afc.max(); # nombre max de clusters AFC
    tmp_n_afc_mod = class_afc.shape[0]; # nombre de modeles AFC
    tmp_list_of_afc_models = AFCindnameswnb; # liste avec numero de modele (le N+1 est 'Obs')
    #tmp_list_of_afc_models = AFCindnames; # liste sans numero de modele
    #tmp_list_of_afc_models = NoAFCindnames; # numero de modele seulement (le N+1 est 'Obs')
    
    print('\n {:s}\n -- {:s} --\n -- {:s} --\n {:s}'.format(''.center(56,'-'),
          'Model List by AFC Cluster'.center(50),
          case_name_base.center(50),
          ''.center(56,'-')))
    for iClust in np.arange(tmp_max_cluster_afc) :
        print('\nAFC Cluster {:d}:\n#############\n'.format(iClust+1))
        for iMod in np.arange(tmp_n_afc_mod) :
            if class_afc[iMod] == (iClust + 1) :
                print('{:s}'.format(tmp_list_of_afc_models[iMod]))
    #
    #
    #%% -------------------------------------------------------------------------
    #                        PLOT DE PROJECTION AFC
    # -------------------------------------------------------------------------
    # Figure 4 pour Article : Projection de l'AFC
    # -------------------------------------------------------------------------
    if Visu_UpwellArt :
        nFigArt = 4;
        figfile = "FigArt{:02d}_".format(nFigArt)
        dpi     = FIGARTDPI
        figpdf  = True
    else :
        figfile = "Fig_"
        dpi     = FIGDPI
        figpdf  = False
    #
    res_for_afc_clust = True
    
    # TEST DE PROJ DES OBJ (100% sur toutes les classes)
    Pobs_ = np.ones((1,nb_class),dtype=int)*100; # perfs des OBS = 100% dans toutes les classes
    F1obs = ldef.do_afc_proj(TTperf4afc,Pobs_)
    # Perf du cumul des modeles
    PMAllCum = TTperf_Qm[-1,:].reshape((1,TTperf_Qm.shape[1]))
    mdlmarker='o'; clsmarker='s'
    F1extra = ldef.do_afc_proj(TTperf4afc,PMAllCum)
    #Lblextra = "FMM{}".format(Nmdlok); xextramarker='*'; Colorextra = [ 1.0, 0.0, 0.0, 1.]; # orange
    Lblextra = "Model-all"; xextramarker='*'; Colorextra = [ 1.0, 0.0, 0.0, 1.]; # orange
    Lblobs="Obs"; obsmarker='D';  Colorobs = [0.80, 1.0, 0.0, 1.] # vert 
    #if AFCWITHOBS :
    #    # si l'AFC est faite avec les OBS aussi
    #    plot_afc_proj(F1U,F2V,CRi,CAj,F1sU,pa,po,class_afc,nb_class,NIJ,Nmdlok,
    #                  indnames=NoAFCindnames,
    #                  figdir=case_figs_dir,
    #                  figfile=figfile, dpi=dpi, figpdf=figpdf,
    #                  xextraF1=F1extra,
    #                  mdlmarker=mdlmarker,clsmarker=clsmarker,
    #                  obsmarker=obsmarker,obsLbl=Lblobs,obscolor=Colorobs,
    #                  xextramarker=xextramarker,xextraLbl=Lblextra,xextracolor=Colorextra,
    #                  )
    #else :
    #    # si les OBS ne participent pas a l'AFC mais seulement projetés
    #    plot_afc_proj(F1U,F2V,CRi,CAj,F1sU,pa,po,class_afc,nb_class,NIJ,Nmdlok,
    #                  indnames=NoAFCindnames,
    #                  figdir=case_figs_dir,
    #                  figfile=figfile, dpi=dpi, figpdf=figpdf,
    #                  xextraF1=F1extra,
    #                  mdlmarker=mdlmarker,clsmarker=clsmarker,
    #                  obsmarker=obsmarker,obsLbl=Lblobs,obscolor=Colorobs,
    #                  xextramarker=xextramarker,xextraLbl=Lblextra,xextracolor=Colorextra,
    #                  )
    #
    if res_for_afc_clust and len(allclusrPerfG) > 0:
        # allclusrPerfG, allclustTperf
        FGrCumExtra = ldef.do_afc_proj(TTperf4afc,allclustTperf)
        xtracumbygrmarker = '*'
    else:
        FGrCumExtra = None
        xtracumbygrmarker = None
    #
    plot_afc_proj(F1U,F2V,CRi,CAj,F1sU,pa,po,class_afc,nb_class,NIJ,Nmdlok,
                  indnames=NoAFCindnames,
                  figdir=case_figs_dir,
                  figfile=figfile, dpi=dpi, figpdf=figpdf,
                  xextraF1=F1extra,
                  mdlmarker=mdlmarker,clsmarker=clsmarker,
                  obsmarker=obsmarker,obsLbl=Lblobs,obscolor=Colorobs,
                  xextramarker=xextramarker,xextraLbl=Lblextra,xextracolor=Colorextra,
                  xtracumbygrF1=FGrCumExtra,xtracumbygrmarker=xtracumbygrmarker)
    #
    if STOP_BEFORE_GENERAL :
        plt.show(); sys.exit(0)
    #
    #%%
    # 
    # Showing results in Reduced Refion (SIZE_REDUCTION == 'sel') of Best model
    # and Best AFC cluster from  Extended Region experiennces
    # 
    if SIZE_REDUCTION == 'sel' : # plot in Petite-zone the Best model BigZone --------
        if Visu_UpwellArt :
            nFigArt = 8;
            figfile = "FigArt{:02d}_".format(nFigArt)
            dpi     = FIGARTDPI
            figpdf  = True
            notitle = True
            #top = 0.98
            fig = plt.figure(figsize=(8.5, 4));
        else :
            figfile = "Fig_"
            dpi     = FIGDPI
            figpdf  = False
            notitle = False
            fig = plt.figure(figsize=(16, 8));
            
        fignum = fig.number # numero de figure en cours ...
        wspace=0.2; hspace=0.01; top=0.90; bottom=0.1; left=0.04; right=0.92;
        fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top,
                            bottom=bottom, left=left, right=right)

        # -----------------------------------------------------------------
        ax = plt.subplot(121)
        #
        XZBestModName =  'CMCC-CM'
        iBMXZ = np.where(XZBestModName == Tmdlname)[0][0]
        BMXZTmdlname = Tmdlname[iBMXZ]
        BMXZTDmdl4CT = TDmdl4CT[iBMXZ,:]
        XZBestModNameWNbr = TmdlnameX[iBMXZ]   # Name with number
        #
        TMixtMdl = ['CMCC-CM']
        iBMXZ = [np.where(m == Tmdlname)[0][0] for m in TMixtMdl]
        BMXZTmdlname = Tmdlname[iBMXZ]
        BMXZTDmdl4CT = TDmdl4CT[iBMXZ,:]
        XZBestModNameWNbr = ', '.join(np.ndarray.tolist(TmdlnameX[iBMXZ]))   # Name with number
        #
        if Visu_UpwellArt :
            #TMixtMdlLabel = '47multi-model'
            TMixtMdlLabel = XZBestModNameWNbr
            titlefnsz = 14
        else:
            TMixtMdlLabel = 'Mod Cum.'
            titlefnsz = 12
        #ax = plt.subplot(nbsubl,nbsubc,isubplot);
        print("\n{:d}-model(s)' generalization: {} ".format(len(BMXZTmdlname),BMXZTmdlname))
        MdlMoy, IMixtMdl, MGPerfglob, XMgeo, \
            Tperf = ctloop.mixtgeneralisation (sMapO, BMXZTmdlname, BMXZTmdlname, BMXZTDmdl4CT, 
                                               class_ref, classe_Dobs, nb_class, Lobs, Cobs, isnumobs,
                                               lon=lon, lat=lat,
                                               TypePerf=TypePerf,
                                               label=TMixtMdlLabel, shorttitle=True,
                                               fignum=fignum,ax=ax,
                                               title_fontsize=titlefnsz, ytitre=1.00, nticks=nticks,
                                               tickfontsize=10,
                                               cbticklabelsize=12,cblabelsize=12,
                                               show_xylabels=False,
                                               visu=True,
                                               ccmap=cpcmap,
                                               bgmap=grcmap, bgval=bggray,
                                               );
        # -----------------------------------------------------------------
        ax = plt.subplot(122)
        #
        # Grande Zone (All): BEST AFC CLUSTER: -> Cluster 4, 13 Models, performance: 69.3 :
        TMixtMdlLabel = 'Model-group 4'
        TMixtMdl = ['CMCC-CM', 'HadGEM2-ES', 'HadGEM2-AO', 'HadGEM2-CC', 'CMCC-CMS',
                    'CNRM-CM5-2', 'CanESM2', 'CanCM4', 'GFDL-CM3', 'CNRM-CM5', 'FGOALS-s2', 
                    'CSIRO-Mk3-6-0', 'CMCC-CESM']

        iBMXZ = [np.where(m == Tmdlname)[0][0] for m in TMixtMdl]
        BMXZTmdlname = Tmdlname[iBMXZ]
        BMXZTDmdl4CT = TDmdl4CT[iBMXZ,:]

        if Visu_UpwellArt :
            #TMixtMdlLabel = '47multi-model'
            TMixtMdlLabel = TMixtMdlLabel
            titlefnsz = 14
        else:
            TMixtMdlLabel = 'Mod Cum.'
            titlefnsz = 12
        #ax = plt.subplot(nbsubl,nbsubc,isubplot);
        print("\n{:d}-model(s)' generalization: {} ".format(len(BMXZTmdlname),BMXZTmdlname))
        MdlMoy, IMixtMdl, MGPerfglob, XMgeo, \
            Tperf = ctloop.mixtgeneralisation (sMapO, BMXZTmdlname, BMXZTmdlname, BMXZTDmdl4CT, 
                                               class_ref, classe_Dobs, nb_class, Lobs, Cobs, isnumobs,
                                               lon=lon, lat=lat,
                                               TypePerf=TypePerf,
                                               label=TMixtMdlLabel, shorttitle=True,
                                               fignum=fignum,ax=ax,
                                               title_fontsize=titlefnsz, ytitre=1.00, nticks=nticks,
                                               tickfontsize=10,
                                               cbticklabelsize=12,cblabelsize=12,
                                               show_xylabels=False,
                                               visu=True,
                                               ccmap=cpcmap,
                                               bgmap=grcmap, bgval=bggray,
                                               );
        #
        if SAVEFIG : # sauvegarde de la figure
            plt.figure(fignum)
            if figfile is None :
                figfile = "Fig_"
            if dpi is None :
                dpi = FIGDPI
            figfile += "TEST-Best-ModAndAFCClust-for-XR-evaluated-in-RR-{:d}Classes_{:s}".format(nb_class,
                                 fprefixe)
            if AFCWITHOBS :
                figfile += "ObsInAFC_"
            else:
                figfile += "ObsOutAFC_"
            figfile += "{:s}Clim-{:s}".format(fshortcode,datemdl2dateinreval(DATAMDL))
            #
            ctloop.do_save_figure(figfile,dpi=dpi,path=case_figs_dir,ext=FIGEXT,figpdf=figpdf)
    #
    #%%
    #___________
    print(("\n{} {},\n{} {},\n{} {},\n{} {},\n{} {}\n").format(
                   "SIZE_REDUCTION ".ljust(18,'.'),SIZE_REDUCTION,
                   "WITHANO ".ljust(18,'.'),WITHANO,
                   "UISST ".ljust(18,'.'),UISST,
                   "climato ".ljust(18,'.'),climato,
                   "NIJ ".ljust(18,'.'),NIJ))
    #
    ctloop.printwarning([ "    END: WHOLE TIME CODE '{:s}' IN {:.2f} SECONDS".format(
            os.path.basename(sys.argv[0]),time()-tpgm0) ])
    #
    #======================================================================
    #
    if 0: 
        ModelList=np.copy(Tmdlnamewnb)
        for ii,mod in enumerate(Tmdlnamewnb) :
            spd=mod.split('-')
            ij=int(spd[0])-1
            ModelList[ij] = Tmdlname[ii]
            print("Model='{}' --> splited: {}".format(mod,spd))
        for mod in ModelList :
            print(mod)
    #%%
    #**************************************************************************
    #.............................. GENERALISATION ............................
    if generalisation_ok :
        #
        #list_of_eqcmap = [eqcmap, cm.jet]
        list_of_eqcmap = eqcmap
        #list_of_eqcmap = cm.jet
        #
        if generalisa_bestafc_clust_ok :
            generalisation_proc(sMapO, lon, lat, varnames, wvmin, wvmax, nb_class,
                        isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, 
                        TDmdl4CT,Tmdlname,
                        data_period_ident=DATAMDL,
                        eqcmap=list_of_eqcmap,
                        bgmap=grcmap, bgval=bggray,
                        figdir=case_figs_dir,
                        generalisation_type='bestclust',  # 'bestclust', 'bestcum'
                        )
        #
        if generalisa_bestcum_ok :
            generalisation_proc(sMapO, lon, lat, varnames, wvmin, wvmax, nb_class,
                        isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, 
                        TDmdl4CT,Tmdlname,
                        data_period_ident=DATAMDL,
                        eqcmap=list_of_eqcmap,
                        bgmap=grcmap, bgval=bggray,
                        figdir=case_figs_dir,
                        generalisation_type='bestcum',  # 'bestclust', 'bestcum'
                        )
        #
        if generalisa_allcum_ok :
            generalisation_proc(sMapO, lon, lat, varnames, wvmin, wvmax, nb_class,
                        isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, 
                        TDmdl4CT,Tmdlname,
                        data_period_ident=DATAMDL,
                        eqcmap=list_of_eqcmap,
                        bgmap=grcmap, bgval=bggray,
                        figdir=case_figs_dir,
                        generalisation_type='all',  # 'bestclust', 'bestcum', 'all'
                        )
        # all AFC Clusters
        if generalisa_allafc_clust_ok :
            generalisafcclust_proc(sst_obs_coded,Dobs,XC_Ogeo,sMapO,lon,lat,ilat,ilon, varnames, wvmin, wvmax, nb_class,
                            isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, 
                            TDmdl4CT, Tmdlname,
                            data_period_ident=DATAMDL,
                            eqcmap=list_of_eqcmap,
                            bgmap=grcmap, bgval=bggray,
                            figdir=case_figs_dir,
                            )
        #
        if generalisa_other_periods_ok :
            #data_period_ident = "raverage_1975_2005";   #<><><><><><><>
            #data_period_ident = "raverage_1930_1960";   #<><><><><><><>
            #data_period_ident = "raverage_1944_1974";   #<><><><><><><>
            #data_period_ident = "rcp_2006_2017";        #<><><><><><><>
            #data_period_ident = "rcp_2070_2100";        #<><><><><><><>
            
            #list_of_periods = [ "raverage_1930_1960", "raverage_1944_1974", "rcp_2006_2017", "rcp_2070_2100" ]
            list_of_periods = [ "raverage_1930_1960", "raverage_1944_1974", "rcp_2070_2100" ]
            #list_of_periods = [ "rcp_2070_2100" ]
            list_of_scenarios = [ "rcp26", "rcp45", "rcp85" ]
            
            for iper,data_period_ident in enumerate(list_of_periods) :
                if data_period_ident.lower().startswith('rcp_') :
                    all_scenarions = list_of_scenarios
                else :
                    all_scenarions = [ None ];
                for ipscenar,scenario_name in enumerate(all_scenarions) :
                    if scenario_name is None :
                        ctloop.printwarning([ "    MODEL: INITIALIZATION AND FIRST LOOP - NEW SET '{}' ".format(data_period_ident)])
                    else :
                        ctloop.printwarning([ "    MODEL: INITIALIZATION AND FIRST LOOP - NEW SET '{}', SCENARIO '{}'".format(data_period_ident,scenario_name)])
                    #
                    Sfiltre=None
                    #
                    TDmdl4CTx,Tmdlnamex,Tmdlnamewnbx,Tmdlonlynbx,Tperfglob4Sortx,Tclasse_DMdlx,\
                        Tmoymensclassx,NDmdlx,Nmdlokx,Smoy_101x,Tsst_102x = ctloop.do_models_startnloop(sMapO,
                                            Tmodels,Tinstit,ilat,ilon,
                                            isnanobs,isnumobs,nb_class,class_ref,classe_Dobs,
                                            Tnmodel=Tnmodel,
                                            Sfiltre=Sfiltre,
                                            TypePerf=TypePerf,
                                            obs_data_path=obs_data_path,
                                            data_period_ident=data_period_ident,
                                            scenario=scenario_name,
                                            MDLCOMPLETION=MDLCOMPLETION,
                                            SIZE_REDUCTION=SIZE_REDUCTION,
                                            NIJ=NIJ,
                                            )
                    generalisation_proc(sMapO, lon, lat, varnames, wvmin, wvmax, nb_class,
                                        isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, 
                                        TDmdl4CTx,Tmdlnamex,
                                        data_period_ident=data_period_ident,
                                        eqcmap=list_of_eqcmap,
                                        bgmap=grcmap, bgval=bggray,
                                        figdir=case_figs_dir,
                                        generalisation_type='bestclust',  # 'bestclust', 'bestcum'
                                        scenario=scenario_name,
                                        )

    #%%
    return
#%%
if __name__ == "__main__":
    main(sys.argv[1:])


if 0 :
    s = '--conf=all --all t -h -a -c a1 a2'
    s = '--conf=all -v'
    args = s.split()
    args
    optlist, args = getopt.getopt(args, 'hvc:', [
        'conf=', 'all', 'sel'])
    optlist, args


if 0:
    for iclust in np.arange(nb_clust) :
        print(" --> Cluster {:d}:\n  {}\n".format(iclust+1,
              AFCindnames[np.where(class_afc==iclust+1)[0]]))

    '''
    -> Cluster 1, 19 Models, performance: 25.0 :
     ['GISS-E2-R-CC' 'HadCM3' 'MIROC5' 'GISS-E2-R' 'MPI-ESM-MR'
     'CESM1-FASTCHEM' 'CCSM4' 'MPI-ESM-LR' 'MPI-ESM-P' 'inmcm4' 'IPSL-CM5B-LR'
     'ACCESS1-0' 'GISS-E2-H-CC' 'GISS-E2-H' 'CESM1-BGC' 'CESM1-CAM5'
     'CESM1-CAM5-1-FV2' 'CESM1-WACCM' 'bcc-csm1-1-m']
    
    -> Cluster 2, 5 Models, performance: 66.0 :
     ['CNRM-CM5' 'CMCC-CMS' 'CNRM-CM5-2' 'GFDL-CM3' 'FGOALS-s2']
    
    -> Cluster 3, 9 Models, performance: 47.5 :
     ['CanCM4' 'CanESM2' 'NorESM1-ME' 'NorESM1-M' 'CMCC-CM' 'FGOALS-g2'
     'MRI-CGCM3' 'IPSL-CM5A-MR' 'IPSL-CM5A-LR']
    
    -> Cluster 4, 9 Models, performance: 38.7 :
     ['ACCESS1-3' 'HadGEM2-ES' 'CSIRO-Mk3-6-0' 'HadGEM2-AO' 'MIROC-ESM'
     'MRI-ESM1' 'HadGEM2-CC' 'MIROC-ESM-CHEM' 'bcc-csm1-1']
    
    -> Cluster 5, 4 Models, performance: 31.2 :
     ['CMCC-CESM' 'GFDL-ESM2M' 'GFDL-ESM2G' 'GFDL-CM2p1']
     '''
    kclust = 1
    SfiltredMod = AFCindnames[np.where(class_afc==kclust)[0]]
    fileextIV = "_Clust{:d}-{:s}{:s}_{:s}{:s}".format(kclust,fprefixe,
                       SIZE_REDUCTION,fshortcode,method_cah)
    fileextIV79 = "_Clust{:d}-{:s}{:s}_{:s}".format(kclust,fprefixe,
                         SIZE_REDUCTION,fshortcode)
    xTDmdl4CT, xTmdlname, xTmdlnamewnb, xTmdlonlynb, xTTperf, xTTperf_Qm, xNmdlok,\
        xNDmdl = ctloop_model_traitement(
                    sst_obs_coded,Dobs,XC_Ogeo,sMapO,lon,lat,ilat,ilon,
                    nb_class,class_ref,classe_Dobs,NDobs,fond_C,
                    isnanobs,isnumobs,Lobs,Cobs,list_of_plot_colors,
                    Sfiltre=SfiltredMod,
                    varnames=varnames, figdir=case_figs_dir,
                    commonfileext=fileextIV, commonfileext79=fileextIV79,
                    eqcmap=eqcmap, ccmap=ccmap,pcmap=pcmap,
                    bgmap=grcmap, bgval=bggray,
                    obs_data_path=obs_data_path,
                    OK101=False,
                    OK102=False,
                    OK104=OK104,
                    OK105=OK105,
                    OK106=False,
                    OK107=False,
                    OK108=OK108,
                    OK109=False,
                    )
