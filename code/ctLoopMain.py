#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:38:49 2018

Version sur 'Master'

Exemple d'appel depuis python:
      runfile('/Users/carlos/Labo/NN_divers/DeepLearning/Projets/Projet_Upwelling/Charles/PourCarlos2_pour_Article/code/ctLoopMain.py',
              wdir='/Users/carlos/Labo/NN_divers/DeepLearning/Projets/Projet_Upwelling/Charles/PourCarlos2_pour_Article/code',
              args="--case=sel -v")         

@author: carlos
"""
    
import numpy as np
import sys, getopt, os
from   time  import time
import matplotlib.pyplot as plt
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
        #    Longitude : 45W à 9W (-44.5 à -9.5)
        #    Latitude :  30N à 5N ( 29.5 à  5.5)
        frlat =  29.5;  tolat =  4.5; #(excluded)
        #frlon = -44.5;  tolon = -8.5; #(excluded)   #(§:25x35)
        frlon = -44.5;  tolon = -9.5; #(excluded)   #(§:25x35)
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
        #    LON:  28W à 16W (-27.5 to -16.5)
        #    LAT : 23N à 10N ( 22.5 to  10.5)
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
    pcmap = ctloop.build_pcmap(nb_class,ccmap)
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
    return  pcmap,AFC_Visu_Classif_Mdl_Clust, AFC_Visu_Clust_Mdl_Moy_4CT,\
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
        eqcmap = cm.jet
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
    ctloop.plot_obs(sst_obs,Dobs,lon,lat,varnames=varnames,cmap=eqcmap,
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
        localcmap = eqcmap
        stitre = "Observed SST {:s} MEAN ({:s}) - FREE LIMITS".format(fcodage,dataystartend)
        #
        ctloop.plot_obsbis(sst_obs,Dobs,varnames=varnames,cmap=localcmap,
                        title=stitre,
                        figsize=figsize,
                        wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                        )
        #
        if SAVEFIG : # sauvegarde de la figure
            figfile = "Fig_Obs4CT_FREELIMITS"
            figfile += fileext
            # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
            # eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figpath,ext=FIGEXT)
        #
        plt.show(block=blockshow)
    #
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
def plot_ct_Umatrix(sMapO) :
    global blockshow
    # #########################################################################
    # C.T. Visualisation ______________________________________________________
    # #########################################################################
    #==>> la U_matrix
    fig = plt.figure(figsize=(6,8));
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
def plot_ct_map_wei(sMapO) :
    #
    global blockshow
    #
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
def plot_geo_classes(lon,lat,XC_ogeo,nb_class,
                     nticks=2,
                     title="Obs Class Geographical Representation", fileext="", figdir=".",
                     figfile=None, dpi=None, figpdf=False,
                     ccmap=cm.jet,
                     figsize=(9,6),
                     top=0.94, bottom=0.08, left=0.06, right=0.925,
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
    ims = ax.imshow(XC_ogeo, interpolation=None,cmap=ccmap,vmin=1,vmax=nb_class,origin=origin);
    if 0:
        plt.xticks(np.arange(0,Cobs,nticks), lon[np.arange(0,Cobs,nticks)], rotation=45, fontsize=10)
        plt.yticks(np.arange(0,Lobs,nticks), lat[np.arange(0,Lobs,nticks)], fontsize=10)
    else :
        #plt.xticks(np.arange(-0.5,Cobs,lolast), np.round(lon[np.arange(0,Cobs,lolast)]).astype(int), fontsize=12)
        #plt.yticks(np.arange(0.5,Lobs,lolast), np.round(lat[np.arange(0,Lobs,lolast)]).astype(int), fontsize=12)
        ctloop.set_lonlat_ticks(lon,lat,step=nticks,fontsize=10,verbose=False,lengthen=True)
        #set_lonlat_ticks(lon,lat,fontsize=10,londecal=0,latdecal=0,roundlabelok=False,lengthen=False)
    #plt.axis('tight')
    plt.xlabel('Longitude', fontsize=12); plt.ylabel('Latitude', fontsize=12)
    plt.title(title,fontsize=16); 
    #grid(); # for easier check
    # Colorbar
    #cbar_ax,kw = cb.make_axes(ax,orientation="vertical",fraction=0.04,pad=0.03,aspect=20)
    #fig.colorbar(ims, cax=cbar_ax, ticks=ticks,boundaries=bounds,values=bounds, **kw);
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="4%", pad="3%")
    hcb = plt.colorbar(ims,cax=cax,ax=ax,ticks=ticks,boundaries=bounds,values=bounds);
    cax.set_yticklabels(coches);
    cax.tick_params(labelsize=12)
    cax.set_ylabel('Class',size=14)

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
                              pcmap=None,
                              figsize=(12,6),
                              wspace=0.0, hspace=0.0, top=0.96, bottom=0.08, left=0.06, right=0.92,
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
                                    pcmap=pcmap,
                                    figsize=figsize,
                                    wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                                    )
    if SAVEFIG : # sauvegarde de la figure
        if figfile is None :
            figfile = "Fig_"
        if dpi is None :
            dpi = FIGDPI
        figfile += "MeanByClass_{:d}Class{:s}".format(nb_class,fileext)
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
                    ) :
    #
    global SIZE_REDUCTION, SAVEFIG, FIGDPI, FIGEXT, blockshow
    #
    if SIZE_REDUCTION == 'All' :
        figsize = (7.5,12)
        wspace=0.01; hspace=0.05; top=0.945; bottom=0.04; left=0.15; right=0.86;
    elif SIZE_REDUCTION == 'sel' :
        figsize=(8,8)
        wspace=0.01; hspace=0.04; top=0.945; bottom=0.04; left=0.04; right=0.97;
    #
    stitre="SOM Map Profils by Cell ({:s}), (background color represents classes)",

    ctloop.do_plot_ct_profils(sMapO,Dobs,class_ref,varnames=varnames,
                           pcmap=pcmap,
                           figsize=figsize,
                           title=title,
                           titlefntsize=titlefntsize,
                           wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
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
                            obs_data_path=".",
                            OK101=False,
                            OK102=False,
                            OK104=False,
                            OK105=False,
                            OK106=False,
                            OK107=False,
                            OK108=False,
                            OK109=False,
                            ) :
    #
    global SIZE_REDUCTION, MDLCOMPLETION, NIJ, FONDTRANS 
    #global OK101, OK102, OK104, OK105, OK106, OK107, OK108, OK109
    global Tinstit, Tmodels, Tnmodel, TypePerf, mdlnamewnumber_ok
    #
    global fshortcode, fprefixe, fcodage, blockshow
    global SAVEFIG, FIGDPI, FIGEXT
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
        figsize=(18,12);
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
    Tperfglob,Tperfglob_Qm,Tmdlname,Tmdlnamewnb,Tmdlonlynb,TTperf,\
        TDmdl4CT = ctloop.do_models_second_loop(sst_obs,Dobs,lon,lat,sMapO,XC_Ogeo,TDmdl4CT,
                                Tmdlname,Tmdlnamewnb,Tmdlonlynb,
                                Tperfglob4Sort,Tclasse_DMdl,Tmoymensclass,
                                MaxPerfglob_Qm,IMaxPerfglob_Qm,
                                min_moymensclass,max_moymensclass,
                                MCUM,Lobs,Cobs,NDobs,NDmdl,
                                isnumobs,nb_class,class_ref,classe_Dobs,fond_C,
                                ccmap=ccmap,pcmap=pcmap,sztitle=sztitle,ysstitre=ysstitre,
                                ysuptitre=ysuptitre,suptitlefs=suptitlefs,
                                NIJ=NIJ,
                                pair_nsublc=pair_nsublc,
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
            figfile = "Fig-104{:s}{:d}MdlvsObstrans-{:s}_{:d}-mod".format(commonfileext,nb_class,dataystartend,Nmdlok)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figdir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    if OK105 : 
        if SAVEFIG :
            plt.figure(105);
            figfile = "Fig-105{:s}{:d}Mdl-{:s}_{:d}-mod".format(commonfileext,nb_class,dataystartend,Nmdlok)
            # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
            # et eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figdir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
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
    return Tperfglob, Tperfglob_Qm, TDmdl4CT, Tmdlname, Tmdlnamewnb, Tmdlonlynb, TTperf, Nmdlok, NDmdl
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
                       TTperf, Nmdlok, Lobs, Cobs, NDmdl, Nobsc, data_label_base,
                       AFC_Visu_Classif_Mdl_Clust=[],
                       AFC_Visu_Clust_Mdl_Moy_4CT=[],
                       ccmap=cm.jet, sztitle=10,
                       figdir=".",
                       figfile=None, dpi=None, figpdf=False,
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
    VAPT,F1U,F1sU,F2V,CRi,CAj,CAHindnames,CAHindnameswnb,NoCAHindnames,\
        figclustmoynum,class_afc,AFCindnames,AFCindnameswnb,\
        NoAFCindnames = ctloop.do_afc(NIJ,
                          sMapO, TDmdl4CT, lon, lat,
                          Tmdlname, Tmdlnamewnb, Tmdlonlynb, TTperf,
                          Nmdlok, Lobs, Cobs, NDmdl, Nobsc,
                          NBCOORDAFC4CAH, nb_clust,
                          isnumobs, isnanobs, nb_class, class_ref, classe_Dobs,
                          ccmap=ccmap, sztitle=sztitle, ysstitre=ysstitre,
                          AFC_Visu_Classif_Mdl_Clust=AFC_Visu_Classif_Mdl_Clust,
                          AFC_Visu_Clust_Mdl_Moy_4CT=AFC_Visu_Clust_Mdl_Moy_4CT,
                          TypePerf=TypePerf,
                          AFCWITHOBS = AFCWITHOBS, CAHWITHOBS=CAHWITHOBS,
                          SIZE_REDUCTION=SIZE_REDUCTION,
                          mdlnamewnumber_ok=mdlnamewnumber_ok,
                          onlymdlumberAFC_ok=onlymdlumberAFC_ok,
                          )
    #print("--CAHindnames: {}".format(CAHindnames))
    #print("--NoCAHindnames: {}".format(NoCAHindnames))
    #print("--Tmdlname: {}".format(Tmdlname))
    print("\n--Tmdlnamewnb: {}".format(Tmdlnamewnb))
    # reprend la figure de performanes par cluster
    plt.figure(figclustmoynum)
    plt.suptitle("AFC Clusters Class Performance ({}), ({} models)".format(dataystartend,Nmodels),fontsize=18);
    #
    if SAVEFIG : # sauvegarde de la figure
        plt.figure(figclustmoynum)
        if figfile is None :
            figfile = "Fig_"
        if dpi is None :
            dpi = FIGDPI
        figfile += "AFCClustersPerf-{:d}Clust-{:d}Classes_{:s}{:s}Clim-{:s}_{:d}-mod".format(nb_clust,
                    nb_class,fprefixe,fshortcode,dataystartend,Nmdlafc)
        #
        ctloop.do_save_figure(figfile,dpi=dpi,path=figdir,ext=FIGEXT,figpdf=figpdf)
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
        #
        ctloop.do_plot_afc_dendro(F1U,F1sU,nb_clust,Nmdlok,
                      afccoords=coord2take,
                      indnames=indnames,
                      AFCWITHOBS = AFCWITHOBS,CAHWITHOBS = CAHWITHOBS,
                      afc_method=method_afc, afc_metric=dist_afc,
                      truncate_mode=None,
                      xlabel="model",
                      ylabel="AFC inter cluster distance ({}/{})".format(method_afc,dist_afc),
                      title=stitre,
                      titlefnsize=18, ytitle=1.02, labelfnsize=12,
                      labelrotation=90, labelsize=labelsize,
                      axeshiftfactor=axeshiftfactor,
                      figsize=figsize,
                      wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right
                      )
        #
        if SAVEFIG : # sauvegarde de la figure
            figfile = "Fig_"
            dpi = FIGDPI
            figfile += "AFCDendro-{:s}-{:d}Clust_{:d}Class_{:s}{:s}Clim-{:s}_{:s}".format(prtlbl,
                    nb_clust,nb_class,fprefixe,fshortcode,dataystartend,data_label_base)
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
        #
        ctloop.do_plot_afc_inertie(VAPT,
                     title=stitre,
                     figsize=(8,6),
                     top=0.93, bottom=0.08, left=0.08, right=0.98,
        )
        if SAVEFIG : # sauvegarde de la figure de performanes par cluster
            figfile = "Fig_"
            figfile += "Inertia-{:d}Clust-{:d}Classes_{:s}{:s}Clim-{:s}".format(nb_clust,nb_class,fprefixe,fshortcode,dataystartend)
            # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
            # eventuellement en PDF, si SAVEPDF active. 
            ctloop.do_save_figure(figfile,dpi=FIGDPI,path=figdir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    return VAPT,F1U,F1sU,F2V,CRi,CAj,CAHindnames,CAHindnameswnb,NoCAHindnames,\
           class_afc,AFCindnames,AFCindnameswnb,NoAFCindnames
#
#%%
def plot_afc_proj(F1U,F2V,CRi,CAj,pa,po,class_afc,nb_class,NIJ,Nmdlok,indnames=None,
                  figdir=".",
                  figfile=None, dpi=None, figpdf=False,
                  ) :
    global SIZE_REDUCTION, AFCWITHOBS
    global FIGDPI, FIGEXT, Visu_UpwellArt
    #
    Nmdlafc = CRi.shape[0]
    dataystartend = datemdl2dateinreval(DATAMDL)
    ctloop.printwarning([ "    AFC: 2-D PROJECTION" ])
    if Visu_UpwellArt :
        lblfontsize=14;    mdlmarkersize=250;
        lblfontsizeobs=16; obsmarkersize=320;
        lblfontsizecls=16; clsmarkersize=280;
        #
        if SIZE_REDUCTION == 'All' :
            zone_stitre = "Large"
            figsize=(16,12)
            top=0.93; bottom=0.05; left=0.05; right=0.95
            xdeltapos      =0.025; ydeltapos     =-0.002; linewidths   =2.5
            xdeltaposobs   =0.030; ydeltaposobs  =-0.003; linewidthsobs=3
            xdeltaposcls   =0.001; ydeltaposcls  =-0.003; linewidthscls=2.5
            xdeltaposlgnd  =0.03;  ydeltaposlgnd =-0.002
            if Nmdlok == 47 :
                legendXstart   = 0.975; legendYstart  =-0.53;   legendYstep  =0.058
            else :
                legendXstart   =-1.22; legendYstart  =0.88;   legendYstep  =0.06
            legendXstartcls=legendXstart;
            legendYstartcls=legendYstart - legendYstep * (nb_clust + 1.2)
        elif SIZE_REDUCTION == 'sel' :
            zone_stitre = "Selected"
            figsize=(16,12)
            top=0.93; bottom=0.05; left=0.05; right=0.95
            xdeltapos      =0.035; ydeltapos     =-0.002; linewidths   =2.5
            xdeltaposobs   =0.040; ydeltaposobs  =-0.003; linewidthsobs=3
            xdeltaposcls   =0.001; ydeltaposcls  =-0.005; linewidthscls=2.5
            xdeltaposlgnd  =0.040; ydeltaposlgnd =-0.002
            legendXstart   =-0.79; legendYstart  =-0.85;  legendYstep  =0.072
            legendXstartcls=legendXstart;
            legendYstartcls=legendYstart - legendYstep * (nb_clust + 1.2)
        #
        stitre = ("SST {:s} -on zone \"{:s}\"- A.F.C Projection with Models, Observations and Classes ({:s})"+\
                  "\n- {:s}, AFC on {:d} CAH Classes for {} models (in {} AFC clusters) + Obs -").format(fcodage,
                       zone_stitre,dataystartend,method_cah,nb_class,Nmdlok,nb_clust)
        #
        ctloop.do_plotart_afc_projection(F1U,F2V,CRi,CAj,pa,po,class_afc,nb_class,NIJ,Nmdlok,
                    indnames=indnames,
                    title=stitre,
                    Visu4Art=Visu_UpwellArt,
                    AFCWITHOBS = AFCWITHOBS,
                    figsize=figsize,
                    top=top, bottom=bottom, left=left, right=right,
                    lblfontsize=lblfontsize,       mdlmarkersize=mdlmarkersize,
                    lblfontsizeobs=lblfontsizeobs, obsmarkersize=obsmarkersize,
                    lblfontsizecls=lblfontsizecls, clsmarkersize=clsmarkersize,
                    xdeltapos   =xdeltapos ,   ydeltapos   =ydeltapos,
                    xdeltaposobs=xdeltaposobs, ydeltaposobs=ydeltaposobs,
                    xdeltaposcls=xdeltaposcls, ydeltaposcls=ydeltaposcls,
                    linewidths=linewidths, linewidthsobs=linewidthsobs, linewidthscls=linewidthscls,
                    legendok=True,
                    xdeltaposlgnd=xdeltaposlgnd,ydeltaposlgnd=ydeltaposlgnd,
                    legendXstart=legendXstart,legendYstart=legendYstart,legendYstep=legendYstep,
                    legendprefixlbl="AFC Cluster",
                    legendprefixlblobs="Observations",
                    legendokcls=True,
                    legendXstartcls=legendXstartcls,legendYstartcls=legendYstartcls,
                    legendprefixlblcls="CAH Classes",
                    )
    else :
        stitre = ("AFC Projection - {:s} SST ({:s}). {:s}".format(fcodage,
                  dataystartend,method_cah));
        lblfontsize=14; linewidths = 2.0
        #
        ctloop.do_plot_afc_projection(F1U,F2V,CRi,CAj,pa,po,class_afc,nb_class,NIJ,Nmdlok,
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
        figfile += "AFC2DProj-{:d}-{:d}_{:d}Clust-{:d}Classes_{:s}{:s}Clim-{:s}_{:d}-mod".format(
                pa,po,nb_clust,nb_class,fprefixe,fshortcode,dataystartend,Nmdlok)
        if AFCWITHOBS :
            figfile += "+Obs"
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
                          ) :
    #
    global SIZE_REDUCTION
    global SAVEFIG, FIGDPI, FIGEXT, FIGARTDPI, SAVEPDF, VFIGEXT, blockshow
    global TypePerf
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
    
    if SIZE_REDUCTION == 'All' :
        figsize = (9,6)
        wspace=0.0; hspace=0.0; top=0.94; bottom=0.08; left=0.06; right=0.925;
        nticks = 5; # 4
    elif SIZE_REDUCTION == 'sel' :
        figsize=(9,9)
        wspace=0.0; hspace=0.0; top=0.94; bottom=0.08; left=0.05; right=0.96;
        nticks = 2; # 4
    #
    fig = plt.figure(figsize=figsize)
    fignum = fig.number # numero de figure en cours ...
    fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
    #
    print("\n{:d}-model(s)' generalization: {} ".format(len(TMixtMdl),TMixtMdl))
    MdlMoy, IMixtMdl, MGPerfglob = ctloop.mixtgeneralisation (sMapO, TMixtMdl, Tmdlname, TDmdl4CT, 
                                        class_ref, classe_Dobs, nb_class, Lobs, Cobs, isnumobs,
                                        lon=lon, lat=lat,
                                        TypePerf=TypePerf,
                                        label=misttitlelabel,
                                        fignum=fignum,
                                        fsizetitre=14, ytitre=1.01, nticks=nticks);
    #
    if len(IMixtMdl) == 0 :
        print("\n *** PAS DE MODELES POUR GENERALISATION !!! ***\n" )
        return
    else :
        if SAVEFIG : # sauvegarde de la figure
            if figfile1 is None :
                figfile1 = "Fig_"
            if dpi is None :
                dpi = FIGDPI
            figfile1 += "MeanModel_{:s}-{:d}-mod_Mean".format(mistfilelabel,len(Tmdlname[IMixtMdl]))
            figfile1 += figfileext1
            #
            ctloop.do_save_figure(figfile1,dpi=dpi,path=figdir,ext=FIGEXT,figpdf=figpdf)
    
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
        # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
        # eventuellement en PDF, si SAVEPDF active. 
        ctloop.do_save_figure(figfile2,dpi=dpi,path=figdir,ext=FIGEXT,figpdf=figpdf)
    #
    #
    #**********************************************************************
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
    global nb_class, eqcmap, ccmap, nFigArt
    global sst_obs_coded, Dobs, XC_Ogeo, NDobs, fond_C, pcmap, obs_data_path
    global sMapO, lon, lat, ilon, ilat, varnames, wvmin, wvmax
    global AFCindnames, NoAFCindnames, TDmdl4CT, Tmdlname, Tmdlnamewnb, Tmdlonlynb
    global isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, case_figs_dir
    global class_afc, list_of_plot_colors
    #
    caseconfig = ''
    caseconfig_valid_set = ( 'All', 'sel' )   # toutes en minuscules svp !
    verbose = False
    manualmode = True
    #%% NE PAS EXECUTER CE BLOCK EN MODE MANUEL
    manualmode = False
    try:
        opts, args = getopt.getopt(argv,"hvc:",["case=","verbose"])
        #
        for opt, arg in opts:
            if opt == '-h':
                print('ctLoopMain.py -c all -v | --case=all --verbose /* cases are all or sel */')
                sys.exit()
            elif opt in ("-v", "--verbose"):
                verbose = True
            elif opt in ("-c", "--case"):
                caseconfig = arg
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
               ' * {:s} *\n'.format("      <OPTIONS> are -h, -v".ljust(62))+\
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
    #
    print("Case config is '{:s}'".format(caseconfig))
    pcmap,AFC_Visu_Classif_Mdl_Clust, AFC_Visu_Clust_Mdl_Moy_4CT,\
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
    if SAVEMAP :
        if not os.path.exists(MAPSDIR) :
            os.makedirs(MAPSDIR)
        case_maps_dir = os.path.join(MAPSDIR,case_label)
        if not os.path.exists(case_maps_dir) :
            os.makedirs(case_maps_dir)
    # -------------------------------------------------------------------------
    # Repertoire principal des figures et sous-repertoire por le cas en cours 
    if SAVEFIG :
        if not os.path.exists(FIGSDIR) :
            os.makedirs(FIGSDIR)
        case_figs_dir = os.path.join(FIGSDIR,case_label)
        if not os.path.exists(case_figs_dir) :
            os.makedirs(case_figs_dir)
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
        fileext1 = "_Lim{:+.1f}-{:+.1f}".format(wvmin,wvmax)
        fileext2 = "_{:s}_{:s}{:s}Clim-{:s}_{:s}".format(eqcmap.name,fprefixe,
                      fshortcode,dataobsystartend,data_label_base)

        plot_obs4ct(sst_obs_coded,Dobs,lon,lat,isnanobs=isnanobs,isnumobs=isnumobs,
                    varnames=varnames,wvmin=wvmin,wvmax=wvmax,
                    eqcmap=eqcmap,Show_ObsSTD=Show_ObsSTD,
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
        plot_ct_Umatrix(sMapO)
        plot_ct_map_wei(sMapO)
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
    fond_C = ctobs.dto2d(fond_C,Lobs,Cobs,isnumobs,missval=0.5)
    #
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
            figsize = (9,6)
            top=0.94; bottom=0.08; left=0.06; right=0.925;
            nticks = 5; # 4
        elif SIZE_REDUCTION == 'sel' :
            figsize=(9,9)
            top=0.94; bottom=0.08; left=0.05; right=0.96;
            nticks = 2; # 4
        #
        if Visu_UpwellArt :
            nFigArt = 1;
            figfile = "FigArt{:02d}_".format(nFigArt);
            dpi     = FIGARTDPI
            figpdf  = True
        else :
            figfile = "Fig_"
            dpi     = FIGDPI
            figpdf  = False

        plot_geo_classes(lon,lat,XC_Ogeo,nb_class,
                         nticks=nticks,
                         title=stitre,
                         fileext=fileextbis, figdir=case_figs_dir,
                         figfile=figfile, dpi=dpi, figpdf=figpdf,
                         ccmap=ccmap,
                         figsize=figsize,
                         top=top, bottom=bottom, left=left, right=right,
                         )
    #%% -----------------------------------------------------------------------
    # Figure 2 pour Article : profils moyens par classe
    # -------------------------------------------------------------------------
    if Visu_ObsStuff or Visu_UpwellArt :
        stitre = "Observations ({:s}), Monthly Mean by Class (method: {:s})".format(dataobsystartend,method_cah)
        fileextbis = "_{:s}{:s}Clim-{:s}_{:s}".format(fprefixe,
                      fshortcode,dataobsystartend,data_label_base)
        if Visu_UpwellArt :
            nFigArt = 2;
            figfile = "FigArt{:02d}_".format(nFigArt)
            dpi     = FIGARTDPI
            figpdf  = True
        else :
            figfile = "Fig_"
            dpi     = FIGDPI
            figpdf  = False
        plot_mean_profil_by_class(sst_obs_coded,nb_class,classe_Dobs,varnames=varnames,
                                  title=stitre,
                                  fileext=fileextbis, figdir=case_figs_dir,
                                  figfile=figfile, dpi=dpi, figpdf=figpdf,
                                  pcmap=pcmap,
                                  figsize=(12,6),
                                  wspace=0.0, hspace=0.0, top=0.95, bottom=0.08, left=0.06, right=0.92,
                                  )
    #
    #%% -----------------------------------------------------------------------
    if Visu_CTStuff : # Visu des profils des référents de la carte SOM
        stitre="SOM Map Profils by Cell ({:s}) (background color represents classes)".format(dataobsystartend)
        fileextter = "_{:s}{:s}Clim-{:s}_{:s}".format(fprefixe,
                      fshortcode,dataobsystartend,data_label_base)
        plot_ct_profils(sMapO,Dobs,class_ref,varnames=varnames,
                        fileext=fileextter, figdir=case_figs_dir,
                        pcmap=pcmap,
                        title=stitre,
                        titlefntsize=14,
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
        Nmdlok, NDmdl = ctloop_model_traitement(sst_obs_coded,Dobs,XC_Ogeo,sMapO,
                    lon,lat,ilat,ilon,
                    nb_class,class_ref,classe_Dobs,NDobs,fond_C,
                    isnanobs,isnumobs,Lobs,Cobs,list_of_plot_colors,
                    Sfiltre=Sfiltre,
                    data_period_ident=DATAMDL,
                    varnames=varnames, figdir=case_figs_dir,
                    commonfileext=fileextIV, commonfileext79=fileextIV79,
                    eqcmap=eqcmap, ccmap=ccmap,pcmap=pcmap,
                    obs_data_path=obs_data_path,
                    OK101=OK101,
                    OK102=OK102,
                    OK104=OK104,
                    OK105=OK105,
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
            top=0.93; bottom=0.18; left=0.06; right=0.98
        else :
            TmdlnameX = Tmdlname
            figsize=(12,6)
            top=0.93; bottom=0.15; left=0.06; right=0.98
        #
        fignum = ctloop.do_models_after_second_loop(Tperfglob,Tperfglob_Qm,TmdlnameX,
                                           list_of_plot_colors,
                                           title=stitre,
                                           TypePerf=TypePerf,fcodage=fcodage,
                                           figsize=figsize,
                                           top=top, bottom=bottom, left=left, right=right)
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
    #
    #%% #######################################################################
    #
    #     ANALYSE FACTORIELLE DES CORRESPONDANCES (CORRESPONDENCE ANALYSIS)
    #
    # -------------------------------------------------------------------------
    # Figure 3 pour Article : Performances selon les clusters de l'AFC
    # -------------------------------------------------------------------------
    if Visu_UpwellArt :
        nFigArt = 3;
        figfile = "FigArt{:02d}_".format(nFigArt)
        dpi     = FIGARTDPI
        figpdf  = True
    else :
        figfile = "Fig_"
        dpi     = FIGDPI
        figpdf  = False
    #
    VAPT, F1U, F1sU, F2V, CRi, CAj, CAHindnames, CAHindnameswnb, NoCAHindnames,\
        class_afc,AFCindnames,AFCindnameswnb,\
        NoAFCindnames = ctloop_compute_afc(sMapO, lon, lat, TDmdl4CT,
                           Tmdlname, Tmdlnamewnb, Tmdlonlynb,
                           nb_class, nb_clust, isnumobs, isnanobs, class_ref, classe_Dobs,
                           TTperf, Nmdlok, Lobs, Cobs, NDmdl, Nobsc, data_label_base,
                           AFC_Visu_Classif_Mdl_Clust=AFC_Visu_Classif_Mdl_Clust,
                           AFC_Visu_Clust_Mdl_Moy_4CT=AFC_Visu_Clust_Mdl_Moy_4CT,
                           ccmap=ccmap,
                           figdir=case_figs_dir,
                           figfile=figfile, dpi=dpi, figpdf=figpdf,
                           )
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
    plot_afc_proj(F1U,F2V,CRi,CAj,pa,po,class_afc,nb_class,NIJ,Nmdlok,
                  indnames=NoAFCindnames,
                  figdir=case_figs_dir,
                  figfile=figfile, dpi=dpi, figpdf=figpdf,
                  )
    #
    if STOP_BEFORE_GENERAL :
        plt.show(); sys.exit(0)
    #
    #%%
    #___________
    print(("\n{} {},\n{} {},\n{} {},\n{} {},\n{} {}\n").format(
                   "SIZE_REDUCTION ".ljust(18,'.'),SIZE_REDUCTION,
                   "WITHANO ".ljust(18,'.'),WITHANO,
                   "UISST ".ljust(18,'.'),UISST,
                   "climato ".ljust(18,'.'),climato,
                   "NIJ ".ljust(18,'.'),NIJ))

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
    generalisation_ok = True
    generalisafcclust_ok = True
    generalisaotherperiods_ok = False
    #**************************************************************************
    #.............................. GENERALISATION ............................
    if generalisation_ok :
        generalisation_proc(sMapO, lon, lat, varnames, wvmin, wvmax, nb_class,
                    isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, case_figs_dir, 
                    TDmdl4CT,Tmdlname,
                    data_period_ident=DATAMDL,
                    generalisation_type='bestclust',  # 'bestclust', 'bestcum'
                    )
        generalisation_proc(sMapO, lon, lat, varnames, wvmin, wvmax, nb_class,
                    isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, case_figs_dir, 
                    TDmdl4CT,Tmdlname,
                    data_period_ident=DATAMDL,
                    generalisation_type='bestcum',  # 'bestclust', 'bestcum'
                    )
        generalisation_proc(sMapO, lon, lat, varnames, wvmin, wvmax, nb_class,
                    isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, case_figs_dir, 
                    TDmdl4CT,Tmdlname,
                    data_period_ident=DATAMDL,
                    generalisation_type='all',  # 'bestclust', 'bestcum', 'all'
                    )
    if generalisafcclust_ok :
        generalisafcclust_proc(sst_obs_coded,Dobs,XC_Ogeo,sMapO,lon,lat,ilat,ilon, varnames, wvmin, wvmax, nb_class,
                        isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, case_figs_dir, 
                        TDmdl4CT, Tmdlname,
                        data_period_ident=DATAMDL,
                        )

        #generalisation_proc(generalisation_ok=generalisation_ok, generalisafcclust_ok=generalisafcclust_ok)
        # generalisation_proc(generalisation_ok=True, generalisafcclust_ok=True)
    #%%
    if generalisaotherperiods_ok :
    #%%
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
                                    isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, case_figs_dir, 
                                    TDmdl4CTx,Tmdlnamex,
                                    data_period_ident=data_period_ident,
                                    scenario=scenario_name,
                                    )

    #%%
    return
#%%
def generalisation_proc(sMapO, lon, lat, varnames, wvmin, wvmax, nb_class,
                        isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, case_figs_dir, 
                        TDmdl4CT, Tmdlname,
                        data_period_ident=None,
                        scenario=None,
                        generalisation_type=None,  # 'bestclust', 'bestcum'
                        ) :
    #%% #########################################################################
#    global , eqcmap, ccmap, nFigArt
#    global , Dobs, XC_Ogeo, NDobs, fond_C, pcmap, obs_data_path
#    global , ilon, ilat, 
#    global AFCindnames, NoAFCindnames, 
#    global 
#    global class_afc, list_of_plot_colors

    #%%
    global nFigArt
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
    # -------------------------------------------------------------------------
    figfileext1 = "_{:s}{:s}_{:d}Class{:s}".format(fprefixe,fshortcode,nb_class,datafileext)
    figfileext2 = ""
    #if Show_ModSTD :
    #    figfileext2 += "+{:d}ySTD".format(Nda)
    figfileext2 += "_Lim{:+.1f}-{:+.1f}_{:s}{:s}Clim{:s}".format(wvmin,wvmax,
                        fprefixe,fshortcode,datafileext)
    # -------------------------------------------------------------------------
    # Figure 5 pour Article : Classes par cluster sur projectrion geo best cluster AFC
    # -------------------------------------------------------------------------
    if Visu_UpwellArt :
        nFigArt = 5;
        figfile = "FigArt{:02d}_".format(nFigArt)
        dpi     = FIGARTDPI
        figpdf  = True
    else :
        figfile = "Fig_"
        dpi     = FIGDPI
        figpdf  = False
    #
    ctloop_generalisation(sMapO, lon, lat, TMixtMdl, TMixtMdlLabel, TDmdl4CT, Tmdlname,
                      nb_class, isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs,
                      varnames=varnames,
                      modstdflg=Show_ModSTD,
                      figdir=case_figs_dir,
                      figfile1=figfile, figfile2=figfile, figfileext1=figfileext1, figfileext2=figfileext2,
                      dpi=dpi, figpdf=figpdf,
                      wvmin=wvmin,wvmax=wvmax,
                      subtitle=stitre,
                      )
        #
    #%%
    return
#
def generalisafcclust_proc(sst_obs_coded,Dobs,XC_Ogeo,sMapO,lon,lat,ilat,ilon, varnames, wvmin, wvmax, nb_class,
                        isnumobs, isnanobs, Lobs, Cobs, class_ref, classe_Dobs, case_figs_dir, 
                        TDmdl4CT, Tmdlname,
                        data_period_ident=None,
                        ) :
    #%% #########################################################################
#    global , eqcmap, ccmap, nFigArt
#    global , Dobs, XC_Ogeo, NDobs, fond_C, pcmap, obs_data_path
#    global , ilon, ilat, 
#    global AFCindnames, NoAFCindnames, 
#    global 
#    global class_afc, list_of_plot_colors

    #%%
    
    global nFigArt
    #
    if data_period_ident is not None :
        dataystartend = datemdl2dateinreval(data_period_ident)
        datastitre = " ({:s})".format(dataystartend)
        datafigext = "-{:s}".format(dataystartend)
    else :
        datastitre = ""
        datafigext = ""
    #
    #**************************************************************************
    #.............................. GENERALISATION ............................
    #
    #
    #  Generalisation d'une ensemble ou cluster precis
    for kclust in np.arange(1,nb_clust + 1) :
        #kclust = 1
        SfiltredMod = AFCindnames[np.where(class_afc==kclust)[0]]
        print("--Generalizing for AFC Cluster{:s}: {:d} with models:\n  {}".format(
                datastitre,kclust,SfiltredMod))
        stitre = "AFC Cluster: {:d}{:s}".format(kclust,datastitre)
        fileextIV = "_AFCclust{:d}{:s}-{:s}{:s}_{:s}{:s}".format(kclust,datafigext,fprefixe,
                           SIZE_REDUCTION,fshortcode,method_cah)
        fileextIV79 = "_AFCclust{:d}{:s}-{:s}{:s}_{:s}".format(kclust,datafigext,fprefixe,
                             SIZE_REDUCTION,fshortcode)
        xTperfglob, xTperfglob_Qm, xTDmdl4CT, xTmdlname, xTmdlnamewnb, \
            xTmdlonlynb, xTTperf, xNmdlok, \
            xNDmdl = ctloop_model_traitement(
                        sst_obs_coded,Dobs,XC_Ogeo,sMapO,lon,lat,ilat,ilon,
                        nb_class,class_ref,classe_Dobs,NDobs,fond_C,
                        isnanobs,isnumobs,Lobs,Cobs,list_of_plot_colors,
                        Sfiltre=SfiltredMod,
                        varnames=varnames, figdir=case_figs_dir,
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
        ctloop_generalisation(sMapO, lon, lat, TMixtMdl, TMixtMdlLabel,
                          TDmdl4CT, Tmdlname,
                          nb_class, isnumobs, isnanobs, Lobs, Cobs,
                          class_ref, classe_Dobs,
                          varnames=varnames,
                          figdir=case_figs_dir,
                          wvmin=wvmin,wvmax=wvmax,
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
    xTDmdl4CT, xTmdlname, xTmdlnamewnb, xTmdlonlynb, xTTperf, xNmdlok,\
        xNDmdl = ctloop_model_traitement(
                    sst_obs_coded,Dobs,XC_Ogeo,sMapO,lon,lat,ilat,ilon,
                    nb_class,class_ref,classe_Dobs,NDobs,fond_C,
                    isnanobs,isnumobs,Lobs,Cobs,list_of_plot_colors,
                    Sfiltre=SfiltredMod,
                    varnames=varnames, figdir=case_figs_dir,
                    commonfileext=fileextIV, commonfileext79=fileextIV79,
                    eqcmap=eqcmap, ccmap=ccmap,pcmap=pcmap,
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
