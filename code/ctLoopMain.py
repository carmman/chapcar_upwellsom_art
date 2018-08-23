#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:38:49 2018

@author: carlos
"""

import numpy as np
import os
from   matplotlib import cm

from models_def_tb import *

import ctLoopTools as ctloop
##from   ctObsMdldef import *

# PARAMETRAGE (#1) DU CAS
from ParamCas import *
#import ParamCas


obs_data_path = '../Datas'


### INITIALISATION ############################################################
case_name_base,varnames,list_of_plot_colors = ctloop.initialisation(case_label_base,tseed=0)

### ACQUISITION DES DONNEES D'OBSERVATION #####################################
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
Nobs,Lobs,Cobs = np.shape(sst_obs);
print("obs.shape : ", Nobs,Lobs,Cobs);
#%%
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
    stitre = "Observed SST %s MEAN (%d-%d)".format(fcodage,andeb,anfin)
    if Show_ObsSTD :
        stitre += "(monthly %d years STD in contours)".format(Nda)
    #
    wvmin,wvmax = ctloop.plot_obs(sst_obs,lon,lat,varnames=varnames,cmap=localcmap,
                    title=stitre,
                    lolast=lolast,figsize=figsize,
                    wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                    TRENDLESS=TRENDLESS,WITHANO=WITHANO,climato=climato,UISST=UISST,
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
    del wvmin,wvmax,localcmap
