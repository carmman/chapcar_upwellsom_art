# -*- coding: cp1252 -*-
# =============================================================================
# Programme ctLoopTools (Version pour Article)
# Version de ctLoopAnyS purement en fonctions, pilotees depuis le "main" se
# trouvant dans ctLoopMain.py
#
#
# =============================================================================
import sys
import os
#import time as time
from   time  import time
from datetime import datetime
import pickle
import numpy as np
from   matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
from   mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from   scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.io
from   scipy.stats import spearmanr, chi2_contingency
from   sklearn.metrics.cluster import adjusted_rand_score
#
from   triedpy import triedtools as tls
from   triedpy import triedsompy as SOM
from   triedpy import triedacp   as acp
#from  triedpy import triedctk   as ctk
#
import localdef     as ldef
import UW3_triedctk as ctk
import ctObsMdldef  as ctobs
#
#%=====================================================================
def afcnuage (CP,cpa,cpb,Xcol,K=None,xoomK=500,linewidths=1,indname=None,
              cmap=cm.jet,holdon=False,ax=None,gridok=False,
              drawaxes=False,axescolors=None,axlinewidth=1,
              drawtriangref=False,axtight=False,aximage=False,axdecal=False,
              rotlabels=None,randrotlabels=None,randseed=0,
              horizalign='left',vertalign='center',
              lblcolor=None,lblbgcolor=None,lblfontsize=8,lblprefix=None,
              marker='o',obsmarker='o',
              markersize=None,obsmarkersize=None,
              edgecolor=None,
              edgeobscolor='k',obscolor='k',edgeclasscolor='k',faceclasscolor='m',
              article_style=False) :
# pomp� de WORKZONE ... TPA05
    if ax is None :
        fig = plt.figure(figsize=(16,12));
        ax = plt.subplot(111)
    else :
        fig = plt.gcf() # figure en cours ...
        ax = ax
    if holdon == False :
        # j'ai un pb obs \ pas obs qui apparaissent dans la m�me couleur que le dernier cluster
        # quand bien m�me il ne participe pas � la clusterisation
        lenCP = len(CP); lenXcol = len(Xcol);
        if lenCP > lenXcol : # hyp : lenCP>lenXcol
            # Je consid�re que les (LE) surnum�raire de CP sont les obs (ou aut chose), je l'enl�ve,
            # et le met de cot�
            CPobs = CP[lenXcol:lenCP,:];
            CP    = CP[0:lenXcol,:];
            # K et indname suivent CP
            if K is not None :
                if np.ndim(K) == 1 :
                    K = K.reshape(len(K),1)
                Kobs  = K[lenXcol:lenCP,:];
                K     = K[0:lenXcol,:];
            else :
                K = 1; Kobs = 1;
            obsname = indname[lenXcol:lenCP];
            indname = indname[0:lenXcol];
        #print("indname:{}".format(indname))
        #print("obsname:{}".format(obsname))
        #
        my_norm = plt.Normalize()
        my_normed_data = my_norm(Xcol)
        ec_colors = cmap(my_normed_data) # a Nx4 array of rgba value
        if edgecolor is None :
            edge_ec_colors = ec_colors
        else:
            edge_ec_colors = edgecolor
        #? if np.ndim(K) > 1 : # On distingue triangle � droite ou vers le haut selon l'axe
        if K is not None and len(np.shape(K)) > 1:
            n,p = np.shape(K);
        else :
            n,p = (1,1)
        #
        if article_style :
            if markersize is None :
                if p > 1 :
                    msize = K[:,cpa-1]*xoomK
                else :
                    msize = K*xoomK
            else:
                msize = markersize
            if obsmarkersize is None :
                if p > 1 :
                    omsize = Kobs[:,cpa-1]*xoomK
                else :
                    omsize = Kobs*xoomK
            else:
                omsize = obsmarkersize
            ax.scatter(CP[:,cpa-1],CP[:,cpb-1],s=msize,marker=marker,
                            edgecolors=edge_ec_colors,facecolor=ec_colors,linewidths=linewidths)
            if lenCP > lenXcol : # cas des surnumeraire, en principe les obs
                ax.scatter(CPobs[:,cpa-1],CPobs[:,cpb-1],s=omsize,marker=obsmarker,
                                edgecolors=edgeobscolor,facecolor=obscolor,linewidths=linewidths)
        else : # NO article style
            if p > 1 : # On distingue triangle � droite ou vers le haut selon l'axe
                ax.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K[:,cpa-1]*xoomK,
                                marker='>',edgecolors=ec_colors,facecolor='none',linewidths=linewidths)
                ax.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K[:,cpb-1]*xoomK,
                                marker='^',edgecolors=ec_colors,facecolor='none',linewidths=linewidths)
                if lenCP > lenXcol : # cas des surnumeraire, en principe les obs
                    ax.scatter(CPobs[:,cpa-1],CPobs[:,cpb-1],s=Kobs[:,cpa-1]*xoomK,
                                    marker='>',edgecolors=obscolor,facecolor='none',linewidths=linewidths)
                    ax.scatter(CPobs[:,cpa-1],CPobs[:,cpb-1],s=Kobs[:,cpb-1]*xoomK,
                                    marker='^',edgecolors=obscolor,facecolor='none',linewidths=linewidths)            
            else :
                ax.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K*xoomK,
                                marker='s',edgecolors=ec_colors,facecolor='none',linewidths=linewidths);
                if lenCP > lenXcol : # ? cas des surnumeraire, en principe les obs
                    ax.scatter(CPobs[:,cpa-1],CPobs[:,cpb-1],s=Kobs*xoomK,
                                marker='s',edgecolors=obscolor,facecolor='none',linewidths=linewidths);
                    
    else : #(c'est pour lescolonnes -les classes)
        if article_style :
            if markersize is None :
                if p > 1 :
                    msize = K[:,cpa-1]*xoomK
                else :
                    msize = K*xoomK
            else:
                msize = markersize
            ax.scatter(CP[:,cpa-1],CP[:,cpb-1],s=msize,marker='s',
                       edgecolors=edgeclasscolor,facecolor=faceclasscolor)
        else :
            ax.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K[:,cpa-1]*xoomK,
                            marker='o',facecolor='m')
            ax.scatter(CP[:,cpa-1],CP[:,cpb-1],s=K[:,cpb-1]*xoomK,
                            marker='o',facecolor='c',alpha=0.5)
    #plt.axis('tight')
    ax.set_xlabel('axe %d'%cpa); ax.set_ylabel('axe %d'%cpb)
    
    if 0 : # je me rapelle plus tres bien � quoi ca sert; do we need a colorbar here ? may be
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(Xcol), vmax=np.max(Xcol)))
        sm.set_array([])
        #if holdon == False :
        #    ax.colorbar(sm);

    ## Labelisation des points, if not empty
    #if indname is not None :
    #    N,p = np.shape(CP);
    #    for i in np.arange(N) :
    #        plt.text(CP[i,cpa-1],CP[i,cpb-1],indname[i])
    #if holdon == False and lenCP > lenXcol :
    #    N,p = np.shape(CPobs);
    #    for i in np.arange(N) :
    #        plt.text(CPobs[i,cpa-1],CPobs[i,cpb-1],obsname[i])
            
    if rotlabels is None :
        rotlabels = 0
    if lblcolor == None :
        lblcolor = 'k'
    if randrotlabels is None :
        randrotlabels = 0
    else:
        np.random.seed(randseed)
    # Labelisation des points, if not empty
    if indname is not None :
        N,p = np.shape(CP);
        for i in np.arange(N) :
            if lblprefix is None :
                currentlbl = indname[i]
            else :
                currentlbl = lblprefix+indname[i]
            #print("indname[{}]: {}".format(i,indname[i]))
            if randrotlabels != 0 :
                localrotlabels = rotlabels + randrotlabels*np.random.normal()
            else :
                localrotlabels = rotlabels
            if lblbgcolor is None :
                ax.text(CP[i,cpa-1],CP[i,cpb-1],currentlbl,color=lblcolor,
                        fontsize=lblfontsize,
                        horizontalalignment=horizalign,
                        verticalalignment=vertalign,
                        rotation=localrotlabels,rotation_mode="anchor")
            else :
                ax.text(CP[i,cpa-1],CP[i,cpb-1],currentlbl,color=lblcolor,backgroundcolor=lblbgcolor,
                        fontsize=lblfontsize,
                        horizontalalignment=horizalign,
                        verticalalignment=vertalign,
                        rotation=localrotlabels,rotation_mode="anchor")
    if holdon == False and lenCP > lenXcol :
        N,p = np.shape(CPobs);
        for i in np.arange(N) :
            if lblprefix is None :
                currentlbl = obsname[i]
            else :
                currentlbl = lblprefix+obsname[i]
            #print("obsname[{}]: {}".format(i,obsname[i]))
            if randrotlabels != 0 :
                localrotlabels = rotlabels + randrotlabels*np.random.normal()
            else :
                localrotlabels = rotlabels
            if lblbgcolor is None :
                ax.text(CPobs[i,cpa-1],CPobs[i,cpb-1],currentlbl,color=lblcolor,
                        fontsize=lblfontsize,
                        horizontalalignment=horizalign,
                        verticalalignment=vertalign,
                        rotation=localrotlabels,rotation_mode="anchor")
            else :
                ax.text(CPobs[i,cpa-1],CPobs[i,cpb-1],currentlbl,color=lblcolor,backgroundcolor=lblbgcolor,
                        fontsize=lblfontsize,
                        horizontalalignment=horizalign,
                        verticalalignment=vertalign,
                        rotation=localrotlabels,rotation_mode="anchor")
    #
    if axtight :
        ax.axis('tight')
    if aximage :
        ax.axis('image')
    if gridok :
        ax.grid(axis='both')
    #
    # decalage force des axes en hauteur et a droite (pour permettre l'affichage des nom depasant)
    if axdecal is not False :
        if axdecal is True :
            decalfactorx = 5; decalfactory = 5; # nombre de centiemes
        elif np.isscalar(axdecal):
                        decalfactorx = axdecal; decalfactory = axdecal
        else:
                        decalfactorx = axdecal[0]; decalfactory = axdecal[1]
        lax=ax.axis()
        dlaxx = (lax[1]-lax[0])/100; dlaxy = (lax[3]-lax[2])/100; 
        ax.axis([lax[0],lax[1] + (decalfactorx * dlaxx),lax[2],lax[3]+ (decalfactory * dlaxy)])
    
    # recupere les limites des axes ...
    xlim = ax.set_xlim();
    ylim = ax.set_ylim();

    if drawaxes :
        # Tracer les axes
        if axescolors is None :
            plt.plot(xlim, np.zeros(2));
            plt.plot(np.zeros(2),ylim);
        else :
            xaxcolor,yaxcolor = axescolors
            plt.plot(xlim, np.zeros(2),color=xaxcolor,linewidth=axlinewidth);
            plt.plot(np.zeros(2),ylim,color=yaxcolor,linewidth=axlinewidth);

    # Plot en noir des triangles de r�f�rence en bas � gauche
    if drawtriangref :
        dx = xlim[1] - xlim[0];
        dy = ylim[1] - ylim[0];
        px = xlim[0] + dx/(xoomK) + dx/20; # � ajuster +|- en ...
        py = ylim[0] + dy/(xoomK) + dy/20; # ... fonction de xoomK
        ax.scatter(px,py,marker='>',edgecolors='k', s=xoomK,     facecolor='none');
        ax.scatter(px,py,marker='>',edgecolors='k', s=xoomK*0.5, facecolor='none');
        ax.scatter(px,py,marker='>',edgecolors='k', s=xoomK*0.1, facecolor='none');
    # remet les axes aux limites mesures precedemment
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
#
#----------------------------------------------------------------------
def Dgeoclassif(sMap,Data,L,C,isnum,MajorPerf,visu=True,cbticklabelsize=8,cblabel=None,
                cblabelsize=10,old=False,ax=None,nticks=1,tickfontsize=10) :
    bmus_   = ctk.mbmus (sMap,Data); 
    classe_ = class_ref[bmus_].reshape(len(bmus_));   
    X_Mgeo_ = ctobs.dto2d(classe_,L,C,isnum); # Classification g�ographique
    #plt.figure(); g�r� par l'appelant car ce peut �tre une fig d�j� d�finie
    #et en subplot ... ou pas ...
    #classe_DD_, Tperf_, Perfglob_ = ctobs.perfbyclass(classe_Dobs,classe_,nb_class,kperf=kperf);
    classe_DD_, Tperf_ = ctobs.perfbyclass(classe_Dobs, classe_, nb_class);
    Perfglob_ = ctobs.perfglobales([MajorPerf], classe_Dobs, classe_, nb_class)[0];
    if visu :
        if ax is None :
            ax = plt.gca() # current axes
        ims = ax.imshow(X_Mgeo_, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
        Tperf_ = np.round([iperf*100 for iperf in Tperf_]).astype(int); #print(Tperf_)   
        # colorbar
        if not old :
            #cbar_ax,kw = cb.make_axes(ax,orientation="vertical",fraction=0.04,pad=0.03,aspect=20)
            #fig.colorbar(ims, cax=cbar_ax, ticks=ticks,boundaries=bounds,values=bounds, **kw);
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="4%", pad="3%")
            hcb = plt.colorbar(ims,cax=cax,ax=ax,ticks=ticks,boundaries=bounds,values=bounds);
        else :
            hcb = plt.colorbar(ims,ax=ax,ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(Tperf_);
        hcb.ax.tick_params(labelsize=cbticklabelsize);
        if cblabel is not None :
            hcb.set_label(cblabel,size=cblabelsize)
            #cbar_ax.set_ylabel(cblabel,size=14)
        plt.sca(ax) # remet les "current axes" comme axes par defaut
        #plt.axis('off');
        if lon is None or lat is None :
            plt.xticks([]); plt.yticks([])
        else :
            if 0:
                plt.xticks(np.arange(0,C,nticks), lon[np.arange(0,C,nticks)], fontsize=tickfontsize)
                plt.yticks(np.arange(0,L,nticks), lat[np.arange(0,L,nticks)], fontsize=tickfontsize)
            else :
                set_lonlat_ticks(lon,lat,step=nticks,fontsize=tickfontsize,verbose=False,lengthen=True)
        #grid(); # for easier check
    return Perfglob_
#----------------------------------------------------------------------
def _printwarning0(msg):
    if isinstance(msg, (list,)) :
        for m in msg :
            print(" * {:75s} *".format(m)) 
    else:
        print(" * {:75s} *".format(msg))
    #print(" * {:75s} *".format(" "))

def printwarning(msg, msg2=None, msg3=None):
    print("\n{}".format(" ".ljust(80,"*")))
    _printwarning0(msg)
    if msg2 is not None :
        print("{}".format(" ".ljust(80,"*")))
        _printwarning0(msg2)
    if msg3 is not None :
        print("{}".format(" ".ljust(80,"*")))
        _printwarning0(msg3)
    print("{}".format(" ".ljust(80,"*")))
#----------------------------------------------------------------------
def set_lonlat_ticks(lon,lat,fontsize=12,lostep=1,lastep=1,step=None,londecal=None,latdecal=None,
                    roundlabelok=True,lengthen=True,verbose=False) :
    ''' Pour tracer les "ticks" et "ticklabels" des figures de type geographique,
        ou les axes ce sont les Latitudes et Longitudes 
    '''
    if londecal is None :
        londecal = (lon[1] - lon[0])/2
        if lon[0] < lon[1] :
            londecal = -londecal
    if latdecal is None :
        latdecal = (lat[1] - lat[0])/2
        if lat[0] < lat[1] :
            latdecal = -latdecal
    if step is not None :
        # force la m�me valeur de pas dans les ticks x et y
        lostep=step
        lastep=step
    if verbose :
        print('londecal: {}\nlatdecal: {}'.format(londecal,latdecal))
    # ralonge les lon et les lat
    if verbose :
        print('LON-LAT:\n  {}\n  {}'.format(lon,lat))
    if lengthen :
        lon = np.concatenate((lon,[lon[-1] + (lon[1] - lon[0])]))
        lat = np.concatenate((lat,[lat[-1] + (lat[1] - lat[0])]))
        if verbose :
            print('LENGHTED LON-LAT:\n  {}\n  {}'.format(lon,lat))
    nLon = lon.shape[0]
    nLat = lat.shape[0]
    # current axis limits
    lax = plt.axis()
    # Ticks
    xticks = np.arange(londecal,nLon,lostep)
    yticks = np.arange(latdecal,nLat,lastep)
    # Ticklabels
    if 0 :
        xticklabels = lon[np.arange(0,nLon,lostep)]
        yticklabels = lat[np.arange(0,nLat,lastep)]
    else :
        xticklabels = lon[np.arange(0,nLon,lostep)]
        yticklabels = lat[np.arange(0,nLat,lastep)]
        if lon[0] < lon[1] :
            xticklabels += londecal
        else :
            xticklabels -= londecal
        if lat[0] < lat[1] :
            yticklabels += latdecal
        else :
            yticklabels -= latdecal
    if verbose :
        print('Tiks:\n  {}\n  {}'.format(xticks,yticks))
        print('Labels:\n  {}\n  {}'.format(xticklabels,yticklabels))
    if roundlabelok :
        xticklabels = np.round(xticklabels).astype(int)
        yticklabels = np.round(yticklabels).astype(int)
        if verbose :
            print('Rounded Labels:\n  {}\n  {}'.format(xticklabels,yticklabels))
    #
    plt.xticks(xticks,
               xticklabels,
               fontsize=fontsize)
    plt.yticks(yticks,
               yticklabels,
               fontsize=fontsize)
    # set axis limits to previous value
    plt.axis(lax)
#
def do_save_figure(figfile,path=None,ext=None,dpi=100,fig2ok=False,ext2=None):
    ''' DO_SAVE_FIGURE
        sauvegarde de la figure en cours dans un fichier au format donne par 
        l'option 'ext' (PNG par defaut).
        En option, on peut sauver en un deuxieme format, par exemple un format
        vectoriel: PDF ou Postscript Encapsule.? Choisir de peference PDF,
        les NaN apparaissent en Noir en EPS. En PDF il suffit d'ajouter l'option
        transparent=False pour eviter ce probleme.
    '''
    if ext is None :
        ext = '.png'
    elif ext[0] != '.' :
        ext = '.'+ext
    if path is None :
        path = '.'
    if fig2ok :
        if ext2 is None :
            ext2 = '.pdf'
        elif ext2[0] != '.' :
            ext2 = '.'+ext2
    #
    figurefilelname = path+os.sep+figfile+ext;
    print("-- {:->88s}".format(''))
    print("-- saving current figure in path/file: '{}/\n     '{}'".format(
            os.path.dirname(figurefilelname),os.path.basename(figurefilelname)))
    # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg))
    plt.savefig(figurefilelname, dpi=dpi)
    # format2, sauvegarde en fotmat vectoriel, PDF ou Postscript Encapsule
    # de peference PDF, car les NaN apparaissent en Noir en EPS. En PDF il suffit
    # d'ajouter l'option transparent=False pour eviter ce probleme.
    if fig2ok :
        figurefilelname = path+os.sep+figfile+ext2;
        print("   saving also in {} format in file:\n     '{}'".format(ext2.upper(),
              os.path.basename(figurefilelname)))
        plt.savefig(figurefilelname, transparent=False)
#
# #####################################################################
# INITIALISATION
# Des trucs qui pourront servir
#######################################################################
def initialisation(case_label_base,tseed=0):
    # ferme toutes les fenetres de figures en cours
    plt.close('all')
    # Met a BLANC la couleur des valeurs masqu�es dans les masked_array,
    # *** Cela change alors la couleur des NAN dans les fichiers EPS ***
    #cmap.set_bad('w',1.)
    #======================================================================
    plt.rcParams.update({'figure.max_open_warning': 0})
    #======================================================================
    casetime=datetime.now()
    casetimelabel = casetime.strftime("%d %b %Y @ %H:%M:%S")
    casetimeTlabel = casetime.strftime("%Y%m%dT%H%M%S")
    #----------------------------------------------------------------------
    # Des truc qui pourront servir
    # cycle des couleurs par defaut pour de graphiques avec plt.plot()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    list_of_plot_colors = prop_cycle.by_key()['color']
    
    # temps initial
    tpgm0 = time();
    plt.ion()
    # Initialise 'varnames' avec les noms des mois
    if 0: # French
        varnames = np.array(["JAN","FEV","MAR","AVR","MAI","JUI",
                            "JUI","AOU","SEP","OCT","NOV","DEC"]);
    elif 0: # Francais
        import locale
        loc0 = locale.getlocale(locale.LC_ALL)
        locale.setlocale(locale.LC_ALL,'fr_FR.ISO8859-1')
        #locale.setlocale(locale.LC_ALL,'fr_FR.UTF-8')
        # retourne les noms des mois en Francais (en trois lettres) et en Majuscules
        varnames = [ datetime(2000, m+1, 1, 0, 0).strftime("%b").upper() for m in np.arange(12) ]
        locale.setlocale(locale.LC_ALL, loc0); # restore saved locale
    elif 1: # Anglais
        import locale
        loc0 = locale.getlocale(locale.LC_ALL)
        locale.setlocale(locale.LC_ALL,'en_US.UTF-8')
        # retourne les noms des mois en Anglais (en trois lettres) et en Majuscules
        varnames = [ datetime(2000, m+1, 1, 0, 0).strftime("%b").upper() for m in np.arange(12) ]
        locale.setlocale(locale.LC_ALL, loc0); # restore saved locale
    #
    #######################################################################
    #
    # PARAMETRAGE (#1) DU CAS
    #from ParamCas import *
    #
    #======================================================================
    if tseed == 0:  # si tseed est zero on ne fait rien
        case_name_base = case_label_base
    else: # si tseed est different de zero on l'ajoute dans le nom du cas en cours
        case_name_base = "{:s}_s{:03d}".format(case_label_base)
    #
    return case_name_base,casetime,casetimelabel,casetimeTlabel,varnames,list_of_plot_colors
#
#######################################################################
# ACQUISITION DES DONNEES D'OBSERVATION (et application des codifications)
#======================================================================
def read_obs(obs_data_path,DATAOBS) :
    # Lecture des Obs____________________________________
    if 0 : # Ca c'�tait avant
        #lat: 29.5 � 5.5 ; lon: -44.5 � -9.5
        sst_obs  = np.load(os.path.join(obs_data_path,"sst_obs_1854a2005_Y60X315.npy"));
        # Selection___________________________________________
        Nobs,Lobs,Cobs = np.shape(sst_obs);
        if Nda > 0 : # Ne prendre que les Nda derni�res ann�es (rem ATTENTION, toutes les ne commencent
            sst_obs = sst_obs[Nobs-(Nda*12):Nobs,]; #  pas � l'ann�e 1850 ou 1854 ni au mois 01 !!!!!!!
    #
    if DATAOBS == "raverage_1975_2005" :  
        obs_filename = os.path.join(obs_data_path,"Donnees_1975-2005","Obs",
                                    "ersstv3b_1975-2005_extract_LON-315-351_LAT-30-5.nc");
        data_label_base = "ERSSTv3b-1975-2005"
    elif DATAOBS == "raverage_1930_1960" :  
        obs_filename = os.path.join(obs_data_path,"Donnees_1930-1960","Obs",
                                    "ersstv3b_1930-1960_extract_LON-315-351_LAT-30-5.nc");   
        data_label_base = "ERSSTv3b-1930_1960"
    elif DATAOBS == "raverage_1944_1974" :
        obs_filename = os.path.join(obs_data_path,"Donnees_1944-1974","Obs",
                                    "ersstv3b_1944-1974_extract_LON-315-351_LAT-30-5.nc");  
        data_label_base = "ERSSTv3b-1944_1974"
    elif DATAOBS == "rcp_2006_2017" :
        obs_filename = os.path.join(obs_data_path,"Donnees_2006-2017","Obs",
                                    "ersstv3b_2006-2017_extrac_LON-315-351_LAT-30-5.nc");
        data_label_base = "ERSSTv3b-2006_2017"
    else :
        print("*** unknown DATAOBS case <{}> ***",DATAOBS)
        raise
    #
    import netCDF4
    nc      = netCDF4.Dataset(obs_filename);
    liste_var = nc.variables;       # mois par mois de janvier 1930 � decembre 1960 I guess
    sst_var   = liste_var['sst']    # 1960 - 1930 + 1 = 31 ; 31 * 12 = 372
    sst_obs   = sst_var[:];         # np.shape = (372, 1, 25, 36)
    Nobs,Ncan,Lobs,Cobs = np.shape(sst_obs);
    if 0 : # visu obs
        ldef.showimgdata(sst_obs,fr=0,n=Nobs);
        plt.suptitle("Obs raverage 1930 � 1960")
        plt.show(); sys.exit(0)
    sst_obs   = sst_obs.reshape(Nobs,Lobs,Cobs); # np.shape = (372, 25, 36)
    sst_obs   = sst_obs.filled(np.nan);
    #    
    if 0:
        lat      = np.arange(29.5, 4.5, -1);
        lon      = np.arange(-44.5, -8.5, 1);
    else :
        lat       = liste_var['lat'][:]
        lon       = liste_var['lon'][:]
        # SI masked_array alors on recupare uniquement la data, on neglige le mask
        if isinstance(np.ma.array(lat),np.ma.MaskedArray) :
            lat = lat.data
        if isinstance(np.ma.array(lon),np.ma.MaskedArray) :
            lon = lon.data
    #
    if np.max(lon) > 180 : # ATTENTION, SI UN > 180 on considere TOUS > 180 ... 
        print("-- LONGITUDE [0:360] to [-180:180] simple conversion ...")
        lon -= 360
    #
    print("\nData ({}x{}): {}".format(len(lat),len(lon),data_label_base))
    print(" - Dir : {}".format(os.path.dirname(obs_filename)))
    print(" - File: {}".format(os.path.basename(obs_filename)))
    print(" - dimensions of SST Obs : {}".format(sst_obs.shape))
    print(" - Lat : {} values from {} to {}".format(len(lat),lat[0],lat[-1]))
    print(" - Lon : {} values from {} to {}".format(len(lon),lon[0],lon[-1]))
    #
    return data_label_base,sst_obs,lon,lat
#
def get_zone_obs(sst_obs,lon,lat,size_reduction='All',frlon=None,tolon=None,frlat=None,tolat=None) :
    # Param�trage : _____________________________________
    # D�finition d'une zone plus petite
    #
    if frlon is None and tolon is None :
        frlon = np.min(lon)
        tolon = np.max(lon) + 1;
    if frlat is None and tolat is None :
        if lat[0] > lat[1] :
            frlat = np.max(lat)
            tolat = np.min(lat) - 1;
        else:
            frlat = np.min(lat)
            tolat = np.max(lat) + 1;
    #
    print("\nCurrent geographical limits ('to' limit excluded):")
    print(" - size_reduction is '{}'".format(size_reduction))
    print(" - Lat : from {} to {}".format(frlat,tolat))
    print(" - Lon : from {} to {}".format(frlon,tolon))
    #
    # selectionne les LON et LAT selon les limites definies dans ParamCas.py
    # le fait pour tout cas de SIZE_REDUCTION, lat et lon il ne devraient pas
    # changer dans le cas de SIZE_REDUCTION=='All'
    ilat = np.intersect1d(np.where(lat <= frlat),np.where(lat > tolat))
    ilon = np.intersect1d(np.where(lon >= frlon),np.where(lon < tolon))
    lat = lat[np.intersect1d(np.where(lat <= frlat),np.where(lat > tolat))]
    lon = lon[np.intersect1d(np.where(lon >= frlon),np.where(lon < tolon))]
    #
    if size_reduction != 'RED' :
        # Prendre d'entr�e de jeu une zone delimitee
        sst_obs = sst_obs[:,ilat,:];
        sst_obs = sst_obs[:,:,ilon];
        print("\nDefinitive data:")
        print(" - Dimensions of SST Obs : {}".format(sst_obs.shape))
        print(" - Lat : {} values from {} to {}".format(len(lat),lat[0],lat[-1]))
        print(" - Lon : {} values from {} to {}".format(len(lon),lon[0],lon[-1]))
    #
    return sst_obs,lon,lat,ilat,ilon

#def do_prepar_data(sst_obs, TRENDLESS=False, WITHANO=True,
#                   climato=None, UISST=False, NORMMAX=False, CENTRED=False) :
#    # Codification des Obs 4CT 
#    sst_obs_coded, Dobs, NDobs = ctobs.datacodification4CT(sst_obs,
#                TRENDLESS=TRENDLESS, WITHANO=WITHANO, climato=climato, UISST=UISST,
#                NORMMAX=NORMMAX, CENTRED=CENTRED);
#    #
#    return sst_obs_coded, Dobs, NDobs 

def plot_obs(sst_obs,Dobs,lon,lat,varnames=None,title="",wvmin=None,wvmax=None,
             isnanobs=None,isnumobs=None,
             lolast=4,figsize=(12,7),cmap='jet',
             wspace=0.04, hspace=0.12, top=0.925, bottom=0.035, left=0.035, right=0.97,
             climato=None,NORMMAX=False,CENTRED=False,Show_ObsSTD=False) :
    #
    if wvmin is None :
        wvmin = np.nanmin(Dobs)
    if wvmax is None :
        wvmax = np.nanmax(Dobs)
    #
    if isnanobs is None :
        isnanobs = np.where(np.isnan(sst_obs[0].reshape(np.prod(np.shape(sst_obs)[1:]))))[0];
    if isnumobs is None :
        isnumobs = np.where(~np.isnan(sst_obs[0].reshape(np.prod(np.shape(sst_obs)[1:]))))[0];
    # 
    _,Lobs,Cobs = np.shape(sst_obs);
    #
    minDobs = np.min(Dobs);   maxDobs=np.max(Dobs);
    moyDobs = np.mean(Dobs);  stdDobs=np.std(Dobs);
    #
    Dstd_, pipo_, pipo_  = ctobs.Dpixmoymens(sst_obs, stat='std',
                                             NORMMAX=NORMMAX,CENTRED=CENTRED);
    #
    fig = plt.figure(figsize=figsize,facecolor='w')
    fignum = fig.number # numero de figure en cours ...
    if climato != "GRAD" :
        if Show_ObsSTD :
            ctobs.aff2D(Dobs,Lobs,Cobs,isnumobs,isnanobs,
                        wvmin=wvmin,wvmax=wvmax,
                        fignum=fignum,varnames=varnames,cmap=cmap,
                        wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                        noaxes=False,noticks=False,nolabels=False,y=0.98,cblabel="SST Anomaly [�C]",
                        lolast=lolast,lonlat=(lon,lat),
                        vcontour=Dstd_, ncontour=np.arange(0,1,1/20), ccontour='k', lblcontourok=True,
                        ); #...
        else:
            ctobs.aff2D(Dobs,Lobs,Cobs,isnumobs,isnanobs,
                        wvmin=wvmin,wvmax=wvmax,
                        fignum=fignum,varnames=varnames,cmap=cmap,
                        wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                        noaxes=False,noticks=False,nolabels=False,y=0.98,cblabel="SST Anomaly [�C]",
                        lolast=lolast,lonlat=(lon,lat),
                        ); #...
    else :
        ctobs.aff2D(Dobs,Lobs,Cobs,isnumobs,isnanobs,
                    wvmin=0.0,wvmax=0.042,
                    fignum=fignum,varnames=varnames,cmap=cmap, 
                    wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                    noaxes=False,noticks=False,nolabels=False,y=0.98,cblabel="SST Anomaly [�C]",
                    vcontour=Dstd_, ncontour=np.arange(0,1,1/20), ccontour='k', lblcontourok=True,
                    lolast=lolast,lonlat=(lon,lat)); #...
    plt.suptitle("{:s} ({:s})\nmin={:f}, max={:f}, mean={:f}, std={:f}".format(title,
                 cmap.name,minDobs,maxDobs,moyDobs,stdDobs),y=0.995);
    #
    del Dstd_, pipo_
    #
    return wvmin,wvmax
#
def plot_obsbis(sst_obs, Dobs, varnames=None, title="",
                isnanobs=None,isnumobs=None,
                figsize=(12,7),cmap='jet',
                wspace=0.04, hspace=0.12, top=0.925, bottom=0.035, left=0.035, right=0.97,
                ) :
    #
    if isnanobs is None :
        isnanobs = np.where(np.isnan(sst_obs[0].reshape(np.prod(np.shape(sst_obs)[1:]))))[0];
    if isnumobs is None :
        isnumobs = np.where(~np.isnan(sst_obs[0].reshape(np.prod(np.shape(sst_obs)[1:]))))[0];
    #
    _,Lobs,Cobs = np.shape(sst_obs);
    #
    minDobs = np.min(Dobs);   maxDobs=np.max(Dobs);
    moyDobs = np.mean(Dobs);  stdDobs=np.std(Dobs);
    #
    # Figure FREE LIMITS
    localcmap = cmap
    ND,p      = np.shape(Dobs);
    X_        = np.empty((Lobs*Cobs,p));   
    X_[isnumobs] = Dobs   
    X_[isnanobs] = np.nan
    X = X_.T.reshape(p,1,Lobs,Cobs)
    #
    facecolor='w'
    fig = plt.figure(figsize=figsize,facecolor=facecolor)
    fignum = fig.number # numero de figure en cours ...
    fig, axes = plt.subplots(nrows=3, ncols=4, num=fignum,
                        sharex=True, sharey=True, figsize=figsize,facecolor=facecolor)
    fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
    ifig = 0;
    for ax in axes.flat:
        img  = X[ifig].reshape(Lobs,Cobs);
        ims = ax.imshow(img, cmap=localcmap);           
        hcb = plt.colorbar(ims,ax=ax);
        ax.axis("image"); #ax.axis("off");
        ax.set_title(varnames[ifig])
        ifig += 1;
    plt.suptitle("{:s} ({:s})\nglobal min={:f}, max={:f}, mean={:f}, std={:f}".format(title,
                 localcmap.name,minDobs,maxDobs,moyDobs,stdDobs),y=0.995);
#   
# #############################################################################
#                       Carte Topologique
#==============================================================================
def do_ct_map_process(Dobs,name=None,mapsize=None,tseed=0,norm_method=None,
                         initmethod=None, neigh=None,varname=None,
                         etape1=[5,5,2],etape2=[5,2,0.1],
                         verbose='on', retqerrflg=False
                         ) :
    #
    print("Initializing random generator with seed={}".format(tseed))
    print("tseed=",tseed); np.random.seed(tseed);
    #--------------------------------------------------------------------------
    # Cr�ation de la structure de la carte_____________________________________
    if name is None :
        name = 'sMapObs'; 
    #
    if norm_method is None :
        norm_method = 'data'; # je n'utilise pas 'var' mais je fais centred �
                              # la place (ou pas) qui est �quivalent, mais qui
                              # me permetde garder la maitrise du codage
    #
    if initmethod is None :
        initmethod = 'random'; # peut etre ['random', 'pca']
    #
    if neigh is None :
        neigh = 'Guassian'; # peut etre ['Bubble', 'Guassian'] (sic !!!)
    #
    # Initialisation de la SOM ________________________________________________
    sMapO = SOM.SOM(name, Dobs, mapsize=mapsize, norm_method=norm_method, \
                    initmethod='random', varname=varname)
    #
    #print("NDobs(sm.dlen)=%d, dim(Dapp)=%d\nCT : %dx%d=%dunits" \
    #      %(sMapO.dlen,sMapO.dim,nbl,nbc,sMapO.nnodes));
    #!EU-T-IL FALLUT NORMALISER LES DONNEES ; il me semble me rappeler que
    #ca � peut etre a voir avec norm_method='data' ci dessus
    #
    # Apprentissage de la carte _______________________________________________
    #
    ttrain0 = time();
    #
    # Entrainenemt de la SOM __________________________________________________
    eqO = sMapO.train(etape1=etape1, etape2=etape2, verbose=verbose, retqerrflg=retqerrflg);
    #
    print("Training elapsed time {:.4f}s".format(time()-ttrain0));
    # + err topo maison
    bmus2O = ctk.mbmus (sMapO, Data=None, narg=2);
    etO    = ctk.errtopo(sMapO, bmus2O); # dans le cas 'rect' uniquement
    #
    print("Two phases training executed:")
    print(" - Phase 1: {0[0]} epochs for radius variing from {0[1]} to {0[2]}".format(etape1))
    print(" - Phase 2: {0[0]} epochs for radius variing from {0[1]} to {0[2]}".format(etape2))
    #
    return sMapO,eqO,etO
#
def plot_ct_dendro(data, datalinkg, nclass, method='ward', metric='euclidean', criterion='maxclust',
                   title="dendrogram",
                   figsize=(14,6),
                   wspace=0.0, hspace=0.2, top=0.92, bottom=0.12, left=0.05, right=0.99,
                   ):
    Ncell = data.shape[0]
    minref = np.min(data);
    maxref = np.max(data);
    #
    fig = plt.figure(figsize=figsize,facecolor='w');
    fignum = fig.number # numero de figure en cours ...
    #
    # calcule la limite de decoupage selon le nombre de classes ou clusters
    max_d = np.sum(datalinkg[[-nclass+1,-nclass],2])/2
    color_threshold = max_d
    #
    plt.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
    #
    R_ = dendrogram(datalinkg,p=Ncell,truncate_mode=None,color_threshold=color_threshold,
                    orientation='top',leaf_font_size=6) #,labels=lignames
    #               leaf_rotation=45);
    #
    plt.axhline(y=max_d, c='k')
    #L_ = np.array(lignames)
    #plt.xticks((np.arange(len(TmdlnameArr)+1)*10)+7,L_[R_['leaves']], fontsize=8,
    #       rotation=45,horizontalalignment='right', verticalalignment='baseline')
    #xtickslocs, xtickslabels = plt.xticks()
    #plt.xticks(xtickslocs, xtickslabels)
    plt.tick_params(axis='x',reset=True)
    plt.tick_params(axis='x',which='major',direction='out',length=3,pad=1,top=False,   #otation_mode='anchor',
                    labelrotation=-80,labelsize=8)
    plt.grid(axis='y')
    plt.xlabel('codebook number', labelpad=15, fontsize=12)
    plt.ylabel("inter class distance ({})".format(method), fontsize=12)
    lax=plt.axis(); daxy=(lax[3]-lax[2])/400
    plt.axis([lax[0],lax[1],lax[2]-daxy,lax[3]])
    plt.title("{:s} - ({:s}), nclass={:d}".format(title,method,nclass));

#%%
def plot_fig01_article(xc_ogeo,lon,lat,nb_class,classe_Dobs,title="observations",
                       cmap=None,
                       figsize=(9,6),
                       wspace=0.0, hspace=0.0, top=0.96, bottom=0.08, left=0.06, right=0.92,
                       nticks=5,
                       ) :
    ''' Figure 1 pour Article 
    '''
    #
    if cmap is None :
        cmap = cm.jet;       # Accent, Set1, Set1_r, gist_ncar; jet, ...
    #
    Lobs, Cobs = xc_ogeo.shape
    #
    coches = np.arange(nb_class)+1;   # ex 6 classes : [1,2,3,4,5,6]
    ticks  = coches + 0.5;            # [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    bounds = np.arange(nb_class+1)+1; # pour bounds faut une fronti�re de plus [1, 2, 3, 4, 5, 6, 7]
    #
    fig = plt.figure(figsize=figsize)
    fignum = fig.number # numero de figure en cours ...
    fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
    if lat[0] < lat[1] :
        origin = 'lower'
    else :
        origin = 'upper'
    fig, ax = plt.subplots(nrows=1, ncols=1, num=fignum,facecolor='w')
    ims = ax.imshow(xc_ogeo, interpolation='none',cmap=cmap,vmin=1,vmax=nb_class,origin=origin);
    if 0:
        plt.xticks(np.arange(0,Cobs,nticks), lon[np.arange(0,Cobs,nticks)], rotation=45, fontsize=10)
        plt.yticks(np.arange(0,Lobs,nticks), lat[np.arange(0,Lobs,nticks)], fontsize=10)
    else :
        #plt.xticks(np.arange(-0.5,Cobs,lolast), np.round(lon[np.arange(0,Cobs,lolast)]).astype(int), fontsize=12)
        #plt.yticks(np.arange(0.5,Lobs,lolast), np.round(lat[np.arange(0,Lobs,lolast)]).astype(int), fontsize=12)
        set_lonlat_ticks(lon,lat,step=nticks,fontsize=10,verbose=False,lengthen=True)
        #set_lonlat_ticks(lon,lat,fontsize=10,londecal=0,latdecal=0,roundlabelok=False,lengthen=False)
    #plt.axis('tight')
    plt.xlabel('Longitude', fontsize=12); plt.ylabel('Latitude', fontsize=12)
    plt.title(title,fontsize=16);
    #
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="4%", pad="3%")
    hcb = plt.colorbar(ims,cax=cax,ax=ax,ticks=ticks,boundaries=bounds,values=bounds);
    cax.set_yticklabels(coches);
    cax.tick_params(labelsize=12)
    cax.set_ylabel('Class',size=14)
#
def plot_mean_curve_by_class(sst_obs,nb_class,classe_Dobs,isnumobs=None,varnames=None,
                             title="observations mean by class",
                             pcmap=None,
                             figsize=(12,6),
                             wspace=0.0, hspace=0.0, top=0.96, bottom=0.08, left=0.06, right=0.92,
                             ):
    #
    if isnumobs is None :
        isnumobs = np.where(~np.isnan(sst_obs[0].reshape(np.prod(np.shape(sst_obs)[1:]))))[0];
    #
    if pcmap is None :
        tmpcmap = cm.jet;       # Accent, Set1, Set1_r, gist_ncar; jet, ...
        # pour avoir des couleurs � peu pr�s equivalente pour les plots
        pcmap  = tmpcmap(np.arange(0,320,round(320/nb_class))); #ok?
        pcmap *= 0.95 # fonce les legerement tous les couleurs ...
        if tmpcmap.name == 'jet' and nb_class == 4:
            pcmap[2] *= 0.9 # fonce la couleur de la classe qui est trop claire(si ccmap = jet)
    #
    # Courbes des Moyennes Mensuelles par Classe
    fig = plt.figure(figsize=figsize)
    fignum = fig.number # numero de figure en cours ...
    fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
    TmoymensclassObs = ctobs.moymensclass(sst_obs,isnumobs,classe_Dobs,nb_class)
    #plt.plot(TmoymensclassObs); plt.axis('tight');
    for i in np.arange(nb_class) :
        plt.plot(TmoymensclassObs[:,i],'.-',color=pcmap[i]);
    plt.grid(axis='y')
    plt.axis('tight');
    #
    if varnames is not None :
        plt.xticks(np.arange(len(varnames)),varnames)
    #
    plt.ylabel('Mean SST Anomaly [�C]', fontsize=12);
    plt.xlabel('Month', fontsize=12);
    legax=plt.legend(np.arange(nb_class)+1,loc=2,fontsize=10);
    legax.set_title('Class')
    plt.title(title,fontsize=16); #,fontweigth='bold');
    #plt.show(); sys.exit(0)
#

def plot_ct_profils(sMapO,Dobs,class_ref,
                    varnames=None,
                    pcmap=None,
                    figsize=(7.5,12),
                    wspace=0.01, hspace=0.05, top=0.95, bottom=0.04, left=0.15, right=0.86,
                    ):
        fig = plt.figure(figsize=figsize)
        fignum = fig.number
        fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
        ctk.showprofils(sMapO, fignum=fignum, Data=Dobs,
                        visu=3, scale=2,Clevel=class_ref-1,Gscale=0.5,
                        axsztext=6,marker='.',markrsz=4,pltcolor='r',
                        ColorClass=pcmap,ticklabels=varnames,xticks=np.arange(0,len(varnames),2),verbose=False);
        plt.suptitle("SOM Map Profils by Cell (background color represents classes)",fontsize=16); #,fontweigth='bold');
    #
    #######################################################################
    #
    #raise

def red_climato(Dmdl,L,C,isnum,isnum_red,frl=None,tol=None,frc=None,toc=None,ix=None,iy=None) :
    ''' Appel:
            X_ = red_climato(Dmdl,L,C,isnum,isnum_red,frl,tol,frc,toc);
            
        ou frl,tol,frc,toc sont les indices limite des x (lon) pour frc et toc
        et des y (lat) pour frl et tol
               
        ou bien,
            X_ = red_climato(Dmdl,L,C,isnum,isnum_red,ix=ilon,ix=lat);
            
        ou ix et iy ce sont les listes d'indices en x (lon) et en y (lat) a preserver.
            
        Attention, il faut soit utiliser les 4 variables : frl,tol,frc,toc
        ou bien les deux autres, ix et iy, on les nommant: ix=ilon,ix=lat
    '''
    X_ = np.empty((L*C,12))
    X_[isnum] = Dmdl;
    X_ = X_.reshape(L,C,12);
    if ix is not None and iy is not None :
        X_ = X_[iy,:,:]
        X_ = X_[:,ix,:]
    else :
        X_ = X_[frl:tol,frc:toc,:]
    l,c,m = np.shape(X_);
    X_ = X_.reshape(l*c,m)
    X_ = X_[isnum_red]
    return X_
#
def do_models_startnloop(sMapO,Tmodels,Tinstit,ilat,ilon,isnanobs,isnumobs,nb_class,class_ref,classe_Dobs,
                             Tnmodel=None,Nmodels=None,Sfiltre=None,
                             TypePerf = ["MeanClassAccuracy"],
                             obs_data_path=".",
                             DATAMDL="raverage_1975_2005",
                             MDLCOMPLETION=True,
                             SIZE_REDUCTION="All",
                             NIJ=0,
                             OK101='False',
                             OK102='False',
                             OK106='False',
                             ) :
    #######################################################################
    #                        MODELS STUFFS START HERE
    #======================================================================
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #       INITILISATIONS EN AMONT de LA BOUCLE SUR LES MODELES
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if Nmodels is None :
        Nmodels = len(Tmodels)
    #
    if Tnmodel is None :
        Tnmodel = []
        for imodel in np.arange(Nmodels) :
            Tnmodel.append("{:d}".format(imodel+1))
        Tnmodel = np.array(Tnmodel)
    #
    # For (sub)plot by modele
    nsub   = Nmodels + 1; # actuellement au plus 48 mod�les + 1 pour les obs.      
    #nsub  = 9;  # Pour MICHEL (8+1pour les obs)
    #def lcsub(nsub,minncol=None) :   # <-- NON, utiliser plutot nl,nc = nsublc() qui est dans localdef.py
    #    if minncol is not None:
    #        nbsubc = minncol
    #    else :
    #        nbsubc = np.ceil(np.sqrt(nsub));
    #    nbsubl = np.ceil(1.0*nsub/nbsubc).as;
    #    return nbsubc, nbsubl
    #nbsubc, nbsubl = lcsub(nsub);
    nbsubl, nbsubc = ldef.nsublc(nsub);
    isubplot=0;
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    min_moymensclass = 999999.999999; # sert pour avoir tous les ...
    max_moymensclass = 000000.000000; # ... subplots � la m�me �chelles
    #
    TDmdl4CT         = []; # Stockage des mod�les 4CT pour AFC-CAH ...
    #
    Nmdlok           = 0;  # Pour si y'a cumul ainsi connaitre l'indice de mod�le valide 
                           # courant, puis au final, le nombre de mod�les valides
                           # quoique ca dependra aussi que SUMi(Ni.) soit > 0                   
    Tperfglob4Sort   = [];
    Tclasse_DMdl     = [];
    Tmdlname         = []; # Table des mod�les
    Tmdlnamewnb      = []; # Table des mod�les avec numero
    Tmdlonlynb       = []; # Table des mod�les avec seulement le numero
    Tmoymensclass    = [];
    #
    Smoy_101 = None
    Tsst_102 = None
    #
    #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    #           PREMIERE BOUCLE SUR LES MODELES START HERE
    #           PREMIERE BOUCLE SUR LES MODELES START HERE
    #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    print("\n {:s}".format(" first loop ".center(79,'o')));
    #
    for imodel in np.arange(Nmodels) :
        nomdl    = Tnmodel[imodel];   # numero de modele
        instname = Tinstit[imodel];   # institution du modele
        mdlname  = Tmodels[imodel,0]; # nom du modele
        anstart  = Tmodels[imodel,1]; # (utile pour rmean seulement)
        #
        # >>> Filtre (selection)de mod�les en entr�e ; Mettre 0 dans le if pour ne pas filtrer
        if Sfiltre is not None and mdlname not in Sfiltre :
            continue;
        #print(" using model '{}' ...".format(mdlname))
        # <<<<< 
        #______________________________________________________
        # Lecture des donn�es (fichiers.mat g�n�r�s par Carlos)
        if  DATAMDL=="raverage_1975_2005" : 
            subdatadir = "Donnees_1975-2005"
            # ###################################
            # EXCEPTION pour FGOALS-s2
            # ###################################
            if mdlname == "FGOALS-s2" :
                mdl_filename = os.path.join(obs_data_path,subdatadir,
                                        "all_data_historical_raverage_1975-2005",
                                        'Data',
                                        instname+'_'+mdlname,
                                        "sst_"+mdlname+"_raverage_1975-2004.mat")
            else :
                mdl_filename = os.path.join(obs_data_path,subdatadir,
                                        "all_data_historical_raverage_1975-2005",
                                        'Data',
                                        instname+'_'+mdlname,
                                        "sst_"+mdlname+"_raverage_1975-2005.mat")
        elif DATAMDL=="raverage_1930_1960" : 
            subdatadir = "Donnees_1930-1960"
            mdl_filename = os.path.join(obs_data_path,subdatadir,
                                    "all_data_historical_raverage_1930-1960",
                                    'Data',
                                    instname+'_'+mdlname,
                                    "sst_"+mdlname+"_raverage_1930-1960.mat")
        elif DATAMDL=="raverage_1944_1974" : 
            subdatadir = "Donnees_1944-1974"
            mdl_filename = os.path.join(obs_data_path,subdatadir,
                                    "all_data_historical_raverage_1944-1974",
                                    'Data',
                                    instname+'_'+mdlname,
                                    "sst_"+mdlname+"_raverage_1944-1974.mat")
        elif DATAMDL == "rcp_2006_2017":
            subdatadir = "Donnees_2006-2017"
            mdl_filename = os.path.join(obs_data_path,subdatadir,
                                    "all_data_"+scenar+"_raverage_2006-2017",
                                    'Data',
                                    instname+'_'+mdlname,
                                    "sst_"+scenar+mdlname+"_raverage_2006-2017.mat")
        elif DATAMDL == "rcp_2070_2100":
            subdatadir = "Donnees_2070-2100"
            mdl_filename = os.path.join(obs_data_path,subdatadir,
                                    "all_data_"+scenar+"_raverage_2070-2100",
                                    'Data',
                                    instname+'_'+mdlname,
                                    "sst_"+scenar+mdlname+"_raverage_2070-2100.mat")
        else :
            print("*** unknown DATAMDL case <{}> for model '{}' ***".format(DATAMDL,mdlname))
            raise
        if imodel == 0:
            print(" using model '{}' with data '{}' ...".format(mdlname,DATAMDL))
        else :
            print(" using model '{}' ...".format(mdlname))
        
        try :
            sst_mat = scipy.io.loadmat(mdl_filename);
        except :
            print(" ** model '{}' not found **".format(mdlname));
            continue;
        sst_mdl = sst_mat['SST'];
        #
        Nmdl, Lmdl, Cmdl = np.shape(sst_mdl); #print("mdl.shape : ", Nmdl, Lmdl, Cmdl);
        #
        Nmdlok += 1; # Pour si y'a cumul ainsi connaitre l'indice de mod�le 
                 # valide courant, puis au final, le nombre de mod�les valides.
                 # (mais ...)
        #
        if MDLCOMPLETION : # Compl�mentation des donn�es mod�les de sorte � ce que seul
            nnan=1;        # le mappage des nans d'obs soit utilis�
            while nnan > 0 :
                sst_mdl, nnan = ctobs.nan2moy(sst_mdl, Left=1, Above=0, Right=0, Below=0)
        #________________________________________________`
        if SIZE_REDUCTION != 'RED' :
            # Prendre d'entr�e de jeu une zone delimitee
            sst_mdl = sst_mdl[:,ilat,:];
            sst_mdl = sst_mdl[:,:,ilon];
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
        # Codification du mod�le (4CT)            
        sst_mdl_coded, Dmdl, NDmdl = ctobs.datacodification4CT(sst_mdl);
        #________________________________________________________
        # Ecart type des mod�les en entr�es cumul�s et moyenn�s
        # (�v�ntuellement control� par Sfiltre ci-dessus)
        if OK101 :
            # Moyenne
            Dmoy_, pipo_, pipo_  = ctobs.Dpixmoymens(sst_mdl_coded);
            if Nmdlok == 1 :     
                Smoy_101 = Dmoy_;        
            else :
                Smoy_101 += Dmoy_;
        #
        if OK102 :
            # Ecart type (� cause de FGOALS-s2 ca complique tout ...
            # (Il y aura un d�calage entre moyenne et ecart type qui sera un peu
            # fauss�, mais ainsi je retrouverai les m�me r�sultats qu'avant ...
            sst_ = sst_mdl_coded;
            # correction manuelle pour FGOALS-s2 (1975_2005) pour avoir le m�me nombre
            # de donnees que pour les autres modeles (c-a-d, le nombre d'annees, car
            # FGOALS-s2 n'as pas de donnees pour 2005, il n'a donc que 30 annees et
            # non 31 comme les autres.
            if mdlname == "FGOALS-s2(2004)" and DATAMDL == "raverage_1975_2005" :
                sst_ = np.concatenate((sst_, sst_[360-12:360]))
            #
            if Nmdlok == 1 :
                Tsst_102 = sst_;        
            else :
                Tsst_102 += sst_;
        #________________________________________________________
        TDmdl4CT.append(Dmdl);  # stockage des mod�les climatologiques 4CT pour AFC-CAH ...
        #
        Tmdlname.append(mdlname)
        Tmdlnamewnb.append("{:s}-{:s}".format(nomdl,mdlname))
        Tmdlonlynb.append("{:s}".format(nomdl))
        #
        #calcul de la perf glob du mod�le et stockage pour tri
        bmusM       = ctk.mbmus(sMapO, Data=Dmdl);
        classe_DMdl = class_ref[bmusM].reshape(NDmdl);
        perfglob    = ctobs.perfglobales([TypePerf[0]], classe_Dobs, classe_DMdl, nb_class)
        Tperfglob4Sort.append(perfglob[0])
        Tclasse_DMdl.append(classe_DMdl)
        #
        if OK106 : # Stockage (if required) pour la Courbes des moyennes mensuelles par classe
            Tmoymensclass.append(ctobs.moymensclass(sst_mdl_coded,isnumobs,classe_Dobs,nb_class)); ##!!?? 
    #
    # Fin de la PREMIERE boucle sur les mod�les
    #
    return TDmdl4CT,Tmdlname,Tmdlnamewnb,Tmdlonlynb,Tperfglob4Sort,Tclasse_DMdl,\
           Tmoymensclass,Dmdl,NDmdl,Nmdlok,Smoy_101,Tsst_102
#
def print_tb_models(mdlname,mdlonlynb,ncol_prnt=1):
    Nmodels = len(mdlname)
    for col in np.arange(ncol_prnt) :
        print(" {:->5s} {:->5s} {:->16s} ".format('','',''), end='')
    print("")
    for col in np.arange(ncol_prnt) :
        print(" {:5s} {:5s} {:16s} ".format('Order'.center(5,'-'),' No. '.center(5,'-'),' Model name '.center(16,'-')), end='')
    print("")
    for col in np.arange(ncol_prnt) :
        print(" {:->5s} {:->5s} {:->16s} ".format('','',''), end='')
    print("")
    for imodel in np.arange(Nmodels) :
        print(" {:5d} {:5s} {:16s} ".format(imodel+1,mdlonlynb[imodel].rjust(5),mdlname[imodel]), end='')
        if np.mod(imodel+1,ncol_prnt) == 0 or imodel == (Nmodels - 1):
            print("")
#
def do_models_plot101et102_past_fl(Nmdlok,Lobs,Cobs,isnumobs,isnanobs,
                              cmap='jet',varnames=None,
                              wvmin=None, wvmax=None,
                              ecvmin=None, ecvmax=None,
                              title101="Mdl MOY SST Mod",
                              title102="EcrtType SST Mod",
                              Smoy_101=None,
                              Tsst_102=None,
                              OK101='False',
                              OK102='False',
                              ) :
    #
    if OK101 :
        # Moyenne des mod�les en entr�e, moyenn�es
        # (Il devrait suffire de refaire la m�me chose pour Sall-cum)
        Smoy_ = Smoy_101 / Nmdlok; # Moyenne des moyennes cumul�es
        if wvmin is None :
            wvmin = np.nanmin(Smoy_)
        if wvmax is None : 
            wvmax = np.nanmax(Smoy_)
        #
        fig = plt.figure(101,figsize=(12,9))
        fignum = fig.number
        plt.clf() # efface une eventuelle figure existente 
        #
        ctobs.aff2D(Smoy_,Lobs,Cobs,isnumobs,isnanobs,
                    fignum=fignum,cmap=cmap,varnames=varnames,
                    wvmin=wvmin,wvmax=wvmax);
        #
        plt.suptitle("%s. Moyenne par mois et par pixel (Before Climatologie)\nmin=%f, max=%f, moy=%f, std=%f" \
                         %(title101,
                           np.nanmin(Smoy_),np.nanmax(Smoy_),np.nanmean(Smoy_),np.nanstd(Smoy_),));
    if OK102 :
        # Ecart type des mod�les en entr�e moyenn�es
        Tsst_ = Tsst_102 / Nmdlok; # Moyenne des cumuls des animalies (mod�le moyen des annomalies mais en fait avant climato))
        Dstd_, pipo_, pipo_  = ctobs.Dpixmoymens(Tsst_,stat='std'); # cliamtologie
        if ecvmin is None or ecvmin < 0 :
            ecvmin = np.nanmin(Dstd_)
        if ecvmax is None or ecvmin < 0 : 
            ecvmax = np.nanmax(Dstd_)
        #
        fig = plt.figure(102,figsize=(12,9))
        fignum = fig.number
        plt.clf() # efface une eventuelle figure existente 
        #
        ctobs.aff2D(Dstd_,Lobs,Cobs,isnumobs,isnanobs,
                    fignum=fignum,cmap=cmap,varnames=varnames,
                    wvmin=ecvmin,wvmax=ecvmax);
        #
        plt.suptitle("%s. \nEcarts Types par mois et par pixel (Before Climatologie)" \
                         %(title102));
        #
        del Smoy_, Tsst_, Dstd_, pipo_
        #plt.show(); sys.exit(0)
#
def do_models_next_first_loop(TDmdl4CT,Tmdlname,Tmdlnamewnb,Tmdlonlynb,Tperfglob4Sort,
                              Tclasse_DMdl,Tmoymensclass,
                              same_minmax_ok=True,
                              OK106='False',
                              ) :
    #_____________________________________________________________________
    ######################################################################
    # Reprise de la boucle (avec les mod�les valides du coup).
    # (question l'emplacement des mod�les sur les figures ne devrait pas etre un probl�me ?)
    #*****************************************
    #del Tmodels ##!!??
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
    # Not sorted model lists
    NSTmdlname    = np.copy(Tmdlname)
    NSTmdlnamewnb = np.copy(Tmdlnamewnb)
    NSTmdlonlynb  = np.copy(Tmdlonlynb)
    printwarning([ "    List of Models" ])
    print_tb_models(NSTmdlname,NSTmdlonlynb,ncol_prnt=3)
    #
    X1_ = np.copy(TDmdl4CT)
    X2_ = np.copy(Tmdlname)
    X3_ = np.copy(Tmdlnamewnb)
    X4_ = np.copy(Tmdlonlynb)
    X5_ = np.copy(Tclasse_DMdl)
    for i in np.arange(Nmodels) : # TDmdl4CT = TDmdl4CT[I_];
        TDmdl4CT[i]     = X1_[IS_[i]]
        Tmdlname[i]     = X2_[IS_[i]]  
        Tmdlnamewnb[i]  = X3_[IS_[i]]  
        Tmdlonlynb[i]   = X4_[IS_[i]]
        Tclasse_DMdl[i] = X5_[IS_[i]]
    del X1_, X2_, X3_, X4_, X5_;
    printwarning([ "    Reorderer List of Models" ])
    print_tb_models(Tmdlname,Tmdlonlynb,ncol_prnt=3)
    #
    min_moymensclass=None;
    max_moymensclass=None; 
    if OK106 :
        X1_ = np.copy(Tmoymensclass);
        for i in np.arange(Nmodels) :
            Tmoymensclass[i] = X1_[IS_[i]]
        del X1_   
        Tmoymensclass    = np.array(Tmoymensclass);
        min_moymensclass = np.nanmin(Tmoymensclass); ##!!??
        max_moymensclass = np.nanmax(Tmoymensclass); ##!!??
        delta_moymensclass = max_moymensclass - min_moymensclass;
        min_moymensclass -= delta_moymensclass/30; # diminue legerement le MIN pour que la figure ne toucha pas le bord
        max_moymensclass += delta_moymensclass/30; # aumente legerement le MAX pour que la figure ne toucha pas le bord
        if same_minmax_ok and min_moymensclass < 0:
            max_moymensclass = max(-1.0*min_moymensclass,max_moymensclass)
            min_moymensclass = -1.0*max_moymensclass
    #
    return TDmdl4CT,Tmdlname,Tmdlnamewnb,Tmdlonlynb,Tperfglob4Sort,Tclasse_DMdl,\
           Tmoymensclass,min_moymensclass,max_moymensclass
#
def do_models_pior_second_loop(TDmdl4CT,Tmdlname,Tmdlnamewnb,Tmdlonlynb,Tperfglob4Sort,
                    Tclasse_DMdl,Tmoymensclass,
                    MCUM,
                    same_minmax_ok=True,
                    OK106='False',
                    ):
    #*****************************************
    MaxPerfglob_Qm  = 0.0; # Utilis� pour savoir les quels premiers mod�les
    IMaxPerfglob_Qm = 0;   # prendre dans la strat�gie du "meilleur cumul moyen"
    #*****************************************
    # l'Init des figures � produire doit pouvoir etre plac� ici (sauf la 106) -----
    facecolor='w'
    #
    ######################################################################
    # Reprise de la boucle (avec les mod�les valides du coup).
    # (question l'emplacement des mod�les sur les figures ne devrait pas etre un probl�me ?)
    #*****************************************
    #del Tmodels ##!!??
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
    # Not sorted model lists
    NSTmdlname    = np.copy(Tmdlname)
    NSTmdlnamewnb = np.copy(Tmdlnamewnb)
    NSTmdlonlynb  = np.copy(Tmdlonlynb)
    printwarning([ "    List of Models" ])
    print_tb_models(NSTmdlname,NSTmdlonlynb,ncol_prnt=3)
    #
    X1_ = np.copy(TDmdl4CT)
    X2_ = np.copy(Tmdlname)
    X3_ = np.copy(Tmdlnamewnb)
    X4_ = np.copy(Tmdlonlynb)
    X5_ = np.copy(Tclasse_DMdl)
    for i in np.arange(Nmodels) : # TDmdl4CT = TDmdl4CT[I_];
        TDmdl4CT[i]     = X1_[IS_[i]]
        Tmdlname[i]     = X2_[IS_[i]]  
        Tmdlnamewnb[i]  = X3_[IS_[i]]  
        Tmdlonlynb[i]   = X4_[IS_[i]]
        Tclasse_DMdl[i] = X5_[IS_[i]]
    del X1_, X2_, X3_, X4_, X5_;
    printwarning([ "    Reorderer List of Models" ])
    print_tb_models(Tmdlname,Tmdlonlynb,ncol_prnt=3)
    #
    min_moymensclass=None;
    max_moymensclass=None; 
    if OK106 :
        X1_ = np.copy(Tmoymensclass);
        for i in np.arange(Nmodels) :
            Tmoymensclass[i] = X1_[IS_[i]]
        del X1_   
        Tmoymensclass    = np.array(Tmoymensclass);
        min_moymensclass = np.nanmin(Tmoymensclass); ##!!??
        max_moymensclass = np.nanmax(Tmoymensclass); ##!!??
        delta_moymensclass = max_moymensclass - min_moymensclass;
        min_moymensclass -= delta_moymensclass/30; # diminue legerement le MIN pour que la figure ne toucha pas le bord
        max_moymensclass += delta_moymensclass/30; # aumente legerement le MAX pour que la figure ne toucha pas le bord
        if same_minmax_ok and min_moymensclass < 0:
            max_moymensclass = max(-1.0*min_moymensclass,max_moymensclass)
            min_moymensclass = -1.0*max_moymensclass
    #
    # -----------------------------------------------------------------------------
    #
        return TDmdl4CT,Tmdlname,Tmdlnamewnb,Tmdlonlynb,Tperfglob4Sort,Tclasse_DMdl,\
               Tmoymensclass,min_moymensclass,max_moymensclass,\
               MaxPerfglob_Qm,IMaxPerfglob_Qm
#Dmdl_TVar,DMdl_Q,DMdl_Qm,Dmdl_TVm
def do_models_second_loop(sst_obs,Dobs,lon,lat,sMapO,XC_ogeo,TDmdl4CT,
                          Tmdlname,Tmdlnamewnb,Tmdlonlynb,
                          Tperfglob4Sort,Tclasse_DMdl,Tmoymensclass,
                          MaxPerfglob_Qm,IMaxPerfglob_Qm,
                          min_moymensclass,max_moymensclass,
                          MCUM,Lobs,Cobs,NDobs,NDmdl,
                          isnumobs,nb_class,class_ref,classe_Dobs,fond_C,
                          ccmap='jet',pcmap=None,sztitle=10,ysstitre=0.98,
                          ysuptitre=14,suptitlefs=0.98,
                          NIJ=0,
                          FONDTRANS="Obs",
                          TypePerf = ["MeanClassAccuracy"],
                          mdlnamewnumber_ok=False,
                          figsize=(7.5,12),
                          wspace=0.01, hspace=0.05, top=0.95, bottom=0.04, left=0.15, right=0.86,
                          OK104='False', OK105='False', OK106='False', OK107='False', OK108='False', OK109='False',
                          suptitle104='fig104',
                          suptitle105='fig105',
                          suptitle106='fig106',
                          suptitle107='fig107',
                          suptitle108='fig108',
                          suptitle109='fig109',
                          ):
    #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    #           DEUXIEME BOUCLE SUR LES MODELES START HERE
    #           DEUXIEME BOUCLE SUR LES MODELES START HERE
    #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    
    if pcmap is None :
        tmpcmap = cm.jet;       # Accent, Set1, Set1_r, gist_ncar; jet, ...
        # pour avoir des couleurs � peu pr�s equivalente pour les plots
        pcmap  = tmpcmap(np.arange(0,320,round(320/nb_class))); #ok?
        pcmap *= 0.95 # fonce les legerement tous les couleurs ...
        if tmpcmap.name == 'jet' and nb_class == 4:
            pcmap[2] *= 0.9 # fonce la couleur de la classe qui est trop claire(si ccmap = jet)
    #
    # Attention 'RED' ne marche pas ... ---------
    # on remet *Obs* = *obs*
    LObs     = Lobs
    CObs     = Cobs
    isnumObs = isnumobs
    XC_Ogeo  = XC_ogeo
    # -------------------------------------------
    #
    Nmodels = len(Tmdlname)
    #
    print("\n {:s}".format(" 2nd loop ".center(79,'o')));
    #
    facecolor = 'w'
    Dmdl_TVar=None; DMdl_Q=None; DMdl_Qm=None; Dmdl_TVm=None
    if OK104 : # Classification avec, "en transparance", les mals class�s
               # par rapport aux obs
        fig = plt.figure(104,figsize=figsize,facecolor=facecolor)
        plt.clf() # efface une eventuelle figure existente 
        plt.subplots_adjust(wspace=wspace,hspace=hspace,top=top,bottom=bottom,left=left,right=right)
    #                     
    if OK105 : #Classification
        plt.figure(105,figsize=figsize,facecolor=facecolor)
        plt.clf() # efface une eventuelle figure existente 
        plt.subplots_adjust(wspace=wspace,hspace=hspace,top=top,bottom=bottom,left=left,right=right)
    #                     
    if OK106 : # Courbes des moyennes mensuelles par classe
        plt.figure(106,figsize=figsize,facecolor=facecolor); # Moyennes mensuelles par classe
        plt.clf() # efface une eventuelle figure existente 
        plt.subplots_adjust(wspace=wspace+0.015,hspace=hspace,top=top,bottom=bottom+0.00,left=left+0.02,right=right-0.025)
    if OK107 : # Variance (not 'RED' compatible)
        plt.figure(107,figsize=figsize,facecolor=facecolor); # Moyennes mensuelles par classe
        plt.clf() # efface une eventuelle figure existente 
        Dmdl_TVar  = np.ones((Nmodels,NDobs))*np.nan; # Tableau des Variance par pixel sur climatologie
                       # J'utiliserais ainsi showimgdata pour avoir une colorbar commune
    if MCUM > 0 :
        # Moyenne CUMulative
        if OK108 : # Classification en Model Cumul� Moyen
            plt.figure(108,figsize=figsize,facecolor=facecolor); # Moyennes mensuelles par classe
            plt.clf() # efface une eventuelle figure existente 
            plt.subplots_adjust(wspace=wspace,hspace=hspace,top=top,bottom=bottom,left=left,right=right)
        #
        DMdl_Q  = np.zeros((NDmdl,12));  # f(mod�le climatologique Cumul�) #+
        DMdl_Qm = np.zeros((NDmdl,12));  # f(mod�le climatologique Cumul� moyen) #+
        #
        # Variance CUMulative
        if OK109 : # Variance sur les Models Cumul�s Moyens (not 'RED' compatible)
            plt.figure(109,figsize=figsize,facecolor=facecolor); # Moyennes mensuelles par classe
            plt.clf() # efface une eventuelle figure existente 
            #plt.subplots_adjust(wspace=wspace,hspace=hspace,top=top,bottom=bottom-0.12,left=left,right=right)
            Dmdl_TVm = np.ones((Nmodels,NDobs))*np.nan; # Tableau des Variance sur climatologie
                       # cumul�e, moyenn�e par pixel. J'utiliserais ainsi showimgdata pour avoir
                       # une colorbar commune
    #
    coches = np.arange(nb_class)+1;   # ex 6 classes : [1,2,3,4,5,6]
    ticks  = coches + 0.5;            # [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    bounds = np.arange(nb_class+1)+1; # pour bounds faut une fronti�re de plus [1, 2, 3, 4, 5, 6, 7]
    nsub   = Nmodels + 1; # actuellement au plus 48 mod�les + 1 pour les obs.      
    nbsubl, nbsubc = ldef.nsublc(nsub);
    nsubmax = nbsubl * nbsubc; # derniere casse subplot, pour les OBS
    if mdlnamewnumber_ok :
        Tmdlname10X = Tmdlnamewnb;
    else :
        Tmdlname10X = Tmdlname;
    isubplot = 0;
    Tperfglob_Qm     = np.zeros((Nmodels,)); # Tableau des Perf globales des mod�les cumulees
    Tperfglob        = np.zeros((Nmodels,len(TypePerf))); # Tableau des Perf globales des mod�les
    if NIJ==1 :
        TNIJ         = [];
    TTperf           = [];
    for imodel in np.arange(Nmodels) :
        isubplot=isubplot+1;
        #
        Dmdl    = TDmdl4CT[imodel];
        mdlname = Tmdlname10X[imodel];
        # 
        classe_DMdl = Tclasse_DMdl[imodel];
        XC_Mgeo     = ctobs.dto2d(classe_DMdl,Lobs,Cobs,isnumObs); # Classification g�ographique
        #
        #>>>>>>>>>>>
        classe_Dmdl = np.copy(classe_DMdl); # ... because RED ... du coup je duplique           pour avoir les memes
        XC_mgeo     = np.copy(XC_Mgeo);     # ... because RED ... du coup je duplique           noms de variables.
        #
        #if SIZE_REDUCTION == 'RED' :
        #    print("RED> %s"%(mdlname), np.shape(sst_mdl), len(isnumObs),len(classe_DMdl))
        #    pipo, XC_mgeo, classe_Dmdl, isnum_red = red_classgeo(sst_mdl,isnumObs,classe_DMdl,frl,tol,frc,toc); 
        #    print("<RED %s"%mdlname)
        #<<<<<<<<<<<
        #
        # Perf par classe
        classe_DD, Tperf = ctobs.perfbyclass(classe_Dobs,classe_Dmdl,nb_class);
        Tperf = np.round([i*100 for i in Tperf]).astype(int); 
        TTperf.append(Tperf); # !!! rem AFC est faite avec ca
        #
        # Perf globales 
        Perfglob             = Tperfglob4Sort[imodel];
        Tperfglob[imodel,0]  = Perfglob;
        Nperf_ = len(TypePerf)
        if Nperf_ > 1 :
            # On calcule les autres perf
            T_ = ctobs.perfglobales(TypePerf[1:Nperf_], classe_Dobs, classe_Dmdl, nb_class)
            Tperfglob[imodel,1:Nperf_] = T_
        #
        #Nmdl, Lmdl, Cmdl = np.shape(sst_mdl)
        NDmdl = len(classe_Dmdl); # needed for 108, ...? 
        #
        # l'AFC pourrait aussi etre faite avec �a
        if NIJ == 1 : # Nij = card|classes| ; Pourrait sembl� inappropri� car les classes
                      # peuvent g�ographiquement �tre n'importe o�, mais ...               
            Znij_ = [];
            for c in np.arange(nb_class) :
                imdlc = np.where(classe_Dmdl==c+1)[0]; # Indices des classes c du model
                Znij_.append(len(imdlc));
            TNIJ.append(Znij_);
        #
        #<:::------------------------------------------------------------------
        if OK104 : # Classification avec, "en transparance", les pixels mals
                   # class�s par rapport aux obs. (pour les mod�les les Perf
                   # par classe sont en colorbar)
            plt.figure(104); plt.subplot(nbsubl,nbsubc,isubplot);
            X_ = ctobs.dto2d(classe_DD,Lobs,Cobs,isnumobs); #X_= classgeo(sst_obs, classe_DD);
            plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
            if FONDTRANS == "Obs" :
                plt.imshow(XC_ogeo, interpolation='none', cmap=ccmap, alpha=0.2,vmin=1,vmax=nb_class);
            elif FONDTRANS == "Mdl" :
                plt.imshow(XC_mgeo, interpolation='none', cmap=ccmap, alpha=0.2,vmin=1,vmax=nb_class);
            plt.imshow(X_, interpolation='none', cmap=ccmap,vmin=1,vmax=nb_class);
            del X_
            plt.axis('off'); #grid(); # for easier check
            plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob,'%'),fontsize=sztitle,y=ysstitre); 
            hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
            hcb.set_ticklabels(Tperf);
            hcb.ax.tick_params(labelsize=8)
            #
        if OK105 : # Classification (pour les mod�les les Perf par classe sont en colorbar)
            plt.figure(105); plt.subplot(nbsubl,nbsubc,isubplot);
            plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
            plt.imshow(XC_mgeo, interpolation='none',cmap=ccmap, vmin=1,vmax=nb_class);
            hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
            hcb.set_ticklabels(Tperf);
            hcb.ax.tick_params(labelsize=8);
            plt.axis('off');
            #grid(); # for easier check
            plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob,'%'),fontsize=sztitle,y=ysstitre);
            #
        if OK106 : # Courbes des moyennes mensuelles par classe
            plt.figure(106); plt.subplot(nbsubl,nbsubc,isubplot);
            # trace l'axe des abscices (a Y=0) en noir
            plt.plot([-0.5,Tmoymensclass.shape[1]-1+0.5],[0,0],'-',color='k',linewidth=1.0);
            for i in np.arange(nb_class) :
                plt.plot(Tmoymensclass[imodel,:,i],'.-',color=pcmap[i]);
            plt.axis([-0.5, 11.5, min_moymensclass, max_moymensclass]);
            plt.xticks([]); #plt.axis('off');     
            plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob,'%'),fontsize=sztitle,y=ysstitre);
            #       
        if OK107 : # Variance (not 'RED' compatible)
            Dmdl_TVar[imodel] = np.var(Dmdl, axis=1, ddof=0);
            #
        if MCUM>0 and (OK108 or OK109) : # Cumul et moyenne
            DMdl_Q  = DMdl_Q + Dmdl;     # Cumul Zone
            DMdl_Qm = DMdl_Q / (imodel+1);   # Moyenne Zone ##!!?? 
            #
        if MCUM>0 and OK108 : # Classification en Model Cumul� Moyen (Perf par classe en colorbar)
            plt.figure(108); plt.subplot(nbsubl,nbsubc,isubplot);
            bmusMdl_Qm = ctk.mbmus (sMapO, Data=DMdl_Qm);
            classe_DMdl_Qm= class_ref[bmusMdl_Qm].reshape(NDmdl);
                           # Ici classe_D* correspond � un r�sultats de classification
                           # (bon ou mauvais ; donc sans nan)
            XC_mgeo_Qm    = ctobs.dto2d(classe_DMdl_Qm,Lobs,Cobs,isnumobs); # Classification g�ographique
                           # Mise sous forme 2D de classe_D*, en mettant nan pour les
                           # pixels mas class�s
            classe_DD_Qm, Tperf_Qm = ctobs.perfbyclass(classe_Dobs, classe_DMdl_Qm, nb_class);
            Perfglob_Qm = ctobs.perfglobales([TypePerf[0]], classe_Dobs, classe_DMdl_Qm, nb_class)[0];  
                           # Ici pour classe_DD* : les pixels bien class�s sont valoris�s avec
                           # leur classe, et les mals class�s ont nan
            Tperfglob_Qm[imodel] = Perfglob_Qm
            Tperf_Qm = np.round([i*100 for i in Tperf_Qm]).astype(int);
    
            plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
            plt.imshow(XC_mgeo_Qm, interpolation='none',cmap=ccmap, vmin=1,vmax=nb_class);
            hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
            hcb.set_ticklabels(Tperf_Qm);
            hcb.ax.tick_params(labelsize=8);
            plt.axis('off');
            #grid(); # for easier check
            plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob_Qm,'%'),fontsize=sztitle,y=ysstitre);
            #
            pgqm_ = np.round_(Perfglob_Qm*100)
            if pgqm_ >= MaxPerfglob_Qm :
                MaxPerfglob_Qm = pgqm_; # Utilis� pour savoir les quels premiers mod�les
                IMaxPerfglob_Qm = imodel+1;   # prendre dans la strat�gie du "meilleur cumul moyen"
                print(" New best cumul perf for {:d} models : {:.0f}% ...".format(imodel+1,pgqm_))
            else :
                print(".",end="")
         #
        if MCUM>0 and OK109 : # Variance sur les Models Cumul�s Moyens (not 'RED' compatible)
                              # Perf par classe en colorbar)
            Dmdl_TVm[imodel] = np.var(DMdl_Qm, axis=1, ddof=0);
    print("\nMaxPerfglob_Qm: {}% for {} model(s)".format(MaxPerfglob_Qm,IMaxPerfglob_Qm))
    #
    # Fin de la DEUXIEME boucle sur les mod�les
    #__________________________________________
    #
    # Les Obs � la fin 
    #isubplot = 49;
    isubplot = nsubmax
    #isubplot = isubplot + 1; # Michel (ou pas ?)
    if OK104 : # Obs for 104
        plt.figure(104); plt.subplot(nbsubl,nbsubc,isubplot);
        plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
        plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
        hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(coches);
        hcb.ax.tick_params(labelsize=8)
        plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle,y=ysstitre);
        if 0 :
            plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
            plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
        else :
            set_lonlat_ticks(lon,lat,step=4,fontsize=8,verbose=False,lengthen=True)
        #grid(); # for easier check
        plt.suptitle(suptitle104,fontsize=suptitlefs,y=ysuptitre)
    #
    if OK105 : # Obs for 105
        plt.figure(105); plt.subplot(nbsubl,nbsubc,isubplot);
        plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
        plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
        hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(coches);
        hcb.ax.tick_params(labelsize=8)
        plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle,y=ysstitre);
        if 0 :
            plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
            plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
        else :
            set_lonlat_ticks(lon,lat,step=4,fontsize=8,verbose=False,lengthen=True)
        #grid(); # for easier check
        plt.suptitle(suptitle105,fontsize=suptitlefs,y=ysuptitre)
    #
    if OK106 : # Obs for 106
        plt.figure(106);
        plt.subplot(nbsubl,nbsubc,isubplot);
        # plt.figure();plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.80)
        # trace l'axe des abscices (a Y=0) en noir
        plt.plot([-0.5,Tmoymensclass.shape[1]-1+0.5],[0,0],'-',color='k',linewidth=1.0);
        TmoymensclassObs = ctobs.moymensclass(sst_obs,isnumobs,classe_Dobs,nb_class);
        Legendline = []
        for i in np.arange(nb_class) :
            line = plt.plot(TmoymensclassObs[:,i],'.-',color=pcmap[i]);
            Legendline.append(line[0])
        plt.axis([-0.5, 11.5, min_moymensclass, max_moymensclass]);
        plt.xlabel('mois');
        plt.xticks(np.arange(12), np.arange(12)+1, fontsize=8)
        plt.legend(Legendline,np.arange(nb_class)+1,loc=2,fontsize=6,numpoints=1,bbox_to_anchor=(1.02, 1.0),title="Class");
        #plt.legend([x[0] for x in Legendline],np.arange(nb_class)+1,loc=2,fontsize=6,numpoints=1,bbox_to_anchor=(1.1, 1.0),title="Class");
        plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle,y=ysstitre);
        #
        # On repasse sur tous les supblots pour les mettre � la m�me echelle.
        print("min_moymensclass= {}\nmax_moymensclass= {}".format(min_moymensclass, max_moymensclass));
        plt.suptitle(suptitle106,fontsize=suptitlefs,y=ysuptitre)
    #
    if OK107 or OK109 : # Calcul de la variance des obs par pixel de la climatologie
        Tlabs = np.copy(Tmdlname10X);  
        Tlabs = np.append(Tlabs,'');                # Pour le subplot vide
        Tlabs = np.append(Tlabs,'Observations');    # Pour les Obs
        varobs= np.ones(Lobs*Cobs)*np.nan;          # Variances des ...
        varobs[isnumobs] = np.var(Dobs, axis=1, ddof=0); # ... Obs par pixel
    #
    if OK107 : # Variance par pixels des mod�les
        plt.figure(107);
        X_ = np.ones((Nmodels,Lobs*Cobs))*np.nan;
        X_[:,isnumobs] = Dmdl_TVar
        for iextra in np.arange(nsubmax - Nmodels - 1):
            print()
            # Rajouter nan pour le subplot vide
            X_ = np.concatenate((X_, np.ones((1,Lobs*Cobs))*np.nan))
        # Rajout de la variance des obs
        X_    = np.concatenate((X_, varobs.reshape(1,Lobs*Cobs)))
        print("X_.shape = {} ({})".format(X_.shape,nsubmax))
        #
        ldef.showimgdata(X_.reshape(nsubmax,1,Lobs,Cobs), Labels=Tlabs, n=nsubmax,fr=0,
                    vmin=np.nanmin(Dmdl_TVar),vmax=np.nanmax(Dmdl_TVar),fignum=107,
                    wspace=wspace,hspace=hspace+0.03,top=top,bottom=bottom,left=left,right=right,
                    );
#                    wspace=wspace,hspace=hspace+0.10,top=top-0.01,bottom=bottom-0.01,left=left+0.14,right=right-0.13,
        del X_
        plt.suptitle(suptitle107,fontsize=suptitlefs,y=ysuptitre);
    #
    if MCUM>0 and OK108 : # idem OK105, but ...
        plt.figure(108); plt.subplot(nbsubl,nbsubc,isubplot);
        plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
        plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
        hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(coches);
        hcb.ax.tick_params(labelsize=8)
        plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle,y=ysstitre);
        if 0 :
            plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
            plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
        else :
            set_lonlat_ticks(lon,lat,step=4,fontsize=8,verbose=False,lengthen=True)
        #grid(); # for easier check
        plt.suptitle(suptitle108,fontsize=suptitlefs,y=ysuptitre);
    #
    if MCUM>0 and OK109 : # Variance par pixels des moyenne des mod�les cumul�s
        plt.figure(109);
        X_ = np.ones((Nmodels,Lobs*Cobs))*np.nan;
        X_[:,isnumobs] = Dmdl_TVm
        # Rajouter nan pour le subplot vide
        X_ = np.concatenate(( X_, np.ones((1,Lobs*Cobs))*np.nan))
        # Rajout de la variance des obs
        X_ = np.concatenate((X_, varobs.reshape(1,Lobs*Cobs)))
        #
        ldef.showimgdata(X_.reshape(Nmodels+2,1,Lobs,Cobs), Labels=Tlabs, n=Nmodels+2,fr=0,
                    vmin=np.nanmin(Dmdl_TVm),vmax=np.nanmax(Dmdl_TVm),fignum=109,
                    wspace=wspace,hspace=hspace+0.03,top=top,bottom=bottom,left=left,right=right,
                    );
        del X_
        plt.suptitle(suptitle109,fontsize=suptitlefs,y=ysuptitre);
    #
    #___________________________________________
    # Mettre les Tableaux-Liste en tableau Numpy
    Tmdlname    = np.array(Tmdlname);
    Tmdlnamewnb = np.array(Tmdlnamewnb);
    Tmdlonlynb  = np.array(Tmdlonlynb);
    TTperf      = np.array(TTperf);
    TDmdl4CT    = np.array(TDmdl4CT);
    if NIJ == 1 :
        TNIJ    = np.array(TNIJ);
    #
    return Tperfglob,Tperfglob_Qm,Tmdlname,Tmdlnamewnb,Tmdlonlynb,TTperf,TDmdl4CT
#
def do_models_after_second_loop(Tperfglob,Tperfglob_Qm,Tmdlname,list_of_plot_colors,  
                                title="SST - Classification Indices of Completed Models",
                                TypePerf = ["MeanClassAccuracy"],
                                fcodage="",
                                ) :
    ##
    ##---------------------------------------------------------------------
    # Redimensionnement de Tperfglob au nombre de mod�les effectif
    Nmodels = Tperfglob.shape[0]
    Tperfglob = Tperfglob[0:Nmodels];
    #
    # Edition des r�sultats
    local_legend_labels = np.copy(TypePerf)
    local_legend_labels = np.concatenate((local_legend_labels,["Cumulated MeanClassAccuracy"]))
    fig = plt.figure(figsize=(12,6),facecolor='w');
    if len(Tperfglob.shape) > 1 :
        for icol in np.arange(len(Tperfglob.shape)) :
            plt.plot(100*Tperfglob[:,icol],'.-',color=list_of_plot_colors[icol]);
    kcol = len(Tperfglob.shape)
    plt.plot(100*Tperfglob_Qm,'.-',color=list_of_plot_colors[kcol]);
    plt.subplots_adjust(wspace=0.0, hspace=0.2, top=0.93, bottom=0.15, left=0.05, right=0.98)
    fignum = fig.number
    #plt.plot(Tperfglob,'.-');
    if 1:
        lax=plt.axis()
        plt.axis([lax[0],lax[1],0,100]); # axis fixex pour l'affichage d'un pourcentage
    else :
        plt.axis("tight");
    plt.grid(axis='both')
    plt.xticks(np.arange(Nmodels),Tmdlname, fontsize=8, rotation=45,
               horizontalalignment='right', verticalalignment='baseline');
    plt.ylabel('performance by Model [%]')
    plt.legend(local_legend_labels,numpoints=1,loc=3)
    plt.title(title);
    #
    #
    
#%%
if 0:
    #%=========================================================================
    if NIJ > 0 : # A.F.C
        #Ajout de l'indice dans le nom du mod�le
        Tm_ = np.empty(len(Tmdlname),dtype='<U32');
        for i in np.arange(Nmdlok) :
            if onlymdlumberAFC_ok : 
                Tm_[i] = Tmdlonlynb[i];
            elif mdlnamewnumber_ok :
                Tm_[i] = Tmdlnamewnb[i];
            else :
                Tm_[i] = str(i+1) + '-' +Tmdlname[i];
        #
        # Harmonaiser la variable servant de tableau de contingence (->Tp_), selon le cas
        if NIJ==1 : # Nij = card|classes| ; Pourrait sembl� inappropri� car les classes
                    # peuvent g�ographiquement �tre n'importe o�, mais ...
            Tp_ = TNIJ; # TNIJ dans ce cas a �t� pr�par� en amont
            #
        elif NIJ==2 or NIJ==3 :
            Tp_ = TTperf; # Pourcentages des biens class�s par classe (transform� ci apr�s
                          # pour NIJ==3 en effectif apr�s �ventuel ajout des obs dans l'afc)
        # On supprime les lignes dont la somme est 0 car l'afc n'aime pas ca.
        # (en esp�rant que la suite du code reste coh�rente !!!???)
        som_   = np.sum(Tp_, axis=1);
        Iok_   = np.where(som_>0)[0]; # Indice des mod�les valides pour l'AFC
        Tp_    = Tp_[Iok_];
        Nmdlok = len(Iok_); # !!! ATTENTION !!! red�finition du nombre de mod�les valides
        #  
        # Tableau (ou liste) des Noms des individus (Mod�les valides et Obs)
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
                # Obs have 100% for any class par d�finition
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
        else : # Les obs en suppl�mentaires
            VAPT, F1U, CAi, CRi, F2V, CAj, F1sU = afaco(Tp_, Xs=[Nobsc]);
            XoU = F1sU; # coord des Obs (ok j'aurais pu mettre directement en retour de fonction...)
        #
        #-----------------------------------------
        # MODELE MOYEN (pond�r� ou pas) PAR CLUSTER D'UNE CAH
        if 1 : # CAH on afc Models's coordinates (without obs !!!???)
            if SIZE_REDUCTION == 'All' :
                nticks = 5; # 4
            elif SIZE_REDUCTION == 'sel' :
                nticks = 2; # 4
            metho_ = 'ward'; #'complete' 'single' 'average' 'ward'; #<><><><><><<><><>
            dist_  = 'euclidean';
            coord2take = np.arange(NBCOORDAFC4CAH); # Coordonn�es de l'AFC �prendre pour la CAH
            if AFCWITHOBS :
                if CAHWITHOBS : # Garder les Obs pour la CAH
                    Z_ = linkage(F1U[:,coord2take], metho_, dist_);
                else : # Ne pas prendre les Obs dans la CAH (ne prendre que les mod�les)
                    Z_ = linkage(F1U[0:Nmdlok,coord2take], metho_, dist_);
                #
            else : # Cas AFC sans les Obs
                if CAHWITHOBS : # Alors rajouter les Obs en Suppl�mentaire
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
                L_ = np.array(NoCAHindnames) # when AFCWITHOBS, "Obs" � d�j� �t� rajout� � la fin
                plt.xticks((np.arange(Nleaves_)*10)+7,L_[R_['leaves']], fontsize=11,
                            rotation=45,horizontalalignment='right', verticalalignment='baseline')
                plt.title("AFC: Coord(%s), dendro. M�tho=%s, dist=%s, nb_clust=%d"
                          %((coord2take+1).astype(str),metho_,dist_,nb_clust))
            #
            if nb_clust < 0 :
                Loop_nb_clust = np.arange(-nb_clust-1)+2;   MultiLevel = True;
            else :
                Loop_nb_clust = np.array([nb_clust]);       MultiLevel = False;
            if max(Loop_nb_clust) > Nmdlok :
                print("Warning : You should not require more clusters level than the number of (valid) models");
            #subc_, subl_ = lcsub(max(Loop_nb_clust)); # <-- NON, utiliser plutot nl,nc = nsublc() qui est dans localdef.py
            subl_, subc_ = nsublc(max(Loop_nb_clust),nsubc=5);
            if MultiLevel == True :
                bestglob_ = 0.0;
                bestloc_  = []; # meilleure perf localement (i.e pour un niveau de coupe)
                ninbest_  = []; # nombre de mod�le concern�s par bestloc_
            for nb_clust in Loop_nb_clust :
                best_ = 0.0;
                class_afc = fcluster(Z_,nb_clust,'maxclust'); 
                if CAHWITHOBS : # On enl�ve le dernier indice de classe qui correspond
                    # aux obs (je crois !!!???) parce qu'on ne veut pas un mod�le moyen
                    # qui tienne compte des obs, ce serait tricher et donc faux
                    class_afc = class_afc[0:Nleaves_-1];
                #
                if MultiLevel == False : # Si il n'y a qu'un seul niveau de d�coupe, on fera la figure 
                    # Nouvelle figure:  performanes par cluster
                    if SIZE_REDUCTION == 'All' :
                        figsize = (4*subc_,2+2*subl_)
                        wspace=0.25; hspace=0.02; top=0.94; bottom=0.01; left=0.02; right=0.94;
                    elif SIZE_REDUCTION == 'sel' :
                        figsize = (3.5*subc_,1+3*subl_)
                        wspace=0.30; hspace=0.02; top=0.94; bottom=0.01; left=0.03; right=0.94;
                    figclustmoy = plt.figure(figsize=figsize,facecolor="w"); # pour les diff�rents cluster induit par ce niveau.
                    figclustmoy.subplots_adjust(wspace=wspace, hspace=hspace, top=top,
                                                bottom=bottom, left=left, right=right)
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
                            XCM_     = ctobs.dto2d(classej_,LObs,CObs,isnumObs); # Classification g�ographique
                            plt.subplot(8,6,jj+1); # plt.subplot(7,7,jj+1);
                            plt.imshow(XCM_, interpolation='none',cmap=ccmap, vmin=1,vmax=nb_class);
                            plt.axis('off');
                            plt.title("%s(%.0f%c)"%(CAHindnames[ijj],100*Tperfglob[ijj,0],'%'),
                                      fontsize=sztitle,y=ysstitre);
                                # m�me si y'a 'Obs' dans CAHindnames, ca devrait pas apparaitre
                                # car elles sont � la fin et ont �t� retir�es de class_afc
                        plt.suptitle("Classification des mod�les du cluster %d"%(ii+1));
                    #if MultiLevel == False :    
                    #    print("%d Mod�les du cluster %d, performance: %d :\n"%(len(iclust),ii+1,Perfglob_), CAHindnames[iclust]);
                    #
                    # Mod�le Moyen d'un cluster (plus de gestion de pond�ration)
                    CmdlMoy  = Dmdlmoy4CT(TDmdl4CT,iclust,pond=None);                
                    #
                    #if 1 : # Affichage Data cluster moyen for CT
                    if  ii+1 in AFC_Visu_Clust_Mdl_Moy_4CT :
                        aff2D(CmdlMoy,Lobs,Cobs,isnumobs,isnanobs,
                              wvmin=wvmin,wvmax=wvmax,
                              figsize=(12,9),cmap=eqcmap, varnames=varnames);
                        plt.suptitle("MdlMoy[%s]\nclust%d %s(%d-%d) for CT\nmin=%f, max=%f, moy=%f, std=%f"
                                %(Tmdlname[Iok_][iclust],ii+1,fcodage,andeb,anfin,np.min(CmdlMoy),
                                  np.max(CmdlMoy),np.mean(CmdlMoy),np.std(CmdlMoy)))    
                    #
                    # Classification du, des mod�les moyen d'un cluster
                    if MultiLevel : # Plusieurs niveaux de d�coupe, c'est pas la peine de faire toutes ces
                                    # figures, mais on a besoin de la perf
                        Perfglob_ = Dgeoclassif(sMapO,CmdlMoy,LObs,CObs,isnumObs,TypePerf[0],
                                                visu=False,nticks=nticks);
                    else : # 1 seul niveau de d�coupe, on fait la figure
                        plt.figure(figclustmoy.number)
                        ax = plt.subplot(subl_,subc_,ii+1);
                        Perfglob_ = Dgeoclassif(sMapO,CmdlMoy,LObs,CObs,isnumObs,TypePerf[0],
                                                ax = ax,
                                                cblabel="performance [%]",cblabelsize=8,
                                                cbticklabelsize=10,nticks=nticks);
                        plt.title("Cluster %d (%d mod.), mean perf=%.0f%c"%(ii+1,len(iclust),
                                                   100*Perfglob_,'%'),fontsize=12);
                    #
                    if MultiLevel :
                        if Perfglob_ > bestglob_ :
                            print("\n-> Cluster {:d}, {:d} Models, performance: {:.1f} :\n {}".format(
                                    ii+1,len(iclust),100*Perfglob_,Tmdlname[iclust]));
                            print("      >>>>>>>> clust %d-%d : new best perf = %f <<<<<<<<"
                                  %(nb_clust, ii+1, Perfglob_));
                            bestglob_ = Perfglob_
                        if Perfglob_ > best_ :
                            best_ = Perfglob_
                            nbest_ = len(iclust)
                    else :    
                        print("\n-> Cluster {:d}, {:d} Models, performance: {:.1f} :\n {}".format(
                                ii+1,len(iclust),100*Perfglob_,CAHindnames[iclust]));
                #        
                if MultiLevel :
                    bestloc_.append(best_)
                    ninbest_.append(nbest_)
                # FIN de la boucle sur le nombre de cluster
            # FIN de la boucle sur les diff�rents niveau de cluster
            if MultiLevel :
                plt.figure();
                plt.subplot(2,1,1); plt.plot(bestloc_,'-*');
                plt.xticks(np.arange(len(Loop_nb_clust)),Loop_nb_clust);
                plt.title("Meilleures perf pour chaque cas de d�coupe");
                
                plt.subplot(2,1,2); plt.plot(ninbest_,'-*');
                plt.xticks(np.arange(len(Loop_nb_clust)),Loop_nb_clust);
                plt.xlabel("nombre de d�coupe (clusters)");
                plt.title("Nombre de mod�les dans le meilleur cluster de la d�coupe");
                plt.suptitle("best cluster depending nb cluster");
            #
            del metho_, dist_, Z_
            if MultiLevel == True :
                plt.show(); sys.exit(0)
        # reprend la figure de performanes par cluster
        plt.figure(figclustmoy.number)
        plt.suptitle("AFC Clusters Class Performance ({} models)".format(Nmodels),fontsize=18);
        #plt.suptitle("AFC Clusters Class Performance ({:d} Classes)".format(nb_class),fontsize=18);
        if SAVEFIG : # sauvegarde de la figure de performanes par cluster
            plt.figure(figclustmoy.number)
            if Visu_UpwellArt :
                figfile = "FigArt_"
            else :
                figfile = "Fig_"
            figfile += "{:d}Clust-{:d}Classes_{:s}{:s}Clim-{:d}-{:d}_{:d}-mod".format(nb_clust,
                        nb_class,fprefixe,fshortcode,andeb,anfin,Nmodels)
            # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
            # eventuellement en PDF, si SAVEPDF active. 
            do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT,fig2ok=SAVEPDF,ext2=VFIGEXT)
    
    #%% pwd
    if NIJ > 0 : # A.F.C (suite)
        if 0 :
            plt.figure()
            plt.scatter(np.random.randn(1,100),np.random.randn(1,100),edgecolors='r',marker='o',linewidth=0.5)
        #
        # FIN du if 1 : MODELE MOYEN (pond�r� ou pas) PAR CLUSTER D'UNE CAH
        #-----------------------------------------
        #      
        # choisir K et son facteur de zoom
        #K=CAi; xoomK=3000; # Pour les contrib Abs (CAi)
        if Visu_AFC_in_one or Visu_UpwellArt: # plot afc en une seule image
            # 1- NOUVELLE FIGURE POUR PROJECTIONS DE L'AFC
            fig = plt.figure(figsize=(16,12));
            plt.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.95)
            ax = plt.subplot(111)
            fignum = fig.number # numero de figure en cours ...
            if Visu_UpwellArt :
                afcnuage(F1U,cpa=pa,cpb=po,Xcol=class_afc,linewidths=1.5,
                         indname=NoAFCindnames,
                         ax=ax,article_style=True,
                         marker='o',obsmarker='o',
                         markersize=60,obsmarkersize=75,
                         edgecolor='k',edgeobscolor='k',obscolor=[ 0.90, 0.90, 0.90, 1.],
                         edgeclasscolor='k',faceclasscolor='m',
                         lblfontsize=14,horizalign='left',vertalign='center',lblprefix=' ',
                         );
                # 3- AJOUT ou pas des colonnes (i.e. des classes)
                colnames = (np.arange(nb_class)+1).astype(str)
                afcnuage(F2V,cpa=pa,cpb=po,Xcol=np.arange(len(F2V)),
                         gridok=True,aximage=True,axtight=False,
                         linewidths=1.5,indname=colnames,holdon=True,drawtriangref=False,
                         ax=ax,article_style=True,
                         drawaxes=True, axescolors=('k','k'),
                         markersize=140,lblcolor='w',
                         lblfontsize=14,horizalign='center',vertalign='center',
                         )
                plt.title(("2 axes AFC of SST {:s} Projection with Models, Observations and Classes ({:s})"+\
                          "\n- {:s}, AFC on {:d} Classes ({} models+Obs) -").format(
                                  fcodage,DATAMDL,method_cah,nb_class,Nmodels),fontsize=14,y=1.02);
            else :
                K=CRi; xoomK=1000;  # Pour les contrib Rel (CRi)
                afcnuage(F1U,cpa=pa,cpb=po,Xcol=class_afc,K=K,xoomK=xoomK,linewidths=2,
                         indname=NoAFCindnames,
                         drawaxes=True, gridok=True,
                         ax=ax);
                if NIJ==1 :
                    plt.title("AFC Projection - %s SST (%s). %s%d AFC on classes of Completed Models (vs Obs)" \
                         %(fcodage,DATAMDL,method_cah,nb_class));
                elif NIJ==3 or NIJ==2 :
                    plt.title("AFC Projection - %s SST (%s). %s%d AFC on good classes of Completed Models (vs Obs)" \
                         %(fcodage,DATAMDL,method_cah,nb_class));
                #
                # 2- MET EN EVIDENCE LES OBS DANS LA FIGURE ...
                if AFCWITHOBS  :
                    ax.plot(F1U[Nmdlok,pa-1],F1U[Nmdlok,po-1], 'oc', markersize=20,
                            markerfacecolor='none',markeredgecolor='m',markeredgewidth=2);    
                else : # Obs en suppl�mentaire
                    ax.text(F1sU[0,0],F1sU[0,1], ".Obs")
                    ax.plot(F1sU[0,0],F1sU[0,1], 'oc', markersize=20,
                            markerfacecolor='none',markeredgecolor='m',markeredgewidth=2);
                #
                # 3- AJOUT ou pas des colonnes (i.e. des classes)
                colnames = (np.arange(nb_class)+1).astype(str)
                afcnuage(F2V,cpa=pa,cpb=po,Xcol=np.arange(len(F2V)),K=CAj,xoomK=xoomK,
                         linewidths=2,indname=colnames,holdon=True,drawtriangref=True,
                         ax=ax,drawaxes=True,aximage=True,axtight=False) 
            #plt.axis("tight"); #?
            if SAVEFIG : # sauvegarde de la figure
                if Visu_UpwellArt :
                    figfile = "FigArt_"
                else :
                    figfile = "Fig_"
                figfile += "AFC2DProj-{:d}-{:d}_{:d}Clust-{:d}Classes_{:s}{:s}Clim-{:d}-{:d}_{:d}-mod".format(
                        pa,po,nb_clust,nb_class,fprefixe,fshortcode,andeb,anfin,Nmodels)
                # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
                # eventuellement en PDF, si SAVEPDF active. 
                do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #%% pwd
    if NIJ > 0 : # A.F.C (suite bis)
        if Visu_afcnu_det : # plot afc etape par �tape"
            # Que les points lignes (mod�les)
            K=CRi; xoomK=1000
            fig = plt.figure(figsize=(12,8));
            plt.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.95)
            ax = plt.subplot(111)
            fignum = fig.number # numero de figure en cours ...
            afcnuage(F1U,cpa=pa,cpb=po,Xcol=class_afc,K=K,xoomK=xoomK,linewidths=2,indname=NoAFCindnames,
                     ax=ax);
            plt.title("%s SST (%s). \n%s%d AFC (nij=%d) of Completed Models (vs Obs)" \
                     %(fcodage,DATAMDL,method_cah,nb_class,NIJ));
            # Que les points colonnes (Classe)
            fig = plt.figure(figsize=(12,8));
            plt.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.95)
            ax = plt.subplot(111)
            afcnuage(F2V,cpa=pa,cpb=po,Xcol=np.arange(len(F2V)),K=CAj,xoomK=xoomK,
                     linewidths=2,indname=colnames,holdon=True,
                     ax=ax, gridok=True)
            plt.title("%s SST (%s). \n%s%d AFC (nij=%d) of Completed Models (vs Obs)" \
                     %(fcodage,DATAMDL,method_cah,nb_class,NIJ));          
        #
        if Visu_Inertie : # Inertie
            #inertie, icum = acp.phinertie(VAPT); 
            #if NIJ==1 :
            #    plt.title("%sSST(%s)). \n%s%d AFC on classes of Completed Models (vs Obs)" \
            #             %(fcodage,DATAMDL,method_cah,nb_class));
            #elif NIJ==3 :
            #    plt.title("%sSST(%s)). \n%s%d AFC on good classes of Completed Models (vs Obs)" \
            #             %(fcodage,DATAMDL,method_cah,nb_class));
            fig = plt.figure(figsize=(8,6));
            fignum = fig.number # numero de figure en cours ...
            plt.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.98)
            ax = plt.subplot(111)
            inertie, icum = acp.phinertie(VAPT,ax=ax,ygapbar=0.01, ygapcum=0.01); #print("inertie=:"); tls.tprin(inertie," %6.3f ")
            ax.grid(axis='y')
            if NIJ==1 :
                ax.set_title("%sSST(%s)) [%s]\n%s%d AFC on classes of Completed Models (vs Obs)" \
                         %(fcodage,DATAMDL,case_label,method_cah,nb_class));
            elif NIJ==2 :
                ax.set_title("%sSST(%s))\n%s%d AFC on good classes of Completed Models (vs Obs)" \
                         %(fcodage,DATAMDL,method_cah,nb_class));
            elif NIJ==3 :
                ax.set_title("%sSST(%s)) [%s]\n%s%d AFC on good classes of Completed Models (vs Obs)" \
                         %(fcodage,DATAMDL,case_label,method_cah,nb_class));
            if SAVEFIG : # sauvegarde de la figure de performanes par cluster
                plt.figure(figclustmoy.number)
                figfile = "Fig_"
                figfile += "Inertia-{:d}Clust-{:d}Classes_{:s}{:s}Clim-{:d}-{:d}".format(nb_clust,nb_class,fprefixe,fshortcode,andeb,anfin)
                # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
                # eventuellement en PDF, si SAVEPDF active. 
                do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    #%%======================================================================
    #
    if STOP_BEFORE_GENERAL :
        plt.show(); sys.exit(0)
    #**********************************************************************
    #............................. GENERALISATION .........................
    def mixtgeneralisation (TMixtMdl, lon=None, lat=None,
                            label=None, cblabel=None, fignum=None,
                            nticks=1,
                            ytitre=0.98, fsizetitre=14) :
        ''' Ici, j'ai :
                 - Tmdlname : une table de NOMS de N mod�les valides ; ##!!??? 
                 - TDmdl4ct : la table correspondante des mod�les 4CT (N, v,12)
            D'autre part, je vais d�finir d'une mani�re ou d'une autre les
            mod�les devant participer � la d�finition du mod�les moyen.
            Ici, il y a plusieurs possibilit�s :
            
            1) Une seule ligne de mod�les � utiliser sans dicernement de classe
               par exemple :
            TMixtMdl = ['ACCESS1-3','NorESM1-ME','bcc-csm1-1','NorESM1-M','CESM1-BGC']
            
            2) Plusieurs lignes de mod�le : une par classe, par exemple
            TMixtMdl = [['ACCESS1-3','NorESM1-ME','bcc-csm1-1','NorESM1-M','CESM1-BGC'],
                 ['IPSL-CM5B-LR','ACCESS1-3','MPI-ESM-P','CMCC-CMS','GISS-E2-R-CC'],
                 ['bcc-csm1-1-m','MIROC-ESM-CHEM','MIROC-ESM','CSIRO-Mk3-6-0','CanESM2'],
                 ['MPI-ESM-MR','CMCC-CM','IPSL-CM5A-MR','FGOALS-g2','MPI-ESM-LR'],
                 ['IPSL-CM5A-MR','CNRM-CM5-2','MPI-ESM-MR','MRI-ESM1','MRI-CGCM3'],
                 ['FGOALS-s2','CNRM-CM5','CNRM-CM5-2','GFDL-CM3','CMCC-CM'],
                 ['GFDL-CM3','GFDL-CM2p1','GFDL-ESM2G','CNRM-CM5','GFDL-ESM2M']];
    
            Dans les 2 cas, il faut :
            - Prendre les mod�les de TMixtMdl � condition qu'ils soient aussi
              dans Tmdlname
            - Envisager ou pas une 2�me phase ...  
        '''
        # Je commence par le plus simple : Une ligne de mod�le sans classe en une phase
        # Je prend le cas : CAH effectu�e sur les 6 coordonn�es d�une AFC  nij=3 ... 
        # TMixtMdl = ['CMCC-CM',   'MRI-ESM1',    'HadGEM2-AO','MRI-CGCM3',   'HadGEM2-ES',
        #             'HadGEM2-CC','FGOALS-g2',   'CMCC-CMS',  'GISS-E2-R-CC','IPSL-CM5B-LR',
        #             'GISS-E2-R', 'IPSL-CM5A-LR','FGOALS-s2', 'bcc-csm1-1'];
        #
        # d�terminer l'indice des mod�les de TMixtMdl dans Tmdlname
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
            print("\n%d mod�les disponibles (sur %d) pour la generalisation : %s"
                  %(len(IMixtMdl),len(TMixtMdl),Tmdlname[IMixtMdl]));
        #
        # Mod�le moyen
        MdlMoy = Dmdlmoy4CT(TDmdl4CT,IMixtMdl);
        #if 1 : # Affichage du moyen for CT
        #    aff2D(MdlMoy,Lobs,Cobs,isnumobs,isnanobs,
        #          wvmin=wvmin,wvmax=wvmax,figsize=(12,9),cmap=eqcmap,fignum=fignum);
        #    plt.suptitle("MdlMoy (%s) \n%s(%d-%d) for CT\nmin=%f, max=%f, moy=%f, std=%f"
        #                %(Tmdlname[IMixtMdl], fcodage,andeb,anfin,np.min(MdlMoy),
        #                 np.max(MdlMoy),np.mean(MdlMoy),np.std(MdlMoy)))
        #
        # Classification du mod�les moyen
        if fignum is None :
            fig = plt.figure();
            fignum = fig.number # numero de figure en cours ...
        else :
            fig = plt.figure(fignum);
            fig, ax = plt.subplots(nrows=1, ncols=1, num=fignum,facecolor='w')
        #
        #def Dgeoclassif(sMap,Data,L,C,isnum,MajorPerf,visu=True,cbticklabelsize=8,cblabel=None,
        #                cblabelsize=10,old=False,ax=None,nticks=1,tickfontsize=10) :
        Perfglob_ = Dgeoclassif(sMapO,MdlMoy,LObs,CObs,isnumObs,TypePerf[0],
                                cblabel="class performance [%]",
                                ax=ax,nticks=nticks,tickfontsize=10,
                                cbticklabelsize=12,cblabelsize=12); #use perfbyclass
        plt.xlabel('Longitude', fontsize=12); plt.ylabel('Latitude', fontsize=12)
        if label is None :
            titre = "MdlMoy ({} models: {})".format(len(Tmdlname[IMixtMdl]),Tmdlname[IMixtMdl])
        else :
            titre = "{} ({} models)".format(label,len(Tmdlname[IMixtMdl]))
        titre += ", mean perf={:.0f}%".format(100*Perfglob_)
        #
        plt.title(titre,fontsize=fsizetitre,y=ytitre);
        #tls.klavier();
        return MdlMoy, IMixtMdl
    #
    #%% ---------------------------------------------------------------------------
    # Mod�les Optimaux (Sopt) ;  Avec "NEW Obs - v3b " 1975-2005
    # Je commence par le plus simple : Une ligne de mod�le sans classe en une phase
    # et une seule codification � la fois
    # Sopt-1975-2005 : Les meilleurs mod�les de la p�riode "de r�f�rence" 1975-2005
    #TMixtMdl= [];
    #TMixtMdl =['CNRM-CM5', 'CMCC-CMS', 'CNRM-CM5-2', 'GFDL-CM3', 'FGOALS-s2']; 
    #TMixtMdl = Sfiltre;
    if 1 : # Best AFC Clusters
        if SIZE_REDUCTION == 'All' :
            # Grande Zone (All): BEST AFC CLUSTER:
            TMixtMdlLabel = 'Best AFC Cluster'
            TMixtMdl = ['CMCC-CM', 'HadGEM2-ES', 'HadGEM2-AO', 'HadGEM2-CC', 'CMCC-CMS',
                        'CNRM-CM5-2', 'CanESM2', 'CanCM4', 'GFDL-CM3', 'CNRM-CM5', 'FGOALS-s2(2004)', 
                        'CSIRO-Mk3-6-0', 'CMCC-CESM']
        elif SIZE_REDUCTION == 'sel' :
            # Petite Zone (sel): BEST AFC CLUSTER:
            TMixtMdlLabel = 'Best AFC Cluster'
            TMixtMdl = ['CNRM-CM5', 'CMCC-CMS', 'CNRM-CM5-2', 'GFDL-CM3', 'FGOALS-s2(2004)']
    elif 1 : # Best Cum Clusters Mopr
        if SIZE_REDUCTION == 'All' :
            # Grande Zone (All): BEST CUM GROUP OF MODELS:
            TMixtMdlLabel = 'Best Cumulated Models Group'
            TMixtMdl = ['CMCC-CM']
        elif SIZE_REDUCTION == 'sel' :
            # Petite Zone (sel): BEST CUM GROUP OF MODELS:
            TMixtMdlLabel = 'Best Cumulated Models Group'
            TMixtMdl = ['CanCM4', 'CNRM-CM5', 'CMCC-CMS', 'CNRM-CM5-2', 'GFDL-CM3', 'CanESM2', 'NorESM1-ME']
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
    #------------------------------------------------------------------------
    if SIZE_REDUCTION == 'All' :
        misttitlelabel = TMixtMdlLabel+" (Big Zone)"
    elif SIZE_REDUCTION == 'sel' :
        misttitlelabel = TMixtMdlLabel+" (Small Zone)"
    mistfilelabel = misttitlelabel.replace(' ','').replace('(','').replace(')','')
    #------------------------------------------------------------------------
    # PZ: BEST AFC CLUSTER:
    #TMixtMdl = []
    #
    if TMixtMdl == [] :
        print("\nSopt non renseign� ; Ce Cas n'a pas encore �t� pr�vu")
    else :
        if 1 :
            if SIZE_REDUCTION == 'All' :
                figsize = (9,6)
                wspace=0.0; hspace=0.0; top=0.94; bottom=0.08; left=0.06; right=0.925;
                nticks = 5; # 4
            elif SIZE_REDUCTION == 'sel' :
                figsize=(9,9)
                wspace=0.0; hspace=0.0; top=0.94; bottom=0.08; left=0.05; right=0.96;
                nticks = 2; # 4
            fig = plt.figure(figsize=figsize)
            fignum = fig.number # numero de figure en cours ...
            fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
        
            print("\n{:d}-model(s)' generalization: {} ".format(len(TMixtMdl),TMixtMdl))
            MdlMoy, IMixtMdl = mixtgeneralisation (TMixtMdl, label=misttitlelabel, fignum=fignum,
                                                   fsizetitre=14,ytitre=1.01,nticks=nticks);
            #
            if SAVEFIG : # sauvegarde de la figure
                if Visu_UpwellArt :
                    figfile = "FigArt_"
                else :
                    figfile = "Fig_"
                figfile += "MeanModel_{:s}-{:d}-mod_Mean".format(mistfilelabel,len(Tmdlname[IMixtMdl]))
                figfile += "_{:s}{:s}_{:d}Class".format(fprefixe,fshortcode,nb_class)
                do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT,fig2ok=SAVEPDF,ext2=VFIGEXT)
    
        if 1 : # Affichage du moyen for CT
            if SIZE_REDUCTION == 'All' :
                figsize = (12,7)
                wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.97;
                nticks = 5; # 4
            elif SIZE_REDUCTION == 'sel' :
                figsize=(10,8.5)
                wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.965;
                nticks = 2; # 4
            fig = plt.figure(figsize=figsize)
            fignum = fig.number # numero de figure en cours ...
            if Show_ModSTD and len(Tmdlname[IMixtMdl]) > 2:
                std_ = np.std(TDmdl4CT[IMixtMdl],axis=0);
                print(np.min(std_),np.max(std_),np.mean(std_),np.std(std_))
                aff2D(MdlMoy,Lobs,Cobs,isnumobs,isnanobs,
                      wvmin=wvmin,wvmax=wvmax,
                      fignum=fignum,varnames=varnames,cmap=eqcmap,
                      wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                      noaxes=False,noticks=False,nolabels=False,y=0.98,cblabel="SST Anomaly [�C]",
                      lolast=lolast,lonlat=(lon,lat),
                      vcontour=std_, ncontour=np.arange(0,2,1/10), ccontour='k', lblcontourok=True,
                      ); #...
            else :
                aff2D(MdlMoy,Lobs,Cobs,isnumobs,isnanobs,
                      wvmin=wvmin,wvmax=wvmax,
                      fignum=fignum,varnames=varnames,cmap=eqcmap,
                      wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                      noaxes=False,noticks=False,nolabels=False,y=0.98,cblabel="SST Anomaly [�C]",
                      lolast=lolast,lonlat=(lon,lat),
                      ); #...
            sptitre = "{}".format(misttitlelabel)
            if Show_ModSTD :
                if len(Tmdlname[IMixtMdl]) > 2 :
                    sptitre += " and inter-model STD in contours"
                else:
                    sptitre += " (not enouth models STD)"
            sptitre += " ({} models)".format(len(Tmdlname[IMixtMdl]))
            sptitre += ", mean perf={:.0f}%".format(100*Perfglob_)
            sptitre += "\nmin=%f, max=%f, moy=%f, std=%f"%(np.min(MdlMoy),np.max(MdlMoy),
                                                           np.mean(MdlMoy),np.std(MdlMoy))
            plt.suptitle(sptitre,fontsize=14,y=0.995);
            #
            if SAVEFIG : # sauvegarde de la figure
                figfile = "Fig_{:s}-{:d}-mod_Mean".format(mistfilelabel,len(Tmdlname[IMixtMdl]))
                if Show_ModSTD :
                    figfile += "+{:d}ySTD".format(Nda)
                figfile += "_Lim{:+.1f}-{:+.1f}_{:s}{:s}Clim-{:d}-{:d}".format(wvmin,wvmax,
                                fprefixe,fshortcode,andeb,anfin)
                # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
                # eventuellement en PDF, si SAVEPDF active. 
                do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT,fig2ok=SAVEPDF,ext2=VFIGEXT)
        #
    #
    #**********************************************************************
    plt.show();
    #___________
    print("\nWITHANO,UISST,climato,NIJ :\n", WITHANO, UISST,climato,NIJ)
    import os
    print("whole time code %s: %f" %(os.path.basename(sys.argv[0]), time()-tpgm0));
    #
    #======================================================================
    #
