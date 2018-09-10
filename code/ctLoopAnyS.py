# -*- coding: cp1252 -*-
# =============================================================================
# Programme ctLoopAnyS (Version pour Article)
#
#  ***************************************************************************
# ( VIEILLE VERSION, NE PAS UTILISER                                          )
# ( CE MODULE A ETE REMPLACE PAR CtLoopMain.py et CtLoopTools.py              )
#  ***************************************************************************
#
# =============================================================================

print("\n *** {} ***".format("".center(75,'*')))
print(" *** {} ***".format("".center(75)))
print(" *** {} ***".format("VIEILLE VERSION, NE PAS UTILISER".center(75)))
print(" *** {} ***".format("CE PROGRAMME/FICHIER A ETE REMPLACE PAR".center(75)))
print(" *** {} ***".format("ctLoopMain.py et ctLoopTools.py".center(75)))
print(" *** {} ***".format("".center(75)))
print(" *** {} ***\n".format("".center(75,'*')))
raise



import sys
import os
import time as time
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
import UW3_triedctk   as ctk
#
from   localdef    import *
from   ctObsMdldef import *
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
# pompé de WORKZONE ... TPA05
    if ax is None :
        fig = plt.figure(figsize=(16,12));
        ax = plt.subplot(111)
    else :
        fig = plt.gcf() # figure en cours ...
        ax = ax
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
        #? if np.ndim(K) > 1 : # On distingue triangle à droite ou vers le haut selon l'axe
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
            if p > 1 : # On distingue triangle à droite ou vers le haut selon l'axe
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
    
    if 0 : # je me rapelle plus tres bien à quoi ca sert; do we need a colorbar here ? may be
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

    # Plot en noir des triangles de référence en bas à gauche
    if drawtriangref :
        dx = xlim[1] - xlim[0];
        dy = ylim[1] - ylim[0];
        px = xlim[0] + dx/(xoomK) + dx/20; # à ajuster +|- en ...
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
    X_Mgeo_ = dto2d(classe_,L,C,isnum); # Classification géographique
    #plt.figure(); géré par l'appelant car ce peut être une fig déjà définie
    #et en subplot ... ou pas ...
    #classe_DD_, Tperf_, Perfglob_ = perfbyclass(classe_Dobs,classe_,nb_class,kperf=kperf);
    classe_DD_, Tperf_ = perfbyclass(classe_Dobs, classe_, nb_class);
    Perfglob_ = perfglobales([MajorPerf], classe_Dobs, classe_, nb_class)[0];
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
            print(" * {:74s} *".format(m)) 
    else:
        print(" * {:74s} *".format(msg))
    print(" * {:74s} *".format(" "))

def printwarning(msg, msg2=None, msg3=None):
    print("\n ******************************************************************************")
    _printwarning0(msg)
    if msg2 is not None :
        _printwarning0(msg2)
    if msg3 is not None :
        _printwarning0(msg3)
    print(" ******************************************************************************\n")
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
        # force la même valeur de pas dans les ticks x et y
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
#%% ###################################################################
# INITIALISATION
# Des trucs qui pourront servir
#######################################################################
# ferme toutes les fenetres de figures en cours
plt.close('all')
# Met a BLANC la couleur des valeurs masquées dans les masked_array,
# *** Cela change alors la couleur des NAN dans les fichiers EPS ***
#cmap.set_bad('w',1.)
#======================================================================
# Pour initialiser le generateur de nombres aleatoires utilise 
# Reset effectué juste avant l'initialisation de la Carte Topologique:
# (si tseed est diff de 0 alors il est ajouté dans le nom du cas)
tseed = 0;
#tseed = 9;
#tseed = np.long(time());
#tseed = np.long(np.mod(time()*1e6,1e3)); # un chiffre aleatoire entre 0 et 999
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
obs_data_path = '../Datas'
#######################################################################
#
# PARAMETRAGE (#1) DU CAS
from ParamCas import *
#
#======================================================================
if tseed == 0:  # si tseed est zero on ne fait rien
    case_name_base = case_label_base
else: # si tseed est different de zero on l'ajoute dans le nom du cas en cours
    case_name_base = "{:s}_s{:03d}".format(case_label_base,tseed)
#
print("\n{:*>86s}\nInitial case label: {}\n".format('',case_name_base))

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
if lon[0] > 180 : # ATTENTION, SI UN > 180 on considere TOUS > 180 ... 
    lon -= 360
# -----------------------------------------------------------------------------
# Complete le Nom du Cas
case_label = case_name_base+"_"+data_label_base
print("\n{:*>86s}\nCase label with data version: {}\n".format('',case_label))
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
print("\nData ({}x{}): {}".format(len(lat),len(lon),data_label_base))
print(" - Dir : {}".format(os.path.dirname(obs_filename)))
print(" - File: {}".format(os.path.basename(obs_filename)))
print(" - dimensions of SST Obs : {}".format(sst_obs.shape))
print(" - Lat : {} values from {} to {}".format(len(lat),lat[0],lat[-1]))
print(" - Lon : {} values from {} to {}".format(len(lon),lon[0],lon[-1]))
print("\nCurrent geographical limits ('to' limit excluded):")
print(" - Lat : from {} to {}".format(frlat,tolat))
print(" - Lon : from {} to {}".format(frlon,tolon))
Nobs,Lobs,Cobs = np.shape(sst_obs); print("obs.shape : ", Nobs,Lobs,Cobs);
#
# Paramétrage : _____________________________________
# Définition d'une zone plus petite
#
if 0: # OLD fashion
    if SIZE_REDUCTION == 'sel' or SIZE_REDUCTION == 'RED':
        frl = int(np.where(lat == frlat)[0]);
        tol = int(np.where(lat == tolat)[0]); # pour avoir 10.5, faut mettre 9.5
        frc = int(np.where(lon == frlon)[0]);
        toc = int(np.where(lon == tolon)[0]); # pour avoir 16.5, faut mettre 15.5
        #
        lat = lat[frl:tol];
        lon = lon[frc:toc];
else :
    # selectionne les LON et LAT selon les limites definies dans ParamCas.py
    # le fait pour tout cas de SIZE_REDUCTION, lat et lon il ne devraient pas
    # changer dans le cas de SIZE_REDUCTION=='All'
    ilat = np.intersect1d(np.where(lat <= frlat),np.where(lat > tolat))
    ilon = np.intersect1d(np.where(lon >= frlon),np.where(lon < tolon))
    lat = lat[np.intersect1d(np.where(lat <= frlat),np.where(lat > tolat))]
    lon = lon[np.intersect1d(np.where(lon >= frlon),np.where(lon < tolon))]
if 0: # OLD fashion
    if SIZE_REDUCTION == 'sel' :
        # Prendre d'entrée de jeu une zone plus petite
        sst_obs = sst_obs[:,frl:tol,frc:toc];
else :
    if SIZE_REDUCTION != 'RED' :
        # Prendre d'entrée de jeu une zone delimitee
        sst_obs = sst_obs[:,ilat,:];
        sst_obs = sst_obs[:,:,ilon];
        print("\nDefinitive data:")
        print(" - Dimensions of SST Obs : {}".format(sst_obs.shape))
        print(" - Lat : {} values from {} to {}".format(len(lat),lat[0],lat[-1]))
        print(" - Lon : {} values from {} to {}".format(len(lon),lon[0],lon[-1]))
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
    #wvmin = -4.9; wvmax = 4.9; # pour mettre tout le monde d'accord ?
    #wvmin = -4.0; wvmax = 4.0; # pour mettre tout le monde d'accord ?
    #wvmin = -3.0; wvmax = 3.0; # pour mettre tout le monde d'accord ?
    wvmin = -2.5; wvmax = 2.5; # pour mettre tout le monde d'accord ?
else : # On suppose qu'il s'agit du brute ...
    wvmin =16.0; wvmax =30.0; # ok pour obs 1975-2005 : SST 4CT: min=16.8666; max=29.029
#    
if Visu_ObsStuff : # Visu (et sauvegarde éventuelle de la figure) des données
    # telles qu'elles vont etre utilisées par la Carte Topologique
    minDobs = np.min(Dobs);   maxDobs=np.max(Dobs);
    moyDobs = np.mean(Dobs);  stdDobs=np.std(Dobs);
    #
    Dstd_, pipo_, pipo_  = Dpixmoymens(sst_obs, stat='std');
    #
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
    facecolor='w'
    fig = plt.figure(figsize=figsize,facecolor=facecolor)
    fignum = fig.number # numero de figure en cours ...
    localcmap = eqcmap
    if climato != "GRAD" :
        if Show_ObsSTD :
            aff2D(Dobs,Lobs,Cobs,isnumobs,isnanobs,
                  wvmin=wvmin,wvmax=wvmax,
                  fignum=fignum,varnames=varnames,cmap=localcmap,
                  wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                  noaxes=False,noticks=False,nolabels=False,y=0.98,cblabel="SST Anomaly [°C]",
                  lolast=lolast,lonlat=(lon,lat),
                  vcontour=Dstd_, ncontour=np.arange(0,1,1/20), ccontour='k', lblcontourok=True,
                  ); #...
        else:
            aff2D(Dobs,Lobs,Cobs,isnumobs,isnanobs,
                  wvmin=wvmin,wvmax=wvmax,
                  fignum=fignum,varnames=varnames,cmap=localcmap,
                  wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                  noaxes=False,noticks=False,nolabels=False,y=0.98,cblabel="SST Anomaly [°C]",
                  lolast=lolast,lonlat=(lon,lat),
                  ); #...
    else :
        aff2D(Dobs,Lobs,Cobs,isnumobs,isnanobs,
              wvmin=0.0,wvmax=0.042,
              fignum=fignum,varnames=varnames,cmap=localcmap, 
              wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
              noaxes=False,noticks=False,nolabels=False,y=0.98,cblabel="SST Anomaly [°C]",
              vcontour=Dstd_, ncontour=np.arange(0,1,1/20), ccontour='k', lblcontourok=True,
              lolast=lolast,lonlat=(lon,lat)); #...
    if Show_ObsSTD :
        plt.suptitle("Observed SST %s MEAN (%d-%d) (monthly %d years STD in contours)\nmin=%f, max=%f, mean=%f, std=%f"
                 %(fcodage,andeb,anfin,Nda,minDobs,maxDobs,moyDobs,stdDobs),y=0.995);
    else :
        plt.suptitle("Observed SST %s MEAN (%d-%d) '%s'\nmin=%f, max=%f, mean=%f, std=%f"
                 %(fcodage,andeb,anfin,localcmap.name,minDobs,maxDobs,moyDobs,stdDobs),y=0.995);
    #
    if SAVEFIG : # sauvegarde de la figure
        figfile = "Fig_Obs4CT"
        if Show_ObsSTD :
            figfile += "+{:d}ySTD".format(Nda)
        figfile += "_Lim{:+.1f}-{:+.1f}_{:s}_{:s}{:s}Clim-{:d}-{:d}_{:s}".format(wvmin,wvmax,localcmap.name,fprefixe,fshortcode,andeb,anfin,data_label_base)
        # sauvegarde en format FIGFMT (normalement BITMAP (png,jpg)) et
        # eventuellement en PDF, si SAVEPDF active. 
        do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
    #plt.show(); sys.exit(0);
    #
    del Dstd_, pipo_
#%%
if 0 :
    # Figure FREE LIMITS
    localcmap = eqcmap
    ND,p      = np.shape(Dobs);
    X_        = np.empty((Lobs*Cobs,p));   
    X_[isnumobs] = Dobs   
    X_[isnanobs] = np.nan
    X = X_.T.reshape(p,1,Lobs,Cobs)
    
    if SIZE_REDUCTION == 'All' :
        figsize = (12,7)
        wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.97;
    elif SIZE_REDUCTION == 'sel' :
        figsize=(10,8.5)
        wspace=0.04; hspace=0.12; top=0.925; bottom=0.035; left=0.035; right=0.965;
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
    plt.suptitle("Observed SST %s MEAN (%d-%d) '%s' FREE LIMITS\nmin=%f, max=%f, mean=%f, std=%f"
             %(fcodage,andeb,anfin,localcmap.name,minDobs,maxDobs,moyDobs,stdDobs),y=0.995);
    figfile = "Fig_Obs4CT_FREELIMITS_{:s}_{:s}{:s}Clim-{:d}-{:d}_{:s}".format(localcmap.name,fprefixe,fshortcode,andeb,anfin,data_label_base)
    do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
#######################################################################
#
if STOP_BEFORE_CT :
    plt.show(); sys.exit(0);
#%% ###################################################################
#                       Carte Topologique
#======================================================================
DO_NEXT = True
if SAVEMAP : # SI sauvegarde de la Map de SOM est ACTIVE
    mapfile = "Map_{:s}{:s}Clim-{:d}-{:d}_{:s}_ts-{}{}".format(fprefixe,fshortcode,
                   andeb,anfin,data_label_base,tseed,mapfileext)
    mapPathAndFile = case_maps_dir+os.sep+mapfile
    if os.path.exists(mapPathAndFile) and not REWRITEMAP :
        printwarning([ u"Attention, le fichier MAP existe déjà, ",
                       "    {}/".format(os.path.dirname(mapPathAndFile)),
                       "         {}".format(os.path.basename(mapPathAndFile)),
                       "",
                       u"on saute le processus d'entrainemant de la MAP." ],
                     u"Activez REWRITEMAP pour reecrire.")
        DO_NEXT = False
if DO_NEXT :
    print("Initializing random generator with seed={}".format(tseed))
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
    ttrain0 = time();
    eqO = sMapO.train(etape1=etape1,etape2=etape2, verbose='on', retqerrflg=True);
    print("Training elapsed time {:.4f}s".format(time()-ttrain0));
    # + err topo maison
    bmus2O = ctk.mbmus (sMapO, Data=None, narg=2);
    etO    = ctk.errtopo(sMapO, bmus2O); # dans le cas 'rect' uniquement
    somtime = casetime
    print("Two phases training executed:")
    print(" - Phase 1: {0[0]} epochs for radius variing from {0[1]} to {0[2]}".format(Parm_app))
    print(" - Phase 2: {0[3]} epochs for radius variing from {0[4]} to {0[5]}".format(Parm_app))
    print("Obs case: {}\n          date ... {}]\n          tseed={}\n          Qerr={:8.6f} ... Terr={:.6f}".format(case_label,
          casetimelabel,tseed,eqO,etO))
    if SAVEMAP : # sauvegarde de la Map de SOM
        printwarning([ "==> Saving MAP in file :",
                       "    {}/".format(os.path.dirname(mapPathAndFile)),
                       "         {}".format(os.path.basename(mapPathAndFile)) ])
        map_d ={ "map" : sMapO, "tseed" : tseed, "somtime" : somtime }
        map_f = open(mapPathAndFile, 'wb')
        pickle.dump(map_d, map_f)
        map_f.close()
elif os.path.exists(mapPathAndFile) and RELOADMAP :
        #reload object from file
        printwarning([ "==> Loading MAP from file :",
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

del DO_NEXT
#
#%%
# Visualisation______________________________________
if Visu_CTStuff : #==>> la U_matrix
    fig = plt.figure(figsize=(6,8));
    fignum = fig.number
    #wspace=0.02; hspace=0.14; top=0.80; bottom=0.02; left=0.01; right=0.95;
    #fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
    a=sMapO.view_U_matrix(fignum=fignum, distance2=2, row_normalized='No', show_data='Yes', \
                      contooor='Yes', blob='No', save='No', save_dir='');
    plt.suptitle("Obs, The U-MATRIX", fontsize=16,y=1.0);
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
bmusO     = ctk.mbmus (sMapO, Data=Dobs); # déjà vu ? conditionnellement ?
minref    = np.min(sMapO.codebook);
maxref    = np.max(sMapO.codebook);
Z_        = linkage(sMapO.codebook, method_cah, dist_cah);
class_ref = fcluster(Z_,nb_class,'maxclust'); # Classes des referents
#
if Visu_Dendro :
    fig = plt.figure(figsize=(14,6),facecolor='w');
    fignum = fig.number # numero de figure en cours ...
    Ncell=np.int(np.prod(sMapO.mapsize))
    max_d = np.sum(Z_[[-nb_class+1,-nb_class],2])/2
    color_threshold = max_d
    if 1:
        plt.subplots_adjust(wspace=0.0, hspace=0.2, top=0.92, bottom=0.12, left=0.05, right=0.99)
        #R_ = dendrogram(Z_,sMapO.dlen,'lastp');
        #dendrogram(Z_,sMapO.dlen,'lastp');
        R_ = dendrogram(Z_,p=Ncell,truncate_mode=None,color_threshold=color_threshold,
                        orientation='top',leaf_font_size=6) #,labels=lignames
        #               leaf_rotation=45);
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
        plt.ylabel("inter class distance ({})".format(method_cah), fontsize=12)
        lax=plt.axis(); daxy=(lax[3]-lax[2])/400
        plt.axis([lax[0],lax[1],lax[2]-daxy,lax[3]])
        plt.title("SOM Codebook Dendrogram for HAC (map size={:d}x{:d}, {:s}, nb_class={:d})".format(nbl,nbc,method_cah,nb_class));
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
    if 0: # OLD fashion
        # using frl,tol,frc,toc
        sst_obs, XC_ogeo, classe_Dobs, isnumobs = red_classgeo(sst_Obs,isnumObs,classe_DObs,frl,tol,frc,toc);
    else :
        # using ilon,ilat
        sst_obs, XC_ogeo, classe_Dobs, isnumobs = red_classgeo(sst_Obs,isnumObs,classe_DObs,ix=ilon,iy=ilat);
        Wsst_obs, WXC_ogeo, Wclasse_Dobs, Wisnumobs = red_classgeo(sst_Obs,isnumObs,classe_DObs,ix=ilon,iy=ilat);
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
# Figure 1 pour Article 
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
if Visu_ObsStuff or Visu_UpwellArt : # Visualisation de truc liés au Obs
    # Classification
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
    if lat[0] < lat[1] :
        origin = 'lower'
    else :
        origin = 'upper'
    fig, ax = plt.subplots(nrows=1, ncols=1, num=fignum,facecolor='w')
    ims = ax.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class,origin=origin);
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
    plt.title("Observations ({:d}-{:d}), {} Class Geographical Representation".format(andeb,anfin,nb_class),fontsize=16); 
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
        if Visu_UpwellArt :
            figfile = "FigArt_"
        else :
            figfile = "Fig_"
        figfile += "ObsGeo{:d}Class_{:s}{:s}Clim-{:d}-{:d}_{:s}".format(nb_class,fprefixe,fshortcode,andeb,anfin,data_label_base)
        do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
#
if Visu_ObsStuff or Visu_UpwellArt : # Visualisation de truc liés au Obs
    # Courbes des Moyennes Mensuelles par Classe
    fig = plt.figure(figsize=(12,6));
    fignum = fig.number # numero de figure en cours ...
    TmoymensclassObs = moymensclass(sst_obs,isnumobs,classe_Dobs,nb_class)
    #plt.plot(TmoymensclassObs); plt.axis('tight');
    for i in np.arange(nb_class) :
        plt.plot(TmoymensclassObs[:,i],'.-',color=pcmap[i]);
    plt.grid(axis='y')
    plt.axis('tight');
    plt.xticks(np.arange(12),varnames)
    plt.ylabel('Mean SST Anomaly [°C]', fontsize=12);
    plt.xlabel('Month', fontsize=12);
    legax=plt.legend(np.arange(nb_class)+1,loc=2,fontsize=10);
    legax.set_title('Class')
    plt.title("Observations, Monthly Mean by Class (method: %s)"%(method_cah),fontsize=16); #,fontweigth='bold');
    #plt.show(); sys.exit(0)
    if SAVEFIG : # sauvegarde de la figure
        if Visu_UpwellArt :
            figfile = "FigArt_"
        else :
            figfile = "Fig_"
        figfile += "MeanByClass_{:d}Class_{:s}{:s}Clim-{:d}-{:d}_{:s}".format(nb_class,fprefixe,fshortcode,andeb,anfin,data_label_base)
        do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT,fig2ok=SAVEPDF,ext2=VFIGEXT)
#
if Visu_CTStuff : # Visu des profils des référents de la carte
    if SIZE_REDUCTION == 'All' :
        figsize = (7.5,12)
        wspace=0.01; hspace=0.05; top=0.945; bottom=0.04; left=0.15; right=0.86;
    elif SIZE_REDUCTION == 'sel' :
        figsize=(8,8)
        wspace=0.01; hspace=0.04; top=0.945; bottom=0.04; left=0.04; right=0.97;
    fig = plt.figure(figsize=figsize)
    fignum = fig.number
    #wspace=0.015; hspace=0.04; top=0.92; bottom=0.05; left=0.05; right=0.98;
    fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
    ctk.showprofils(sMapO, fignum=fignum, Data=Dobs,
                    visu=3, scale=2,Clevel=class_ref-1,Gscale=0.5,
                    axsztext=6,marker='.',markrsz=4,pltcolor='r',
                    ColorClass=pcmap,ticklabels=varnames,verbose=False);
    plt.suptitle("SOM Map Profils by Cell (background color represents classes)",fontsize=16); #,fontweigth='bold');
#
#######################################################################
#
#raise
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
#def lcsub(nsub,minncol=None) :   # <-- NON, utiliser plutot nl,nc = nsublc() qui est dans localdef.py
#    if minncol is not None:
#        nbsubc = minncol
#    else :
#        nbsubc = np.ceil(np.sqrt(nsub));
#    nbsubl = np.ceil(1.0*nsub/nbsubc).as;
#    return nbsubc, nbsubl
#nbsubc, nbsubl = lcsub(nsub);
nbsubl, nbsubc = nsublc(nsub);
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
Tmdlnamewnb      = []; # Table des modèles avec numero
Tmdlonlynb       = []; # Table des modèles avec seulement le numero
Tmoymensclass    = [];
#
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
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#           PREMIERE BOUCLE SUR LES MODELES START HERE
#           PREMIERE BOUCLE SUR LES MODELES START HERE
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
print("\nooooooooooooooooooooooooooooo first loop ooooooooooooooooooooooooooooo");
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
    nomdl    = Tnmodel[imodel];   # numero de modele
    instname = Tinstit[imodel];   # institution du modele
    mdlname  = Tmodels[imodel,0]; # nom du modele
    anstart  = Tmodels[imodel,1]; # (utile pour rmean seulement)
    #
    # >>> Filtre (selection)de modèles en entrée ; Mettre 0 dans le if pour ne pas filtrer
    if 0 and mdlname not in Sfiltre :
        continue;
    #print(" using model '{}' ...".format(mdlname))
    # <<<<< 
    #______________________________________________________
    # Lecture des données (fichiers.mat générés par Carlos)
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
        print(" using model '{}'' ...".format(mdlname))
    
    try :
        sst_mat = scipy.io.loadmat(mdl_filename);
    except :
        print(" ** model '{}' not found **".format(mdlname));
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
    #________________________________________________`
    if 0: # OLD fashion
        if SIZE_REDUCTION=='sel' : # Prendre une zone plus petite
            sst_mdl = sst_mdl[:,frl:tol,frc:toc];
    else :
        if SIZE_REDUCTION != 'RED' :
            # Prendre d'entrée de jeu une zone delimitee
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
        # correction manuelle pour FGOALS-s2 (1975_2005) pour avoir le même nombre
        # de donnees que pour les autres modeles (c-a-d, le nombre d'annees, car
        # FGOALS-s2 n'as pas de donnees pour 2005, il n'a donc que 30 annees et
        # non 31 comme les autres.
        if mdlname == "FGOALS-s2(2004)" and DATAMDL == "raverage_1975_2005" :
            sst_ = np.concatenate((sst_, sst_[360-12:360]))
        #
        if Nmdlok == 1 :
            Tsst_ = sst_;        
        else :
            Tsst_ = Tsst_ + sst_;
    #________________________________________________________
    TDmdl4CT.append(Dmdl);  # stockage des modèles climatologiques 4CT pour AFC-CAH ...
    #
    Tmdlname.append(mdlname)
    Tmdlnamewnb.append("{:s}-{:s}".format(nomdl,mdlname))
    Tmdlonlynb.append("{:s}".format(nomdl))
    #
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
if mdlnamewnumber_ok :
    Tmdlname10X = Tmdlnamewnb;
else :
    Tmdlname10X = Tmdlname;
#____________________________
if OK101 :
    if mdlnamewnumber_ok :
        Tmdlname10X = Tmdlnamewnb;
    else :
        Tmdlname10X = Tmdlname;
    #
    if len(Tmdlname) > 6 :            # Sous forme de liste, la liste des noms de modèles
        Tnames_ = Tmdlname10X;           # n'est pas coupé dans l'affichage du titre de la figure
    else :                            # par contre il l'est sous forme d'array; selon le cas, ou 
        Tnames_ = np.array(Tmdlname10X); # le nombre de modèles, il faut adapter comme on peut
    #
    # Moyenne des modèles en entrée, moyennées
    # (Il devrait suffire de refaire la même chose pour Sall-cum)
    Smoy_ = Smoy_ / Nmdlok; # Moyenne des moyennes cumulées
    aff2D(Smoy_,Lobs,Cobs,isnumobs,isnanobs,
          figsize=(12,9),cmap=eqcmap,varnames=varnames,
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
        aff2D(Dstd_,Lobs,Cobs,isnumobs,isnanobs, figsize=(12,9),cmap=eqcmap,varnames=varnames,
              wvmin=ecvmin,wvmax=ecvmax);
    else :
        aff2D(Dstd_,Lobs,Cobs,isnumobs,isnanobs,
              figsize=(12,9),cmap=eqcmap,varnames=varnames,
              wvmin=np.nanmin(Dstd_),wvmax=np.nanmax(Dstd_));
    #plt.suptitle("MCUMMOY%s\n%sSST(%s)). \nEcarts Types par mois et par pixel (Before Climatologie)" \
    plt.suptitle("Mdl_MOY%s\n%sSST(%s)). \nEcarts Types par mois et par pixel (Before Climatologie)" \
                     %(Tnames_,fcodage,DATAMDL));
    #
    del Smoy_, Tsst_, Dstd_, Tnames_, pipo_, Tmdlname10X
    #plt.show(); sys.exit(0)
#_____________________________________________________________________
######################################################################
# Reprise de la boucle (avec les modèles valides du coup).
# (question l'emplacement des modèles sur les figures ne devrait pas etre un problème ?)
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
print(" {:5s} {:s}".format('No.','Model name'))
print(" {:->5s} {:->15s}".format('',''))
for imodel in np.arange(Nmodels) :
    print(" {:5s} {:s}".format(NSTmdlonlynb[imodel],NSTmdlname[imodel]))
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
#*****************************************
MaxPerfglob_Qm  = 0.0; # Utilisé pour savoir les quels premiers modèles
IMaxPerfglob_Qm = 0;   # prendre dans la stratégie du "meilleur cumul moyen"
#*****************************************
# l'Init des figures à produire doit pouvoir etre placé ici (sauf la 106) -----
if SIZE_REDUCTION == 'All' :
    figsize10X=(18,11);
    wspace10X=0.01; hspace10X=0.15; top10X=0.94; bottom10X=0.05; left10X=0.01; right10X=0.99;
elif SIZE_REDUCTION == 'sel' :
    figsize10X=(18,12);
    wspace10X=0.01; hspace10X=0.14; top10X=0.94; bottom10X=0.04; left10X=0.01; right10X=0.99;
ysstitre10X = ysstitre
suptitlefs10X = 16
ysuptitre10X = 0.99
facecolor10X='w'
# -----------------------------------------------------------------------------
if OK104 : # Classification avec, "en transparance", les mals classés
           # par rapport aux obs
    plt.figure(104,figsize=figsize10X,facecolor=facecolor10X)
    plt.subplots_adjust(wspace=wspace10X,hspace=hspace10X,top=top10X,bottom=bottom10X,left=left10X,right=right10X)
    suptitle104="%sSST(%s)). %s Classification of Completed Models (vs Obs) (%d models)" \
                 %(fcodage,DATAMDL,method_cah,Nmodels);
if OK105 : #Classification
    plt.figure(105,figsize=figsize10X,facecolor=facecolor10X)
    plt.subplots_adjust(wspace=wspace10X,hspace=hspace10X,top=top10X,bottom=bottom10X,left=left10X,right=right10X)
    suptitle105="%sSST(%s)). %s Classification of Completed Models (vs Obs) (%d models)" \
                 %(fcodage,DATAMDL,method_cah,Nmodels);
if OK106 : # Courbes des moyennes mensuelles par classe
    plt.figure(106,figsize=figsize10X,facecolor=facecolor10X); # Moyennes mensuelles par classe
    plt.subplots_adjust(wspace=wspace10X+0.015,hspace=hspace10X,top=top10X,bottom=bottom10X+0.00,left=left10X+0.02,right=right10X-0.025)
    if 0 : #MICHEL
        suptitle106="MoyMensClass(%sSST(%s)).) %s Classification of Completed Models (vs Obs) (%d models)" \
                     %(fcodage,DATAMDL,method_cah,Nmodels);
    else : # pas MICHEL
        suptitle106="MOY - MoyMensClass(%sSST(%s)).) %s Classification of Completed Models (vs Obs) (%d models)" \
                     %(fcodage,DATAMDL,method_cah,Nmodels);
if OK107 : # Variance (not 'RED' compatible)
    plt.figure(107,figsize=figsize10X,facecolor=facecolor10X); # Moyennes mensuelles par classe
    #plt.subplots_adjust(wspace=wspace10X,hspace=hspace10X,top=top10X,bottom=bottom10X-0.12,left=left10X,right=right10X)
    suptitle107="VARiance(%sSST(%s)).) Variance (by pixel) on Completed Models (%d models)" \
                 %(fcodage,DATAMDL,Nmodels);
    Dmdl_TVar  = np.ones((Nmodels,NDobs))*np.nan; # Tableau des Variance par pixel sur climatologie
                   # J'utiliserais ainsi showimgdata pour avoir une colorbar commune
if MCUM > 0 :
    # Moyenne CUMulative
    if OK108 : # Classification en Model Cumulé Moyen
        plt.figure(108,figsize=figsize10X,facecolor=facecolor10X); # Moyennes mensuelles par classe
        plt.subplots_adjust(wspace=wspace10X,hspace=hspace10X,top=top10X,bottom=bottom10X,left=left10X,right=right10X)
        suptitle108="MCUM - %sSST(%s)). %s Classification of Completed Models (vs Obs) (%d models)" \
                     %(fcodage,DATAMDL,method_cah,Nmodels);
    #
    DMdl_Q  = np.zeros((NDmdl,12));  # f(modèle climatologique Cumulé) #+
    DMdl_Qm = np.zeros((NDmdl,12));  # f(modèle climatologique Cumulé moyen) #+
    #
    # Variance CUMulative
    if OK109 : # Variance sur les Models Cumulés Moyens (not 'RED' compatible)
        plt.figure(109,figsize=figsize10X,facecolor=facecolor10X); # Moyennes mensuelles par classe
        #plt.subplots_adjust(wspace=wspace10X,hspace=hspace10X,top=top10X,bottom=bottom10X-0.12,left=left10X,right=right10X)
        Dmdl_TVm = np.ones((Nmodels,NDobs))*np.nan; # Tableau des Variance sur climatologie
                   # cumulée, moyennée par pixel. J'utiliserais ainsi showimgdata pour avoir
                   # une colorbar commune
        suptitle109="VCUM - %sSST(%s)). Variance sur la Moyenne Cumulée de Modeles complétés (%d models)" \
                     %(fcodage,DATAMDL,Nmodels);
#
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#           DEUXIEME BOUCLE SUR LES MODELES START HERE
#           DEUXIEME BOUCLE SUR LES MODELES START HERE
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
if mdlnamewnumber_ok :
    Tmdlname10X = Tmdlnamewnb;
else :
    Tmdlname10X = Tmdlname;
isubplot = 0;
Tperfglob_Qm = np.zeros((Nmodels,)); # Tableau des Perf globales des modèles cumulees
print("\nooooooooooooooooooooooooooooo 2nd loop ooooooooooooooooooooooooooooo")
for imodel in np.arange(Nmodels) :
    isubplot=isubplot+1;
    #
    Dmdl    = TDmdl4CT[imodel];
    mdlname = Tmdlname10X[imodel];
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
        plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob,'%'),fontsize=sztitle,y=ysstitre10X); 
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
        plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob,'%'),fontsize=sztitle,y=ysstitre10X);
        #
    if OK106 : # Courbes des moyennes mensuelles par classe
        plt.figure(106); plt.subplot(nbsubl,nbsubc,isubplot);
        # trace l'axe des abscices (a Y=0) en noir
        plt.plot([-0.5,Tmoymensclass.shape[1]-1+0.5],[0,0],'-',color='k',linewidth=1.0);
        for i in np.arange(nb_class) :
            plt.plot(Tmoymensclass[imodel,:,i],'.-',color=pcmap[i]);
        plt.axis([-0.5, 11.5, min_moymensclass, max_moymensclass]);
        plt.xticks([]); #plt.axis('off');     
        plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob,'%'),fontsize=sztitle,y=ysstitre10X);
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
        Tperfglob_Qm[imodel] = Perfglob_Qm
        Tperf_Qm = np.round([i*100 for i in Tperf_Qm]).astype(int);

        plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
        plt.imshow(XC_mgeo_Qm, interpolation='none',cmap=ccmap, vmin=1,vmax=nb_class);
        hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
        hcb.set_ticklabels(Tperf_Qm);
        hcb.ax.tick_params(labelsize=8);
        plt.axis('off');
        #grid(); # for easier check
        plt.title("%s, perf=%.0f%c"%(mdlname,100*Perfglob_Qm,'%'),fontsize=sztitle,y=ysstitre10X);
        #
        pgqm_ = np.round_(Perfglob_Qm*100)
        if pgqm_ >= MaxPerfglob_Qm :
            MaxPerfglob_Qm = pgqm_; # Utilisé pour savoir les quels premiers modèles
            IMaxPerfglob_Qm = imodel+1;   # prendre dans la stratégie du "meilleur cumul moyen"
            print(" New best cumul perf for {:d} models : {:.0f}% ...".format(imodel+1,pgqm_))
     #
    if MCUM>0 and OK109 : # Variance sur les Models Cumulés Moyens (not 'RED' compatible)
                          # Perf par classe en colorbar)
        Dmdl_TVm[imodel] = np.var(DMdl_Qm, axis=1, ddof=0);
print("MaxPerfglob_Qm: {}% for {} model(s)".format(MaxPerfglob_Qm,IMaxPerfglob_Qm))
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
    plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle,y=ysstitre10X);
    if 0 :
        plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
        plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
    else :
        set_lonlat_ticks(lon,lat,step=4,fontsize=8,verbose=False,lengthen=True)
    #grid(); # for easier check
    plt.suptitle(suptitle104,fontsize=suptitlefs10X,y=ysuptitre10X)
    #
    if SAVEFIG : # sauvegarde de la figure
        figfile = "Fig-104_{:s}{:s}_{:s}{:s}{:d}MdlvsObstrans_{:d}-mod".format(fprefixe,
                           SIZE_REDUCTION,fshortcode,method_cah,nb_class,Nmodels)
        # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
        # et eventuellement en PDF, si SAVEPDF active. 
        do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
if OK105 : # Obs for 105
    plt.figure(105); plt.subplot(nbsubl,nbsubc,isubplot);
    plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
    plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
    hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
    hcb.set_ticklabels(coches);
    hcb.ax.tick_params(labelsize=8)
    plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle,y=ysstitre10X);
    if 0 :
        plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
        plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
    else :
        set_lonlat_ticks(lon,lat,step=4,fontsize=8,verbose=False,lengthen=True)
    #grid(); # for easier check
    plt.suptitle(suptitle105,fontsize=suptitlefs10X,y=ysuptitre10X)
    if SAVEFIG :
        figfile = "Fig-105_{:s}{:s}_{:s}{:s}{:d}Mdl_{:d}-mod".format(fprefixe,
                           SIZE_REDUCTION,fshortcode,method_cah,nb_class,Nmodels)
        # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
        # et eventuellement en PDF, si SAVEPDF active. 
        do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
    #
if OK106 : # Obs for 106
    plt.figure(106);
    plt.subplot(nbsubl,nbsubc,isubplot);
    # plt.figure();plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.93, bottom=0.05, left=0.05, right=0.80)
    # trace l'axe des abscices (a Y=0) en noir
    plt.plot([-0.5,Tmoymensclass.shape[1]-1+0.5],[0,0],'-',color='k',linewidth=1.0);
    TmoymensclassObs = moymensclass(sst_obs,isnumobs,classe_Dobs,nb_class);
    Legendline = []
    for i in np.arange(nb_class) :
        line = plt.plot(TmoymensclassObs[:,i],'.-',color=pcmap[i]);
        Legendline.append(line[0])
    plt.axis([-0.5, 11.5, min_moymensclass, max_moymensclass]);
    plt.xlabel('mois');
    plt.xticks(np.arange(12), np.arange(12)+1, fontsize=8)
    plt.legend(Legendline,np.arange(nb_class)+1,loc=2,fontsize=6,numpoints=1,bbox_to_anchor=(1.02, 1.0),title="Class");
    #plt.legend([x[0] for x in Legendline],np.arange(nb_class)+1,loc=2,fontsize=6,numpoints=1,bbox_to_anchor=(1.1, 1.0),title="Class");
    plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle,y=ysstitre10X);
    #
    # On repasse sur tous les supblots pour les mettre à la même echelle.
    print("min_moymensclass= {}\nmax_moymensclass= {}".format(min_moymensclass, max_moymensclass));
    plt.suptitle(suptitle106,fontsize=suptitlefs10X,y=ysuptitre10X)
    if SAVEFIG :
        figfile = "Fig-106_{:s}{:s}_{:s}{:s}{:d}moymensclass_{:d}-mod".format(fprefixe,
                           SIZE_REDUCTION,fshortcode,method_cah,nb_class,Nmodels)
        # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
        # et eventuellement en PDF, si SAVEPDF active. 
        do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
#
if OK107 or OK109 : # Calcul de la variance des obs par pixel de la climatologie
    Tlabs = np.copy(Tmdlname10X);  
    Tlabs = np.append(Tlabs,'');                # Pour le subplot vide
    Tlabs = np.append(Tlabs,'Observations');    # Pour les Obs
    varobs= np.ones(Lobs*Cobs)*np.nan;          # Variances des ...
    varobs[isnumobs] = np.var(Dobs, axis=1, ddof=0); # ... Obs par pixel
#
if OK107 : # Variance par pixels des modèles
    plt.figure(107);
    X_ = np.ones((Nmodels,Lobs*Cobs))*np.nan;
    X_[:,isnumobs] = Dmdl_TVar
    # Rajouter nan pour le subplot vide
    X_    = np.concatenate(( X_, np.ones((1,Lobs*Cobs))*np.nan))
    # Rajout de la variance des obs
    X_    = np.concatenate((X_, varobs.reshape(1,Lobs*Cobs)))
    #
    showimgdata(X_.reshape(Nmodels+2,1,Lobs,Cobs), Labels=Tlabs, n=Nmodels+2,fr=0,
                vmin=np.nanmin(Dmdl_TVar),vmax=np.nanmax(Dmdl_TVar),fignum=107,
                wspace=wspace10X,hspace=hspace10X+0.03,top=top10X,bottom=bottom10X,left=left10X,right=right10X,
                );
    del X_
    plt.suptitle(suptitle107,fontsize=suptitlefs10X,y=ysuptitre10X);
    #
    if SAVEFIG :
        figfile = "Fig-107_{:s}VAR_{:s}_{:s}Mdl_{:d}-mod".format(fprefixe,
                           SIZE_REDUCTION,fshortcode,Nmodels)
        # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
        # et eventuellement en PDF, si SAVEPDF active. 
        do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
#
if MCUM>0 and OK108 : # idem OK105, but ...
    plt.figure(108); plt.subplot(nbsubl,nbsubc,isubplot);
    plt.imshow(fond_C, interpolation='none', cmap=cm.gray,vmin=0,vmax=1)
    plt.imshow(XC_ogeo, interpolation='none',cmap=ccmap,vmin=1,vmax=nb_class);
    hcb = plt.colorbar(ticks=ticks,boundaries=bounds,values=bounds);
    hcb.set_ticklabels(coches);
    hcb.ax.tick_params(labelsize=8)
    plt.title("Obs, %d classes "%(nb_class),fontsize=sztitle,y=ysstitre10X);
    if 0 :
        plt.xticks(np.arange(0,Cobs,4), lon[np.arange(0,Cobs,4)], rotation=45, fontsize=8)
        plt.yticks(np.arange(0,Lobs,4), lat[np.arange(0,Lobs,4)], fontsize=8)
    else :
        set_lonlat_ticks(lon,lat,step=4,fontsize=8,verbose=False,lengthen=True)
    #grid(); # for easier check
    plt.suptitle(suptitle108,fontsize=suptitlefs10X,y=ysuptitre10X);
    #
    if SAVEFIG :
        figfile = "Fig-108_{:s}MCUM_{:s}_{:s}{:s}{:d}Mdl_{:d}-mod".format(fprefixe,
                           SIZE_REDUCTION,fshortcode,method_cah,nb_class,Nmodels)
        # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
        # et eventuellement en PDF, si SAVEPDF active. 
        do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
#
if MCUM>0 and OK109 : # Variance par pixels des moyenne des modèles cumulés
    plt.figure(109);
    X_ = np.ones((Nmodels,Lobs*Cobs))*np.nan;
    X_[:,isnumobs] = Dmdl_TVm
    # Rajouter nan pour le subplot vide
    X_ = np.concatenate(( X_, np.ones((1,Lobs*Cobs))*np.nan))
    # Rajout de la variance des obs
    X_ = np.concatenate((X_, varobs.reshape(1,Lobs*Cobs)))
    #
    showimgdata(X_.reshape(Nmodels+2,1,Lobs,Cobs), Labels=Tlabs, n=Nmodels+2,fr=0,
                vmin=np.nanmin(Dmdl_TVm),vmax=np.nanmax(Dmdl_TVm),fignum=109,
                wspace=wspace10X,hspace=hspace10X+0.03,top=top10X,bottom=bottom10X,left=left10X,right=right10X,
                );
    del X_
    plt.suptitle(suptitle109,fontsize=suptitlefs10X,y=ysuptitre10X);
    #
    if SAVEFIG :
        figfile = "Fig-109_{:s}VCUM_{:s}_{:s}Mdl_{:d}-mod".format(fprefixe,
                           SIZE_REDUCTION,fshortcode,Nmodels)
        # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
        # et eventuellement en PDF, si SAVEPDF active. 
        do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
##
##---------------------------------------------------------------------
# Redimensionnement de Tperfglob au nombre de modèles effectif
Tperfglob = Tperfglob[0:Nmodels];
#
# Edition des résultats
if Visu_preACFperf : # Tableau des performances en figure de courbes
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
    plt.xticks(np.arange(Nmodels),Tmdlname10X, fontsize=8, rotation=45,
               horizontalalignment='right', verticalalignment='baseline');
    plt.ylabel('performance by Model [%]')
    plt.legend(local_legend_labels,numpoints=1,loc=3)
    plt.title("SST %s (%d-%d) %d Classes -  Classification Indices of Completed Models (vs Obs) (%d models)"\
                 %(fcodage,andeb,anfin,nb_class,Nmodels));
    #             %(fcodage,DATAMDL,method_cah,nb_class));
    if SAVEFIG :
        figfile = "Fig_{:s}perf-by_model_{:s}_{:s}{:s}{:d}Mdl_{:d}-mod".format(fprefixe,
                       SIZE_REDUCTION,fshortcode,method_cah,nb_class,Nmodels)
        # sauvegarde de la figure ... en format FIGFMT (normalement BITMAP (png,jpg))
        # et eventuellement en PDF, si SAVEPDF active. 
        do_save_figure(figfile,path=case_figs_dir,ext=FIGEXT) #,fig2ok=SAVEPDF,ext2=VFIGEXT)
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
#======================================================================
#
if STOP_BEFORE_AFC :
    plt.show(); sys.exit(0)
#
#%%======================================================================
if NIJ > 0 : # A.F.C
    #Ajout de l'indice dans le nom du modèle
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
        if SIZE_REDUCTION == 'All' :
            nticks = 5; # 4
        elif SIZE_REDUCTION == 'sel' :
            nticks = 2; # 4
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
        #subc_, subl_ = lcsub(max(Loop_nb_clust)); # <-- NON, utiliser plutot nl,nc = nsublc() qui est dans localdef.py
        subl_, subc_ = nsublc(max(Loop_nb_clust),nsubc=5);
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
                # Nouvelle figure:  performanes par cluster
                if SIZE_REDUCTION == 'All' :
                    figsize = (4*subc_,2+2*subl_)
                    wspace=0.25; hspace=0.02; top=0.94; bottom=0.01; left=0.02; right=0.94;
                elif SIZE_REDUCTION == 'sel' :
                    figsize = (3.5*subc_,1+3*subl_)
                    wspace=0.30; hspace=0.02; top=0.94; bottom=0.01; left=0.03; right=0.94;
                figclustmoy = plt.figure(figsize=figsize,facecolor="w"); # pour les différents cluster induit par ce niveau.
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
                        XCM_     = dto2d(classej_,LObs,CObs,isnumObs); # Classification géographique
                        plt.subplot(8,6,jj+1); # plt.subplot(7,7,jj+1);
                        plt.imshow(XCM_, interpolation='none',cmap=ccmap, vmin=1,vmax=nb_class);
                        plt.axis('off');
                        plt.title("%s(%.0f%c)"%(CAHindnames[ijj],100*Tperfglob[ijj,0],'%'),
                                  fontsize=sztitle,y=ysstitre);
                            # même si y'a 'Obs' dans CAHindnames, ca devrait pas apparaitre
                            # car elles sont à la fin et ont été retirées de class_afc
                    plt.suptitle("Classification des modèles du cluster %d"%(ii+1));
                #if MultiLevel == False :    
                #    print("%d Modèles du cluster %d, performance: %d :\n"%(len(iclust),ii+1,Perfglob_), CAHindnames[iclust]);
                #
                # Modèle Moyen d'un cluster (plus de gestion de pondération)
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
                # Classification du, des modèles moyen d'un cluster
                if MultiLevel : # Plusieurs niveaux de découpe, c'est pas la peine de faire toutes ces
                                # figures, mais on a besoin de la perf
                    Perfglob_ = Dgeoclassif(sMapO,CmdlMoy,LObs,CObs,isnumObs,TypePerf[0],
                                            visu=False,nticks=nticks);
                else : # 1 seul niveau de découpe, on fait la figure
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
        # FIN de la boucle sur les différents niveau de cluster
        if MultiLevel :
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
    # FIN du if 1 : MODELE MOYEN (pondéré ou pas) PAR CLUSTER D'UNE CAH
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
            else : # Obs en supplémentaire
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
    if Visu_afcnu_det : # plot afc etape par étape"
        # Que les points lignes (modèles)
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
    #if 1 : # Affichage du moyen for CT
    #    aff2D(MdlMoy,Lobs,Cobs,isnumobs,isnanobs,
    #          wvmin=wvmin,wvmax=wvmax,figsize=(12,9),cmap=eqcmap,fignum=fignum);
    #    plt.suptitle("MdlMoy (%s) \n%s(%d-%d) for CT\nmin=%f, max=%f, moy=%f, std=%f"
    #                %(Tmdlname[IMixtMdl], fcodage,andeb,anfin,np.min(MdlMoy),
    #                 np.max(MdlMoy),np.mean(MdlMoy),np.std(MdlMoy)))
    #
    # Classification du modèles moyen
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
# Modèles Optimaux (Sopt) ;  Avec "NEW Obs - v3b " 1975-2005
# Je commence par le plus simple : Une ligne de modèle sans classe en une phase
# et une seule codification à la fois
# Sopt-1975-2005 : Les meilleurs modèles de la période "de référence" 1975-2005
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
    print("\nSopt non renseigné ; Ce Cas n'a pas encore été prévu")
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
                  noaxes=False,noticks=False,nolabels=False,y=0.98,cblabel="SST Anomaly [°C]",
                  lolast=lolast,lonlat=(lon,lat),
                  vcontour=std_, ncontour=np.arange(0,2,1/10), ccontour='k', lblcontourok=True,
                  ); #...
        else :
            aff2D(MdlMoy,Lobs,Cobs,isnumobs,isnanobs,
                  wvmin=wvmin,wvmax=wvmax,
                  fignum=fignum,varnames=varnames,cmap=eqcmap,
                  wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right,
                  noaxes=False,noticks=False,nolabels=False,y=0.98,cblabel="SST Anomaly [°C]",
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
