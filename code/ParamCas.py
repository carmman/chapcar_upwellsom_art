# -*- coding: utf-8 -*-
"""
Created on ???  2018

Tables de Modeles

@author: charles
"""
from   matplotlib import cm
##from   ctObsMdldef import *
from models_def_tb import *

#######################################################################
# FLAGS DE COMPORTEMENT
#======================================================================
SAVEFIG    = True;
#SAVEFIG    = False;
# -----------------------------------------------------------------------------
SAVEPDF    = True;
#SAVEPDF    = False;
# -----------------------------------------------------------------------------
SAVEMAP    = True;
#SAVEMAP    = False;
# -----------------------------------------------------------------------------
#REWRITEMAP = True;
REWRITEMAP = False;
# -----------------------------------------------------------------------------
RELOADMAP = True;
#RELOADMAP = False;
# -----------------------------------------------------------------------------
#RERUNTRAINMAP = True;
#RERUNTRAINMAP = False;
# -----------------------------------------------------------------------------
#######################################################################
# PARAMETRAGE (#1) DU CAS
#======================================================================
# Choix du jeu de données
#
DATAOBS = "raverage_1975_2005";   #<><><><><><><>
#DATAOBS = "raverage_1930_1960";   #<><><><><><><>
#DATAOBS = "raverage_1944_1974";   #<><><><><><><>
#DATAOBS = "rcp_2006_2017";        #<><><><><><><>
#DATAOBS = "rcp_2070_2100";        #<><><><><><><>
#
DATAMDL = "raverage_1975_2005";   #<><><><><><><>
#DATAMDL = "raverage_1930_1960";   #<><><><><><><>
#DATAMDL = "raverage_1944_1974";   #<><><><><><><>
#DATAMDL = "rcp_2006_2017";        #<><><><><><><>
#DATAMDL = "rcp_2070_2100";        #<><><><><><><>
#
if DATAMDL=="rcp_2006_2017" \
or DATAMDL=="rcp_2070_2100" :      # on précise le scénario
    scenar = "rcp85";              # rcp26 rcp45 rcp85
#
# Tableau des modèles (cf dans ctObsMdldef.py; avant il était différent
# selon DATARUN, maintenant, ca ne devrait plus etre le cas).
Tinstit = Tinstitut_anyall;
Tmodels = Tmodels_anyall;
#Tmodels= Tmodels[2:12];  # Pour limiter le nombre de modèle en phase de mise au point
Nmodels = len(Tmodels); # print(Nmodels); sys.exit(0)
# tableau de numero de modele
Tnmodel = []
for imodel in np.arange(Nmodels) :
    Tnmodel.append("{:d}".format(imodel+1))
Tnmodel = np.array(Tnmodel)

# -----------------------------------------------------------------------------
# Conditions d'execution:
#    | --------------- CARACTERISTIQUES ----------------- | -- VARIABLES ---- |
#  - Architecture de la carte SOM ......................... nbl, nbc
#  - Parametres d'entrainement en deux phases ............. Parm_app
#  - Zone geographique consideree (toute, reduite, ...) ... SIZE_REDUCTION
#  - Nombre de classes .................................... nb_class
#  - Nombre de clusters et de coordonnees pour l'AFC ...... nb_clust, NBCOORDAFC4CAH
#  - Critere pour evaluation des performances pour l'AFC .. NIJ
# -----------------------------------------------------------------------------
# Prendre une zone plus petite (concerne aussi l'entrainement)
    # 'All' : Pas de réduction d'aucune sorte
    # 'sel' : On sélectionne, d'entrée de jeu une zone plus petite,
    #         c'est à dire à la lecture des données. Zone sur laquelle
    #         tous les traitement seront effectues (codification, CT, Classif ...)
    #         ('sel' à la place de 'mini' dans les version précédantes);
    # 'RED' : Seule la classification est faite sur la zone REDuite
    # rem   : PLM 'sel' et 'RED' ne sont pas compatibles; voir ci-après pour
    #         la définition de la zone qui sera concernée
    # AFC
# -----------------------------------------------------------------------------
# NIJ = 0 : Pas d'AFC
#     = 1 : nombre d'elt par classe
#     = 2 : perf par classe
#     = 3 : nombre d'elt bien classés par classe
#           (le seul qui devrait survivre à mon sens)
# -----------------------------------------------------------------------------
# NBCOORDAFC4CAH ... les n premières coordonnées de l'afc
#   # à utiliser pour faire la CAH (limité à nb_class-1).
#   # AFC.NBCOORDAFC4CAH; par exemple AFC.6
#
# ATTENTION, ne changez plus ici les conditions pour 'ALL' ou 'sel', ...
#            utilisez les arguments d'appel au programme: p. exemple=
#      runfile('/Users/carlos/Labo/NN_divers/DeepLearning/Projets/Projet_Upwelling/Charles/PourCarlos2_pour_Article/code/ctLoopMain.py',
#              wdir='/Users/carlos/Labo/NN_divers/DeepLearning/Projets/Projet_Upwelling/Charles/PourCarlos2_pour_Article/code',
#              args="--case=sel -v")         
if 0 : # conditions Code Charles: GRANDE ZONE
    SIZE_REDUCTION = 'All';
    # A - Grande zone de l’upwelling (25x36) :
    #    Longitude : 45W à 9W (-44.5 à -9.5)
    #    Latitude :  30N à 5N ( 29.5 à  5.5)
    frlat =  29.5;  tolat =  4.5; #(excluded)
    frlon = -44.5;  tolon = -8.5; #(excluded)   #(§:25x35)
    #   * Carte topologique et CAH : 30x4 (5, 5, 1, - 16, 1, 0.1) : TE=0.6824 ; QE=0.153757
    #   Nb_classe = 7
    nbl            = 30;  nbc =  4;  # Taille de la carte 30x4=120
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
elif 0 : # conditions Code Charles: PETITE ZONE
    SIZE_REDUCTION = 'sel';
    # B - Sous-zone ciblant l’upwelling (13x12) :
    #    LON:  28W à 16W (-27.5 to -16.5)
    #    LAT : 23N à 10N ( 22.5 to  10.5)
    frlat =  22.5;  tolat =   9.5; #(excluded)
    frlon = -27.5;  tolon = -15.5; #(excluded)   #(§:13x12)
    #   * Carte topologique et CAH : 17x6 (4, 4, 1, - 16, 1, 0.1) : TE=0.6067 ; QE=0.082044
    #   Nb_classe = 4
    nbl            = 17;  nbc =  6;  # Taille de la carte 17x6=102
    Parm_app       = ( 4, 4., 1.,  16, 1., 0.1); # Température ini, fin, nb_it
    nb_class       = 4; #6, 7, 8  # Nombre de classes retenu
    # et CAH for cluster with AFC
    NIJ            = 2;
    nb_clust       = 5; # Nombre de cluster
    NBCOORDAFC4CAH = nb_class - 1; # n premières coordonnées de l'afc à
else : # Autres cas, valeurs par defaut
    #SIZE_REDUCTION = 'All';
    SIZE_REDUCTION = 'sel'; # selectionne une zone reduite  
    #SIZE_REDUCTION = 'RED'; # Ne pas utiliser
    # -------------------------------------------------------------------------
    # Définition d'une la sous-zone :
    # Once upon a time
    #frlat=20.5; tolat=11.5; frlon=-20.5; tolon=-11.5   #(once upon a time)
    #LON 16W a 28W  (donc -28 a -16) LAT 10N a 23N
    #frlat=23.5; tolat=10.5; frlon=-28.5; tolon=-15.5;  #(¤:13x13)
    frlat=22.5; tolat= 9.5; frlon=-27.5; tolon=-15.5;   #(§:13x12)
    # J'essaye d'équilibrer les classes
    #frlat=22.5; tolat=10.5; frlon=-24.5; tolon=-15.5   #(a)
    #frlat=21.5; tolat=10.5; frlon=-23.5; tolon=-15.5   #(b)
    # -------------------------------------------------------------------------
    #nbl      = 6;  nbc =  6;  # Taille de la carte
    #nbl      = 30;  nbc =  4;  # Taille de la carte
    nbl       = 36;  nbc =  6;  # Taille de la carte 36x6=216
    #nbl      = 52;  nbc =  8;  # Taille de la carte
    # -------------------------------------------------------------------------
    #Parm_app = ( 5, 5., 1.,  16, 1., 0.1); # Température ini, fin, nb_it
    Parm_app = ( 50, 5., 1.,  100, 1., 0.1); # Température ini, fin, nb_it
    #Parm_app = ( 500, 5., 1.,  1000, 1., 0.1); # Température ini, fin, nb_it
    #Parm_app = ( 2000, 5., 1.,  5000, 1., 0.1); # Température ini, fin, nb_it
    # -------------------------------------------------------------------------
    nb_class   = 7; #6, 7, 8  # Nombre de classes retenu
    # -------------------------------------------------------------------------
    NIJ        = 2; # cas de
    # -------------------------------------------------------------------------
    nb_clust   = 7; # Nombre de cluster
    NBCOORDAFC4CAH = nb_class - 1; # n premières coordonnées de l'afc à
# -----------------------------------------------------------------------------

#______________________________
#______________________________
# Complémentation des nan pour les modèles
MDLCOMPLETION = True; # True=>Cas1
#______________________________
# Transcodage des classes 
TRANSCOCLASSE = 'STD'; # Permet le transcodage des classes ...
#TRANSCOCLASSE = 'GRAD'; # Permet le transcodage des classes de façon à ce
    # que les (indices de) classes correspondent à l'un des critères :
    # 'GAP' 'GAPNM' 'STD' 'MOY' 'MAX' 'MIN' 'GRAD'. Attention ce critère
    # est appliqué sur les référents ...
    # Avec la valeur '' le transcodage n'est pas requis.
#______________________________
#______________________________
MCUM = 3; # Moyenne des Models climatologiques CUmulés
          # Pour une vérif avec une Moyenne des Models CUmulés (avant
          # climatologie), c.f. ctLoopMdlCum.py
#______________________________
# Choix des figures à produire
# fig Pour chaque modèle et par pixel :
# 104 : Classification avec, "en transparance", les mals classés
#       par rapport aux obs (*1)
# 105 : Classification (*1)
# 106 : Courbes des moyennes mensuelles par classe
# 107 : Variance (not 'RED' compatible)
# 108 : Classification en Model Cumulé Moyen (*1)
# 109 : Variance sur les Models Cumulés Moyens (not 'RED' compatible)
# (*1): (Pour les modèles les Perf par classe sont en colorbar)
# rem : Les classes peuvent être transcodée de sorte à ce que leurs couleurs 
#       correspondent à un critère (cf paramètre TRANSCOCLASSE)
# PLM, le figures 107 et 109 n'ont pas été adaptées
# pour 'RED'. Je ne sais pas si c'est vraiment intéressant de le faire,
# attendre que le besoin émerge.
#>>>
OK101 = True; # Pour produire la Moyenne des modèles en entrée moyennées par
               # pixel et par mois (eventuellement controlé par Sopt...)
OK102 = True; # Pour produire l'Ecart type des modèles en entrée moyennées par
               # pixel et par mois (eventuellement controlé par Sopt...)
ecvmin = 0.10; ecvmax = 0.70; # si ecvmin<0 on utilise les min et max
ecvmin= -1.0                 # des valeurs à afficher: si negatif prend les limites de chaque figure
#<<<
OK105=True;
OK104=OK106=OK108=True;
if SIZE_REDUCTION == 'RED' :
    OK107=OK109=False;
else :
    OK107=OK109=False;
#OK104=OK105=OK106=OK107=OK108=OK109=True;
#OK104=OK105=OK106=OK107=OK109=False;
#
# Other stuff
FONDTRANS = "Obs"; # "Obs", "Mdl"
# -----------------------------------------------------------------------------
FIGSDIR    = 'figs'
FIGEXT     = '.png'
FIGDPI     = 144  # le defaut semble etre 100 dpi (uniquement pour format bitmap)
#VFIGEXT    = '.eps'
VFIGEXT    = '.pdf'
# -----------------------------------------------------------------------------
MAPSDIR    = 'maps'
# -----------------------------------------------------------------------------
mapfileext = ".pkl" # exten,sion du fichier des MAP
# -----------------------------------------------------------------------------
#if SIZE_REDUCTION == 'All' :
#    fprefixe  = 'Zall_'
#elif SIZE_REDUCTION == 'sel' :
#    fprefixe  = 'Zsel_'
#elif SIZE_REDUCTION == 'RED' :
#    fprefixe  = 'Zred_'
#else :
#    print(" *** unknown SIZE_REDUCTION <{}> ***".format(SIZE_REDUCTION))
#    raise
#______________________________
#if DATARUN=="rmean" : caduc ; # Les moyennes des runs des modèles que j'avais calculées
#    Nda  =30; #!!! Prendre que les Nda dernières années (All Mdls Compatibles)
#    anfin=2005; andeb = anfin-Nda; # avec .NPY andeb n'est pas inclus
#    #(rem ATTENTION, toutes les données ne commencent pas à l'année 1850
#    # ou 1854 ni au mois 01 !!!!!!!
#    #
# Les runs moyens des modèles calculés par Carlos
#!!! Prendre que les Nda dernières années (All Mdls Compatibles)
# avec .MAT andeb est inclus (Caduc, maintenant on a que des .nc)
if DATAMDL=="raverage_1975_2005" :  
    Nda  = 31; 
    anfin= 2005; andeb = anfin-Nda+1; 
elif DATAMDL=="raverage_1930_1960" :  # Les runs moyens des modèles calculés par Carlos
    Nda  = 31; #!!! Prendre que les Nda dernières années (All Mdls Compatibles)
    anfin= 1960; andeb = anfin-Nda+1; # avec .MAT andeb est inclus
elif DATAMDL=="raverage_1944_1974" :  # Les runs moyens des modèles calculés par Carlos
    Nda  = 31; #!!! Prendre que les Nda dernières années (All Mdls Compatibles)
    anfin= 1974; andeb = anfin-Nda+1; # avec .MAT andeb est inclus
elif DATAMDL=="rcp_2006_2017" :  # Les sénarios rcp26, rcp45 et rcp85 de 2006 à 2017
    Nda  = 12; # on prend tout
    anfin= 2017; andeb = anfin-Nda+1; # avec .MAT andeb est inclus
elif DATAMDL=="rcp_2070_2100" :  # Les sénarios rcp26, rcp45 et rcp85 de 2070 à 2100
    Nda  = 31; # on prend tout
    anfin= 2100; andeb = anfin-Nda+1; # avec .MAT andeb est inclus
#______________________________
INDSC     = False;  #!!! IndSC : Indicateur de Saisonalité Climatologique
#
# Transfo des données-séries brutes, dans l'ordre :
TRENDLESS = False;  # Suppression de la tendance
WITHANO   = True;   # False,  True   #<><><><><><><>
#
climato   = None;   # None  : climato "normale" : moyenne mensuelle par pixel et par mois
                    # "GRAD": pente b1 par pixel et par mois
#
# b) Transfo des moyennes mensuelles par pixel dans l'ordre :
UISST     = False;  # "after", "before" (Som(Diff) = Diff(Som))
NORMMAX   = False;  # Dobs =  Dobs / Max(Dobs)
CENTRED   = False ;
#______________________________
# for CAH for classif with CT (see ctObsMdl for more)
method_cah = 'ward'; # 'average', 'ward', 'complete','weighted'    #KKKKKKK
dist_cah   = 'euclidean'; #
#nb_class ... DECLAREE CI-DESSUS  # Nombre de classes retenu #KKKKKKK   ##@@
#______________________________
# for AFC ... (see ctObsMdl for more)
method_afc = 'ward'; # 'average', 'ward', 'complete','weighted'    #KKKKKKK
dist_afc   = 'euclidean'; #
# -------------------------
# map de couleur pour les classes
ccmap      = cm.jet;       # Accent, Set1, Set1_r, gist_ncar; jet, ...
# -------------------------
# map de couleur par defaut
dcmap      = cm.gist_ncar; 
# -------------------------
# map de couleur pour donnes negative/positives
eqcmap = cm.RdYlBu
#eqcmap = cm.coolwarm
#eqcmap = cm.bwr
#eqcmap = cm.seismic
#eqcmap = cm.RdGy
#eqcmap = cm.RdBu
#eqcmap = cm.RdYlGn
#eqcmap = cm.Spectral
#eqcmap = cm.BrBG
#______________________________
# Calcul de la performance globale
#kperf = 1; # 1     : # Les bien classés / effectif total sans distinction de classe
#           # sinon : # Moyenne des biens classés par classe (sans doute mieux
#                       adapté avec NIJ=2, mais ne concerne pas que l'afc)
TypePerf = ["MeanClassAccuracy","GlobalAccuracy"]; #,"SpearmanCorr","Index2Rand","ContengencyCoef","GlobalAccuracy"];
#______________________________
# AFC
#NIJ ... DECLAREE CI-DESSUS
AFCWITHOBS = True; #False True : afc avec ou sans les Obs ?
#AFCWITHOBS = False; # <<<<<<<<<<<<<<  ##@@
pa=1; po=2; # Choix du plan factoriel
#pa=3; po=4; # Choix du plan factoriel
#
#______________________________
# et CAH for cluster with AFC
CAHWITHOBS = True; #True; #False
#nb_clust ... DECLAREE CI-DESSUS   # 5 8, 7, 6, 4, 3, -20   WARD.nbclust; par exemple WARD.5 
#CAHCOORD  = 1; # 0:-> CAH avec les coordonnées du plan de l'afc  #<><><><><><><>
#              # sinon : CAH avec TOUTES les coordonnées de l'afc
#CAHCOORD  = np.array([1,2,3,4,5,6]); # [list] des coordonnées de l'afc à
#CAHCOORD  = np.arange(4); # les n premières coordonnées de l'afc à
              # utiliser pour faire la CAH (limité à nb_class-1).
#NBCOORDAFC4CAH  ... VOIR CI-DESSUS
#
#______________________________
# Points d'arrets
STOP_BEFORE_CT       = False;
STOP_BEFORE_MDLSTUFF = False;
STOP_BEFORE_AFC      = False; 
STOP_BEFORE_GENERAL  = False;
#
# Flag de visualisation
Show_ObsSTD     = False;    # Flag de visu des la STD des Obs (si Visu_ObsStuff est True)
Show_ModSTD     = True;    # Flag de visu des la STD des Obs (si Visu_ObsStuff est True)

Visu_ObsStuff   = False;    # Flag de visu des Obs  : 4CT, classif et courbes moy. mens.
Visu_CTStuff    = False;    # Flag de visu de la CT : Umat, Map, Profils (sauf Dendro) 
Visu_Dendro     = False;    # Flag de visualisation des dendrogrammes
Visu_preACFperf = False;    # performances avant l'AFC: MeanClassAccuracy, GlobalAccuracy, ...
Visu_AFC_in_one = False;    # plot afc en une seule image
Visu_afcnu_det  = False;    # Sylvie: plot afc etape par étape
Visu_Inertie    = False;    # Flag de visualisation de l'Inertie
#
# FLAGS en vue de l'article
Visu_UpwellArt  = True;     # Flag de visu des figures pour article avec Juliette et Adama
FIGARTDPI     = 300  # DPI pour les figures bitmap de l'article
if Visu_UpwellArt :
    OK101 = False; # Pour produire la Moyenne d'un modèle moyen par ...
    OK102 = False; # Pour produire les Ecarts types d'un modèle moyen 
ysstitre        = 0.96   # position vertical des soustitres dans figures de groupe
same_minmax_ok     = True;  # MIN = -MAX
mdlnamewnumber_ok  = True; # fait apparaitre le numero de modele dans fogures 10X (104, 105, ...)
onlymdlumberAFC_ok = True; # identifie les modeles uniquement par leur numero dans la projection AFC
#
# POUT TEST:
#OK101=OK102=OK104=OK105=OK106=OK107=OK108=OK109=True;
#OK104=OK105=OK106=OK107=OK108=OK109=True;
Visu_ObsStuff=Visu_CTStuff=Visu_Dendro=Visu_preACFperf=Visu_AFC_in_one=Visu_afcnu_det=Visu_Inertie=True;

#######################################################################
