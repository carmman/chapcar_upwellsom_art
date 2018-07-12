from   ctObsMdldef import *
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
Tmodels = Tmodels_anyall;
#Tmodels= Tmodels[2:12];  # Pour limiter le nombre de modèle en phase de mise au point
Nmodels = len(Tmodels); # print(Nmodels); sys.exit(0)
#______________________________
# For the Carte Topo (see also ctObsMdl)
if 1 : # pour 'All' : 25x36 or 'sel' : 13x13 - 13x12
    nbl      = 30;  nbc =  4;  # Taille de la carte
    Parm_app = (5, 5., 1.,  16, 1., 0.1);
elif 0 : # pour 'sel' : 13x13 - 13x12
    nbl      = 12;  nbc =  8;   # Taille de la carte
    Parm_app = (3, 3.,1.,  12,1.,1.0);
    #---
    #nbl      = 20;  nbc =  5;  # Taille de la carte
    #Parm_app = (10, 5., 1.,  20, 1., 0.1);
    #nbl      = 28;  nbc =  4;  # Taille de la carte
    #Parm_app = (6, 5., 1.,  16, 1., 0.1);
    #---
elif 0 : # || pour 'sel' : 13x13 - 13x12
    nbl      = 17;  nbc =  6;  # Taille de la carte
    Parm_app = (4, 4., 1.,  16, 1., 0.1);
#
epoch1,radini1,radfin1,epoch2,radini2,radfin2 = Parm_app
#______________________________
# Complémentation des nan pour les modèles
MDLCOMPLETION = True; # True=>Cas1
#______________________________
# Transcodage des classes 
TRANSCOCLASSE = 'STD'; # Permet le transcodage des classes de façon à ce
#TRANSCOCLASSE = 'GRAD'; # Permet le transcodage des classes de façon à ce
    # que les (indices de) classes correspondent à l'un des critères :
    # 'GAP' 'GAPNM' 'STD' 'MOY' 'MAX' 'MIN' 'GRAD'. Attention ce critère
    # est appliqué sur les référents ...
    # Avec la valeur '' le transcodage n'est pas requis.
#______________________________
# Prendre une zone plus petite
SIZE_REDUCTION = 'All'; # 'sel', 'All';
    # 'All' : Pas de réduction d'aucune sorte
    # 'sel' : On sélectionne, d'entrée de jeu une zone plus petite,
    #         c'est à dire à la lecture des données. Zone sur laquelle
    #         tous les traitement seront effectues (codification, CT, Classif ...)
    # 'RED' : Seule la classification est faite sur la zone REDuite
    # rem   : PLM 'sel' et 'RED' ne sont pas compatibles.
# Définition d'une la sous-zone :
# Once upon a time
#frlat=20.5; tolat=11.5; frlon=-20.5; tolon=-11.5   #(once upon a time)
#LON 16W a 28W  (donc -28 a -16) LAT 10N a 23N
#frlat=23.5; tolat=10.5; frlon=-28.5; tolon=-15.5;  #(¤:13x13)
frlat=22.5; tolat= 9.5; frlon=-27.5; tolon=-15.5;   #(§:13x12)
# J'essaye d'équilibrer les classes
#frlat=22.5; tolat=10.5; frlon=-24.5; tolon=-15.5   #(a)
#frlat=21.5; tolat=10.5; frlon=-23.5; tolon=-15.5   #(b)
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
OK101 = True; # Pour produire les Ecarts types d'un modèle moyen par
              # pixel et par mois (eventuellement controlé par Sopt...)
ecvmin = 0.10; ecvmax = 0.70; # si ecvmin<0 on utilise les min et max
ecvmin= -1.0                 # des valeurs à afficher
#<<<
OK105=True;
OK104=OK106=OK108=True;
if SIZE_REDUCTION == 'RED' :
    OK107=OK109=False;
else :
    OK107=OK109=False;
#OK104=OK105=OK106=OK107=OK108=OK109=True;
#
# Other stuff
FONDTRANS = "Obs"; # "Obs", "Mdl"
SAVEFIG   = False; # True; False;
fprefixe  = 'Z_'
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
fcodage=""; fshortcode="";
if climato=="GRAD" :
    fcodage=fcodage+"GRAD(";        fshortcode=fshortcode+"Grad"
if INDSC :
    fcodage=fcodage+"INDSC(";       fshortcode=fshortcode+"Indsc"
if TRENDLESS :
    fcodage=fcodage+"TRENDLESS(";   fshortcode=fshortcode+"Tless"
if WITHANO :
    fcodage=fcodage+"ANOMALIE(";    fshortcode=fshortcode+"Ano"
#if CENTREE : fcodage=fcodage+"CENTREE(";
#-> Climatologie (Moyenne mensuelle par pixel)
if UISST :
    fcodage=fcodage+"UI(";          fshortcode=fshortcode+"Ui"
if NORMMAX :
    fcodage=fcodage+"NORMMAX(";     fshortcode=fshortcode+"Nmax"
if CENTRED :
    fcodage=fcodage+"CENTRED(";     fshortcode=fshortcode+"Ctred"
#print(fcodage); sys.exit(0);
#______________________________
# for CAH for classif with CT (see ctObsMdl for more)
method_cah = 'ward'; # 'average', 'ward', 'complete','weighted'    #KKKKKKK
dist_cah   = 'euclidean'; #
nb_class   = 7; # 4, 5, 7; # Nombre de classes retenu #KKKKKKK   ##@@
ccmap      = cm.jet;       # Accent, Set1, Set1_r, gist_ncar; jet, ... : map de couleur pour les classes
# pour avoir des couleurs à peu près equivalente pour les plots
#pcmap     = ccmap(np.arange(1,256,round(256/nb_class)));ko 512ko, 384ko
pcmap      = ccmap(np.arange(0,320,round(320/nb_class))); #ok?
#
#______________________________
# Calcul de la performance globale
#kperf = 1; # 1     : # Les bien classés / effectif total sans distinction de classe
#           # sinon : # Moyenne des biens classés par classe (sans doute mieux
#                       adapté avec NIJ=2, mais ne concerne pas que l'afc)
TypePerf = ["MeanClassAccuracy","GlobalAccuracy"]; #,"SpearmanCorr","Index2Rand","ContengencyCoef","GlobalAccuracy"];
#______________________________
# AFC
NIJ = 2; # 0 : Pas d'AFC
         # 1 : nombre d'elt par classe
         # 2 : perf par classe
         # 3 : nombre d'elt bien classés par classe
         # voir aussi avec kperf plus haut
AFCWITHOBS = True; #False True : afc avec ou sans les Obs ?
#AFCWITHOBS = False; # <<<<<<<<<<<<<<  ##@@
pa=1; po=2; # Choix du plan factoriel
#pa=3; po=4; # Choix du plan factoriel
#
#______________________________
# et CAH for cluster with AFC
CAHWITHOBS = True; #True; #False
nb_clust   = 4; # 5 8, 7, 6, 4, 3, -20   WARD.nbclust; par exemple WARD.5     #<><><><><><><>
#CAHCOORD  = 1; # 0:-> CAH avec les coordonnées du plan de l'afc  #<><><><><><><>
#              # sinon : CAH avec TOUTES les coordonnées de l'afc
#CAHCOORD  = np.array([1,2,3,4,5,6]); # [list] des coordonnées de l'afc à
#CAHCOORD  = np.arange(4); # les n premières coordonnées de l'afc à
              # utiliser pour faire la CAH (limité à nb_class-1).
NBCOORDAFC4CAH = 5; # nb_class-1 = les n premières coordonnées de l'afc
#^v^v^v^v^v^v
#NBCOORDAFC4CAH = nb_class-1; # les n premières coordonnées de l'afc
#NBCOORDAFC4CAH = 1; # ##@@
                    # à utiliser pour faire la CAH (limité à nb_class-1).
                    # AFC.NBCOORDAFC4CAH; par exemple AFC.6
#
#______________________________
# Points d'arrets
STOP_BEFORE_CT       = False;
STOP_BEFORE_MDLSTUFF = False;
STOP_BEFORE_AFC      = False; 
STOP_BEFORE_GENERAL  = False;
#
# Flag de visualisation
Visu_ObsStuff  = False;   # Flag de visu des Obs  : 4CT, classif et courbes moy. mens.
Visu_CTStuff   = False;   # Flag de visu de la CT : Umat, Map, Profils (sauf Dendro) 
Visu_Dendro    = False;   # Flag de visualisation des dendrogrammes
Visu_afcnu_det = False;   # Sylvie
Visu_Inertie   = True;    # Flag de visualisation de l'Inertie
#Flag visu classif des modèles des cluster
AFC_Visu_Classif_Mdl_Clust  = []; # liste des cluster a afficher (à partir de 1)
#AFC_Visu_Classif_Mdl_Clust = [1,2,3,4,5,6,7]; 
#Flag visu Modèles Moyen 4CT des cluster
AFC_Visu_Clust_Mdl_Moy_4CT  = []; # liste des cluster a afficher (à partir de 1)
#AFC_Visu_Clust_Mdl_Moy_4CT = [1,2,3,4,5,6,7];
#
#######################################################################
