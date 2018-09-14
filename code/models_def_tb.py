# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:21:37 2018

Tables de Modeles

@author: carlos
"""
import numpy as np

#======================================================================
# Table(s) des Modèles
#Tmodels_rmean_ Tmodels_raverage_ (in _SV0, but caduc now)
Tmodels_anyall_OUT = np.array([     # Mettre de coté certains modèles; on peut
        ["Observations",    ""],    # aussi les déplacer dans un repertoire sst_OUT/     
]);
if 0 :
    Tmodels_anyall = np.array([
        ["bcc-csm1-1",      ""],    #( 3)
        ["bcc-csm1-1-m",    ""],    #( 4)
        #["BNU-ESM",         ""],        # initialement non fourni, mais present actuellement
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
    Tinstitut_anyall = np.repeat('', Tmodels_anyall.shape[0])
else :
    Tmodels_and_institut_anyall = np.array([
        ['BCC',          'bcc-csm1-1',       ''],   #( 3)
        ['BCC',          'bcc-csm1-1-m',     ''],   #( 4)
        ['BNU',          'BNU-ESM',          ''],   # initialement non fourni, mais present actuellement
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
        ['FIO',          'FIO-ESM',          ''],    # pas de donnees 'tos', en 'historical', uniquement 'so', mais a des donnees en scenarios ...
        #['',             'OBS',              ''],    #(??)    # par exemple.
    ]);
    Tinstitut_anyall = Tmodels_and_institut_anyall[:,0]
    Tmodels_anyall = Tmodels_and_institut_anyall[:,(1,2)]
