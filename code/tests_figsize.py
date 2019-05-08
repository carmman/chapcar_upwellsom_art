#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:44:15 2019

@author: carlos
"""

import numpy as np
#import sys, getopt, os
#from   time  import time
import matplotlib.pyplot as plt
#from   matplotlib import colors
from   matplotlib import cm
#from   mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#import pickle
#from   scipy.cluster.hierarchy import dendrogram, linkage, fcluster
#%%
tseed = 0;
np.random.seed(tseed);
N = 30
CP = np.random.rand(N,2)*N
CP
#%%
markersize = 150;
marker     = 'o';
linewidths = 2;
face_colors = 'y' # a Nx4 array of rgba value
edge_colors = 'k'
#%%
fW = 8;fH = 6; 
#fW = 12;fH = 9; 
#fW = 16;fH = 12; 
fdpi = 50
fig = plt.figure(figsize=(fW,fH),dpi=fdpi,facecolor="w"); # pour les différents cluster induit par ce niveau.
#%%
wspace=0.35; hspace=0.02; top=0.94; bottom=0.10; left=0.10; right=0.98;
fignum = fig.number
fig.subplots_adjust(wspace=wspace, hspace=hspace, top=top,
                    bottom=bottom, left=left, right=right)
ax = plt.subplot(111)

#ax.scatter(CP[:,0], CP[:,1],
#           s=6, marker=marker,
#           linewidths=linewidths)

ax.scatter(CP[:,0], CP[:,1],
           s=markersize, marker=marker,
           facecolor=face_colors,
           edgecolors=edge_colors,
           linewidths=linewidths)
plt.ylabel('ordenada')
plt.xlabel('abscisa')
plt.title('Titulo')
plt.show()
#%%
dpi = 300
figurefilelname = "fig_{}x{}-{}dpi_saved{}dpi.jpg".format(fW,fH,fdpi,dpi)
plt.savefig(figurefilelname, dpi=dpi, transparent=False)
