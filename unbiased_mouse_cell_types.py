import sys
import warnings
warnings.filterwarnings("ignore")
from multi_dim_reduction import *
from file_process import *
from sample_cluster import *
from sample_filter import *
from vis_tools import *
import pandas as pd
import numpy as np
import os
from IPython.display import Image
import re
from matplotlib_venn import venn2
import rpy2.robjects as ro
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

indir = 'unbiased_mouse_cell_types'
outdir = 'unbiased_mouse_cell_types/results'

## Input data
rpkms = pd.read_table('%s/rpkms.csv' %(indir), header=0, index_col=0, sep="\t")
counts = pd.read_table('%s/counts.csv' %(indir), header=0, index_col=0, sep="\t")
cell_type_annot = pd.read_table('%s/cell_type_annotation.csv' %(indir), header=0, index_col=None, sep="\t")
cell_type_annot.index = cell_type_annot['sampleName']
var_genes = pd.read_table('%s/mvg.csv' %(indir), header=0, index_col=0, sep="\t")['g'].to_list()

rpkm_filter = 1
count_filter = 2
ncell = 3
fdr = 0.01

## Process
n_cells = cell_type_annot.groupby(by="sampleCluster").apply(lambda x: x.shape[0])
new_ctype_labels = {ctype: '%s (%s)' %(ctype, n_cells[ctype]) for ctype in n_cells.keys()}
cell_type_annot['sampleCluster'] = [new_ctype_labels[item] for item in cell_type_annot['sampleCluster'].values]
cells_in_use = list(cell_type_annot['sampleName'].values)
rpkms = rpkms[cells_in_use]
counts = counts[cells_in_use]
flag = rpkms > rpkm_filter
rpkms = rpkms[flag[flag].sum(axis=1) > ncell]
flag = counts > count_filter
counts = counts[flag[flag].sum(axis=1) > ncell]
spikeGenes = [ii for ii in counts.index if re.match('ERCC', ii) is not None]
skipGenes = [ii for ii in counts.index if re.match('eGFP', ii) is not None]

## most var genes
sigVarGenes = variable_genes(count_df, None, None, fdr, outdir, nTopGenes=1000)
sigVarGenes = list(set(sigVarGenes + var_genes))
comp = principle_component_analysis(rpkms.T, sigVarGenes, n_comp=30, 
                                    annot=None, annoGE=None, 
                                    pcPlot=False,markerPlot=False)
n_comp = choose_dims_N(comp)

## Plot by donors
plate = np.array([donor.split('_')[0] for donor in rpkms.columns])
reduce_plot(umap_loc_new.values, annot=plate, title='', prefix='UMAP', 
            size=50, fontsize=10, pcX=1, pcY=2, color_palette=None,
            figsize=(6,6), legend_loc='bottom', legend_ncol=3,
            add_sample_label=False, ordered_sample_labels=None, edgecolors='none')

log2rpkm = np.log2(rpkms.T+1)
reduce_plot_marker_expr(umap_loc_new.values, log2rpkm, geneNames=['Nphs1','Pecam1','Pdgfrb','Cldn1'], 
                        ncol=4, prefix='UMAP', pcX=1, pcY=2,
                        facecolor='lightgrey', size=50, colorbar_label = 'Expression (log2 RPKM)',
                        title_size=30, edgecolors='none', marker_cmap_str='YlOrRd')

reduce_plot_marker_expr(umap_loc_new.values, log2rpkm, geneNames=['Ptprc','Pck1','Aqp2','Umod'], 
                        ncol=2, prefix='UMAP', pcX=1, pcY=2,
                        facecolor='white', size=50, colorbar_label = 'Expression (log2 RPKM)',
                        title_size=30, edgecolors='none', marker_cmap_str='YlOrRd')

reduce_plot_marker_expr(umap_loc_new.values, log2rpkm, geneNames=['Wt1'], 
                        ncol=2, prefix='UMAP', pcX=1, pcY=2,
                        facecolor='white', size=50, colorbar_label = 'Expression (log2 RPKM)',
                        title_size=30, edgecolors='none', marker_cmap_str='YlOrRd')

## Plot by cell types
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

colot_panel = {'T+NK cells': colors['blueviolet'],
               'CD': colors['blue'],
               'PEC': colors['darkorange'],
               'Podocyte': colors['skyblue'],
               'MNP': colors['c'],
               'Neutrophil': colors['olive'],
               'MLC': colors['royalblue'],
               'PTC': colors['tan'],
               'EC 2': colors['purple'],
               'EC 1': colors['magenta'],
               'DCT 1': colors['salmon'],
               'DCT 2': colors['pink'],
               'DCT 3': colors['plum'],
               'DCT 4': colors['lime'],
               'cTAL': colors['aqua'],
               'B cells': colors['brown']}

new_color_panel = {new_ctype_labels[ctype]: colot_panel[ctype]  for ctype in colot_panel.keys()}
reduce_plot(umap_loc_new.values, annot=cell_type_annot.loc[rpkms.columns]['sampleCluster'], 
            title='', prefix='UMAP', 
            size=50, fontsize=10, pcX=1, pcY=2, color_palette=new_color_panel,
            figsize=(6,6), legend_loc='bottom', legend_ncol=3,
            add_sample_label=False, ordered_sample_labels=None, edgecolors='none')

## The number of detected genes in each cell type
flag = rpkms > 1
ngenes = flag[flag].sum(axis=0)
curr_ctypes = pd.concat([cell_type_annot.loc[ngenes.index], pd.DataFrame(ngenes, columns=['N_genes'])], axis=1)
curr_ctypes.columns = ['Cell types','Cell Name','N detected genes']

curr_ctypes['N detected genes'] = curr_ctypes['N detected genes'].astype('int')
curr_ctype_list = curr_ctypes['Cell types'].drop_duplicates().sort_values().to_list()
curr_col_list = [new_color_panel[ctype] for ctype in curr_ctype_list]

plt.figure(figsize=(10,3))
plt.rcParams['pdf.fonttype'] = 42
plt.tick_params(axis="both",bottom=True,top=False,left=True,right=False)

ax = sns.violinplot(x='Cell types',y='N detected genes',data=curr_ctypes, scale='width', 
                    cut=0, order=curr_ctype_list, palette=curr_col_list)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
sns.despine()
plt.savefig('%s/n_genes_in_cell_types.pdf' %(outdir), dpi=300, bbox_inches='tight')



