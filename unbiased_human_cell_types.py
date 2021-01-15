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
from rpy2.robjects import pandas2ri
pandas2ri.activate()

indir = 'unbiased_human_cell_types'
outdir = 'unbiased_human_cell_types/results'

## Input data
rpkm_df = pd.read_table('%s/rpkms.csv' %(indir), header=0, sep="\t", index_col=0)
count_df = pd.read_table('%s/counts.csv' %(indir), header=0, sep="\t", index_col=0)
sample_cluster_annot = pd.read_table('%s/sample_cluster.csv' %(indir), header=0, index_col=None, sep="\t")

## Parameters
rpkm_filter = 1
count_filter = 2
ncell = 3
fdr = 0.05

sample_cluster_annot.index = sample_cluster_annot['sampleName']
cells_in_use = [cell for cell in rpkm_df.columns if cell.split('_')[0] not in ['308s']]

rpkm_df = rpkm_df[cells_in_use]
count_df = count_df[cells_in_use]
sample_cluster_annot = sample_cluster_annot.loc[cells_in_use]

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

## most variable genes
sigVarGenes = variable_genes(count_df, None, None, fdr, outdir, nTopGenes=2000)
comp = principle_component_analysis(rpkm_df.T, sigVarGenes, n_comp=30, 
                                    annot=None, annoGE=None, 
                                    pcPlot=False,markerPlot=False, 
                                    with_mean=True, with_std=False)
n_comp = choose_dims_N(comp)

## plot by donors
donors = [sample.split('_')[0] for sample in rpkm_df.columns]
sample_annot = pd.DataFrame([rpkm_df.columns, donors], index=['SampleName','Donor']).T

comp = principle_component_analysis(rpkm_df.T, sigVarGenes, n_comp=n_comp, 
                                    annot=np.array(donors), 
                                    annoGE=None, pcPlotType='normal',
                                    pcPlot=False,markerPlot=False, log=True,
                                    with_mean=True, with_std=False, figsize=(6,6))

umap_comp = umap_emb(rpkm_df.T, sigVarGenes, n_comp=2, annot=np.array(donors), 
                     annoGE=None, 
                     init='pca', init_n_comp=n_comp,
                     n_neighbors=15, metric='euclidean', min_dist=0.7, initial_embed='spectral',
                     log=True, markerPlot=False, pcPlot=True, pcPlotType='normal', pcX=1, pcY=2,
                     facecolor='lightgrey', markerPlotNcol=5, fontsize=10, random_state=0, size=60, with_mean=True,
                     with_std=False, color_palette=None, figsize=(6,6), title='', prefix='',  
                     legend_loc='bottom', legend_ncol=3)

umap_comp = umap_emb(rpkm_df.T, sigVarGenes, n_comp=2, annot=np.array(donors), 
                     annoGE=['NPHS1','PECAM1','PDGFRB'], 
                     init='pca', init_n_comp=n_comp,
                     n_neighbors=15, metric='euclidean', min_dist=0.7, initial_embed='spectral',
                     log=True, markerPlot=True, pcPlot=False, pcPlotType='normal', pcX=1, pcY=2,
                     facecolor='lightgrey', markerPlotNcol=2, fontsize=10, random_state=0, size=60, with_mean=True,
                     with_std=False, color_palette=None, figsize=(6,6), title='', prefix=''legend_loc='bottom', legend_ncol=3)

umap_comp = umap_emb(rpkm_df.T, sigVarGenes, n_comp=2, annot=np.array(donors), 
                     annoGE=['PTPRC','PCK1','AQP2','UMOD'], 
                     init='pca', init_n_comp=n_comp,
                     n_neighbors=15, metric='euclidean', min_dist=0.7, initial_embed='spectral',
                     log=True, markerPlot=True, pcPlot=False, pcPlotType='normal', pcX=1, pcY=2,
                     facecolor='lightgrey', markerPlotNcol=2, fontsize=10, random_state=0, size=60, with_mean=True,
                     with_std=False, color_palette=None, figsize=(6,6), title='', prefix='',
                     legend_loc='bottom', legend_ncol=3)

## Plot by cell types
comp = principle_component_analysis(rpkm_df.T, sigVarGenes, n_comp=n_comp, 
                                    annot=None, annoGE=None, log=True,
                                    pcPlot=False,pcPlotType='normal',markerPlot=False)

cell_type_annot = sample_cluster_annot.copy()
cell_type_annot.loc[cell_type_annot['sampleCluster'] == 'cluster1','sampleCluster'] = 'MLC'
cell_type_annot.loc[cell_type_annot['sampleCluster'] == 'cluster2','sampleCluster'] = 'T+NK cells'
cell_type_annot.loc[cell_type_annot['sampleCluster'] == 'cluster3','sampleCluster'] = 'GEC'
cell_type_annot.loc[cell_type_annot['sampleCluster'] == 'cluster4','sampleCluster'] = 'cTAL+CD'
cell_type_annot.loc[cell_type_annot['sampleCluster'] == 'cluster5','sampleCluster'] = 'Podocyte'
cell_type_annot.loc[cell_type_annot['sampleCluster'] == 'cluster6','sampleCluster'] = 'DTL'
cell_type_annot.loc[cell_type_annot['sampleCluster'] == 'cluster7','sampleCluster'] = 'unknown'
cell_type_annot.loc[cell_type_annot['sampleCluster'] == 'cluster8','sampleCluster'] = 'MNP'
cell_type_annot.loc[cell_type_annot['sampleCluster'] == 'cluster9','sampleCluster'] = 'PTC'
cell_type_annot.loc[cell_type_annot['sampleCluster'] == 'cluster10','sampleCluster'] = 'B cells'

cell_type_annot.index = cell_type_annot['sampleName']
n_cells = cell_type_annot.groupby(by="sampleCluster").apply(lambda x: x.shape[0])
new_ctype_labels = {ctype: '%s (%s)' %(ctype, n_cells[ctype]) for ctype in n_cells.keys()}
cell_type_annot['sampleCluster'] = [new_ctype_labels[item] for item in cell_type_annot['sampleCluster'].values]

colot_panel = {'T+NK cells': colors['blueviolet'],
               'DTL': colors['orange'],
               'Podocyte': colors['skyblue'],
               'Monocyte': colors['c'],
               'MLC': colors['royalblue'],
               'PTC': colors['tan'],
               'GEC': colors['magenta'],
               'cTAL+CD': colors['darkblue'],
               'B cells': colors['brown']}

new_color_panel = {new_ctype_labels[ctype]: colot_panel[ctype]  for ctype in colot_panel.keys()}

umap_comp = umap_emb(rpkm_df.T, sigVarGenes, n_comp=2, 
                     annot=cell_type_annot.loc[rpkm_df.columns]['sampleCluster'], 
                     annoGE=None, 
                     init='pca', init_n_comp=n_comp,
                     n_neighbors=15, metric='euclidean', min_dist=0.7, initial_embed='spectral',
                     log=True, markerPlot=False, pcPlot=True, pcPlotType='normal', pcX=1, pcY=2,
                     facecolor='black', markerPlotNcol=4, fontsize=10, random_state=0, size=60, with_mean=True,
                     with_std=False, color_palette=new_color_panel, figsize=(6,6), title='', prefix='', 
                     legend_loc='right', legend_ncol=3)

## The number of detected genes in each cell type
flag = rpkm_df > 1
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
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
sns.despine()
plt.savefig('%s/n_genes_in_cell_types.pdf' %(outdir), dpi=300, bbox_inches='tight')

## DEGs among cell types
curr_cell_type_annot = cell_type_annot.copy()
curr_cell_type_annot.index = curr_cell_type_annot['sampleName'].to_list()

if not os.path.exists('%s/DEGs' %outdir): os.makedirs('%s/DEGs' %outdir)
curr_rpkms = rpkm_df[curr_cell_type_annot.index]
curr_cell_type_annot = curr_cell_type_annot.replace(' ','_',regex=True)
grp_strings = ','.join(list(set(curr_cell_type_annot['sampleCluster'].to_list())))
curr_rpkms.to_csv('%s/DEGs/expr.csv' %(outdir), sep='\t')
curr_cell_type_annot.to_csv('%s/DEGs/sample_clusters.csv' %(outdir), sep='\t', index=False)

os.system('Rscript %s/diff_expr.R %s/DEGs/expr.csv %s/DEGs %s/DEGs/sample_clusters.csv %s kw 0.01 5000 TRUE 3 v2 1 conover FALSE median FALSE' %(indir, outdir, outdir, outdir, grp_strings))

sigGenes = pd.read_table('%s/DEGs/rankedGeneStats.csv' %(outdir), index_col=0, header=0)
dtl_genes = sigGenes.query('sigCluster=="dtl"')
sig_gene_expr = np.log2(curr_rpkms.loc[dtl_genes.index]+1)

sample_cluster_df = curr_cell_type_annot.copy()
sample_cluster_df.index = sample_cluster_df['sampleName']
sample_label = sample_cluster_df.loc[sig_gene_expr.columns]['sampleCluster']
sig_gene_expr.index = [gname.split('|')[0] for gname in sig_gene_expr.index]

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy.api as sc
mpl.rcParams['pdf.fonttype'] = 42
sc.set_figure_params(color_map='viridis')

expr_data = sig_gene_expr.copy()

X = expr_data.values
gene_names = list(expr_data.index)
sample_names = list(expr_data.columns)
    
adata = sc.AnnData(X.transpose())
adata.var_names = gene_names
adata.row_names = sample_names
sc.settings.figdir = '%s/DEGs' %outdir
adata.obs['Cell types'] = list(sample_label.values)

plt.rcParams['axes.labelsize'] = 15
sc.pl.dotplot(adata, gene_names, groupby='Cell types', 
              log=False, standard_scale='var',
              expression_cutoff=1, fontsize=10, color_map='RdPu', figsize=(8,2))

# immune cell markers
immune_cell_markers = ['CD3D','CD3G','TRAC','CD4','CD8A','NKG7',
                       'MS4A1','HLA-DRA','CD19','CD14','FCGR3A',
                       'FCN1','VCAN','LYZ','FCGR3B','S100P',
                       'PRF1','NCAM1','KLRK1','KLRD1','CD68','CSF3R']

immune_cells_annot = cell_type_annot.loc[cell_type_annot['sampleCluster'].isin(['B cells (3)','Monocyte (263)','T+NK cells (58)'])].copy()

immune_cell_marker_expr = rpkm_df[immune_cells_annot.index].loc[immune_cell_markers].T.copy()
immune_cell_marker_expr.columns = [gene.split('|')[0] for gene in immune_cell_marker_expr.columns]
immune_data = pd.concat([immune_cells_annot, immune_cell_marker_expr], axis=1)

# boxplot
import seaborn as sns
sns.set_style("ticks")
plt.figure(figsize=(6,4))
cell_type_color_map = {'B cells (3)': 'brown',
                       'Monocyte (263)': 'c',
                       'T+NK cells (58)': 'blueviolet'}

idx = 1
for cell_type in ['B cells (3)','Monocyte (263)','T+NK cells (58)']:
    
    curr_data = immune_data.query('sampleCluster=="%s"' %(cell_type))
    curr_values = curr_data[immune_cell_marker_expr.columns].T.values.tolist()
    curr_values = np.array(sum(curr_values,[]))
    curr_genes = sum([[gene for i in range(curr_data.shape[0])] for gene in immune_cell_marker_expr.columns],[])
    tmp = pd.DataFrame([list(np.log2(curr_values+1)),curr_genes], index=['Expression','Genes']).T
    tmp['Expression'] = tmp['Expression'].astype(float)
    
    plt.subplot(int('14%s' %(idx)))
    ax = sns.violinplot(x="Expression", y="Genes", data=tmp, cut=0, scale='width',#fliersize=0, 
                     color=cell_type_color_map[cell_type], inner=None, linewidth=1)
    
    for mybox in ax.artists: mybox.set_edgecolor('black')
    
    plt.setp(ax.lines, color="black")
    sns.despine()
    ax.set_xlabel(cell_type, fontsize=10)
    ax.set_ylabel('')  
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    if idx>1: 
        ax.set_yticks([])
        #ax.set_xticks([])
    
    idx = idx + 1
    
plt.savefig("%s/immune_genes_boxplot.pdf" %(outdir), dpi=150, format='pdf')
