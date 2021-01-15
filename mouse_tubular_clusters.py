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

indir = 'mouse_tubular_clusters'
outdir = 'mouse_tubular_clusters/results'

## Inputs
rpkms = pd.read_table('%s/rpkms.csv' %(indir), header=0, index_col=0, sep="\t")
counts = pd.read_table('%s/counts.csv' %(indir), header=0, index_col=0, sep="\t")
cell_type_annotation = pd.read_table('%s/tubular_cell_types.csv' %(indir), header=0, index_col=None, sep="\t")

rpkm_filter = 1
count_filter = 2
ncell = 3

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

## Process data
cells_in_use = list(cell_type_annotation['sampleName'].to_list())
rpkms = rpkms[cells_in_use]
counts = counts[cells_in_use]

flag = rpkms > rpkm_filter
rpkms = rpkms[flag[flag].sum(axis=1) > ncell]

flag = counts > count_filter
counts = counts[flag[flag].sum(axis=1) > ncell]

spikeGenes = [ii for ii in counts.index if re.match('ERCC', ii) is not None]
skipGenes = [ii for ii in counts.index if re.match('eGFP', ii) is not None]

## significant variable genes
sigVarGenes = variable_genes(counts, spikeGenes, skipGenes, 0.01, outdir, nTopGenes=1000)

## DEGs between all cell types
if not os.path.exists('%s/DEGs' %(outdir)): os.makedirs('%s/DEGs' %(outdir))
rpkms.to_csv('%s/DEGs/expr.csv' %(outdir), sep="\t", index=True)
cell_type_annotation.index = cell_type_annotation['sampleName']

df = cell_type_annotation.loc[rpkms.columns]
df = df.replace(regex=r' ', value='_').replace(regex=r'/', value='_')
df.to_csv('%s/DEGs/cell_groups.csv' %(outdir), sep="\t", index=False)

grpList = ','.join(list(set(df['sampleCluster'].values)))
os.system('Rscript %s/diff_expr.R %s/DEGs/expr.csv %s/DEGs %s/DEGs/cell_groups.csv %s kw 0.01 5000 TRUE 4 v2 1 conover FALSE median FALSE' %(indir, outdir, outdir, outdir, outdir))

## Trajectory analysis
cell_annot = df.loc[rpkms.columns]

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy.api as sc
rcParams['pdf.fonttype'] = 42
sc.set_figure_params(color_map='viridis')

expr_data = rpkms.copy()
sample_cluster=cell_annot['sampleCluster']

X = expr_data.values
gene_names = list(expr_data.index)
sample_names = list(expr_data.columns)
    
adata = sc.AnnData(X.transpose())
adata.var_names = gene_names
adata.row_names = sample_names

sc.settings.figdir = outdir

genes = list(set(list(set(sigVarGenes))
adata = adata[:, genes] 
adata.var_names = [gene.split('|')[0] for gene in adata.var_names]
adata.obs['Cell_types'] = cell_annot.loc[expr_data.columns]['sampleCluster'].to_list()
                 
adata.uns['Cell_types_colors'] = [colors['blue'],colors['salmon'],colors['pink'],
                                  colors['plum'],colors['lime'],colors['green'], 
                                  colors['goldenrod'], 
                                  colors['aqua']]
                 
root_cell = list(adata.obs['Cell_types'][adata.obs['Cell_types'] == 'PTC'].index)[0]
adata.uns['iroot'] = int(root_cell)
sc.pp.log1p(adata)
                 
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=25)
sc.tl.louvain(adata, resolution=1)

rcParams['figure.figsize'] = [6,6]
sc.tl.umap(adata, n_components=2, random_state=0)
sc.pl.umap(adata, color=['Cell_types','louvain'],projection='2d', legend_loc='on data')
                 
sc.pl.umap(adata, color=['Cell_types'],projection='2d', legend_loc='on data',frameon=False)

### violin plot
curr_markers = ['Slc12a1','Nos1','Ptgs2','Slc9a2']
marker_gid = [gene for gene in expr_data.index if gene.split('|')[0] in curr_markers]
tmp_expr = expr_data.loc[marker_gid]
tmp_expr.index = [gene.split('|')[0] for gene in tmp_expr.index]

tmp_expr_list = []
tmp_gname_list = []
tmp_ctype_list = []
for gene in tmp_expr.index:
    curr_values = list(np.log2(tmp_expr.loc[gene].values+1))
    curr_genes = [gene for val in curr_values]
    curr_ctypes = cell_annot.loc[tmp_expr.columns]['sampleCluster'].to_list()
    
    tmp_expr_list.extend(curr_values)
    tmp_gname_list.extend(curr_genes)
    tmp_ctype_list.extend(curr_ctypes)

curr_df = pd.DataFrame([tmp_expr_list, tmp_gname_list, tmp_ctype_list], index=['Expression','Gene','CellType']).T
curr_df['Expression'] = curr_df['Expression'].astype(float)
                 
plt.figure(figsize=(15,2))
sns.set_style('ticks')
rcParams['pdf.fonttype'] = 42

sns.violinplot(x='Gene',y='Expression',hue='CellType', data=curr_df, scale='width', 
               cut=0, linewidth=0.6, inner='quartile',
               palette=['lime','pink','goldenrod','aqua','salmon','green','blue','plum'])
sns.despine()
plt.legend(bbox_to_anchor=(1., 1.1))
plt.savefig('%s/Macula_densa_marker_expression_violin.pdf' %(outdir), bbox_inches='tight', dpi=300)

### trajectory
sc.tl.diffmap(adata)
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_diffmap')
sc.tl.dpt(adata)
sc.pl.diffmap(adata, color=['Cell_types'], projection='2d', 
              save='dpt_plot_color_by_cell_types.pdf', frameon=False)

sc.pl.diffmap(adata, color=['dpt_pseudotime'], projection='2d', 
              frameon=False, color_map='viridis')

sc.tl.paga(adata, groups='Cell_types')
                 
sc.pl.paga(adata, color=['Cell_types'], threshold=0.05, edge_width_scale=0.8, 
           node_size_scale=5, 
           layout='eq_tree', random_state=0, frameon=False,
           fontsize=20)

