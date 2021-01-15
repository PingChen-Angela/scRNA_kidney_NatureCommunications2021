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

indir = 'pdgfrb_mice_cell_types'
outdir = 'pdgfrb_mice_cell_types/results'

## Input data
rpkm_file = '%s/refseq_rpkms.txt' %(indir)
qc_star_folder = '%s/qc_star' %(indir)
rpkm_filter = 1
count_filter = 2
ncell = 3
fdr = 0.01

## Data preparation
expr_dict = read_rpkmforgenes_exprMatrix(rpkm_file, None, None, rpkm_filter, count_filter, ncell)
org_rpkm_df = expr_dict['rpkm']
org_count_df = expr_dict['count']

spikeGenes = [ii for ii in org_count_df.index if re.match('ERCC_', ii) is not None]
skipGenes = [ii for ii in org_count_df.index if re.match('eGFP', ii) is not None]

## Select good quality cells
selected = filter_QCstats(qc_star_folder, org_rpkm_df, nreads=50000, 
                          unique_mapping=0.4, exon_mapping=0.4,
                          n_genes_detected=500, min_rpkm=1, rm_ercc=False,  
                          bar_width = 0.3, bins=50)

cells_in_use = [cell for cell in selected['samples'] if re.match('665s_',cell)]

rpkm_df = org_rpkm_df[cells_in_use]
count_df = org_count_df[cells_in_use]

flag = rpkm_df > rpkm_filter
rpkm_df = rpkm_df[flag[flag].sum(axis=1) > ncell]

flag = count_df > count_filter
count_df = count_df[flag[flag].sum(axis=1) > ncell]

## most variable genes
sigVarGenes = variable_genes(count_df, spikeGenes, skipGenes, fdr, outdir, nTopGenes=1000)
comp = principle_component_analysis(rpkm_df.T, sigVarGenes, n_comp=30, 
                                    annot=None, annoGE=marker_index_names, 
                                    pcPlot=False,markerPlot=False)
n_comp = choose_dims_N(comp['PCmatrix'])

## Cell type annotation
comp = principle_component_analysis(rpkm_df.T, sigVarGenes, n_comp=n_comp, 
                                    annot=None, annoGE=None, log=True,
                                    pcPlot=False,pcPlotType='normal',markerPlot=False,
                                    pcX=1, pcY=2, figsize=(6,6))

cell_type_annot = clustering(comp, 'AP')
cell_type_annot.index = cell_type_annot['sampleName']
cell_type_annot = cell_type_annot.replace('cluster1','GEC')
cell_type_annot = cell_type_annot.replace('cluster2','MC')
cell_type_annot = cell_type_annot.replace('cluster3','MLC')
cell_type_annot = cell_type_annot.replace('cluster4','TubuleC')
cell_type_annot = cell_type_annot.replace('cluster5','PEC')

genes = ['Pdgfrb']
gene_ids = [gene for gene in rpkm_df.index if gene.split('|')[0] in genes]

## UMAP
plt.rcParams["axes.grid"] = False

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

color_panel = {
               'PEC': colors['darkorange'],
               'MLC': colors['cyan'],
               'TubuleC': colors['brown'],
               'GEC': colors['magenta'],
               'MC': colors['blue']}

umap_comp = umap_emb(rpkm_df.T, sigVarGenes, n_comp=2, 
                annot=cell_type_annot.loc[rpkm_df.columns]['sampleCluster'], 
                annoGE=gene_ids, init='pca', init_n_comp=n_comp,
                n_neighbors=15, metric='euclidean', min_dist=0.5, 
                initial_embed='spectral',
                log=True, markerPlot=True, pcPlot=True, pcPlotType='normal', 
                pcX=1, pcY=2, prefix='',
                facecolor='lightgrey', markerPlotNcol=5, fontsize=10, 
                random_state=0, size=100, with_mean=True,
                color_palette=color_panel,
                with_std=False, figsize=(6,6), legend_loc='bottom', legend_ncol=3)

new_ctype_annot = cell_type_annot.copy()
n_cells = new_ctype_annot.groupby(by="sampleCluster").apply(lambda x: x.shape[0])
new_ctype_labels = {ctype: '%s (%s)' %(ctype, n_cells[ctype]) for ctype in n_cells.keys()}
new_ctype_annot['sampleCluster'] = [new_ctype_labels[item] for item in new_ctype_annot['sampleCluster'].values]
new_color_panel = {new_ctype_labels[ctype]: color_panel[ctype]  for ctype in color_panel.keys()}

umap_comp = umap_emb(rpkm_df.T, sigVarGenes, n_comp=2, 
                annot=new_ctype_annot.loc[rpkm_df.columns]['sampleCluster'], 
                annoGE=gene_ids, init='pca', init_n_comp=n_comp,
                n_neighbors=15, metric='euclidean', min_dist=0.5, 
                initial_embed='spectral',
                log=True, markerPlot=False, pcPlot=True, pcPlotType='normal', 
                pcX=1, pcY=2, prefix='',
                facecolor='lightgrey', markerPlotNcol=5, fontsize=10, 
                random_state=0, size=100, with_mean=True,
                color_palette=new_color_panel,
                with_std=False, figsize=(6,6), legend_loc='bottom', legend_ncol=3)

## The number of detected genes in each cell type
flag = rpkm_df > 1
ngenes = flag[flag].sum(axis=0)
curr_ctypes = pd.concat([new_ctype_annot.loc[ngenes.index], pd.DataFrame(ngenes, columns=['N_genes'])], axis=1)
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

## SCENIC TF
pdbfrb_indir = "SCENIC/results/pdgfrb_mmu/output"
pdbfrb_regulon_binary_file = "%s/regulons_binary_subset.csv" %(pdbfrb_indir)
pdbfrb_cell_type_file = "%s/cell_type_info.csv" %(pdbfrb_indir)

pdbfrb_regulon_binary = pd.read_table(pdbfrb_regulon_binary_file, header=0, index_col=0, sep="\t")
pdbfrb_cell_type_annot = pd.read_table(pdbfrb_cell_type_file, header=0, index_col=0, sep="\t")
pdbfrb_pec = list(pdbfrb_cell_type_annot.query('CellType=="PEC"').index)

curr_pdbfrb_data = pdbfrb_regulon_binary.copy()
curr_pdbfrb_data.index = [gene.split(' (')[0] for gene in curr_pdbfrb_data.index]
curr_pdbfrb_data.index = [gene.replace('_extended','') for gene in curr_pdbfrb_data.index]

plt.rcParams["axes.grid"] = False
reduce_plot_marker_expr(umap_comp.values, curr_pdbfrb_data[umap_comp.index].T, 
                        geneNames=['Wt1'], ncol=3, prefix='PC', pcX=1, pcY=2,
                        facecolor='white', size=100, colorbar_label = 'Binary regulon activity',
                        title_size=30, edgecolors='gainsboro', marker_cmap_str='cool')