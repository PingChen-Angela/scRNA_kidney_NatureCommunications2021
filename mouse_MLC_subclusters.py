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

indir = 'mouse_MLC_subcluster'
outdir = 'mouse_MLC_subcluster/results'

# - negative selected data
neg_rpkms = pd.read_table('%s/neg_mlc_rpkms.csv' %(indir), sep="\t", header=0, index_col=0)
neg_counts = pd.read_table('%s/neg_mlc_counts.csv' %(indir), sep="\t", header=0, index_col=0)
neg_cell_annot = pd.read_table('%s/neg_mlc_annotation.csv' %(indir), sep="\t", header=0, index_col=0)

# - unbiased sorted data
unb_rpkms = pd.read_table('%s/mlc_rpkms.csv' %(indir), header=0, index_col=0, sep="\t")
unb_counts = pd.read_table('%s/mlc_counts.csv' %(indir), header=0, index_col=0, sep="\t")
cells_in_use = list(unb_cell_annot.query('sampleCluster=="Mesangial/SMC/Renin"')['sampleName'].values)
unbiased_rpkms = unb_rpkms[cells_in_use]
unbiased_counts = unb_counts[cells_in_use]
unbiased_cell_annot = pd.read_table('%s/unb_ctype_annotation.csv', header=0, index_col=None, sep="\t")

# process data
comm_cells = list(set(unbiased_rpkms.columns) & set(unbiased_cell_annot['sampleName']))
unbiased_rpkms = unbiased_rpkms[comm_cells]
unbiased_counts = unbiased_counts[comm_cells]
unbiased_cell_annot.index = unbiased_cell_annot['sampleName']
unbiased_cell_annot = unbiased_cell_annot.loc[comm_cells]

comm_genes = list(set(unbiased_rpkms.index) & set(neg_rpkms.index))
unbiased_rpkms = unbiased_rpkms.loc[comm_genes]
neg_rpkms = neg_rpkms.loc[comm_genes]
rpkms = pd.concat([unbiased_rpkms,neg_rpkms], axis=1)

comm_genes = list(set(unbiased_counts.index) & set(neg_counts.index))
unbiased_counts = unbiased_counts.loc[comm_genes]
neg_counts = neg_counts.loc[comm_genes]
counts = pd.concat([unbiased_counts,neg_counts], axis=1)

cell_annot = pd.concat([unbiased_cell_annot,neg_cell_annot], axis=0)
cell_annot = cell_annot.loc[rpkms.columns]

rpkm_filter = 1
count_filter = 2
ncell = 3
fdr = 0.01

flag = rpkms > rpkm_filter
rpkms = rpkms[flag[flag].sum(axis=1) > ncell]

flag = counts > count_filter
counts = counts[flag[flag].sum(axis=1) > ncell]

spikeGenes = [ii for ii in counts.index if re.match('ERCC_', ii) is not None]
skipGenes = [ii for ii in counts.index if re.match('eGFP', ii) is not None]

# Batch correction
batch_ids = [sname.split('_')[0] for sname in counts.columns]
sample_annot = pd.DataFrame(np.array([list(counts.columns),batch_ids]), index=['SampleName','Batch']).T

selected_batches = sample_annot['Batch'].value_counts()
selected_batches = list(selected_batches[selected_batches>2].index)
selected_sample_annot = sample_annot[sample_annot['Batch'].isin(selected_batches)]
selected_sample_annot.to_csv('%s/sample_batch.csv' %(outdir), sep='\t', index=None)

new_counts = counts[list(selected_sample_annot['SampleName'].values)]
new_rpkms = rpkms[list(selected_sample_annot['SampleName'].values)]

os.system('Rscript %s/batch_effects.R %s/messangial_like_count.csv %s %s/sample_batch.csv %s %s TRUE FALSE NULL' %(indir, outdir, outdir, outdir, 'count', 'Batch'))
corrected_count_df = pd.read_table('%s/combat_corrected_count.csv' %(outdir), index_col=0, header=0)

os.system('Rscript %s/batch_effects.R %s/messangial_like_expr.csv %s %s/sample_batch.csv %s %s TRUE FALSE NULL' %(indir, outdir, outdir, outdir, 'rpkm', 'Batch'))
corrected_rpkm_df = pd.read_table('%s/combat_corrected_rpkm.csv' %(outputDIR), index_col=0, header=0)

flag = corrected_rpkm_df > rpkm_filter
corrected_rpkm_df = corrected_rpkm_df[flag[flag].sum(axis=1) > ncell]

flag = corrected_count_df > count_filter
corrected_count_df = corrected_count_df[flag[flag].sum(axis=1) > ncell]

counts = corrected_count_df.copy()
rpkms = corrected_rpkm_df.copy()

# Select top biological variable genes from all cells
sigVarGenes = variable_genes(counts, spikeGenes, skipGenes, fdr, outdir, nTopGenes=1500)

# clustering
comp = principle_component_analysis(rpkms.T, sigVarGenes, n_comp=30, 
                                    annot=None, annoGE=None, 
                                    pcPlot=False,markerPlot=False)

n_comp = choose_dims_N(comp)

comp = principle_component_analysis(rpkms.T, sigVarGenes, n_comp=n_comp, 
                                    annot=None, annoGE=None, log=True,
                                    pcPlot=False,pcPlotType='normal',markerPlot=False,
                                    pcX=1, pcY=2, figsize=(6,6))

sample_clusters = clustering(comp)
sample_annot = pd.DataFrame.from_dict(sample_clusters)
sample_annot.index = sample_annot['sampleName']

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colot_panel = {'MLC-C1': colors['darkorange'],
               'MLC-C4': colors['red'],
               'MLC-C2': colors['cyan'],
               'MLC-C3': colors['lime']}

# DEGs between 4 clusters
if not os.path.exists('%s/DEGs_mesang_clusters' %(outdir)): 
    os.makedirs('%s/DEGs_mesang_clusters' %(outdir))
    
sample_annot.to_csv('%s/DEGs_mesang_clusters/merged_cell_cluster_annot.csv' %(outdir), 
                    index=False, sep="\t")

rpkms[sample_annot.index].to_csv('%s/DEGs_mesang_clusters/expr.csv' %(outdir), index=True, sep="\t")

os.system('Rscript %s/diff_expr.R %s/DEGs_mesang_clusters/expr.csv %s/DEGs_mesang_clusters %s/DEGs_mesang_clusters/merged_cell_cluster_annot.csv MLC-mix-C1,MLC-mix-C2,MLC-mix-C3,MLC-mix-C4 kw 0.05 5000 TRUE 1.5 v2 1 conover FALSE mean FALSE' %(indir, outdir, outdir, outdir))

sigGenes = pd.read_table('%s/DEGs_mesang_clusters/sigGeneStats.csv' %(outdir), index_col=0, header=0)
rankT = pd.read_table('%s/DEGs_mesang_clusters/rankedGeneStats.csv' %(outdir), index_col=0, header=0)

##top 15 DEGs
degT = rankT.copy()
genes = [gene for gene in degT.index if not re.match('ERCC_', gene)]
degT = degT.loc[genes]

nn = 15
rankType = 'range'
all_grps = ['mlc-mix-c1','mlc-mix-c2','mlc-mix-c3','mlc-mix-c4']
c1_genes = get_top_genes(degT, all_grps=all_grps, targetGroup='mlc-mix-c1', ntop=nn, rankType=rankType)
c2_genes = get_top_genes(degT, all_grps=all_grps, targetGroup='mlc-mix-c2', ntop=nn, rankType=rankType)
c3_genes = get_top_genes(degT, all_grps=all_grps, targetGroup='mlc-mix-c3', ntop=nn, rankType=rankType)
c4_genes = get_top_genes(degT, all_grps=all_grps, targetGroup='mlc-mix-c4', ntop=nn, rankType=rankType)

topGenesT = pd.concat([c1_genes, c2_genes, c3_genes, c4_genes], axis=0)
sig_gene_expr = np.log2(rpkms.loc[topGenesT.index]+1)

sample_cluster_df = sample_annot.copy()
sample_cluster_df.index = sample_annot['sampleName']
sample_label = sample_cluster_df.loc[sig_gene_expr.columns]['sampleCluster']

sig_gene_expr.index = [gname.split('|')[0] for gname in sig_gene_expr.index]
heatmap_gene_expr(sig_gene_expr, clust_method='ward', sample_label=sample_label,
                  fig_width=6, fig_height=12, font_scale=1, 
                  color_palette=[colors['darkorange'],colors['cyan'],colors['lime'],colors['red']], 
                  legend_x=0.75, legend_y=4.5, 
                  fontsize=12,linewidths=0,
                  cbar=[1.05, .13, .03, .1], cmap_str='jet')

# DEGs between cluster2, cluster3 and cluster4
if not os.path.exists('%s/DEGs_mesang_cluster2_vs_cluster3_vs_cluster4' %(outdir)): 
    os.makedirs('%s/DEGs_mesang_cluster2_vs_cluster3_vs_cluster4' %(outdir))
    
os.system('Rscript %s/diff_expr.R %s/DEGs_mesang_cluster1_vs_cluster4/expr.csv %s/DEGs_mesang_cluster2_vs_cluster3_vs_cluster4 %s/DEGs_mesang_cluster2_vs_cluster3_vs_cluster4/merged_cell_cluster_annot.csv MLC-mix-C2,MLC-mix-C3,MLC-mix-C4 kw 0.05 5000 TRUE 1 v2 1 conover FALSE mean FALSE' %(indir, outdir, outdir, outdir))

# DEGs between cluster1 and cluster4
if not os.path.exists('%s/DEGs_mesang_cluster1_vs_cluster4' %(outdir)): 
    os.makedirs('%s/DEGs_mesang_cluster1_vs_cluster4' %(outdir))
    
sample_annot.to_csv('%s/DEGs_mesang_cluster1_vs_cluster4/merged_cell_cluster_annot.csv' %(outdir), index=False, sep="\t")

os.system('Rscript %s/diff_expr.R %s/DEGs_mesang_clusters/expr.csv %s/DEGs_mesang_cluster1_vs_cluster4 %s/DEGs_mesang_cluster1_vs_cluster4/merged_cell_cluster_annot.csv MLC-mix-C1,MLC-mix-C4 kw 0.05 5000 TRUE 1 v2 1 conover FALSE mean FALSE' %(indir, outdir, outdir, outdir))

sigGenes = pd.read_table('%s/DEGs_mesang_cluster1_vs_cluster4/sigGeneStats.csv' %(outputDIR), index_col=0, header=0)
rankT = pd.read_table('%s/DEGs_mesang_cluster1_vs_cluster4/rankedGeneStats.csv' %(outputDIR), index_col=0, header=0)

## top 30 DEGs
degT = rankT.copy()
genes = [gene for gene in degT.index if not re.match('ERCC_', gene)]
degT = degT.loc[genes]

nn = 30
rankType = 'range'
all_grps = ['mlc-mix-c1','mlc-mix-c4']
c1_genes = get_top_genes(degT, all_grps=all_grps, targetGroup='mlc-mix-c1', ntop=nn, rankType=rankType)
c4_genes = get_top_genes(degT, all_grps=all_grps, targetGroup='mlc-mix-c4', ntop=nn, rankType=rankType)

topGenesT = pd.concat([c1_genes, c4_genes], axis=0)

expr = np.log2(rpkms.loc[topGenesT.index]+1)
expr.index = [gname.split('|')[0] for gname in expr.index]
annotation = pd.DataFrame(sample_annot.copy()['sampleCluster'])
annotation = annotation.loc[expr.columns]
annotation.columns = ['Level1']

annotation = annotation.loc[annotation['Level1'].isin(['MLC-C1','MLC-C4'])]
expr = expr[annotation.index]

l1_color = {'MLC-C1': colors['darkorange'], 
            'MLC-C4': colors['red']}

heatmap_gene_expr_by_individual(expr, pd.DataFrame(annotation), cm.jet,
                                l1_color=l1_color, 
                                yaxis_label_x_coords=-0.4,
                                legend_x=1.2, legend_y=0.75, l2_legend=False, 
                                colorbar_rect=[0.95, 0.15, 0.02, 0.1], spine_linewidth=0.1,
                                figsize=(3,10), fontsize=10)

# Trajectory analysis
cell_subcluster_annot = sample_annot.loc[rpkms.columns]

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy.api as sc
sc.set_figure_params(color_map='viridis')
rcParams['pdf.fonttype'] = 42

expr_data = rpkms.copy()
sample_cluster=cell_annot['sampleCluster']

X = expr_data.values
gene_names = list(expr_data.index)
sample_names = list(expr_data.columns)
    
adata = sc.AnnData(X.transpose())
adata.var_names = gene_names
adata.row_names = sample_names

sc.settings.figdir = outdir
adata = adata[:, sigVarGenes] 
adata.var_names = [gene.split('|')[0] for gene in adata.var_names]
adata.obs['Cell_types'] = [cell_subcluster_annot.loc[sample]['sampleCluster'] for sample in sample_names]
adata.obs['Smartseq2 plates'] = [donor.split('_')[0] for donor in sample_names]
adata.obs['MLC subclusters in each dataset'] = [cell_annot.loc[sample]['sampleCluster'] for sample in sample_names] 

colot_panel = {'MLC-C1': colors['darkorange'],
               'MLC-C4': colors['red'],
               'MLC-C2': colors['cyan'],
               'MLC-C3': colors['lime']}

adata.uns['Cell_types_colors'] = [colors['darkorange'],colors['cyan'],colors['lime'],colors['red']]
root_cell = list(adata.obs['Cell_types'][adata.obs['Cell_types'] == 'MLC-C2'].index)[0]

adata.uns['iroot'] = int(root_cell)
sc.pp.log1p(adata) 
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=10)
sc.tl.louvain(adata, resolution=1)

rcParams['figure.figsize'] = [6,6]
sc.tl.umap(adata, n_components=2, random_state=1)
sc.pl.umap(adata, color=['Cell_types','louvain'],projection='2d', legend_loc='on data')
sc.pl.umap(adata, color=['Smartseq2 plates','MLC subclusters in each dataset'],projection='2d')
sc.pl.umap(adata, color=['Cnn1','Acta2','Ren1'],projection='2d', cmap='YlOrRd',legend_loc='on data')
sc.pl.umap(adata, color=['Pdgfrb','Pdgfra','Gata3'],projection='2d', cmap='YlOrRd',legend_loc='on data')

sc.tl.diffmap(adata)
sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_diffmap')
sc.tl.dpt(adata)
sc.tl.paga(adata, groups='Cell_types')

sc.pl.paga_compare(adata, threshold=0.03, title='', right_margin=0, size=150, edge_width_scale=1.5, node_size_scale=5,
                   legend_fontsize=15, fontsize=15, frameon=False, edges=True, save=True, legend_loc='on data')