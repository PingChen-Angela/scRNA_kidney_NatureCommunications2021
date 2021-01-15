library(SCENIC)
library(SingleCellExperiment, verbose=FALSE)

ncores = 35

## prepare input data
## load input data
load(file = "SCENIC/data/kidney_mouse_data.rda")

# outdir
outdir = 'SCENIC/results/mouse'
setwd(outdir)

# select cells
selected_cells = cell_type_annot[,"sampleName"]
exprMat = rpkms[,selected_cells]
exprMat = as.matrix(exprMat)
rownames(cell_type_annot) = cell_type_annot[,'sampleName']
cellInfo = cell_type_annot[selected_cells,]
colnames(cellInfo) = c("CellType","Cell")
cellInfo$Cell = NULL

# Color to assign to the variables (same format as for NMF::aheatmap)
colVars <- list(CellType=c("T+NK cells"="blueviolet", 
                           "CD"="blue", 
                           "PEC"="darkorange", 
                           "Podocyte"="skyblue", 
                           "MNP"="cyan3", 
                           "Neutrophil"="olivedrab",
                           "MLC"="royalblue",
                           "PTC"="tan",
                           "EC 2"="purple4",
                           "EC 1"="magenta",
                           "DCT 1"="salmon",
                           "DCT 2"="pink",
                           "DCT 3"="plum",
                           "DCT 4"="lawngreen",
                           "cTAL"="cyan",
                           "B cells"="brown"))

## Initialize SCENIC settings
dbDir = "SCENIC/resources/"
org = "mgi"
myDatasetTitle = "Mouse glom"
data(defaultDbNames)
dbs <- defaultDbNames[[org]]
scenicOptions <- initializeScenic(org=org, dbDir=dbDir, dbs=dbs, 
                                  datasetTitle=myDatasetTitle, nCores=ncores) 

saveRDS(cellInfo, file="int/cellInfo.Rds")
colVars$CellType <- colVars$CellType[intersect(names(colVars$CellType), cellInfo$CellType)]
saveRDS(colVars, file="int/colVars.Rds")
plot.new(); legend(0,1, fill=colVars$CellType, legend=names(colVars$CellType))

scenicOptions@inputDatasetInfo$cellInfo <- "int/cellInfo.Rds"
scenicOptions@inputDatasetInfo$colVars <- "int/colVars.Rds"
saveRDS(scenicOptions, file="int/scenicOptions.Rds") 

## Co-expression network
rownames(exprMat) = unname(sapply(rownames(exprMat), function(x){strsplit(x,"\\|")[[1]][1]}))
gnames = rownames(exprMat)
gname_counts = as.data.frame(table(gnames))

uniq = gname_counts[gname_counts[,'Freq'] == 1,]
uniq_gnames = as.character(uniq[,'gnames'])
new_exprMat = exprMat[uniq_gnames,]

genesKept <- geneFiltering(new_exprMat, scenicOptions=scenicOptions,
                           minCountsPerGene=5,
                           minSamples=ncol(exprMat)*.05)

exprMat_filtered <- new_exprMat[genesKept, ]

## Correction
runCorrelation(exprMat_filtered, scenicOptions)

## GENIE3
exprMat_filtered <- log2(exprMat_filtered+1) 
runGenie3(exprMat_filtered, scenicOptions)

## Build and score the GRN
logMat = exprMat_filtered
saveRDS(logMat, file="int/logMat.Rds")
scenicOptions@settings$verbose <- TRUE
scenicOptions@settings$nCores <- ncores
scenicOptions@settings$seed <- 123

runSCENIC_1_coexNetwork2modules(scenicOptions)
runSCENIC_2_createRegulons(scenicOptions) 
runSCENIC_3_scoreCells(scenicOptions, logMat)

## Binarize the network activity (regulon on/off)
aucellApp <- plotTsne_AUCellApp(scenicOptions, logMat)
scenicOptions@settings$devType="pdf"
runSCENIC_4_aucell_binarize(scenicOptions)

## Clustering / dimensionality reduction on the regulon activity
scenicOptions@settings$seed <- 123
nPcs <- c(10,20,30,40,50)
fileNames <- tsneAUC(scenicOptions, aucType="AUC", nPcs=nPcs, perpl=c(5,15,25,40,50))
fileNames <- tsneAUC(scenicOptions, aucType="AUC", nPcs=nPcs, perpl=c(5,15,25,40,50), onlyHighConf=TRUE, filePrefix="int/tSNE_oHC")

# Plot as pdf (individual files in int/):
fileNames <- paste0("int/",grep(".Rds", grep("tSNE_", list.files("int"), value=T), value=T))

# view and compare
pdf("output/tSNE_compare_color_by_cell_type.pdf", width=20,height=20)
par(mfrow=c(length(nPcs), 5))
fileNames <- paste0("int/",grep(".Rds", grep("tSNE_AUC", list.files("int"), value=T, perl = T), value=T))
plotTsne_compareSettings(fileNames, scenicOptions, showLegend=FALSE, varName="CellType", cex=.5)
dev.off()

# Using "high-confidence" regulons
pdf("output/tSNE_compare_color_by_cell_type_oHC.pdf", width=20,height=20)
par(mfrow=c(length(nPcs), 5))
fileNames <- paste0("int/",grep(".Rds", grep("tSNE_oHC_AUC", list.files("int"), value=T, perl = T), value=T))
plotTsne_compareSettings(fileNames, scenicOptions, showLegend=FALSE, varName="CellType", cex=.5)
dev.off()

# keep best parameters for later plotting
scenicOptions@settings$defaultTsne$aucType <- "AUC"
scenicOptions@settings$defaultTsne$dims <- 30
scenicOptions@settings$defaultTsne$perpl <- 50
saveRDS(scenicOptions, file="int/scenicOptions.Rds")

## Export to loom/SCope
# Export:
fileName <- getOutName(scenicOptions, "loomFile")
scenicOptions@fileNames$output["loomFile",] <- "output/mouse_SCENIC.loom"
export2scope(scenicOptions, new_exprMat, addAllTsnes=TRUE)
saveRDS(scenicOptions, file="int/scenicOptions_final.Rds")

## output different tables for downstream analysis in python
library(SCopeLoomR)
scenicLoomPath <- getOutName(scenicOptions, "loomFile")
loom <- open_loom(scenicLoomPath)
regulons_incidMat <- get_regulons(loom)
regulons <- regulonsToGeneLists(regulons_incidMat)
regulonsAUC <- get_regulonsAuc(loom)
regulonsAucThresholds <- get_regulonThresholds(loom)

# the subset of regulon for plotting
# - AUC
mat4tsne <- getAUC(loadInt(scenicOptions, "aucell_regulonAUC"))
mat4tsne_subset <- mat4tsne[onlyNonDuplicatedExtended(rownames(mat4tsne)),]
write.table(mat4tsne_subset, "output/regulonsAUC_subset.csv", sep="\t", row.names=TRUE, quote=FALSE)
write.table(mat4tsne, "output/regulonsAUC.csv", sep="\t", row.names=TRUE, quote=FALSE)

# - Binary
mat4tsne_binary <- loadInt(scenicOptions, "aucell_binary_nonDupl") 
mat4tsne_binary_subset <- mat4tsne_binary[onlyNonDuplicatedExtended(rownames(mat4tsne_binary)),] 
write.table(mat4tsne_binary_subset, "output/regulons_binary_subset.csv", sep="\t", row.names=TRUE, quote=FALSE)

write.table(cellInfo, "output/cell_type_info.csv", sep="\t", row.names=TRUE, quote=FALSE)

## export regulons in heatmap
tmp = loadInt(scenicOptions, "aucell_regulonSelection") 
write.table(data.frame(tmp$all), "output/aucell_regulonSelection_all.csv", sep="\t", 
            row.names=FALSE, quote=FALSE)
write.table(data.frame(tmp$onePercent), "output/aucell_regulonSelection_onePercent.csv", 
            sep="\t", row.names=FALSE, quote=FALSE)
write.table(data.frame(tmp$corr), "output/aucell_regulonSelection_corr.csv", sep="\t", 
            row.names=FALSE, quote=FALSE)

tmp2 = loadInt(scenicOptions, "aucell_binaryRegulonOrder") 
write.table(data.frame(tmp2), "output/aucell_binaryRegulonOrder.csv", sep="\t", 
            row.names=FALSE, quote=FALSE)

# downstream analysis
scenicOptions <- readRDS("int/scenicOptions_final.Rds")
regulons <- loadInt(scenicOptions, "aucell_regulons")
regulonTargetsInfo <- loadInt(scenicOptions, "regulonTargetsInfo")
write.table(regulonTargetsInfo, "output/regulonTargetsInfo.csv", sep="\t", row.names=FALSE, quote=FALSE)
