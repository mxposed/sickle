require(Seurat)
require(dplyr)


thisFile <- function() {
  cmdArgs <- commandArgs(trailingOnly = FALSE)
  needle <- "--file="
  match <- grep(needle, cmdArgs)
  if (length(match) > 0) {
    # Rscript
    return(normalizePath(sub(needle, "", cmdArgs[match])))
  } else {
    # 'source'd via R console
    return(normalizePath(sys.frames()[[1]]$ofile))
  }
}


CURRENT_DIR <- dirname(thisFile())
ROOT <- dirname(dirname(CURRENT_DIR))


main <- function() {
  l1 <- read.csv(file.path(ROOT, 'rmbatch_dge', 'Lung1_rm.batch_dge.txt'), sep=' ')
  l2 <- read.csv(file.path(ROOT, 'rmbatch_dge', 'Lung2_rm.batch_dge.txt'), sep=' ')
  l3 <- read.csv(file.path(ROOT, 'rmbatch_dge', 'Lung3_rm.batch_dge.txt'), sep=' ')
  
  l1 <- as.data.frame(t(l1))
  l2 <- as.data.frame(t(l2))
  l3 <- as.data.frame(t(l3))
  
  lung <- bind_rows(lapply(list(l1, l2, l3), add_rownames))
  rownames(lung) <- lung$rowname
  lung$rowname <- NULL
  lung[is.na(lung)] <- 0
  lungs <- CreateSeuratObject(as.data.frame(t(lung)))
  
  cell_types <- read.csv(file.path(ROOT, 'MCA_assign.csv'))
  cell_types <- cell_types[cell_types$Tissue == 'Lung',]
  cell_types$ClusterID <- as.numeric(sub('Lung_', '', cell_types$ClusterID))
  cell_types$ClusterID <- cell_types$ClusterID - 1
  rownames(cell_types) <- cell_types$Cell.name

  for (i in 1:5) {
    train_cells <- read.csv(file.path(
      CURRENT_DIR, 
      sprintf('cv%d-train.csv', i)
    ), header=FALSE)
    test_cells <- read.csv(file.path(
      CURRENT_DIR, 
      sprintf('cv%d-test.csv', i)
    ), header=FALSE)
    train <- SubsetData(lungs, cells.use = as.vector(train_cells$V2))
    train <- CreateSeuratObject(train@data)
    test <- SubsetData(lungs, cells.use = as.vector(test_cells$V2))
    test <- CreateSeuratObject(test@data)
    pred <- ClassifyCells(
      train,
      training.classes = cell_types[train@cell.names, 'ClusterID'],
      new.data = test@data
    )
    pred <- data.frame(pred = pred)
    rownames(pred) <- test@cell.names
    write.csv(pred, file.path(
      CURRENT_DIR, 
      sprintf('cv%d-preds-seurat.csv', i)
    ))
  }
}

main()