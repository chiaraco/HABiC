#!/usr/bin/Rscript

#===============================================================================================
#=========================================================================================

#       Cross-platform normalization

#=========================================================================================
#===============================================================================================


# load library 
#===========================

library(TDM)      #remotes::install_github("greenelab/TDM")
library(limma)   # RBE
library(sva)   # Combat
library(tidyr)
library(plyr)
library(ggplot2)
   

# violinplot  function
#===================

violinPlot = function (df, mainTitle,dotsize,binwidth,Xangle =0,ordered_x=unique(dataf$data)  ){
  dataf <- gather(gene_df ,key="Data", value="Val")
  dataf$Data <- factor(dataf$Data , levels=ordered_x)
  ggplot(dataf, aes(x=Data, y=Val, fill=Data)) +
    theme(axis.text.x = element_text(angle = Xangle, hjust = 1),
          axis.title.x=element_blank()) +
    ggtitle(mainTitle)+
    scale_x_discrete(limits=names(df))+    #to avoid ggplot to reorder alph automatiqualy  
    geom_violin(trim = FALSE)+
    geom_dotplot(binaxis='y', stackdir='center',dotsize=dotsize,fill = "black",binwidth = binwidth)
}

violinPlot_Ylim = function (df, mainTitle="",dotsize,binwidth,Xangle =0,ylo,yhi,ordered_x=unique(dataf$data)){
  dataf <- gather(df ,key="Data", value="Val")
  dataf$Data <- factor(dataf$Data , levels=ordered_x)
  ggplot(dataf, aes(x=Data, y=Val, fill=Data)) +
    theme(axis.text.x = element_text(angle = Xangle, hjust = 1),
          axis.title.x=element_blank())+
    ggtitle(mainTitle)+ ylim(ylo,yhi)+
    scale_x_discrete(limits=names(df))+    #to avoid ggplot to reorder alph automatiqualy  
    geom_violin(trim = FALSE) + 
    geom_dotplot(binaxis='y', stackdir='center',dotsize=dotsize,fill = "black",binwidth = binwidth)
}

# normalization custom function
#===================

# These functions were copied from  https://github.com/dy16b/Cross-Platform-Normalization 
# publication: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7868049/


processplatforms = function(datalist, namesvec=NULL, skip.match=FALSE){
  #Convert data from various formats to the proper format for use 
  #with all the crossnorm normalization functions
  
  for(i in 1:length(datalist)){
    if(is.matrix(datalist[[i]])){
      datalist[[i]] <- as.data.frame(datalist[[i]])
    }
  }
  
  if (is.null(namesvec)){
    namesvec <- numeric(length(datalist))
    for (i in 1:length(datalist)){
      namesvec[i] <- 0
    }
  }
  
  #Put the row names in their places
  for (i in 1:length(namesvec)){
    if(namesvec[i] != 0){
      rownames(datalist[[i]]) = datalist[[i]][,namesvec[i]]
      datalist[[i]] = datalist[[i]][,-1*namesvec[i],drop=FALSE]
    }	
  }
  
  if(!skip.match){
    #Create the common genes list
    commongenes <- rownames(datalist[[1]])
    for (i in 2:length(datalist)){
      commongenes <- intersect(commongenes,rownames(datalist[[i]]))
    }
    
    
    #Put it all together
    for (i in 1:length(datalist)){
      datalist[[i]] <- datalist[[i]][commongenes,,drop=FALSE]
    }
  }
  return(datalist)
}


gq = function(platform1.data, platform2.data, p1.names=0, p2.names=0, skip.match=FALSE){
  #This function is basically a wrapper for normalizeGQ
  
  #Match names
  input = processplatforms(list(x=platform1.data,y=platform2.data),namesvec = c(p1.names, p2.names), skip.match=skip.match)
  
  #Prepare for normalizeGQ
  combined = cbind(input$x,input$y)
  pf = c(seq(1,1,length.out=dim(input$x)[2]),seq(2,2,length.out=dim(input$y)[2]))
  
  #Call normalizeGQ
  ngq = normalizeGQ(combined,pf)
  
  #Split the results and return
  out=split(seq(pf),pf)
  out[[1]] = ngq[,out[[1]]]
  out[[2]] = ngq[,out[[2]]]
  names(out) <- c("x","y")
  return(out)
}


normalizeGQ <- function(M, pf, ...) { 
  #This function was provided by Xiao-Qin Xia, one of the authors of webarraydb modified MRS
  # M is the data matrix
  # pf is the vector to specify the platform for each column of M.
  idx <- split(seq(pf), pf)
  if (length(pf)<=1) return(M)
  imax <- which.max(sapply(idx, length)) # for reference
  ref_med <- apply(M[, idx[[imax]]], 1, function(x) median(x, na.rm=TRUE))
  ref_med_srt <- sort(ref_med)
  idx[imax] <- NULL
  lapply(idx, function(i) {
    MTMP <- sapply(i, function(x) ref_med_srt[rank(M[,x])]); 
    M[,i] <<- MTMP - apply(MTMP, 1, median) + ref_med 
  } )
  invisible(M)
}



#========================================================================================


# load dataset


#======================================================================================

setwd("tcga")

exprSet <- readRDS("exprSet.rds")   # from STEP1
exprSet <- exprSet[-1]   # remove "count" dataset from the list (1st one)
sampleAnnot <- readRDS("sampleAnnot.rds")

sampleAnnot_All <- list()
for (i in names(exprSet) ){   
  sampleAnnot_All[[i]] <- sampleAnnot
  sampleAnnot_All[[i]]$batch <- i
}

BatchCol<-"batch" 


#==========================================================

#     Merging each RNAseq table with array

#=========================================================


exprSet_array <- exprSet[["array"]]
exprSet_Rseq <- exprSet[-length(exprSet)]  # remove the last one = "array" one
sampleAnnot_array<- sampleAnnot_All[["array"]]
sampleAnnot_Rseq <- sampleAnnot_All[-length(exprSet)]

# select randomly 4 genes and 4 samples to plot  for each cross pltform normalization
random_genes <-sample(row.names(exprSet_array),size = 4,replace =F)                                
random_sples<-sample(names(exprSet_array[,1:ncol(exprSet_array)]),size = 4,replace =F)
                     
                     
#===============================================
                     
 # no crossplatform normalization (noCPN) reference dataset
                     
#======================================================
                     
                     
CombSampleAnnot <-list()
for (i in names(sampleAnnot_Rseq) ){
sampleAnnot_array2 <-sampleAnnot_array
rownames(sampleAnnot_array2) <- paste0(rownames(sampleAnnot_array2),"_arr")
CombSampleAnnot[[i]] <- rbind(sampleAnnot_array2,sampleAnnot_Rseq[[i]])
CombSampleAnnot[[i]]$subtype <- "Lum"
CombSampleAnnot[[i]]$subtype[CombSampleAnnot[[i]]$er.status.by.ihc == "Negative" & CombSampleAnnot[[i]]$pr.status.by.ihc == "Negative" & CombSampleAnnot[[i]]$her2.status.by.ihc == "Negative"]<-"TripleNeg"
CombSampleAnnot[[i]]$subtype[CombSampleAnnot[[i]]$her2.status.by.ihc == "Positive"]<-"HER2"
 }
                     
                     
exprSet_noCPN<-list()
 for (i in names(exprSet_Rseq ) ){
print(paste0("nrow before merge:",nrow(exprSet_Rseq[[i]])))
print(paste0("ncol before merge:", ncol(exprSet_Rseq[[i]]), "  expected after: ", (ncol(exprSet_Rseq[[i]])*2)))
exprSet_noCPN[[i]] <- merge(exprSet_array, exprSet_Rseq[[i]],
                           by= "row.names")
row.names(exprSet_noCPN[[i]]) <-exprSet_noCPN[[i]][,1]; exprSet_noCPN[[i]] <-exprSet_noCPN[[i]][,-1]
names(exprSet_noCPN[[i]])<- gsub (".x$","_arr",names(exprSet_noCPN[[i]]),ignore.case = FALSE)
names(exprSet_noCPN[[i]])<- gsub (".y$","",names(exprSet_noCPN[[i]]),ignore.case = FALSE)
print(paste0("nrow after merge:",nrow(exprSet_noCPN[[i]])))
print(paste0("ncol after merge:",ncol(exprSet_noCPN[[i]])))
print("-------------------------")
}
                     
 #  violin plot
#===============
                     
                
exprSet_toplot <- exprSet_noCPN
name_toplot <-"noCPN"
n <-ncol(exprSet_toplot[[1]])  # same length for all df
                     
pdf(paste0("after_CPN_violinPlot_4genes_",name_toplot,".pdf"),height = 4, width = 6)       
                     
for (j in random_genes){
                       
gene_df <-do.call(rbind, (lapply(exprSet_toplot, function(x) x[j,((n/2)+1):n])))   #because "NOT array samples" are at the second part of the dataframe
gene_df<- as.data.frame(t(gene_df))
gene_df$array <-as.numeric(  exprSet_toplot[[1]][j,1:(n/2)]  )  # because array samples are at the first part of the dataframe
p<-violinPlot(gene_df, mainTitle=paste0(name_toplot," - gene = ",j),
                                     dotsize=0,binwidth=0.2,Xangle=45, ordered_x = names(gene_df))      
 print (p)
                       
}
dev.off()
                     
pdf(paste0("after_CPN_violinPlot_4sples_",name_toplot,".pdf"),height = 4, width = 6)   
                     
for (j in random_sples){
        
denst <-lapply( lapply( exprSet_toplot, function(y)  y=y[, ((n/2)+1):n]) , 
                                       function(x) density(x[,j])  )  
                       
                       ind_df <-do.call(rbind, (lapply(denst, function(dens) dens$y)  ) )
                       ind_df<- as.data.frame(t(ind_df))
                       densarray <- density(exprSet_array[,paste0(j)])
                       ind_df$array <- densarray$y
p<-violinPlot_Ylim(ind_df, mainTitle=paste0(name_toplot, " - pat = ",j),
                     dotsize=0,binwidth=0.2,Xangle=45, ylo = -0.1, yhi = 0.5 ,
                     ordered_x = names(ind_df))      
print (p)
}
dev.off()
                     
                     
saveRDS(CombSampleAnnot,file="sampleAnnot_noCPN.rds")
saveRDS(exprSet_noCPN,file="exprSet_noCPN.rds")
                     
 
 
 #=============================
 
 #  GQ
 
 #=============================
 
 
 exprSet_GQ<-list()
 
 for (i in names(exprSet_noCPN)){
   
   Babatch <- as.factor ( CombSampleAnnot[[i]][,BatchCol] )
   contrasts(Babatch) <- contr.sum(levels(Babatch))
   Babatch <- model.matrix(~Babatch)[, -1, drop = FALSE]
   arrayMat <- exprSet_noCPN[[i]][,Babatch==1]
   RseqMat <- exprSet_noCPN[[i]][,Babatch==-1]
   
   exprSet_GQ[[i]]<- gq(platform1.data=arrayMat, platform2.data=RseqMat)
   exprSet_GQ[[i]]<-cbind(arrayMat,exprSet_GQ[[i]]$y)
   
   print(paste0(i, ": done"))
 }
 
 
 #  violin plot
 #===============                          
 
 exprSet_toplot <- exprSet_GQ
 name_toplot <-"GQ"
 n <-ncol(exprSet_toplot[[1]])
 
 pdf(paste0("after_CPN_violinPlot_4genes_",name_toplot,".pdf"),height = 4, width = 6)       
 
 for (j in random_genes){
   
   gene_df <-do.call(rbind, (lapply(exprSet_toplot, function(x) x[j,((n/2)+1):n])))   #because "NOT array samples" are at the second part of the dataframe
   gene_df<- as.data.frame(t(gene_df))
   gene_df$array <-as.numeric(  exprSet_toplot[[1]][j,1:(n/2)]  )  # because array samples are at the first part of the dataframe
   p<-violinPlot(gene_df, mainTitle=paste0(name_toplot," - gene = ",j),
                 dotsize=0,binwidth=0.2,Xangle=45, ordered_x = names(gene_df))      
   print (p)
   
 }
 dev.off()
 
 pdf(paste0("after_CPN_violinPlot_4sples_",name_toplot,".pdf"),height = 4, width = 6)   
 
 for (j in random_sples){
   denst <-lapply( lapply( exprSet_toplot, function(y)  y=y[, ((n/2)+1):n]) , 
                   function(x) density(x[,j])  )  
   
   ind_df <-do.call(rbind, (lapply(denst, function(dens) dens$y)  ) )
   ind_df<- as.data.frame(t(ind_df))
   densarray <- density(exprSet_array[,paste0(j)])
   ind_df$array <- densarray$y
   p<-violinPlot_Ylim(ind_df, mainTitle=paste0(name_toplot, " - pat = ",j),
                      dotsize=0,binwidth=0.2,Xangle=45, ylo = -0.1, yhi = 0.5 ,
                      ordered_x = names(ind_df))      
   print (p)
 }
 dev.off()
                     
saveRDS(exprSet_GQ, file="exprSet_GQ.rds")
rm(exprSet_GQ)
                                
                                
