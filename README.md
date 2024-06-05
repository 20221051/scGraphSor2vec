# scGraphSor2vec

This study aims to accurately understand the characteristics of single-cell data <br/>
and develop a model that functions even with **unseen external data**. <br/>

We introduce single-cell graph samples and aggregation with cor2vec (scGraphSor2vec). scGraphSor2vec is structured into four key stages: <br/>
embedding, cor2vec, weighted graph sample and aggregation (GraphSAGE), and a linear classifier. <br/>

* Four key steps <br/>
  Embedding : PCA (each cell and gene) <br/>
  Cor2vec : Cor2vec specifically adopted the skip-gram approach of Word2vec.<br/>
  We aimed to select cells and genes with a PCC (Pearson Correlation Coefficient) above a certain level to provide weighted information.<br/>
  GraphSAGE : Cells and genes were configured as nodes, with correlations as edges.<br/>
  Linear classifier : Classification of cell types based on the characteristics of single-cell data.<br/>

We developed scGraphSor2vec, a method that combined supervised deep learning with weighted GraphSAGE and cor2vec approaches, <br/>
for **classifying cell types** in human and mouse single-cell RNA-seq data.<br/>

We can classify 

## Data
* Single-cell data <br/>
Human : 10 tissues with internal and external data <br/>
Mouse : 12 tissues with internal and external data <br/>

## Main Architecture
<img src='./images/main_architecture.png' width="400" height="300"/> <img src='./images/cor2vec.png' width="400" height="300"/>

## UMAP
<img src='./images/internal_umap.png' width="400" height="300"/> <img src='./images/external_umap.png' width="400" height="300"/>
