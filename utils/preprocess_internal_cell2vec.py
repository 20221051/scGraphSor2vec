import argparse
import pandas as pd
import dgl
import torch
import torch.nn.functional as F
import collections
from scipy.sparse import csr_matrix, vstack, save_npz
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import numpy as np
from pprint import pprint
import json


def normalize_weight(graph: dgl.DGLGraph):
    # normalize weight & add self-loop
    in_degrees = graph.in_degrees()
    for i in range(graph.number_of_nodes()):
        src, dst, in_edge_id = graph.in_edges(i, form='all')
        if src.shape[0] == 0:
            continue
        edge_w = graph.edata['weight'][in_edge_id]
        graph.edata['weight'][in_edge_id] = in_degrees[i] * edge_w / torch.sum(edge_w)


def get_id_2_gene(species_data_path, species, tissue, filetype):
    data_path = species_data_path
    if tissue == 'Colorectum':
        data_files = [
                      data_path / f'{species}_Duodenum4681_data.{filetype}',
                      data_path / f'{species}_Ileum3367_data.{filetype}',
                      data_path / f'{species}_JeJunum5549_data.{filetype}',
                      data_path / f'{species}_Rectum5718_data.{filetype}',
                      data_path / f'{species}_Sigmoid_colon3281_data.{filetype}',
                      data_path / f'{species}_Transverse_colon5765_data.{filetype}',
                      data_path / f'{species}_Transverse_colon11229_data.{filetype}'
                    ]
    else:
        data_files = data_path.glob(f'{species}_{tissue}*_data.{filetype}')
    genes = None
    for file in data_files:
        if filetype == 'csv':
            data = pd.read_csv(file, dtype=np.str, header=0,low_memory=False).values[:, 0]
        else:
            data = pd.read_csv(file, compression='gzip', header=0,low_memory=False).values[:, 0]
        if genes is None:
            genes = set(data)
        else:
            genes = genes | set(data)
    id2gene = list(genes)
    id2gene.sort()
    return id2gene


def get_id_2_label_and_label_statistics(species_data_path, species, tissue):
    data_path = species_data_path
    if tissue == 'Colorectum':
        cell_files = [
                      data_path / f'{species}_Duodenum4681_celltype.csv',
                      data_path / f'{species}_Ileum3367_celltype.csv',
                      data_path / f'{species}_JeJunum5549_celltype.csv',
                      data_path / f'{species}_Rectum5718_celltype.csv',
                      data_path / f'{species}_Sigmoid_colon3281_celltype.csv',
                      data_path / f'{species}_Transverse_colon5765_celltype.csv',
                      data_path / f'{species}_Transverse_colon11229_celltype.csv'
                    ]
    else:
        cell_files = data_path.glob(f'{species}_{tissue}*_celltype.csv')
        
    cell_types = set()
    cell_type_list = list()
    for file in cell_files:
        df = pd.read_csv(file, dtype=np.str, header=0,low_memory=False)
        df['Cell_type'] = df['Cell_type'].map(str.strip)
        cell_types = set(df.values[:, 2]) | cell_types
        cell_type_list.extend(df.values[:, 2].tolist())
    id2label = list(cell_types)
    label_statistics = dict(collections.Counter(cell_type_list))
    return id2label, label_statistics


def save_statistics(statistics_path, id2label, id2gene, tissue):
    gene_path = statistics_path / f'{tissue}_genes.txt'
    label_path = statistics_path / f'{tissue}_cell_type.txt'
    with open(gene_path, 'w', encoding='utf-8') as f:
        for gene in id2gene:
            f.write(gene + '\r\n')
    with open(label_path, 'w', encoding='utf-8') as f:
        for label in id2label:
            f.write(label + '\r\n')


def load_data_internal_cell2vec(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    species = params.species
    tissue = params.tissue
    np.random.seed(random_seed)
    device = torch.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')

    proj_path = Path(__file__).parent.resolve().parent.resolve()
    species_data_path = proj_path / 'train' / species
    graph_path = proj_path / 'pretrained_cell2vec' / species / 'graphs'
    statistics_path = proj_path / 'pretrained_cell2vec' / species / 'statistics'

    if not species_data_path.exists():
        raise NotImplementedError

    if not statistics_path.exists():
        statistics_path.mkdir(parents=True)
    if not graph_path.exists():
        graph_path.mkdir(parents=True)

    # generate gene statistics file
    id2gene = get_id_2_gene(species_data_path, species, tissue, filetype=params.filetype)
    # generate cell label statistics file
    id2label, label_statistics = get_id_2_label_and_label_statistics(species_data_path, species, tissue)
    total_cell = sum(label_statistics.values())
    for label, num in label_statistics.items():
        if num / total_cell <= params.exclude_rate:
            id2label.remove(label)  # remove exclusive labels
    # prepare unified genes
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)
    # prepare unified labels
    num_labels = len(id2label)
    label2id = {label: idx for idx, label in enumerate(id2label)}
    save_statistics(statistics_path, id2label, id2gene, tissue)
    print(f"The build graph contains {num_genes} genes with {num_labels} labels supported.")

    graph = dgl.DGLGraph()

    gene_ids = torch.arange(num_genes, dtype=torch.int32, device=device).unsqueeze(-1)
    graph.add_nodes(num_genes, {'id': gene_ids})

    all_labels = []
    matrices = []
    num_cells = 0

    data_path = species_data_path
    if tissue == 'Colorectum':
        col_species = ['Duodenum','Ileum', 'JeJunum', 'Rectum', 'Sigmoid_colon','Transverse_colon','Transverse_colon']
        data_files = [
                      data_path / f'{species}_Duodenum4681_data.{params.filetype}',
                      data_path / f'{species}_Ileum3367_data.{params.filetype}',
                      data_path / f'{species}_JeJunum5549_data.{params.filetype}',
                      data_path / f'{species}_Rectum5718_data.{params.filetype}',
                      data_path / f'{species}_Sigmoid_colon3281_data.{params.filetype}',
                      data_path / f'{species}_Transverse_colon5765_data.{params.filetype}',
                      data_path / f'{species}_Transverse_colon11229_data.{params.filetype}'
                    ]

    else:
        data_files = data_path.glob(f'*{species}_{tissue}*_data.{params.filetype}')

    for data_file in data_files:
        if tissue == 'Colorectum':
            ind = data_files.index(data_file)
            Col_tissue = col_species[ind]
            number = ''.join(list(filter(str.isdigit, data_file.name)))
            type_file = species_data_path / f'{species}_{Col_tissue}{number}_celltype.csv'
        else:
            number = ''.join(list(filter(str.isdigit, data_file.name)))
            type_file = species_data_path / f'{species}_{tissue}{number}_celltype.csv'

        # load celltype file then update labels accordingly
        cell2type = pd.read_csv(type_file, index_col=0,low_memory=False)
        cell2type.columns = ['cell', 'type']
        cell2type['type'] = cell2type['type'].map(str.strip)
        cell2type['id'] = cell2type['type'].map(label2id)
        # filter out cells not in label-text
        filter_cell = np.where(pd.isnull(cell2type['id']) == False)[0]
        cell2type = cell2type.iloc[filter_cell]

        assert not cell2type['id'].isnull().any(), 'something wrong about celltype file.'
        all_labels += cell2type['id'].tolist()

        # load data file then update graph
        # df = pd.read_csv(data_file, index_col=0)  # (gene, cell)

        if params.filetype == 'csv':
            df = pd.read_csv(data_file, index_col=0,low_memory=False)  # (gene, cell)
        elif params.filetype == 'gz':
            df = pd.read_csv(data_file, compression='gzip', index_col=0,low_memory=False)
        else:
            print(f'Not supported type for {data_path}. Please verify your data file')

        df = df.transpose(copy=True)  # (cell, gene)
        # filter out cells not in label-text
        df = df.iloc[filter_cell]
        assert cell2type['cell'].tolist() == df.index.tolist()
        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]
       

        if tissue == 'Colorectum':
            print(
                f'{species}_{Col_tissue}{number}_data.{params.filetype} -> Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')
        else:
            print(
                f'{species}_{tissue}{number}_data.{params.filetype} -> Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')

        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        row_idx, col_idx = np.nonzero(arr > params.threshold)  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values
        cell_idx = row_idx + graph.number_of_nodes()  # cell_index
        gene_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, gene_idx)), shape=info_shape)
        matrices.append(info)
        
        num_cells += len(df)

        ids = torch.tensor([-1] * len(df), dtype=torch.int32, device=device).unsqueeze(-1)
        graph.add_nodes(len(df), {'id': ids})
        graph.add_edges(cell_idx, gene_idx,
                        {'weight': torch.tensor(non_zeros, dtype=torch.float32, device=device).unsqueeze(1)})
        graph.add_edges(gene_idx, cell_idx,
                        {'weight': torch.tensor(non_zeros, dtype=torch.float32, device=device).unsqueeze(1)})

        print(f'Added {len(df)} nodes and {len(cell_idx)} edges.')
        print(f'#Nodes in Graph: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}.')

    assert len(all_labels) == num_cells

    save_npz(graph_path / f'{species}_{tissue}_data', vstack(matrices))

    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    assert sparse_feat.shape[0] == num_cells
    
    
    #load gene2vec embedding
    gene2vec = pd.read_table(f"/home/psy/scDeepSort/gene2vec0.1/gene2vec_dim_200_{species}_{tissue}_iter_10.txt", sep = ' ', header = None,low_memory=False)
    gene2vec = gene2vec.sort_values(0)    
    gene2vec = gene2vec.set_index(0)
    gene2vec_emb = gene2vec.iloc[:,:-1]
    gene2vec_emb = np.array(gene2vec_emb)   
    
    #load cell2vec embedding
    cell2vec = pd.read_table(f"/home/psy/scDeepSort/cell2vec0.1/cell2vec_dim_200_{species}_{tissue}_iter_10.txt", sep = ' ', header = None,low_memory=False)
    cell2vec['list'] = cell2vec[0].str.extract(r'(\d+)')
    cell2vec = cell2vec.astype({'list':'int'})
    cell2vec = cell2vec.sort_values('list')
    cell2vec = cell2vec.set_index(0)
    cell2vec_emb = cell2vec.iloc[:,:-2]
    cell2vec_emb = np.array(cell2vec_emb)
    
    # transpose to gene-wise
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat.T)
    gene_feat = gene_pca.transform(sparse_feat.T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')

    print('------Train label statistics------')
    for i, label in enumerate(id2label, start=1):
        print(f"#{i} [{label}]: {label_statistics[label]}")

    # do normalization
    sparse_feat = sparse_feat / (np.sum(sparse_feat, axis=1, keepdims=True) + 1e-6)
    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat.dot(gene_feat)
    #ref_cell_mat = np.load(f'/home/psy/scDeepSort/ref_cell_mat/{species}_{tissue}_ref_cell_mat.npy')
    
    cell_feat = np.concatenate((sparse_feat.dot(gene_feat),cell2vec_emb), axis = 1)
    gene_feat = np.concatenate((gene_feat,gene2vec_emb), axis = 1)
#     train_cell_mat = pd.DataFrame(cell_feat)
#     train_type = pd.read_csv(f'/home/psy/scDeepSort/labels/{species}_{tissue}_labels.csv')
#     train_cell_mat['label'] = train_type.iloc[:,1]
    
    gene_feat = torch.from_numpy(gene_feat)  # use shared storage
    cell_feat = torch.from_numpy(cell_feat)

    graph.ndata['features'] = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float)
    graph = graph.to(device)
    graph.ndata['features'] = graph.ndata['features'].to(device)
    labels = torch.tensor([-1] * num_genes + all_labels, dtype=torch.long, device=device)  # [gene_num+train_num]

#     # StratifiedShuffleSplit train set and test set
#     cell_id = list(range(num_cells))
#     split = StratifiedShuffleSplit(n_splits=1, test_size=params.test_rate, random_state=random_seed)
#     for train_index, test_index in split.split(cell_id,all_labels):
#         train_ids = train_index
#         test_ids = test_index
    
#     train_ids = torch.tensor(train_ids + num_genes).to(device)
#     test_ids = torch.tensor(test_ids + num_genes).to(device)
    
    # split train set and test set
    test_ids = np.load(proj_path / 'ids' / f'{params.species}_{tissue}_test_ids.npy')
    train_ids = np.load(proj_path / 'ids' / f'{params.species}_{tissue}_train_ids.npy')
    test_ids = torch.tensor(test_ids).to(device)
    train_ids = torch.tensor(train_ids).to(device)


    # normalize weight
    normalize_weight(graph)
    # add self-loop
    graph.add_edges(graph.nodes(), graph.nodes(),
                    {'weight': torch.ones(graph.number_of_nodes(), dtype=torch.float, device=device).unsqueeze(1)})
    graph.readonly()
    
    return num_cells, num_genes, num_labels, graph, train_ids, test_ids, labels

