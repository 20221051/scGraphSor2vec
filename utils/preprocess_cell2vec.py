import argparse
import pandas as pd
import dgl
import torch
import torch.nn.functional as F
import collections
from scipy.sparse import csr_matrix, vstack, load_npz
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import numpy as np
from time import time


def get_map_dict(map_path: Path, tissue):
    map_df = pd.read_excel(map_path / 'map.xlsx')
    # {num: {test_cell1: {train_cell1, train_cell2}, {test_cell2:....}}, num_2:{}...}
    map_dic = dict()
    for idx, row in enumerate(map_df.itertuples()):
        if getattr(row, 'Tissue') == tissue:
            num = getattr(row, 'num')
            test_celltype = getattr(row, 'Celltype')
            train_celltype = getattr(row, '_5')
            if map_dic.get(getattr(row, 'num')) is None:
                map_dic[num] = dict()
                map_dic[num][test_celltype] = set()
            elif map_dic[num].get(test_celltype) is None:
                map_dic[num][test_celltype] = set()
            map_dic[num][test_celltype].add(train_celltype)
    return map_dic


def normalize_weight(graph: dgl.DGLGraph):
    # normalize weight & add self-loop
    in_degrees = graph.in_degrees()
    for i in range(graph.number_of_nodes()):
        src, dst, in_edge_id = graph.in_edges(i, form='all')
        if src.shape[0] == 0:
            continue
        edge_w = graph.edata['weight'][in_edge_id]
        graph.edata['weight'][in_edge_id] = in_degrees[i] * edge_w / torch.sum(edge_w)


def get_id_2_gene(gene_statistics_path):
    id2gene = []
    with open(gene_statistics_path, 'r', encoding='utf-8') as f:
        for line in f:
            id2gene.append(line.strip())
    return id2gene


def get_id_2_label(cell_statistics_path):
    id2label = []
    with open(cell_statistics_path, 'r', encoding='utf-8') as f:
        for line in f:
            id2label.append(line.strip())
    return id2label


def load_data_cell2vec(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    test = params.test_dataset
    tissue = params.tissue
    np.random.seed(random_seed)
    device = torch.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')

    proj_path = Path(__file__).parent.resolve().parent.resolve()
    species_data_path = proj_path / 'pretrained_cell2vec' / params.species
    statistics_path = species_data_path / 'statistics'

    if params.evaluate:
        map_path = proj_path / 'map' / params.species
        map_dict = get_map_dict(map_path, tissue)
        
        
    if params.species == 'mouse':
        if tissue == 'Blood':
            gene_statistics_path = statistics_path / ('Peripheral_blood' + '_genes.txt')  # train+test gene
            cell_statistics_path = statistics_path / ('Peripheral_blood' + '_cell_type.txt')  # train labels
        elif tissue == 'Intestine':
            gene_statistics_path = statistics_path / ('Small_intestine' + '_genes.txt')  # train+test gene
            cell_statistics_path = statistics_path / ('Small_intestine' + '_cell_type.txt')  # train labels
        else:
            gene_statistics_path = statistics_path / (tissue + '_genes.txt')  # train+test gene
            cell_statistics_path = statistics_path / (tissue + '_cell_type.txt')  # train labels
    elif params.species == 'human':
        if tissue == 'Brain':
            gene_statistics_path = statistics_path / ('Cerebellum' + '_genes.txt')  # train+test gene
            cell_statistics_path = statistics_path / ('Cerebellum' + '_cell_type.txt')  # train labels
        elif tissue == 'Blood':
            gene_statistics_path = statistics_path / ('Peripheral_blood' + '_genes.txt')  # train+test gene
            cell_statistics_path = statistics_path / ('Peripheral_blood' + '_cell_type.txt')  # train labels
        else:
            gene_statistics_path = statistics_path / (tissue + '_genes.txt')  # train+test gene
            cell_statistics_path = statistics_path / (tissue + '_cell_type.txt')  # train labels
    if not statistics_path.exists():
        statistics_path.mkdir()

    # generate gene statistics file
    id2gene = get_id_2_gene(gene_statistics_path)
    # generate cell label statistics file
    id2label = get_id_2_label(cell_statistics_path)

    test_num = 0
    # prepare unified genes
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)
    # prepare unified labels
    num_labels = len(id2label)
    label2id = {label: idx for idx, label in enumerate(id2label)}
    print(f"The build graph contains {num_genes} gene nodes with {num_labels} labels supported.")

    test_graph_dict = dict()  # test-graph dict
    if params.evaluate:
        test_label_dict = dict()  # test label dict
    test_index_dict = dict()  # test feature indices in all features
    test_mask_dict = dict()
    test_nid_dict = dict()
    test_cell_origin_id_dict = dict()
    test_gene_dict = dict()

    ids = torch.arange(num_genes, dtype=torch.int32, device=device).unsqueeze(-1)

    # ==================================================
    # add all genes as nodes

    for num in test:
        test_graph_dict[num] = dgl.DGLGraph()
        test_graph_dict[num].add_nodes(num_genes, {'id': ids})
    # ====================================================

    matrices = []
    test_cell2vec = []

    if params.species == 'mouse':
        if tissue == 'Blood':
            support_data = proj_path / 'pretrained_cell2vec' / f'{params.species}' / 'graphs' / f'{params.species}_Peripheral_blood_data.npz'
        elif tissue == 'Intestine':
            support_data = proj_path / 'pretrained_cell2vec' / f'{params.species}' / 'graphs' / f'{params.species}_Small_intestine_data.npz'
        else:
            support_data = proj_path / 'pretrained_cell2vec' / f'{params.species}' / 'graphs' / f'{params.species}_{tissue}_data.npz'
    elif params.species == 'human':
        if tissue == 'Brain':
            support_data = proj_path / 'pretrained_cell2vec' / f'{params.species}' / 'graphs' / f'{params.species}_Cerebellum_data.npz'
        elif tissue == 'Blood':
            support_data = proj_path / 'pretrained_cell2vec' / f'{params.species}' / 'graphs' / f'{params.species}_Peripheral_blood_data.npz'  
        else:
            support_data = proj_path / 'pretrained_cell2vec' / f'{params.species}' / 'graphs' / f'{params.species}_{tissue}_data.npz'
    
    support_num = 0
    info = load_npz(support_data)
    
    print(f"load {support_data.name}")
    row_idx, gene_idx = np.nonzero(info > 0)
    non_zeros = info.data
    cell_num = info.shape[0]
    support_num += cell_num
    matrices.append(info)
    ids = torch.tensor([-1] * cell_num, device=device, dtype=torch.int32).unsqueeze(-1)
    total_cell = support_num
    
    if params.species == 'mouse':
        if tissue == 'Blood':
            cell2vec = pd.read_table(f"/home/psy/scDeepSort/cell2vec0.1/cell2vec_dim_200_{params.species}_Peripheral_blood_iter_10.txt", sep = ' ', header = None,low_memory=False)
        elif tissue == 'Intestine':
            cell2vec = pd.read_table(f"/home/psy/scDeepSort/cell2vec0.1/cell2vec_dim_200_{params.species}_Small_intestine_iter_10.txt", sep = ' ', header = None,low_memory=False)
        else:
            cell2vec = pd.read_table(f"/home/psy/scDeepSort/cell2vec0.1/cell2vec_dim_200_{params.species}_{tissue}_iter_10.txt", sep = ' ', header = None,low_memory=False)
    elif params.species == 'human':
        if tissue == 'Brain':
            cell2vec = pd.read_table(f"/home/psy/scDeepSort/cell2vec0.1/cell2vec_dim_200_{params.species}_Cerebellum_iter_10.txt", sep = ' ', header = None,low_memory=False)
        elif tissue == 'Blood':
            cell2vec = pd.read_table(f"/home/psy/scDeepSort/cell2vec0.1/cell2vec_dim_200_{params.species}_Peripheral_blood_iter_10.txt", sep = ' ', header = None,low_memory=False)                  
        else:
            cell2vec = pd.read_table(f"/home/psy/scDeepSort/cell2vec0.1/cell2vec_dim_200_{params.species}_{tissue}_iter_10.txt", sep = ' ', header = None,low_memory=False)
    
    cell2vec['list'] = cell2vec[0].str.extract(r'(\d+)')
    cell2vec = cell2vec.astype({'list':'int'})
    cell2vec = cell2vec.sort_values('list')
    cell2vec = cell2vec.set_index(0)
    cell2vec_support_emb = cell2vec.iloc[:,:-2]
    cell2vec_support_emb = np.array(cell2vec_support_emb)
    test_cell2vec.append(cell2vec_support_emb)

    for n in test:  # training cell also in test graph
        cell_idx = row_idx + test_graph_dict[n].number_of_nodes()
        test_graph_dict[n].add_nodes(cell_num, {'id': ids})
        test_graph_dict[n].add_edges(cell_idx, gene_idx,
                                     {'weight': torch.tensor(non_zeros, dtype=torch.float32,
                                                             device=device).unsqueeze(1)})
        test_graph_dict[n].add_edges(gene_idx, cell_idx,
                                     {'weight': torch.tensor(non_zeros, dtype=torch.float32,
                                                             device=device).unsqueeze(1)})

    for num in test:
        data_path = proj_path / params.test_dir / params.species / f'{params.species}_{tissue}{num}_data.{params.filetype}'
        if params.evaluate:
            type_path = proj_path / params.test_dir / params.species / f'{params.species}_{tissue}{num}_celltype.csv'
            # load celltype file then update labels accordingly
            cell2type = pd.read_csv(type_path, index_col=0,low_memory=False)
            cell2type.columns = ['cell', 'type']
            cell2type['type'] = cell2type['type'].map(str.strip)
            # test_labels += cell2type['type'].tolist()
            test_label_dict[num] = cell2type['type'].tolist()
        
        #load cell2vec embedding
        cell2vec = pd.read_table(f"/home/psy/scDeepSort/cell2vec0.1_test/cell2vec_dim_200_{params.species}_{tissue}{num}_iter_10.txt", sep = ' ', header = None,low_memory=False)
        
        cell2vec['list'] = cell2vec[0].str.extract(r'(\d+)')
        cell2vec = cell2vec.astype({'list':'int'})
        cell2vec = cell2vec.sort_values('list')
        cell2vec = cell2vec.set_index(0)
        cell2vec_emb = cell2vec.iloc[:,:-2]
        cell2vec_emb = np.array(cell2vec_emb)
        test_cell2vec.append(cell2vec_emb)
            
        # load data file then update graph
        if params.filetype == 'csv':
            df = pd.read_csv(data_path, index_col=0,low_memory=False)  # (gene, cell)
        elif params.filetype == 'gz':
            df = pd.read_csv(data_path, compression='gzip', index_col=0,low_memory=False)
        else:
            print(f'Not supported type for {data_path}. Please verify your data file')

        test_cell_origin_id_dict[num] = list(df.columns)
        df = df.transpose(copy=True)  # (cell, gene)

        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]

        print(f'{params.species}_{tissue}{num}_data.{params.filetype} -> Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')
        tic = time()
        print(f'Begin to cumulate time of training/testing ...')
        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        row_idx, col_idx = np.nonzero(arr > params.threshold)  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values
        # inter-dataset index
        cell_idx = row_idx + test_graph_dict[num].number_of_nodes()
        gene_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, gene_idx)), shape=info_shape)
        matrices.append(info)
 
        # test_nodes_index_dict[num] = list(range(graph.number_of_nodes(), graph.number_of_nodes() + len(df)))
        ids = torch.tensor([-1] * len(df), device=device, dtype=torch.int32).unsqueeze(-1)
        test_index_dict[num] = list(range(num_genes + support_num + test_num, num_genes + support_num + test_num + len(df)))
        test_nid_dict[num] = list(
            range(test_graph_dict[num].number_of_nodes(), test_graph_dict[num].number_of_nodes() + len(df)))
        test_num += len(df)
        test_graph_dict[num].add_nodes(len(df), {'id': ids})
        # for the test cells, only gene-cell edges are in the test graph
        test_graph_dict[num].add_edges(gene_idx, cell_idx,
                                       {'weight': torch.tensor(non_zeros, dtype=torch.float32,
                                                               device=device).unsqueeze(1)})

        print(f'Added {len(df)} nodes and {len(cell_idx)} edges.')
        total_cell += num
    
    #load gene2vec embedding
    if params.species == 'mouse':
        if tissue == 'Blood':
            gene2vec = pd.read_table(f"/home/psy/scDeepSort/gene2vec0.1/gene2vec_dim_200_{params.species}_Peripheral_blood_iter_10.txt", sep = ' ', header = None,low_memory=False)
        elif tissue == 'Intestine':
            gene2vec = pd.read_table(f"/home/psy/scDeepSort/gene2vec0.1/gene2vec_dim_200_{params.species}_Small_intestine_iter_10.txt", sep = ' ', header = None,low_memory=False)
        else:
            gene2vec = pd.read_table(f"/home/psy/scDeepSort/gene2vec0.1/gene2vec_dim_200_{params.species}_{tissue}_iter_10.txt", sep = ' ', header = None,low_memory=False)
    elif params.species == 'human':
        if tissue == 'Brain':
            gene2vec = pd.read_table(f"/home/psy/scDeepSort/gene2vec0.1/gene2vec_dim_200_{params.species}_Cerebellum_iter_10.txt", sep = ' ', header = None,low_memory=False)
        elif tissue == 'Blood':
            gene2vec = pd.read_table(f"/home/psy/scDeepSort/gene2vec0.1/gene2vec_dim_200_{params.species}_Peripheral_blood_iter_10.txt", sep = ' ', header = None,low_memory=False)                
        else:
            gene2vec = pd.read_table(f"/home/psy/scDeepSort/gene2vec0.1/gene2vec_dim_200_{params.species}_{tissue}_iter_10.txt", sep = ' ', header = None,low_memory=False)

    gene2vec = gene2vec.sort_values(0)    
    gene2vec = gene2vec.set_index(0)
    gene2vec_emb = gene2vec.iloc[:,:-1]
    gene2vec_emb = np.array(gene2vec_emb)   
        
    #stack cell2vec
    cell2vec_feat = vstack(test_cell2vec).toarray() # (test_cell,200)
    print(cell2vec_feat.shape)
    support_index = list(range(num_genes + support_num))
    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)    

    # transpose to gene-wise
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat[:support_num].T)
    gene_feat = gene_pca.transform(sparse_feat[:support_num].T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')

    # do normalization
    sparse_feat = sparse_feat / (np.sum(sparse_feat, axis=1, keepdims=True) + 1e-6)                                     
    print(sparse_feat.shape)
    
    # use weighted gene_feat as cell_feat 
    cell_feat = np.concatenate((sparse_feat.dot(gene_feat),cell2vec_feat), axis = 1)
    print(cell_feat.shape)
    gene_sup_feat = np.concatenate((gene_feat,gene2vec_emb), axis = 1)
    print(gene_sup_feat.shape)
    #load gene2vec_test embedding
    # for num in test:
    #     gene2vec = pd.read_table(f"/home/psy/scDeepSort/gene2vec_test/gene2vec_dim_200_{params.species}_{tissue}{num}_iter_10.txt", sep = ' ', header = None)
    #     gene2vec = gene2vec.sort_values(0)    
    #     gene2vec = gene2vec.set_index(0)
    #     gene2vec_emb = gene2vec.iloc[:,:-1]
    #     gene2vec_emb = np.array(gene2vec_emb)
    #     test_gene_dict[num] = np.concatenate((gene_feat,gene2vec_emb), axis = 1)
     
    
    # use shared storage
    cell_feat = torch.from_numpy(cell_feat)
    gene_sup_feat = torch.from_numpy(gene_sup_feat)
    features = torch.cat([gene_sup_feat, cell_feat], dim=0).type(torch.float).to(device)
    
    for num in test:
        # gene_index = list(range(num_genes))
        # gene_test_feat = torch.from_numpy(test_gene_dict[num]).to(device)
        # features[gene_index] = gene_test_feat
        test_graph_dict[num].ndata['features'] = features[support_index + test_index_dict[num]]

    for num in test:
        test_mask_dict[num] = torch.zeros(test_graph_dict[num].number_of_nodes(), dtype=torch.bool, device=device)
        test_mask_dict[num][test_nid_dict[num]] = 1
        test_nid_dict[num] = torch.tensor(test_nid_dict[num], dtype=torch.int64)
        # normalize weight & add self-loop
        normalize_weight(test_graph_dict[num])
        test_graph_dict[num].add_edges(test_graph_dict[num].nodes(), test_graph_dict[num].nodes(), {
            'weight': torch.ones(test_graph_dict[num].number_of_nodes(), dtype=torch.float, device=device).unsqueeze(
                1)})
        test_graph_dict[num].readonly()

    if params.evaluate:
        test_dict = {
            'graph': test_graph_dict,
            'label': test_label_dict,
            'nid': test_nid_dict,
            'mask': test_mask_dict,
            'origin_id': test_cell_origin_id_dict
        }
        time_used = time() - tic
        return total_cell, num_genes, num_labels, np.array(id2label, dtype=np.str), test_dict, map_dict, time_used
    else:
        test_dict = {
            'graph': test_graph_dict,
            'nid': test_nid_dict,
            'mask': test_mask_dict,
            'origin_id': test_cell_origin_id_dict
        }
        time_used = time() - tic
        return total_cell, num_genes, num_labels, np.array(id2label, dtype=np.str), test_dict, map_dict, time_used

