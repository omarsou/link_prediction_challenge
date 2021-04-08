import os
import random
import numpy as np
import gzip
import pickle
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


def get_training_graph(graph, edges_to_remove):
    residual_g = graph.copy()
    for edge in edges_to_remove:
        residual_g.remove_edge(edge[0], edge[1])
    return residual_g


def compute_matrix_similarity(node_ids, txt2feat):
    node_paper_embedding = []

    for node in node_ids:
        node_paper_embedding.append(txt2feat[node]['paper_embedding'])

    node_paper_embedding = np.stack(node_paper_embedding, axis=0)

    # Calculate similarity
    paper_sim = cosine_similarity(node_paper_embedding, node_paper_embedding)

    return paper_sim


def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr-(1-fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])[0]


def return_metrics(true, preds):
    preds = torch.cat(preds, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()
    thres = find_optimal_cutoff(true, preds)
    preds_label = np.where(preds > thres, 1, 0)
    f1 = f1_score(true, preds_label)
    auc = roc_auc_score(true, preds)
    return f1, auc, thres
