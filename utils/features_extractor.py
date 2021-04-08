import numpy as np
import networkx as nx
from tqdm.notebook import tqdm


# XGBOOST FEATURES APPROACH

def compute_features(graph, n2v, t2v, samples, paper_sim, node_mapping, pagerank, katz):
    feature_func = lambda x, y: np.concatenate([x, y])

    undirected_graph = graph.to_undirected()
    features = list()

    for edge in tqdm(samples):
        node_left, node_right = edge[0], edge[1]

        ### GRAPH FEATURES ###

        # Retrieve features related to the node2vec embedding
        diff_n2v = feature_func(n2v[node_left], n2v[node_right])

        # Resource Allocation Index
        RAI = list(nx.resource_allocation_index(undirected_graph, [(node_left, node_right)]))[0][2]
        # Jaccard Coefficient
        JC = list(nx.jaccard_coefficient(undirected_graph, [(node_left, node_right)]))[0][2]
        # Adamic Adar Index
        AAI = list(nx.adamic_adar_index(undirected_graph, [(node_left, node_right)]))[0][2]
        # Preferential Attachment
        PA = list(nx.preferential_attachment(undirected_graph, [(node_left, node_right)]))[0][2]
        # Common Neighbors
        CN = len(list(nx.common_neighbors(undirected_graph, u=node_left, v=node_right)))
        # Page Rank
        PR = np.log(pagerank[node_left] * pagerank[node_right])
        # Katz
        KZ = np.log(katz[node_left] * katz[node_right])

        graph_features = list(diff_n2v) + [PR, KZ, RAI, JC, AAI, PA, CN]

        ### Textual/Meta Features ###

        # Retrieve cosine similarity between asbtract
        cos_sim = paper_sim[node_mapping[node_left], node_mapping[node_right]]

        # Common Authors
        authors_left = t2v[node_left]['authors']
        authors_right = t2v[node_right]['authors']

        if authors_left is None or authors_right is None:
            common_authors = float('nan')
        else:
            common_authors = len(list(set(authors_left).intersection(authors_right)))

        # Difference Date
        diff_date = t2v[node_left]['date'] - t2v[node_right]['date']
        if diff_date < 0:
            diff_date = -10

        # Common Journal
        journal_left = t2v[node_left]['journal']
        journal_right = t2v[node_right]['journal']

        if journal_left is None or journal_right is None:
            is_common_journal = float('nan')
        else:
            journal_left = set([a for a in journal_left.split('.') if a.isalpha()])
            journal_right = set([a for a in journal_right.split('.') if a.isalpha()])
            is_common_journal = len(journal_right.intersection(journal_left))

        text_features = [cos_sim, common_authors, diff_date, is_common_journal]

        total_features = graph_features + text_features

        features.append(total_features)

    return features

# Transformers Features Extraction


def compute_data_nodes(n2v, samples):
    features_nodes = list()

    for edge in tqdm(samples):
        node_left, node_right = edge[0], edge[1]
        n_left_embed = np.append(n2v[node_left], np.float32(0.0))
        n_right_embed = np.append(n2v[node_right], np.float32(1.0))
        features = np.concatenate([n_left_embed[None, :], n_right_embed[None, :]])
        features_nodes.append(features)
    return features_nodes


def compute_data_texts(t2v, samples):
    features_text = list()

    for edge in tqdm(samples):
        node_left, node_right = edge[0], edge[1]
        t_left_embed = np.append(t2v[node_left]['paper_embedding'], np.float32(0.0))
        t_right_embed = np.append(t2v[node_right]['paper_embedding'], np.float32(1.0))
        features = np.concatenate([t_left_embed[None, :], t_right_embed[None, :]])
        features_text.append(features)
    return features_text


def compute_features_extra(graph, t2v, samples, pagerank, katz):
    undirected_graph = graph.to_undirected()

    features_graph = list()
    features_extra = list()

    for edge in tqdm(samples):
        node_left, node_right = edge[0], edge[1]

        ### GRAPH FEATURES ###

        # Resource Allocation Index
        RAI = list(nx.resource_allocation_index(undirected_graph, [(node_left, node_right)]))[0][2]

        # Jaccard Coefficient
        JC = list(nx.jaccard_coefficient(undirected_graph, [(node_left, node_right)]))[0][2]

        # Adamic Adar Index
        AAI = list(nx.adamic_adar_index(undirected_graph, [(node_left, node_right)]))[0][2]

        # Preferential Attachment
        PA = list(nx.preferential_attachment(undirected_graph, [(node_left, node_right)]))[0][2]

        # Common Neighbors
        CN = len(list(nx.common_neighbors(undirected_graph, u=node_left, v=node_right)))

        # Page Rank
        PR = np.log(pagerank[node_left] * pagerank[node_right])

        # Katz
        KZ = np.log(katz[node_left] * katz[node_right])

        graph_features = np.array([RAI, JC, AAI, PA, CN, PR, KZ])

        ### Textual/Meta Features ###

        # Common Authors
        authors_left = t2v[node_left]['authors']
        authors_right = t2v[node_right]['authors']
        if authors_left is None or authors_right is None:
            common_authors = -1
        else:
            common_authors = len(list(set(authors_left).intersection(authors_right)))

        # Difference Date
        diff_date = t2v[node_left]['date'] - t2v[node_right]['date']

        # Common Journal
        journal_left = t2v[node_left]['journal']
        journal_right = t2v[node_right]['journal']

        if journal_left is None or journal_right is None:
            is_common_journal = -1
        else:
            journal_left = set([a for a in journal_left.split('.') if a.isalpha()])
            journal_right = set([a for a in journal_right.split('.') if a.isalpha()])
            is_common_journal = len(journal_right.intersection(journal_left))

        extra_features = np.array([common_authors, diff_date, is_common_journal])

        features_extra.append(extra_features)
        features_graph.append(graph_features)

    return features_extra, features_graph
