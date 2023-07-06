# Python imports
import configparser
import json
import argparse

# Third-party imports
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from pathlib import Path
import dgl
import networkx as nx
import sklearn.metrics
import pandas as pd
import numpy.ma as ma

# Package imports
from cython_metric import custom_hake_pairwise_distances

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

EMBEDDINGS_DIR = Path(CONFIG['EMBEDDING FILES']['embeddings_dir'])
NODE_EMBEDDINGS_FILE = Path(CONFIG['EMBEDDING FILES']['node_embeddings_file'])
REL_EMBEDDINGS_FILE = Path(CONFIG['EMBEDDING FILES']['rel_embeddings_file'])
LAMBDA_FILE = Path(CONFIG['EMBEDDING FILES']['lambda_file'])
DATA_FILE = Path(CONFIG['MODEL FILES']['data_file'])
KNOWLEDGE_GRAPH_FILE = Path(CONFIG['MODEL FILES']['knowledge_graph_file'])

KEY_TO_INDEX_FILE = CONFIG['EMBEDDING FILES']['key_to_index_file']
INDEX_TO_KEY_FILE = CONFIG['EMBEDDING FILES']['index_to_key_file']

DENSE_DISTS_MAT_FILE = CONFIG['EMBEDDING FILES']['dense_dists_mat_file']
DENSE_PROBS_MAT_FILE = CONFIG['EMBEDDING FILES']['dense_probs_mat_file']


def translate_hake_embedding(node_embedding, rel_embedding, lam=1.0):
    h_m, h_p = np.split(node_embedding, 2, axis=-1)
    r_m, r_p = np.split(rel_embedding, 2, axis=-1)

    return np.hstack([h_m * r_m, lam * ((h_p + r_p) % (2 * np.pi))])


def get_embedding_distances(rel_embeddings, node_embeddings, lam, lam2):

    dist_mats = []
    print(len(node_embeddings))
    for rel_embedding in tqdm(rel_embeddings, desc='rel_types'):

        transformed_embeddings = translate_hake_embedding(
            node_embeddings, rel_embedding)

        dist_mat = -1 * custom_hake_pairwise_distances(
            transformed_embeddings, lam=lam, lam2=lam2)

        dist_mats.append(-1 * dist_mat)

    dist_mat = -1 * custom_hake_pairwise_distances(
        node_embeddings, lam=lam, lam2=lam2)

    dist_mats.append(-1 * dist_mat)

    dist_mats = np.array(dist_mats)

    return dist_mats


def get_softmax_probs(dist_mats, ratio=0.8, thresh_1=5.0, thresh_2=6.0):
    softmax_mat = []
    for rel in tqdm(range(dist_mats.shape[0])):
        dist_mat = dist_mats[rel]

        softmax_probs = np.zeros(dist_mat.shape)

        for i, row in enumerate(dist_mat):
            s1_indices = np.argwhere(row <= thresh_1)
            s2_indices = np.argwhere((row > thresh_1) & (row <= thresh_2))
            s3_indices = np.argwhere(row > thresh_2)

            alpha = np.log(1 / 10.0) / -thresh_1

            row_softmax_probs = softmax(-1 * alpha * row)

            s1 = np.sum(row_softmax_probs[s1_indices])
            s2 = np.sum(row_softmax_probs[s2_indices])
            s3 = np.sum(row_softmax_probs[s3_indices])

            remainder = (1 - ratio) / 2.0

            row_softmax_probs[s1_indices] *= 0.80 / s1
            row_softmax_probs[s2_indices] *= 0.10 / s2
            row_softmax_probs[s3_indices] *= 0.0 / s3

            row_softmax_probs = row_softmax_probs / np.sum(row_softmax_probs)
            softmax_probs[i] = row_softmax_probs

        softmax_mat.append(softmax_probs)

    softmax_mat = np.array(softmax_mat)
    softmax_mat = softmax_mat / np.sum(softmax_mat, axis=2)[:, :, None]

    return softmax_mat


def get_softmax_probs2(dist_mats,
                       knowledge_graph,
                       index_to_key,
                       ratio=0.8,
                       thresh_1=5.0,
                       thresh_2=6.0):
    softmax_mat = []
    for rel in tqdm(range(dist_mats.shape[0])):
        dist_mat = dist_mats[rel]

        softmax_probs = np.zeros(dist_mat.shape)

        for i, row in enumerate(dist_mat):
            sampling_strategy = False
            if i == (len(dist_mat) - 1):
                sampling_strategy = True
            else:
                all_edges = knowledge_graph.edges(index_to_key[i], data=True)
                all_edge_rel_types = np.array(
                    [edge[2]['rel_type'] for edge in all_edges])
                sampling_strategy = np.any(all_edge_rel_types == rel)

            if sampling_strategy:
                s1_indices = np.argwhere(row <= thresh_1)[:, 0]

                s1_count = len(s1_indices)

                s2_mask = row > thresh_1

                row_filtered_a = row[s2_mask]
                sorted_indices = np.argsort(row_filtered_a)
                top_n_indices = sorted_indices[:s1_count]
                s2_indices = np.nonzero(s2_mask)[0][top_n_indices]

                joint_indices = np.concatenate([s1_indices, s2_indices],
                                               axis=0)

                row_filtered = row[joint_indices]

                alpha = np.log(1 / 5.0) / -thresh_1
                row_softmax_probs_filtered = softmax(-1 * alpha * row_filtered)
                row_softmax_probs = np.zeros_like(row)

                row_softmax_probs[joint_indices] = row_softmax_probs_filtered

                row_softmax_probs = row_softmax_probs / np.sum(
                    row_softmax_probs)
                softmax_probs[i] = row_softmax_probs
            else:
                row_softmax_probs = np.zeros_like(row)
                row_softmax_probs[i] = 1.0
                softmax_probs[i] = row_softmax_probs

        softmax_mat.append(softmax_probs)

    softmax_mat = np.array(softmax_mat)
    softmax_mat = softmax_mat / np.sum(softmax_mat, axis=2)[:, :, None]

    return softmax_mat


def distance_kernel(x, seq_len, threshold, v, u, k_max, k_min):
    b = 10**(np.log10(float(k_max) / k_min) / (seq_len * (u - v)))
    a = float(k_max) / b**(-seq_len * (threshold + v))

    k = a * b**(-x)
    k = np.minimum(np.ones_like(k), k)

    return k


def extract_embedding_subsets(node_embeddings, knowledge_graph,
                              extracted_concept_ids):
    index_to_key = np.array(sorted(list(knowledge_graph.nodes)))
    key_to_index = dict(zip(index_to_key, range(len(index_to_key))))

    extracted_node_embeddings = []
    extracted_index_to_key = np.zeros_like(extracted_concept_ids, dtype=int)
    extracted_key_to_index = {}

    for i, concept_id in enumerate(extracted_concept_ids):
        concept_id_index = key_to_index[concept_id]

        extracted_node_embedding = node_embeddings[concept_id_index]

        extracted_node_embeddings.append(extracted_node_embedding)
        extracted_index_to_key[i] = concept_id
        extracted_key_to_index[concept_id] = i

    extracted_node_embeddings = np.array(extracted_node_embeddings)
    extracted_index_to_key = extracted_index_to_key.tolist()

    return extracted_node_embeddings, extracted_index_to_key, extracted_key_to_index


def save_embedding_distances_and_probs(output_dir,
                                       use_cached_dists=True,
                                       use_training_data=True):

    with open(Path(output_dir) / NODE_EMBEDDINGS_FILE, 'rb') as f:
        node_embeddings = np.load(f)

    with open(Path(output_dir) / REL_EMBEDDINGS_FILE, 'rb') as f:
        rel_embeddings = np.load(f)

    with open(Path(output_dir) / LAMBDA_FILE, 'rb') as f:
        lam_dict = json.load(f)
        lam = lam_dict['lambda']
        lam2 = lam_dict['lambda2']

    knowledge_graph = nx.read_gpickle(Path(output_dir) / KNOWLEDGE_GRAPH_FILE)

    if use_cached_dists and Path.exists(
            Path(output_dir) / DENSE_DISTS_MAT_FILE):
        dist_mats = np.load(Path(output_dir) / DENSE_DISTS_MAT_FILE)
        with open(Path(output_dir) / INDEX_TO_KEY_FILE, 'r') as f:
            index_to_key = json.load(f)
    else:

        if use_training_data:
            data = pd.read_csv(Path(output_dir) / DATA_FILE)

            concept_ids = data['concept_id'].unique().astype(int).tolist()

            node_embeddings, index_to_key, key_to_index = extract_embedding_subsets(
                node_embeddings, knowledge_graph, concept_ids)

            dist_mats = get_embedding_distances(rel_embeddings,
                                                node_embeddings, lam, lam2)

            with open(Path(output_dir) / KEY_TO_INDEX_FILE, 'w') as f:
                json.dump(key_to_index, f)

            with open(Path(output_dir) / INDEX_TO_KEY_FILE, 'w') as f:
                json.dump(index_to_key, f)
        else:
            dist_mats = get_embedding_distances(rel_embeddings,
                                                node_embeddings, lam, lam2)
    softmax_probs = get_softmax_probs2(dist_mats,
                                       knowledge_graph,
                                       index_to_key,
                                       ratio=0.50,
                                       thresh_1=5.0,
                                       thresh_2=6.0)

    # Save dense matrices

    if not (use_cached_dists
            and Path.exists(Path(output_dir) / DENSE_DISTS_MAT_FILE)):
        with open(Path(output_dir) / DENSE_DISTS_MAT_FILE, 'wb') as f:
            np.save(f, dist_mats)

    with open(Path(output_dir) / DENSE_PROBS_MAT_FILE, 'wb') as f:
        np.save(f, softmax_probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Calculate distances and replacement probabilities between node embeddings."
    )
    parser.add_argument('output_dir',
                        help='Directory containing model output.')
    args = parser.parse_args()

    save_embedding_distances_and_probs(args.output_dir,
                                       use_cached_dists=True,
                                       use_training_data=True)
