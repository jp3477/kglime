# Python imports
import collections
import os
from pathlib import Path
import logging
import warnings
import dill as pickle
from datetime import timedelta
import configparser

# Third-party imports
import pandas as pd
from kglime_explainer import KGLIMEExplainer, TableDomainMapper
import numpy as np
import networkx as nx
from pyvis.network import Network
from msticpy.vis.timeline import display_timeline
from bokeh.plotting import output_file, save
from bs4 import BeautifulSoup
from jinja2 import Environment, BaseLoader
from tqdm import tqdm
from sklearn.metrics import label_ranking_average_precision_score, top_k_accuracy_score
from torchmetrics.functional import retrieval_hit_rate
import torch
import sklearn

# Package imports
from risk_score_model.sequencer import build_padded_sequences, condense_sequences

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

MAXLEN = CONFIG['MODEL PARAMETERS']['max_sequence_length']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['BOKEH_PY_LOG_LEVEL'] = 'WARNING'
import tensorflow as tf

tf.config.run_functions_eagerly(False)


def kglime_explain(patient_sequence,
                   ade_model,
                   knowledge_graph,
                   sparse_dist_info,
                   index_date,
                   domains=['Condition', 'Drug', 'Measurement'],
                   domain_relation_map=None,
                   num_samples=1000,
                   num_features=10):
    if domain_relation_map is None:
        domain_relation_map = {
            'Condition': 'inv_has_ae',
            'Drug': 'has_ae',
            'Measurement': 'norel'
        }

    index_to_key = np.array(sorted(list(knowledge_graph.nodes)))
    key_to_index = dict(zip(index_to_key, range(len(index_to_key))))

    def feature_neighbor_fns(concept_id):
        domain_id = knowledge_graph.nodes[concept_id]['domain_id']
        relation = domain_relation_map[domain_id]

        if domain_id in domains:
            sparse_dist_info_rel = sparse_dist_info[relation]
            sparse_softmax_probs = sparse_dist_info_rel['probs']
            dists = sparse_dist_info_rel['dists']

            i = key_to_index[concept_id]

            indices = sparse_softmax_probs.indices[
                sparse_softmax_probs.indptr[i]:sparse_softmax_probs.indptr[i +
                                                                           1]]
            values = index_to_key[indices]
            probs = sparse_softmax_probs.data[
                sparse_softmax_probs.indptr[i]:sparse_softmax_probs.indptr[i +
                                                                           1]]
            dists = dists[i][indices]

        else:
            values = np.array([concept_id])
            probs = np.array([1.0])
            dists = np.array([0.0])

        return values, dists, probs

    categorical_names = {}
    for concept_id in knowledge_graph.nodes:
        categorical_names[concept_id] = knowledge_graph.nodes[concept_id][
            'title']

    categorical_names[0] = 'OOV'
    categorical_names[-1] = 'OOV'

    class_names = ['No AE', 'AE']

    explainer = KGLIMEExplainer(
        CONFIG['MODEL PARAMETERS']['max_sequence_length'],
        2,
        mode="classification",
        feature_names=["concept_id"],
        categorical_features=[0],
        categorical_names=categorical_names,
        class_names=class_names,
        feature_selection='auto',
        feature_neighbor_fns=feature_neighbor_fns,
        n_features=2)

    TableDomainMapper.map_exp_ids = TableDomainMapper._map_exp_ids_with_features
    exp = explainer.explain_instance(
        patient_sequence,
        lambda x: ade_model(x).numpy().reshape(-1, 1),
        num_samples=num_samples,
        num_features=num_features,
        top_labels=None,
        labels=[
            0,
        ])
    TableDomainMapper.map_exp_ids = TableDomainMapper._map_exp_ids

    # Parse explanations
    feat_indexes = [
        feat[0] for feat in exp.as_map()[0]
        if patient_sequence[feat[0]][0] != 0
    ]

    exp_scores = [
        feat[1] for feat in exp.as_map()[0]
        if patient_sequence[feat[0]][0] != 0
    ]
    concept_names = []
    days_backs = []
    concept_ids = []
    concept_dates = []

    for feat_index in feat_indexes:
        if feat_index < MAXLEN:
            concept_id = patient_sequence[feat_index][0]
            item_name = knowledge_graph.nodes[concept_id]['concept_name']

            days_back = int(patient_sequence[feat_index][1] - 1)
            item_date = index_date - timedelta(days=days_back)

            concept_names.append(item_name)
            concept_dates.append(item_date)
            days_backs.append(days_back)
            concept_ids.append(concept_id)
        else:
            days_back = int(patient_sequence[feat_index - MAXLEN][1] - 1)
            item_date = index_date - timedelta(days=days_back)
            item_name = f"days_back={patient_sequence[feat_index - MAXLEN][1] - 1}"

            concept_names.append(item_name)
            concept_dates.append(item_date)
            days_backs.append(days_back)
            concept_ids.append(0)

    exp_df = pd.DataFrame({
        'concept_id': concept_ids,
        'concept_name': concept_names,
        'days_back': days_backs,
        'dates': concept_dates,
        'score': exp_scores
    })

    return exp_df


def explain_patient_sequence(ade_joint_model, patient_sequence_df,
                             knowledge_graph, embedding_distances_matrix):
    condensed_sequence = condense_sequences(patient_sequence_df)
    patient_sequence = build_padded_sequences(condensed_sequence)
    index_date = patient_sequence_df['concept_id'].max()

    ade_models = ade_joint_model.layers[1:-1]

    explanations = {}
    for ade_model in ade_models:
        adverse_effect_name = ade_model.name
        ade_pred = ade_model(patient_sequence)
        explanation_df = kglime_explain(patient_sequence, ade_model,
                                        knowledge_graph,
                                        embedding_distances_matrix, index_date)

        explanations[adverse_effect_name] = {
            'pred': ade_pred,
            'explanation': explanation_df
        }

    return explanations


if __name__ == '__main__':
    pass