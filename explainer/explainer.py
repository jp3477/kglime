# Python imports
import os
from datetime import timedelta
import configparser
from pathlib import Path

# Third-party imports
import pandas as pd
from kglime_explainer import KGLIMEExplainer, KGLIMEDomainMapper
import numpy as np

# Package imports
from risk_score_model.sequencer import build_padded_sequences, condense_sequences
from utils import CONFIG_PATH

CONFIG = configparser.ConfigParser()
CONFIG.read(CONFIG_PATH)

MAXLEN = int(CONFIG['MODEL PARAMETERS']['max_sequence_length'])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['BOKEH_PY_LOG_LEVEL'] = 'WARNING'
import tensorflow as tf

# tf.config.run_functions_eagerly(False)


def kglime_explain(patient_sequence,
                   ade_model,
                   knowledge_graph,
                   dense_dists_mat,
                   dense_probs_mat,
                   rel_key,
                   index_date,
                   num_samples=10000,
                   num_features=10):

    index_to_key = np.array(sorted(list(knowledge_graph.nodes)))
    key_to_index = dict(zip(index_to_key, range(len(index_to_key))))

    categorical_names = {}
    for concept_id in knowledge_graph.nodes:
        categorical_names[concept_id] = knowledge_graph.nodes[concept_id][
            'title']

    categorical_names[0] = 'OOV'
    categorical_names[-1] = 'OOV'

    class_names = ['No AE', 'AE']

    explainer = KGLIMEExplainer(int(
        CONFIG['MODEL PARAMETERS']['max_sequence_length']),
                                2,
                                dense_dists_mat,
                                dense_probs_mat,
                                index_to_key,
                                key_to_index,
                                rel_key,
                                mode="classification",
                                feature_names=["concept_id"],
                                categorical_features=[0],
                                categorical_names=categorical_names,
                                class_names=class_names,
                                feature_selection='auto')

    KGLIMEDomainMapper.map_exp_ids = KGLIMEDomainMapper._map_exp_ids_with_features
    exp = explainer.explain_instance(
        patient_sequence,
        lambda x: ade_model(x).numpy().reshape(-1, 1),
        num_samples=num_samples,
        num_features=num_features,
        top_labels=None,
        labels=[
            0,
        ])
    # KGLIMEDomainMapper.map_exp_ids = KGLIMEDomainMapper._map_exp_ids

    # Parse explanations

    # feat_indexes = [
    #     feat[0] for feat in exp.as_map()[0]
    #     if patient_sequence[feat[0]][0] != 0
    # ]

    feats = [
        feat[0] for feat in exp.as_list(label=0) if feat[0].concept_id != 0
    ]

    exp_scores = [
        feat[1] for feat in exp.as_list(label=0) if feat[0].concept_id != 0
    ]
    concept_names = []
    days_backs = []
    concept_ids = []
    concept_dates = []
    concept_domains = []
    rels = []

    for feat in feats:
        concept_id = feat.concept_id
        item_name = knowledge_graph.nodes[concept_id]['concept_name']
        domain_id = knowledge_graph.nodes[concept_id]['domain_id']

        # days_back = int(patient_sequence[feat_index][1] - 1)
        days_back = feat.days_elapsed
        item_date = index_date - timedelta(days=days_back)
        rel = feat.rel

        concept_names.append(item_name)
        concept_dates.append(item_date)
        days_backs.append(days_back)
        concept_ids.append(concept_id)
        concept_domains.append(domain_id)
        rels.append(rel)
    # else:
    #     days_back = int(patient_sequence[feat_index - MAXLEN][1] - 1)
    #     item_date = index_date - timedelta(days=days_back)
    #     item_name = f"days_back={patient_sequence[feat_index - MAXLEN][1] - 1}"

    #     concept_names.append(item_name)
    #     concept_dates.append(item_date)
    #     days_backs.append(days_back)
    #     concept_ids.append(0)
    #     concept_domains.append('None')

    exp_df = pd.DataFrame({
        'concept_id': concept_ids,
        'concept_name': concept_names,
        'rel': rels,
        'days_back': days_backs,
        'dates': concept_dates,
        'score': exp_scores,
        'concept_domain': concept_domains
    })

    exp_df['dates'] = exp_df['dates'].dt.strftime('%Y-%m-%d')

    return exp_df


def kglime_explain_old(patient_sequence,
                       ade_model,
                       knowledge_graph,
                       sparse_dist_info,
                       index_date,
                       domains=['Condition', 'Drug', 'Measurement'],
                       domain_relation_map=None,
                       num_samples=500,
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

    explainer = KGLIMEExplainer(int(
        CONFIG['MODEL PARAMETERS']['max_sequence_length']),
                                2,
                                mode="classification",
                                feature_names=["concept_id"],
                                categorical_features=[0],
                                categorical_names=categorical_names,
                                class_names=class_names,
                                feature_selection='auto',
                                feature_neighbor_fns=feature_neighbor_fns)

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
    print(patient_sequence.shape)

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
    concept_domains = []

    for feat_index in feat_indexes:
        if feat_index < MAXLEN:
            concept_id = patient_sequence[feat_index][0]
            item_name = knowledge_graph.nodes[concept_id]['concept_name']
            domain_id = knowledge_graph.nodes[concept_id]['domain_id']

            days_back = int(patient_sequence[feat_index][1] - 1)
            item_date = index_date - timedelta(days=days_back)

            concept_names.append(item_name)
            concept_dates.append(item_date)
            days_backs.append(days_back)
            concept_ids.append(concept_id)
            concept_domains.append(domain_id)
        else:
            days_back = int(patient_sequence[feat_index - MAXLEN][1] - 1)
            item_date = index_date - timedelta(days=days_back)
            item_name = f"days_back={patient_sequence[feat_index - MAXLEN][1] - 1}"

            concept_names.append(item_name)
            concept_dates.append(item_date)
            days_backs.append(days_back)
            concept_ids.append(0)
            concept_domains.append('None')

    exp_df = pd.DataFrame({
        'concept_id': concept_ids,
        'concept_name': concept_names,
        'days_back': days_backs,
        'dates': concept_dates,
        'score': exp_scores,
        'concept_domain': concept_domains
    })

    exp_df['dates'] = exp_df['dates'].dt.strftime('%Y-%m-%d')

    return exp_df


# def explain_patient_sequence(ade_joint_model, patient_sequence_df,
#                              knowledge_graph, embedding_distances_matrix):
#     condensed_sequence = condense_sequences(patient_sequence_df)
#     patient_sequence, patient_sequence_dates = build_padded_sequences(
#         condensed_sequence, maxlen=MAXLEN, include_dates=True)
#     patient_sequence = np.stack([patient_sequence, patient_sequence_dates],
#                                 axis=-1)[0]

#     index_date = patient_sequence_df['concept_date'].max()

#     ade_models = ade_joint_model.layers[1:-1]

#     explanations = {}
#     for ade_model in ade_models:
#         adverse_effect_name = ade_model.name
#         ade_pred = round(
#             float(
#                 ade_model(np.expand_dims(patient_sequence, 0)).numpy()[0][0]),
#             2)
#         explanation_df = kglime_explain(patient_sequence, ade_model,
#                                         knowledge_graph,
#                                         embedding_distances_matrix, index_date)

#         explanations[adverse_effect_name] = {
#             'risk_score': ade_pred,
#             'explanation': explanation_df.to_dict(orient='records')
#         }

#     return explanations


def explain_patient_sequence(ade_models, patient_sequence_df, knowledge_graph,
                             dense_dists_mat, dense_probs_mat, rel_key,
                             ae_name):
    condensed_sequence = condense_sequences(patient_sequence_df)
    patient_sequence, patient_sequence_dates = build_padded_sequences(
        condensed_sequence, maxlen=MAXLEN, include_dates=True)
    patient_sequence = np.stack([patient_sequence, patient_sequence_dates],
                                axis=-1)[0]

    index_date = patient_sequence_df['concept_date'].max()

    # ade_models = ade_joint_model.layers[1:-1]
    ade_model = ade_models[ae_name]

    rel_key = list(rel_key.keys())
    rel_key.append('identity')
    explanation = kglime_explain(patient_sequence,
                                 ade_model,
                                 knowledge_graph,
                                 dense_dists_mat,
                                 dense_probs_mat,
                                 rel_key,
                                 index_date,
                                 num_features=20,
                                 num_samples=5000)

    print('explanation_type, ', explanation)
    explanation = explanation.fillna(0)
    explanation['score'] = explanation['score'] / np.sum(explanation['score'])
    explanation = explanation.round(2).to_dict(orient='records')

    for i, _ in enumerate(explanation):
        explanation[i]['id'] = i + 1

    return explanation


def predict_risk_scores(ade_models, patient_sequence_df):
    condensed_sequence = condense_sequences(patient_sequence_df)
    patient_sequence, patient_sequence_dates = build_padded_sequences(
        condensed_sequence, maxlen=MAXLEN, include_dates=True)
    patient_sequence = np.stack([patient_sequence, patient_sequence_dates],
                                axis=-1)[0]

    # adverse_effect_names = ade_models.keys()
    ade_preds = []
    for adverse_effect_name, ade_model in ade_models.items():
        ade_pred = round(
            # np.squeeze(ade_model.predict(np.expand_dims(patient_sequence,
            #                                             0))).item(), 2)
            np.squeeze(ade_model(np.expand_dims(patient_sequence, 0))).item(),
            2)
        # ade_preds.append(ade_pred)
        ade_preds.append({
            'adverse_effect_name': adverse_effect_name,
            'pred': ade_pred
        })

    # ade_preds_zipped = dict(zip(adverse_effect_names, ade_preds))

    return ade_preds


# def predict_risk_scores(ade_joint_model, patient_sequence_df):
#     condensed_sequence = condense_sequences(patient_sequence_df)
#     patient_sequence, patient_sequence_dates = build_padded_sequences(
#         condensed_sequence, maxlen=MAXLEN, include_dates=True)
#     patient_sequence = np.stack([patient_sequence, patient_sequence_dates],
#                                 axis=-1)[0]

#     ade_models = ade_joint_model.layers[1:-1]
#     adverse_effect_names = []
#     for ade_model in ade_models:
#         adverse_effect_name = ade_model.name
#         adverse_effect_names.append(adverse_effect_name)

#     ade_preds = ade_joint_model.predict(np.expand_dims(patient_sequence, 0))
#     ade_preds_zipped = dict(zip(adverse_effect_names, ade_preds))

#     return ade_preds_zipped

if __name__ == '__main__':
    from cohort import get_concept_sequences_with_drug_era_ids
    from pathlib import Path
    from tensorflow import keras
    import tensorflow as tf
    import networkx as nx
    import os
    import numpy as np
    import json
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('model_path')
    parser.add_argument('drug_era_id', type=int)
    parser.add_argument('knowledge_graph_path')
    parser.add_argument('rel_key_path')

    args = parser.parse_args()
    saved_model_path = Path(args.model_path) / 'Nausea/calibrated_model'

    print('loading model')
    model = keras.models.load_model(saved_model_path)
    print("getting concept sequence")
    patient_sequence_df = get_concept_sequences_with_drug_era_ids(
        args.drug_era_id)
    print("reading knowledge graph")
    knowledge_graph = nx.read_gpickle(Path(args.knowledge_graph_path))

    print("Opening dense matrices")
    with open(Path(args.model_path) / 'dense_dists_mat.npy', 'rb') as f:
        dense_dists_mat = np.load(f)

    with open(Path(args.model_path) / 'dense_probs_mat.npy', 'rb') as f:
        dense_probs_mat = np.load(f)

    with open(Path(args.rel_key_path), 'r') as f:
        rel_key = json.load(f)

    print("Explaining sequence")
    explanation = explain_patient_sequence({'Nausea': model},
                                           patient_sequence_df,
                                           knowledge_graph, dense_dists_mat,
                                           dense_probs_mat, rel_key, 'Nausea')

    print(explanation)
