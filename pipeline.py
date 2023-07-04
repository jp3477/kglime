# Python imports
import logging
from pathlib import Path
import os
import argparse
import configparser

# Third-party imports
import networkx as nx
import pandas as pd

# Package imports
from knowledge_graph import create_knowledge_graph
from cohort import (build_cohort, write_vocabulary_to_table)
from graph_embeddings.train_graph_embeddings import train_graph_embeddings_mp
from risk_score_model.adverse_event_prediction import train_adverse_event_models
from graph_embeddings.perturbations import save_embedding_distances_and_probs
from utils import CONFIG_PATH

CONFIG = configparser.ConfigParser()
CONFIG.read(CONFIG_PATH)

MS_DMT_DRUGS_CSV = Path(CONFIG['REFERENCE FILES']['ms_dmt_drugs_file'])
ADVERSE_EFFECTS_CSV = Path(CONFIG['REFERENCE FILES']['adverse_effects_file'])


def main(output_dir):
    output_dir = Path(output_dir)

    # Create output dir if does not exist
    if not Path.exists(output_dir):
        Path.mkdir(output_dir, exist_ok=True)

    # Create drug digraph
    print("""
############################################################
Creating drug graph
############################################################
    """)
    knowledge_graph_path = Path(
        output_dir) / CONFIG['MODEL FILES']['knowledge_graph_file']

    # knowledge_graph = create_knowledge_graph("myfakedbstring",
    #                                          knowledge_graph_path,
    #                                          use_cache=True)

    # write_vocabulary_to_table(knowledge_graph_path)

    # Define patient cohort
    print("""
############################################################
Defining patient cohort
############################################################
    """)

    # records = build_cohort(drug_concept_ids, ae_concept_ids,
    #                        CONFIG['MODEL FILES']['drug_eras_file'])

    # Train Graph embeddings
    print("""
############################################################
Training graph embeddings
############################################################
    """)

    # trained_embeddings_dir = Path(
    #     output_dir) / CONFIG['EMBEDDING FILES']['embeddings_dir']

    # train_graph_embeddings_mp(output_dir,
    #                           learning_rate=0.01,
    #                           batch_size=1024,
    #                           epochs=300,
    #                           embedding_size=80,
    #                           n_layers=2,
    #                           negative_samples=128,
    #                           patience=10,
    #                           num_gpus=2,
    #                           regularizer='basis',
    #                           basis=6,
    #                           dropout=0.0)

    # Calculate Embedding Distances and Probabilities
    print("""
############################################################
Calculating embedding distances and probabilities
############################################################
    """)

    # save_embedding_distances_and_probs(output_dir)

    # Train Adverse Event prediction model
    print("""
############################################################
Train adverse event prediction model
############################################################
    """)

    train_adverse_event_models(output_dir,
                               epochs=100,
                               batch_size=256,
                               learning_rate=0.001,
                               patience=20,
                               use_data_cache=True)

    # Perform Anova Analysis on predictions
    print("""
############################################################
Performing an Anova Analysis of model prediction
############################################################
    """)

    # model_dir = ae_prediction_model_path / 'saved_model'
    # graph_path = OUTPUT_DIR / "drug_graph.pkl"
    # test_data_path = ae_prediction_model_path / 'test_data.csv'
    # labels_path = ae_prediction_model_path / 'labels.csv'

    # run_anova_on_model_from_files(model_dir, graph_path, test_data_path,
    #                               labels_path, ae_concepts,
    #                               ae_prediction_model_path)

    # Apply LIME Model
    print("""
############################################################
Running LIME explanation model
############################################################
    """)

    # base_path = Path(OUTPUT_DIR) / f'adverse_event_prediction_model/'

    # model_dir = base_path / 'saved_model'

    # graph_path = OUTPUT_DIR / "drug_graph.pkl"
    # test_data_path = base_path / 'test_data.csv'
    # train_data_path = base_path / 'train_data.csv'
    # labels_path = base_path / 'labels.csv'
    # savepath = base_path

    # explain_model_from_files(model_dir, graph_path, ae_names, test_data_path,
    #                          train_data_path, labels_path, savepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a ML pipeline for adverse drug event predictions")

    parser.add_argument('output_dir', help='Output path for pipeline run.')
    args = parser.parse_args()

    main(args.output_dir)