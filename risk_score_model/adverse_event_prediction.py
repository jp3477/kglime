# Python imports
from pathlib import Path
import logging
import argparse
from itertools import chain
import configparser

# Third-party imports
import numpy as np
from numpy.random import seed
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Concatenate
import pandas as pd
import networkx as nx

# Package imports
from .layers import RobustScalerLayer
from cohort import get_adverse_event_labels, get_concept_sequences_with_drug_era_ids
from .calibration import build_calibrated_model, build_joint_calibrated_model
from .loss import BinaryFocalLoss
from .sequencer import build_padded_sequences, condense_sequences
from .evaluation import evalute_calibrated_model

#Make keras pickalable
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

MAXLEN = int(CONFIG['MODEL PARAMETERS']['max_sequence_length'])

seed(1)
logging.basicConfig(level=logging.INFO)
tf.keras.backend.set_floatx('float32')


# Hotfix function
def make_keras_picklable():
    def unpack(model, training_config, weights):
        restored_model = deserialize(model)
        if training_config is not None:
            restored_model.compile(
                **saving_utils.compile_args_from_training_config(
                    training_config))
        restored_model.set_weights(weights)
        return restored_model

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


make_keras_picklable()



class ReturnBestEarlyStopping(keras.callbacks.EarlyStopping):
    def __init__(self, **kwargs):
        super(ReturnBestEarlyStopping, self).__init__(**kwargs)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose > 0:
                print(f'\nEpoch {self.stopped_epoch + 1}: early stopping')
        elif self.restore_best_weights:
            if self.verbose > 0:
                print(
                    'Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)


def build_model(pretrained_weights,
                vocab,
                embedding_size,
                robust_scaler,
                lstm_units=32,
                depth=2):

    pretrained_weights = pretrained_weights.copy()
    vocab = vocab.copy()

    if 0 in vocab:
        zero_index = vocab.index(0)
        vocab.remove(0)
        pretrained_weights.pop(zero_index)

    pretrained_weights = np.array(
        [np.zeros(embedding_size),
         np.zeros(embedding_size)] + list(pretrained_weights))

    vocab_size = len(vocab)

    inputs = keras.Input(shape=(None, 2), dtype=tf.int32)
    concept_ids = keras.layers.Lambda(lambda x: x[:, :, 0])(inputs)
    concept_date_ints = keras.layers.Lambda(lambda x: x[:, :, 1])(inputs)
    concept_date_ints = keras.layers.Flatten()(concept_date_ints)
    concept_date_ints = tf.cast(concept_date_ints, tf.float32)

    concept_dates = RobustScalerLayer(robust_scaler.center_,
                                      robust_scaler.scale_)(concept_date_ints)
    concept_dates = tf.cast(concept_dates, tf.float32)
    concept_dates = tf.reshape(concept_dates, tf.shape(concept_date_ints))

    x = keras.layers.experimental.preprocessing.IntegerLookup(
        vocabulary=vocab, mask_value=0)(concept_ids)
    x = keras.layers.Embedding(
        input_dim=vocab_size + 2,
        output_dim=embedding_size,
        weights=[pretrained_weights],
        mask_zero=True,
        trainable=False,
        # embeddings_regularizer=keras.regularizers.L2(1e-2)
    )(x)

    y = tf.expand_dims(concept_dates, axis=-1)
    x = Concatenate(axis=-1)([x, y])

    for i in range(depth):
        return_sequences = False if i == depth - 1 else True
        x = keras.layers.LSTM(
            lstm_units,
            return_sequences=return_sequences,
            # kernel_regularizer=keras.regularizers.L2(0.9),
            # recurrent_regularizer=keras.regularizers.L2(1e-2),
            dropout=0.3)(x)

    x = keras.layers.Dense(
        100,
        activation='relu',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(100, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=x)

    return model


def build_submodel(base_model, idx):
    inputs = base_model.input
    output_layer = base_model(inputs, training=False)
    output_layer = keras.layers.Lambda(lambda x: x[:, idx])(output_layer)

    return keras.models.Model(inputs=inputs, outputs=output_layer)


def train_adverse_event_model(adverse_effect_concepts,
                              knowledge_graph_path,
                              output_folder,
                              embeddings_path,
                              learning_rate=0.01,
                              batch_size=128,
                              epochs=30,
                              patience=5,
                              use_data_cache=True):

    Path(output_folder).mkdir(exist_ok=True)

    data_path = Path(output_folder) / 'data.csv'
    labels_path = Path(output_folder) / 'labels.csv'

    adverse_effect_concept_ids = list(adverse_effect_concepts['concept_id'])
    adverse_effect_concept_names = list(
        adverse_effect_concepts['adverse_effect_name'])

    if use_data_cache and data_path.exists():
        logging.info(f"Loading cached data from {Path(data_path)}")
        data = pd.read_csv(Path(data_path), parse_dates=['concept_date'])
        labels = pd.read_csv(labels_path)
    else:
        logging.info("Fetching concept sequences")
        data = get_concept_sequences_with_drug_era_ids()

        logging.info("Fetching labels")

        labels = None
        key_cols = ['person_id', 'drug_era_id', 'hasDx']
        counter = 1
        for adverse_effect_concept_id in adverse_effect_concept_ids:
            ae_labels = get_adverse_event_labels(adverse_effect_concept_id)
            if labels is None:
                labels = ae_labels
            else:
                labels = labels.merge(
                    ae_labels[key_cols],
                    on=['person_id', 'drug_era_id'],
                    suffixes=(None,
                              f'_{adverse_effect_concept_names[counter]}'))

                counter += 1

        labels = labels.rename(
            columns={'hasDx': f'hasDx_{adverse_effect_concept_names[0]}'})

        data.to_csv(data_path, index=False)
        labels.to_csv(labels_path, index=False)

    # Load embeddings
    node_embeddings = np.load(embeddings_path)
    knowledge_graph_path = nx.read_gpickle(knowledge_graph_path)
    vocab = sorted(list(knowledge_graph_path.nodes))
    pretrained_weights = list(node_embeddings)

    # Train and Test sets (split by patient)
    train_patients, test_patients = train_test_split(
        data['person_id'].unique())

    train_data = data[data['person_id'].isin(train_patients)]
    test_data = data[data['person_id'].isin(test_patients)]

    train_data.to_csv(Path(output_folder) / 'train_data.csv', index=False)
    test_data.to_csv(Path(output_folder) / 'test_data.csv', index=False)

    # Condense data
    logging.info("Condensing data")
    data = condense_sequences(data, labels)

    train_data = data[data['person_id'].isin(train_patients)]
    test_data = data[data['person_id'].isin(test_patients)]

    logging.info(
        f"Sequence Length Median: {data['concept_id'].apply(len).median()}")

    logging.info("Padding data")
    train_sequences, robust_scaler = build_padded_sequences(train_data,
                                                            include_dates=True,
                                                            training=True,
                                                            maxlen=MAXLEN,
                                                            padding_how='post')
    train_labels = np.array(train_data.filter(regex='^hasDx', axis=1).values)

    train_sequences, train_concept_dates = train_sequences
    train_sequences = np.array(train_sequences)
    train_concept_dates = np.array(train_concept_dates)

    test_sequences = build_padded_sequences(test_data,
                                            include_dates=True,
                                            training=False,
                                            maxlen=MAXLEN,
                                            padding_how='post')
    test_labels = np.array(test_data.filter(regex='^hasDx', axis=1).values)
    test_sequences, test_concept_dates = test_sequences

    def create_model():
        # Build LSTM model
        model = build_model(pretrained_weights,
                            vocab,
                            len(node_embeddings[0]),
                            robust_scaler,
                            lstm_units=120,
                            depth=2)

        model.compile(
            loss=BinaryFocalLoss(alpha=0.75, gamma=2.5),
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[
                tf.keras.metrics.AUC(curve='PR', name='AUPRC'),
            ],
            run_eagerly=True)

        return model

    # Train LSTM model
    early_stopping = ReturnBestEarlyStopping(monitor='val_AUPRC',
                                             verbose=1,
                                             patience=patience,
                                             mode="max",
                                             restore_best_weights=True)

    x_train = np.stack([train_sequences, train_concept_dates], axis=-1)
    y_train = train_labels
    x_test = np.stack([test_sequences, test_concept_dates], axis=-1)
    y_test = test_labels

    uncalibrated_models = []
    calibrated_models = []
    for i in range(y_train.shape[1]):
        ae_name = adverse_effect_concept_names[i]
        logging.info(f'Training {ae_name} model')

        model = create_model()
        model.fit(x_train,
                  y_train[:, i],
                  validation_data=(x_test, y_test[:, i]),
                  epochs=epochs,
                  batch_size=batch_size,
                  shuffle=True,
                  callbacks=[early_stopping])

        logging.info(f'Calibrating {ae_name} model')
        calibrated_model = build_calibrated_model(model,
                                                  x_test,
                                                  y_test[:, i],
                                                  name=ae_name.replace(
                                                      " ", "_"))

        uncalibrated_models.append(model)
        calibrated_models.append(calibrated_model)

    joint_uncalibrated_model = build_joint_calibrated_model(
        uncalibrated_models)
    joint_calibrated_model = build_joint_calibrated_model(calibrated_models)

    saved_model_path = Path(
        output_folder) / CONFIG['MODEL FILES']['model_binary_dir']
    saved_model_path.mkdir(exist_ok=True)

    logging.info(
        f"Saving model to {Path(saved_model_path) / CONFIG['MODEL FILES']['calibrated_model_file']}"
    )

    # with open(
    #         Path(saved_model_path) /
    #         CONFIG['MODEL FILES']['calibrated_model_file'], 'wb') as f:
    #     pickle.dump(joint_calibrated_model, f)

    # with open(
    #         Path(saved_model_path) /
    #         CONFIG['MODEL FILES']['uncalibrated_model_file'], 'wb') as f:
    #     pickle.dump(joint_uncalibrated_model, f)

    joint_calibrated_model.save(saved_model_path /
                                CONFIG['MODEL FILES']['calibrated_model_file'])
    joint_uncalibrated_model.save(
        saved_model_path / CONFIG['MODEL FILES']['uncalibrated_model_file'])

    # Evaluate model
    evalute_calibrated_model(joint_calibrated_model, joint_uncalibrated_model,
                             x_train, y_train, x_test, y_test,
                             adverse_effect_concept_names, Path(output_folder))

    return joint_calibrated_model


def train_adverse_event_models(adverse_effect_concepts,
                               knowledge_graph_path,
                               output_folder,
                               embeddings_path,
                               learning_rate=0.01,
                               batch_size=128,
                               epochs=30,
                               patience=5,
                               use_data_cache=True):

    logging.info(
        f'Training adverse event model with {len(adverse_effect_concepts)} AEs'
    )
    model = train_adverse_event_model(adverse_effect_concepts,
                                      knowledge_graph_path,
                                      output_folder,
                                      embeddings_path,
                                      learning_rate=learning_rate,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      patience=patience,
                                      use_data_cache=use_data_cache)

    return model


#'natlizumab, interferon 1a, interferon 1b, fingolimod'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Run pipeline for linking drug ingredients and adverse effects")

    parser.add_argument('knowledge_graph')
    parser.add_argument('embeddings', dest='embeddings_path', required=True)
    parser.add_argument('adverse_effect_concept_ids', nargs='+')
    parser.add_argument('adverse_effect_concept_names', nargs='+')
    parser.add_argument('output_folder')

    args = parser.parse_args()

    train_adverse_event_models(args.adverse_effect_concept_ids,
                               args.adverse_effect_concept_names,
                               args.knowledge_graph, args.output_folder,
                               args.embeddings_path)
