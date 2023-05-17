import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask
from flask_cors import CORS
from flask import jsonify
import pandas as pd
from pathlib import Path
from explainer.explainer import explain_patient_sequence, predict_risk_scores
import networkx as nx
import dill as pickle
from flask.json import JSONEncoder
from utils import PROJECT_PATH
# from tensorflow.python.keras.layers import deserialize, serialize
# from tensorflow.python.keras.saving import saving_utils
import pathos.multiprocessing as multiprocessing
import numpy as np
import json

from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

from risk_score_model.layers import RobustScalerLayer, SliceLayer
from tensorflow_lattice.layers import PWLCalibration

app = Flask(__name__)
CORS(app)

DRUG_ERAS_CSV = PROJECT_PATH / 'run_output/drug_eras.csv'
PATIENT_SEQUENCES_CSV = PROJECT_PATH / 'run_output/adverse_event_prediction_model/data.csv'
# JOINT_ADE_MODEL_PATH = PROJECT_PATH / 'run_output/adverse_event_prediction_model/saved_model/calibrated_model'
JOINT_ADE_MODEL_PATH = PROJECT_PATH / 'run_output/adverse_event_prediction_model/'

KNOWLEDGE_GRAPH_PATH = PROJECT_PATH / 'run_output/knowledge_graph.pkl'
EMBEDDINGS_DISTANCES_PATH = PROJECT_PATH / 'run_output/adverse_event_prediction_model/dense_dists_mat.npy'
EMBEDDINGS_PROBS_PATH = PROJECT_PATH / 'run_output/adverse_event_prediction_model/dense_probs_mat.npy'
REL_KEY_PATH = PROJECT_PATH / 'run_output/rel_key.json'

RISK_SCORE_MODELS = {}


def load_model(d):
    from tensorflow import keras
    import tensorflow as tf

    from tensorflow.keras.models import Model
    from tensorflow.python.keras.layers import deserialize, serialize
    from tensorflow.python.keras.saving import saving_utils

    from risk_score_model.layers import RobustScalerLayer, SliceLayer
    from tensorflow_lattice.layers import PWLCalibration

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

    ade_model_path = Path(d) / 'calibrated_model'
    print('Loading model ', d)
    # if Path.exists(ade_model_path):
    # K.clear_session()
    with keras.utils.custom_object_scope({
            'RobustScalerLayer': RobustScalerLayer,
            # 'adverse_event_prediction>PWLCalibration':
            # PWLCalibration,
            # 'adverse_event_prediction>RobustScalerLayer':
            # RobustScalerLayer,
            'PWLCalibration': PWLCalibration,
            'SliceLayer': SliceLayer
    }):
        ade_model = keras.models.load_model(
            ade_model_path,
            custom_objects={
                'RobustScalerLayer': RobustScalerLayer,
                # 'adverse_event_prediction>PWLCalibration':
                # PWLCalibration,
                # 'adverse_event_prediction>RobustScalerLayer':
                # RobustScalerLayer,
                'PWLCalibration': PWLCalibration,
                'SliceLayer': SliceLayer
            })
        # pickle.detect.trace(True)
        # print(pickle.detect.errors(ade_model))
        ae_name = ade_model.name

        print(pickle.pickles(ade_model))

        return (ae_name, ade_model)


# print("Diving into multiprocessing")
# pool = multiprocessing.Pool(processes=10)
# risk_models_list = pool.map(load_model,
#                             list(Path(JOINT_ADE_MODEL_PATH).iterdir()))
# RISK_SCORE_MODELS = dict(risk_models_list)

# SERVED_ADVERSE_EFFECTS = [
#     'Urinary tract infection', 'Depression', 'Leukopenia', 'Hypertension'
# ]
# for d in list(Path(JOINT_ADE_MODEL_PATH).iterdir()):
#     # if d.stem in SERVED_ADVERSE_EFFECTS:
#     if d.is_dir():
#         ade_model_path = d / 'calibrated_model'
#         if Path.exists(ade_model_path):
#             K.clear_session()
#             ade_model = keras.models.load_model(ade_model_path)
#             ae_name = ade_model.name

#             RISK_SCORE_MODELS[ae_name] = ade_model

# JOINT_ADE_MODEL = keras.models.load_model(JOINT_ADE_MODEL_PATH)

KNOWLEDGE_GRAPH = {}
EMBEDDINGS_DISTANCES = {}
REL_KEY = {}


def round_float(data):
    if type(data) is float:
        # rounded = math.ceil(data * 100) / 100
        rounded = round(data, 2)
        return rounded
    else:
        raise TypeError('data must be a float not %s' % type(data))


class CustomJSONEncoder(JSONEncoder):
    def default(self, o):
        #alternatively do the type checking in the default method
        if type(o) is float:
            return round_float(o)
        if type(o) is int:
            return float(o)


app.json_encoder = CustomJSONEncoder


@app.route("/drug_eras")
def get_drug_eras():
    df = pd.read_csv(DRUG_ERAS_CSV)
    return df.to_json(orient='records')


@app.route("/patient_sequence/<drug_era_id>")
def get_patient_sequence(drug_era_id):
    drug_era_id = int(drug_era_id)
    df = pd.read_csv(PATIENT_SEQUENCES_CSV)
    df = df[df['drug_era_id'] == drug_era_id]

    df['id'] = df.index

    return df.to_json(orient='records')


@app.route("/ade_prediction/<drug_era_id>")
def get_predictions(drug_era_id):
    drug_era_id = int(drug_era_id)
    patient_sequence = pd.read_csv(PATIENT_SEQUENCES_CSV,
                                   parse_dates=['concept_date'])
    patient_sequence = patient_sequence[patient_sequence['drug_era_id'] ==
                                        drug_era_id]

    risk_scores = predict_risk_scores(RISK_SCORE_MODELS, patient_sequence)
    # risk_scores = dict(
    #     sorted(risk_scores.items(), key=lambda x: x[1], reverse=True))
    risk_scores = sorted(risk_scores, key=lambda x: x['pred'], reverse=True)

    print(risk_scores)
    # print(risk_scores_and_exp)
    # return jsonify(risk_scores_and_exp)

    return jsonify(risk_scores)


@app.route("/ade_explanation/<drug_era_id>/<ae_name>")
def get_explanation(drug_era_id, ae_name):
    drug_era_id = int(drug_era_id)
    patient_sequence = pd.read_csv(PATIENT_SEQUENCES_CSV,
                                   parse_dates=['concept_date'])
    patient_sequence = patient_sequence[patient_sequence['drug_era_id'] ==
                                        drug_era_id]

    explanation = explain_patient_sequence(RISK_SCORE_MODELS, patient_sequence,
                                           KNOWLEDGE_GRAPH,
                                           EMBEDDINGS_DISTANCES,
                                           EMBEDDINGS_PROBS, REL_KEY, ae_name)

    return jsonify(explanation)


if __name__ == '__main__':
    app.logger.info("Loading risk score models")
    print("Diving into multiprocessing")
    model_dirs = [
        p for p in list(Path(JOINT_ADE_MODEL_PATH).iterdir())
        if p.is_dir() and p.stem in ['Leukopenia', 'Pain', 'Nausea']
        # and p.stem in [
        #     'Nausea', 'Infection', 'Hypertension', 'Headache', 'Rash',
        #     'Pyrexia', 'Diarrhoea', 'Arthralgia', 'Back pain',
        #     'Hypersensitivity', 'Oedema peripheral', 'Depression', 'Chills',
        #     'Cough', 'Thrombocytopenia', 'Asthenia', 'Anaemia', 'Seizure',
        #     'Pruritus', 'Dyspnoea', 'Myalgia', 'Vomiting', 'Lymphopenia',
        #     'Upper respiratory tract infection', 'Tachycardia'
        # ]
    ]

    with multiprocessing.Pool(processes=18) as pool:
        with keras.utils.custom_object_scope({
                'RobustScalerLayer': RobustScalerLayer,
                # 'adverse_event_prediction>PWLCalibration':
                # PWLCalibration,
                # 'adverse_event_prediction>RobustScalerLayer':
                # RobustScalerLayer,
                'PWLCalibration': PWLCalibration,
                'SliceLayer': SliceLayer
        }):
            risk_models_list = pool.map(load_model, model_dirs)
        RISK_SCORE_MODELS = dict(risk_models_list)

    app.logger.info("Loading knowledge graph")
    print("Loading knowledge graph")
    KNOWLEDGE_GRAPH = nx.read_gpickle(KNOWLEDGE_GRAPH_PATH)

    app.logger.info("Loading embedding distances")
    print("Loading embedding distances")
    with open(EMBEDDINGS_DISTANCES_PATH, 'rb') as f:
        EMBEDDINGS_DISTANCES = np.load(f)

    with open(EMBEDDINGS_PROBS_PATH, 'rb') as f:
        EMBEDDINGS_PROBS = np.load(f)

    with open(REL_KEY_PATH, 'r') as f:
        REL_KEY = json.load(f)

    app.run(debug=False)