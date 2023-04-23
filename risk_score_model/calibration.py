# Python imports

# Third-party imports
import tensorflow as tf
from tensorflow import keras
import tensorflow_lattice as tfl
import numpy as np

# Package imports

# tfl.layers.PWLCalibration = keras.utils.register_keras_serializable(
#     'adverse_event_prediction')(tfl.layers.PWLCalibration)


def build_calibrated_model(base_model, x_calib, y_calib, name=None):
    probs = base_model.predict(x_calib)
    input_keypoints = np.linspace(np.min(probs), np.max(probs), num=5)

    base_model.trainable = False
    inputs = base_model.input
    output_layer = base_model(inputs, training=False)

    calibrator_layer = tfl.layers.PWLCalibration(
        input_keypoints=input_keypoints,
        output_min=0.0,
        output_max=1.0,
        monotonicity='increasing')(output_layer)

    calibrated_model = tf.keras.models.Model(inputs=inputs,
                                             outputs=calibrator_layer,
                                             name=name)

    LEARNING_RATE = 0.1
    BATCH_SIZE = 128
    NUM_EPOCHS = 10

    calibrated_model.compile(loss=tf.keras.losses.mean_squared_error,
                             optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    calibrated_model.fit(x_calib,
                         y_calib,
                         batch_size=BATCH_SIZE,
                         epochs=NUM_EPOCHS,
                         shuffle=True,
                         verbose=0)

    return calibrated_model


# def build_joint_calibrated_model(calibrated_models):

#     inputs = tf.keras.Input(shape=(None, 2), dtype=tf.int32)

#     combined_models = []
#     for calibrated_model in calibrated_models:
#         calibrated_model.trainable = False
#         combined_models.append(calibrated_model(inputs, training=False))

#     combined_models = tf.keras.layers.Concatenate(axis=-1)(combined_models)

#     joint_calibrated_model = tf.keras.models.Model(inputs=inputs,
#                                                    outputs=combined_models)

#     calibrated_model.compile(loss=tf.keras.losses.mean_squared_error,
#                              optimizer=tf.keras.optimizers.Adam(0.01))

#     return joint_calibrated_model


def build_joint_calibrated_model(calibrated_models):
    inputs = tf.keras.Input(shape=(None, 2))

    outputs = [model(inputs) for model in calibrated_models]
    combined_outputs = tf.keras.layers.Concatenate(-1)(outputs)

    joint_calibrated_model = tf.keras.models.Model(inputs=inputs,
                                                   outputs=combined_outputs)

    joint_calibrated_model.compile(loss=tf.keras.losses.mean_squared_error,
                                   optimizer=tf.keras.optimizers.Adam(0.01))

    return joint_calibrated_model
