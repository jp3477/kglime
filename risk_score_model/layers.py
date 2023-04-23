# Python imports

# Third-party imports
from tensorflow import keras
import tensorflow as tf

# Package imports


# @keras.utils.register_keras_serializable('adverse_event_prediction')
class RobustScalerLayer(keras.layers.Layer):
    def __init__(self, robust_scaler_center, robust_scaler_scale, **kwargs):
        super().__init__(**kwargs)
        self.robust_scaler_center = robust_scaler_center
        self.robust_scaler_scale = robust_scaler_scale

        # robust_scaler = RobustScaler()
        # robust_scaler.center_ = robust_scaler_center
        # robust_scaler.scale_ = robust_scaler_scale

        # self.robust_scaler = robust_scaler

    def call(self, inputs):
        return (inputs - self.robust_scaler_center) / self.robust_scaler_scale
        # return tf.numpy_function(lambda t: self.robust_scaler.transform(t),
        #                          [inputs], tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "robust_scaler_center": self.robust_scaler_center,
            "robust_scaler_scale": self.robust_scaler_scale
        })
        return config


# @keras.utils.register_keras_serializable('adverse_event_prediction')
class SliceLayer(keras.layers.Layer):
    def __init__(self, indices, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.indices = indices

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=-1)

    # def compute_output_shape(self, input_shape):
    #     return input_shape[:-1] + (len(self.indices), )

    def get_config(self):
        config = super().get_config()
        config.update({"indices": self.indices})
        return config
