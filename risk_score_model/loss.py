# Python imports

# Third-party imports
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

# Package imports


@keras.utils.register_keras_serializable('adverse_event_prediction')
class BinaryFocalLoss(keras.losses.Loss):
    def __init__(self,
                 alpha=0.75,
                 gamma=2.0,
                 name='binary_focal_loss',
                 reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(name=name)

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def call(self, y_true, y_pred):
        fl = tfa.losses.SigmoidFocalCrossEntropy(alpha=self.alpha,
                                                 gamma=self.gamma,
                                                 reduction=self.reduction)
        sample_weight = tf.expand_dims(tf.ones_like(y_true), -1)
        # sample_weight[:, 1:] = 1
        focal_loss = fl(y_true, y_pred, sample_weight=sample_weight)
        return focal_loss