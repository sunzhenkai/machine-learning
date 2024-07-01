import pandas as pd
import tensorflow as tf

from internal.feature.feature_engineering import DataManager


class TensorflowLinearRegression:
    def __init__(self, data_manager: DataManager, epochs: int = 1000):
        self.data: DataManager = data_manager
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(units=1, input_shape=self.data.shape))
        self.model.compile(optimizer='sgd', loss='mean_squared_error')
        self.epochs = epochs

    def run(self):
        x = tf.constant(self.data.train_x_encoded.values, dtype=tf.float32)
        y = tf.constant(self.data.train_y.values, dtype=tf.float32)
        print(x, y)
        self.model.fit(x, y, epochs=self.epochs)
        weights = self.model.get_weights()
        print('weights: ', weights)

    def predict(self, test_x: pd.DataFrame):
        x = tf.constant(test_x.values, dtype=tf.float32)
        return self.model.predict(x)
