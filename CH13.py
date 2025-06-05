import datetime
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook
from itertools import product
from typing import Union
import importlib.util

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell
print(tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
tf.random.set_seed(42)
np.random.seed(42)


class DataWindow():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,stride=1,shuffle=True,label_columns=None):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.stride=stride
        self.shuffle=shuffle

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, self.total_window_size)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:,:,self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, plot_col='y', max_subplots=3):
        inputs, labels = self.sample_batch

        plt.figure(figsize=(9, 6))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [scaled]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
              label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
              label_col_index = plot_col_index

            if label_col_index is None:
              continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', marker='s', label='Labels', c='green', s=64)

            if n == 0:
              plt.legend()

        plt.xlabel('Time (h)')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        shuffle = self.shuffle
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.stride,# ← スライド幅はstride
            shuffle=shuffle,
            batch_size=32
        )

        ds = ds.map(self.split_to_inputs_labels)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def sample_batch(self):
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            #tf.data.Datasetから最初の1バッチだけ取り出す一般的な方法です。
            self._sample_batch = result
        return result
    

class BaselineModel(Model):
    def __init__(self, label_index, mode="last", forecast_horizon=1, input_width=None):
        """
        Multi-step対応ベースラインモデル（last, mean, repeat対応）

        Parameters:
        - label_index : int or list[int]
            予測対象の特徴量のインデックス
        - mode : 'last', 'mean', 'repeat'
        - forecast_horizon : int
            予測ステップ数（ラベル幅）
        - input_width : int
            入力系列の長さ（repeatモードで必須）
        """
        super().__init__()

        if label_index is None:
            raise ValueError("label_index must be specified explicitly.")
        self.label_index = label_index if isinstance(label_index, list) else [label_index]
        self.mode = mode.lower()
        self.forecast_horizon = forecast_horizon
        self.input_width = input_width

        if self.mode == "repeat" and self.input_width is None:
            raise ValueError("In 'repeat' mode, input_width must be specified.")

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        indices = self.label_index

        if self.mode == "last":
            last_values = tf.concat([inputs[:, -1:, i:i+1] for i in indices], axis=-1)
            return tf.tile(last_values, [1, self.forecast_horizon, 1])

        elif self.mode == "mean":
            mean_values = tf.concat([
                tf.reduce_mean(inputs[:, :, i:i+1], axis=1, keepdims=True)
                for i in indices
            ], axis=-1)
            return tf.tile(mean_values, [1, self.forecast_horizon, 1])

        elif self.mode == "repeat":
            values = tf.concat([inputs[:, :, i:i+1] for i in indices], axis=-1)
            repeats = tf.math.floordiv(self.forecast_horizon + self.input_width - 1, self.input_width)
            tiled = tf.tile(values, [1, repeats, 1])
            return tiled[:, :self.forecast_horizon, :]

        else:
            raise ValueError("mode must be 'last', 'mean', or 'repeat'")

def model_predict(model, data_window):
    """
    modelとdata windowのsample_batchの最初のウィンドウに対して予測を行い、
    実測値と予測値をDataFrame にして返す。

    Parameters:
    - model : 学習済みモデル（BaselineModel など）
    - data_window : DataWindowクラスのインスタンス

    Returns:
    - df : pd.DataFrame（t, true_*, pred_*）*は列名
    """
    # ステップ1
    inputs, labels = data_window.sample_batch
    y_true_0 = np.squeeze(labels.numpy()[0])
    y_pred_0 = np.squeeze(model(inputs[0:1]).numpy())
#    y_pred_0 = np.squeeze(model.predict(inputs[0:1]))  # shape: (1, label_width, num_labels)

    # ステップ2: 次元を整える
    if y_true_0.ndim == 1:
        y_true_0 = y_true_0[:, np.newaxis]
    if y_pred_0.ndim == 1:
        y_pred_0 = y_pred_0[:, np.newaxis]

    # ステップ3: カラム名の取得（label_columns が None のときは index 数値）
    if data_window.label_columns is not None:
        col_names = data_window.label_columns
    else:
        col_names = [str(i) for i in range(y_true_0.shape[1])]

    # ステップ4: DataFrame にまとめる
    df = pd.DataFrame({'t': np.arange(y_true_0.shape[0])})
    for i, name in enumerate(col_names):
        df[f'true_{name}'] = y_true_0[:, i]
        df[f'pred_{name}'] = y_pred_0[:, i]

    return df


def plot_model_prediction(df, max_columns=3):
    """
    model_predict() の出力 DataFrame を用いて、実測値と予測値を比較表示する。

    Parameters:
    - df : pd.DataFrame（t, true_*, pred_* を含む）
    - max_columns : int 表示する系列の最大数（デフォルト3）
    """
    # ラベル列名を抽出
    true_cols = [col for col in df.columns if col.startswith("true_")]
    pred_cols = [col.replace("true_", "pred_") for col in true_cols]
    t = df["t"]

    n_plots = min(len(true_cols), max_columns)
    plt.figure(figsize=(8, 3 * n_plots))

    for i in range(n_plots):
        plt.subplot(n_plots, 1, i + 1)
        plt.plot(t, df[true_cols[i]], label="True", marker='o')
        plt.plot(t, df[pred_cols[i]], label="Pred", marker='x')
        plt.title(true_cols[i].replace("true_", ""))
        plt.ylabel("Value")
        plt.xlabel("Time step")
        plt.legend()

    plt.tight_layout()
    plt.show()


def compile_and_fit(model, window, patience=5, max_epochs=100):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')

    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])

    history = model.fit(window.train,
                       epochs=max_epochs,
                       validation_data=window.val,
                       callbacks=[early_stopping])

    return history