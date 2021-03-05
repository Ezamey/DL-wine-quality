"""Handle the models part of our script. Early Recall is set to track false negative"""

# imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# set ups
plt.rcParams["figure.figsize"] = (12, 10)


# Metrics for our models
METRICS = [
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
]


class WineTesting:
    def __init__(self, name, wine, class_weight=None):
        self.name = name
        self.wine = wine
        self.metrics = METRICS
        (
            self.train_features,
            self.test_features,
            self.train_target,
            self.test_target,
        ) = self.create_train_test_split()
        self.input_shape = (self.train_features.shape[1],)
        self.output_shape = len(wine["quality"].unique())  # number of output node
        self.class_weight = class_weight
        self.fit_number = 0

    def do_corr_heatmat(self):
        """Utility function to draw a heatmap"""
        corr = wine.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, linewidths=0.5, ax=ax)
        plt.savefig("../../img/corrheatmap.png", dpi=200)
        return

    def create_train_test_split(self, index_sep=0.8):
        """Split the data into train,test sets

        Args:
            index_sep (float, optional): % for split. Defaults to 0.8.

        Returns:
            [tuple]: 4 np.arrays
        """
        split_index = int(self.wine.shape[0] * index_sep)
        train = self.wine[:split_index]
        test = self.wine[split_index:]
        # Features selection
        X_train = train.drop(["quality"], axis=1).values
        X_test = test.drop(["quality"], axis=1).values
        # Target = Quality
        target = pd.get_dummies(self.wine["quality"]).values  # One hot encode
        y_train = target[:split_index]
        y_test = target[split_index:]
        return (X_train, X_test, y_train, y_test)

    def make_model(self, output_shape, input_shape, metrics=METRICS, output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        model = keras.Sequential(
            [
                keras.layers.Dense(60, activation="sigmoid", input_shape=input_shape),
                keras.layers.Dense(30, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(output_shape, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=self.metrics,
        )

        return model

    def fit_model(self, model, validation_split=0.2, EPOCHS=30, BATCH_SIZE=500):
        """Function fitting the model"""
        self.fit_number += 1

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="fn", verbose=0, patience=10, mode="min", restore_best_weights=True
        )

        model_history = model.fit(
            self.train_features,
            self.train_target,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[early_stopping],
            validation_split=validation_split,
            class_weight=self.class_weight,
        )

        return model_history

    def plot_metrics(self, history):
        metrics = ["loss", "auc", "precision", "recall"]
        for n, metric in enumerate(metrics):
            name = metric.replace("_", " ").capitalize()
            plt.subplot(2, 2, n + 1)
            plt.plot(history.epoch, history.history[metric], color="red", label="Train")
            plt.plot(
                history.epoch,
                history.history["val_" + metric],
                color="blue",
                linestyle="--",
                label="Val",
            )
            plt.xlabel("Epoch")
            plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "auc":
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()
        plt.savefig(
            "img/{}_learning_train_{}.png".format(self.name, self.fit_number), dpi=200
        )
        plt.clf()
        return

    def eval_model(self, model):
        return model.evaluate(self.test_features, self.test_target)
