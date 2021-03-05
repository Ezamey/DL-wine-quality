"""Launch our app"""
import numpy as np
import pandas as pd
from sklearn.utils import class_weight

from scripts.modelkeras import WineTesting

# dataframe
PATH = "data/wine.csv"
wine = pd.read_csv(PATH)

# Some features engineering
wine_modif = wine.copy()
# ratio TSO2 /FSO2
wine_modif["ratioSO2"] = (
    wine_modif["free sulfur dioxide"] / wine_modif["total sulfur dioxide"]
)
# Addingclass weights
labels = wine_modif["quality"].values

class_weights = dict(
    zip(
        [0, 1, 2, 3, 4, 5, 6],
        class_weight.compute_class_weight(None, classes=np.unique(labels), y=labels),
    )
)

# Our class
weighted_model_testing = WineTesting(
    name="Weighted", wine=wine_modif, class_weight=class_weights
)

# Normalize data threw our class attributes
mean = np.mean(weighted_model_testing.train_features, axis=0)
weighted_model_testing.train_features -= mean
weighted_model_testing.test_features -= mean

std = np.std(weighted_model_testing.train_features, axis=0)
weighted_model_testing.train_features /= std
weighted_model_testing.test_features /= std

# Model creation
weighted_model = weighted_model_testing.make_model(
    input_shape=weighted_model_testing.input_shape,
    output_shape=weighted_model_testing.output_shape,
)
# Model training
weight_history = weighted_model_testing.fit_model(model=weighted_model)
# Model plotting images saved in img in format(img/{model.name}_learning_train_{model.train_number}.png)
weighted_model_testing.plot_metrics(weight_history)

r3 = weighted_model.evaluate(
    weighted_model_testing.test_features, weighted_model_testing.test_target
)


def write_details(rapport, for_n):
    return """Rapport for {}
    \nloss : {},
    \ntp: {},
    \nfp:{},
    \ntn : {},
    \nfn : {},
    \naccuracy : {},
    \nprecision : {},
    \nrecall : {},
    \nauc : {}
    """.format(
        for_n,
        rapport[0],
        rapport[1],
        rapport[2],
        rapport[3],
        rapport[4],
        rapport[5],
        rapport[6],
        rapport[7],
        rapport[8],
    )


print("----------------")
print(write_details(r3, weighted_model_testing.name))


def refit():
    while True:
        print("Re-train the model? y/n")
        choice = input()
        if choice == "y":
            weight_history = weighted_model_testing.fit_model(model=weighted_model)
            weighted_model_testing.plot_metrics(weight_history)
            model_eval = weighted_model.evaluate(
                weighted_model_testing.test_features, weighted_model_testing.test_target
            )
            print(write_details(rapport=model_eval, for_n=weighted_model_testing.name))
        else:
            break
    return


refit()
