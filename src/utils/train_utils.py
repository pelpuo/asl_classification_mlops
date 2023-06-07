import numpy as np
import pandas as pd
import tensorflow as tf
from dvclive.keras import DVCLiveCallback
from sklearn import preprocessing


def train_model(train_file_path, epochs, batch_size, validation_split, model_path):
    train_data = pd.read_csv(train_file_path)

    y_train = train_data["label"]
    X_train = train_data.drop("label", axis=1).values.reshape(-1, 28, 28, 1)

    X_train = X_train / 255.0

    labels = y_train.unique().tolist()
    labels.sort()

    classes = []
    for label in labels:
        classes.append(chr(label + 65))
    classes

    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_binarized = lb.transform(y_train)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(24, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train_binarized,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size
    )

    model.save(model_path)
