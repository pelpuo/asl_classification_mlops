import numpy as np
import pandas as pd
from sklearn import preprocessing
from tensorflow.keras.models import load_model


def get_metrics(test_file_path, model_path):
    test_data = pd.read_csv(test_file_path)

    y_test = test_data["label"]
    X_test = test_data.drop("label", axis=1).values.reshape(-1, 28, 28, 1)

    labels = y_test.unique().tolist()
    labels.sort()

    classes = []
    for label in labels:
        classes.append(chr(label + 65))

    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test_binarized = lb.transform(y_test)

    model = load_model(model_path)

    test_loss, test_acc = model.evaluate(X_test, y_test_binarized, verbose=2)

    return {"test_loss": test_loss, "test_acc": test_acc}
