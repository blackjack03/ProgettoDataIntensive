from flask import Flask, request, render_template

import tensorflow as tf
from tensorflow.keras import backend as K
import joblib
import numpy as np


def recall_m(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def predict_impl(kargs):
    try:
        scaler = joblib.load("models/scaler_X.pkl")
        X = scaler.transform(np.asarray([[
                kargs["age"],      # age
                kargs["sex"],      # sex
                kargs["hchol"],    # high_col
                kargs["cholck"],   # col_check
                kargs["bmi"],      # BMI
                kargs["smoke"],    # smoker
                kargs["heartdis"], # heartDis
                kargs["F"],        # fruits
                kargs["V"],        # veg
                kargs["alchool"],  # alchool
                kargs["genheal"],  # gen_health
                kargs["physill"],  # phisical_ill_days
                kargs["diffwalk"], # diff_walk
                kargs["highbp"]    # high_BP
            ]], dtype=np.float32)
        )
        # Load Model
        model = tf.keras.models.load_model(
            'models/best_model.keras',
            compile=False,
            custom_objects={'f1_score': f1_score}
        )
        output = model.predict([X], verbose=0)[0, 0]
        # print("OUTPUT:", output)
        return 'diabetes' if output > 0.5 else 'no_diabetes'
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "error"


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("/index.html")

@app.route("/predict")
def predict():
    D = {
        "age": float(request.args["age"]),
        "sex": float(request.args["sex"]),
        "hchol": float(request.args["hchol"]),
        "cholck": float(request.args["cholck"]),
        "bmi": float(request.args["bmi"]),
        "smoke": float(request.args["smoke"]),
        "heartdis": float(request.args["heartdis"]),
        "F": float(request.args["F"]),
        "V": float(request.args["V"]),
        "alchool": float(request.args["alchool"]),
        "genheal": float(request.args["genheal"]),
        "physill": float(request.args["physill"]),
        "diffwalk": float(request.args["diffwalk"]),
        "highbp": float(request.args["highbp"])
    }
    return predict_impl(D)


if __name__ == '__main__':
    app()
