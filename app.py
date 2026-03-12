from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "insurance_model.pkl"

app = Flask(__name__, template_folder=str(BASE_DIR))

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    features = pd.DataFrame(
        [
            {
                "age": int(payload["age"]),
                "sex": payload["sex"],
                "bmi": float(payload["bmi"]),
                "children": int(payload["children"]),
                "smoker": payload["smoker"],
                "region": payload["region"],
            }
        ]
    )

    predicted_cost = max(0.0, float(model.predict(features)[0]))

    return jsonify(
        {
            "predicted_cost": predicted_cost,
            "formatted_cost": f"${predicted_cost:,.2f}",
        }
    )


if __name__ == "__main__":
    app.run(debug=True)