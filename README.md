# Insurance Predictor

Simple Flask web app for predicting medical insurance cost using Linear Regression.

## Dataset
https://www.kaggle.com/datasets/mirichoi0218/insurance

## Tech Stack

- Python
- Flask
- scikit-learn
- pandas
- NumPy
- HTML
- JavaScript
- Tailwind CSS

## Flow

1. Load the insurance dataset from `dataset/insurance.csv`.
2. Train a Linear Regression model and save it as `insurance_model.pkl`.
3. Open the web form and enter the required user details.
4. Submit the form to get the predicted insurance cost.

## Run

Train the model:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

Start the Flask app:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open:

```text
http://127.0.0.1:5000
```