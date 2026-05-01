from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# load files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# 👇 define input schema
class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    # كمل باقي الأعمدة هنا بنفس الأسماء

@app.get("/")
def home():
    return {"message": "Churn API Running"}

@app.post("/predict")
def predict(data: CustomerData):
    # تحويل لـ dict
    input_dict = data.dict()

    # ترتيب الأعمدة
    row = [input_dict[col] for col in columns]

    features = np.array(row).reshape(1, -1)

    # scaling
    features = scaler.transform(features)

    # prediction
    prob = model.predict_proba(features)[0][1]
    prediction = int(prob >= 0.3)

    return {
        "churn_probability": float(prob),
        "prediction": prediction
    }