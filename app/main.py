from fastapi import FastAPI
import joblib

# Create FastAPI app
app = FastAPI()

# Load model and vectorizer
model = joblib.load("app/model.pkl")
vectorizer = joblib.load("app/vectorizer.pkl")

# Home route
@app.get("/")
def home():
    return {"message": "Spam Detection API is running"}

# Prediction route
@app.post("/predict")
def predict(text: str):
    # Convert text → numbers
    transformed_text = vectorizer.transform([text])

    # Predict
    prediction = model.predict(transformed_text)[0]

    # Convert output
    result = "spam" if prediction == 1 else "not spam"

    return {"prediction": result}