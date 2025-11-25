import pandas as pd
from pathlib import Path
from joblib import load
from utils import basic_clean

DEPARTMENT = load("models/department_classifier.joblib")
SENTIMENT = load("models/sentiment_classifier.joblib")

def predict_one(title: str, body: str):
    text = basic_clean((title or "") + " " + (body or ""))
    dept = DEPARTMENT.predict([text])[0]
    sent = SENTIMENT.predict([text])[0]
    return dept, sent

def predict_csv(input_csv: str, output_csv: str = "outputs/predictions_batch.csv"):
    df = pd.read_csv(input_csv)
    texts = (df["title"].fillna("") + " " + df["body"].fillna("")).map(basic_clean)
    
    df["predicted_department"] = DEPARTMENT.predict(texts)
    df["predicted_sentiment"] = SENTIMENT.predict(texts)
    df["timestamp"] = pd.Timestamp.now().isoformat()
    
    Path("outputs").mkdir(exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    
if __name__ == "__main__":
    import sys
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "data/samples_to_predict.csv"