import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from utils import basic_clean, make_train_test

DATA = "data/synthetic_reviews.csv"
MODEL_DIRECTORY = Path("models"); MODEL_DIRECTORY.mkdir(exist_ok=True)

def build_text(df):
    return (df["title"].fillna("") + " " + df["body"].fillna("")).map(basic_clean)

def make_pipeline(task: str) -> Pipeline:
    vec = TfidfVectorizer(ngram_range=(1,2), analyzer='word', min_df=2)
    if task == "department":
        clf = LinearSVC()
    elif task == "sentiment":
        clf = LogisticRegression(max_iter=200)
    else:
        raise ValueError("Unknown task")
    return Pipeline([("vectorizer", vec), ("classifier", clf)])

def train_model(df, y_col: str, model_name: str):
    train_df, test_df = make_train_test(df, y_col=y_col)
    pipe = make_pipeline(task=y_col)
    pipe.fit(build_text(train_df), train_df[y_col])
    y_pred = pipe.predict(build_text(test_df))
    print (f"\n=== {model_name} ===")
    print (classification_report(test_df[y_col], y_pred, digits=3))
    dump(pipe, MODEL_DIRECTORY / f"{model_name}.joblib")

def main():
    df = pd.read_csv(DATA)
    train_model(df, y_col="department", model_name="department_classifier") # Department
    train_model(df, y_col="sentiment", model_name="sentiment_classifier") # Sentiment
    
if __name__ == "__main__":
    main()