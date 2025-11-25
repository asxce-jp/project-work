import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import basic_clean, make_train_test

DATA = "data/synthetic_reviews.csv"
OUTPUT_DIRECTORY = Path("outputs"); OUTPUT_DIRECTORY.mkdir(exist_ok=True)

def build_text(df):
    return (df["title"].fillna("") + " " + df["body"].fillna("")).map(basic_clean)

def evaluate_task(model_path, df, y_col, out_png):
    model = load(model_path)
    _, test = make_train_test(df, y_col=y_col)
    x_test = build_text(test)
    y_true = test[y_col]
    y_pred = model.predict(x_test)
    print(classification_report(y_true, y_pred, digits=3))
    
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_true.unique()))
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRECTORY / out_png, dpi=150)
    plt.close()
    
def main():
    df = pd.read_csv(DATA)
    
    print("\n=== Evaluating Department Classifier ===")
    evaluate_task("models/department_classifier.joblib", df, "department", "confusion_matrix_department.png")
    
    print("\n=== Evaluating Sentiment Classifier ===")
    evaluate_task("models/sentiment_classifier.joblib", df, "sentiment", "confusion_matrix_sentiment.png")
    
if __name__ == "__main__":
    main()