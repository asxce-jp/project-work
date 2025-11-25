import streamlit as st
import pandas as pd
import datetime as dt
from joblib import load
from pathlib import Path
import io

st.set_page_config(page_title="Hotel Review Classifier", layout="centered")
st.title("Hotel Review Classifier + Sentiment Analysis")

@st.cache_resource
def load_models():
    return load("models/department_classifier.joblib"), load("models/sentiment_classifier.joblib")

DEPARTMENT, SENTIMENT = load_models()

def clean(s):
    import re
    s = s.lower()
    s = re.sub(r"[^\w\sàèéìòóù]", " ", s)
    return " ".join(s.split())

tab1, tab2 = st.tabs(["Single Review Prediction", "Batch CSV"])

with tab1:
    title = st.text_input("Review Title", "")
    body = st.text_area("Review Body", "", height=160, placeholder="Write the review text here...")
    if st.button("Predict"):
        text = clean(f"{title} {body}")
        department = DEPARTMENT.predict([text])[0]
        sentiment = SENTIMENT.predict([text])[0]
        st.success(f"Predicted Department: **{department}** | Predicted Sentiment: **{sentiment}**")
        
with tab2:
    uploaded_file = st.file_uploader("Upload CSV file with 'id','title' and 'body' columns", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        texts = (df["title"].fillna("") + " " + df["body"].fillna("")).map(clean)
        df["predicted_department"] = DEPARTMENT.predict(texts)
        df["predicted_sentiment"] = SENTIMENT.predict(texts)
        df["timestamp"] = pd.Timestamp.now().isoformat()
        st.dataframe(df.head(20))
        buf = io.StringIO(); df.to_csv(buf, index=False)
        st.download_button("Download Predictions CSV", buf.getvalue(), file_name=f"predictions_batch_{dt.datetime.now():%Y-%m-%d_%H-%M-%S}.csv", mime="text/csv")