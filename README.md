# Hotel Reviews Project

## Overview
Questo progetto implementa un sistema di smistamento automatico delle recensioni per strutture alberghiere. L'applicazione analizza un testo (titolo + corpo recensione) e produce 2 previsioni:
1. Reparto reponsabile
- Housekeeping
- Reception
- Food & Beverage (F&B)

2. Setiment della recensione
- Positivo
- Negativo

Il sistema comprende 3 componenti principali:
- Dataset sintetico generato automaticamente
- Modelli di ML addestrati con TF-IDF + Linear SVM / Logistic Regression
- Interfaccia di inferenza:
    - Script CLI
    - UI Streamlit

## Architettura
pw20-hotel-reviews/
├─ data/
│  ├─ synthetic_reviews.csv
│  └─ samples_to_predict.csv
├─ models/
│  ├─ department_classifier.joblib
│  └─ sentiment_classifier.joblib
├─ src/
│  ├─ generate_dataset.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ infer.py
│  └─ utils.py
├─ app/
│  └─ streamlit_app.py
├─ outputs/
│  ├─ confusion_matrix_department.png
│  └─ confusion_matrix_sentiment.png
├─ requirements.txt
└─ README.md

## Installazione
1. Creazione ambiente virtuale
- Per macOS/Linux:
  
    _python -m venv venv_
  
    _source venv/bin/activate_

- Per Windows:
  
    _python -m venv venv_
  
    _venv\Scripts\activate_

2. Installazione dipendenze:
   
    _pip install -r requirements.txt_

## 1. Generazione Dataset
(Il dataset può essere utilizzato anche già pronto in data/)
Per rigenerarlo:
    python3 src/generate_dataset.py

Il file risultante è:
    data/synthetic_reviews.csv
con colonne: id, title, body, department, sentiment.

## 2. Addestramento dei modelli
    python3 src/train.py

Output:
- models/department_classifier.joblib
- models/sentiment_classifier.joblib
- classification report a console

## 3. Valutazione dei modelli
    python3 src/evaluate.py

Output:
- outputs/confusion_matrix_department.png
- outputs/confusion_matrix_sentiment.png
- metriche di F1 / Accuracy stampate a console

## 4. Inferenza (CLI)
Predizione singola da riga di comando.

Prepara un CSV di input come:
id,title,body
101,Check-in rapido,Personale molto gentile e accogliente
102,Camera sporca,cattivo odore in stanza e bagno non pulito
103,Colazione ottima,ristorante eccellente ma parcheggio scomodo

salvato in data/samples_to_predict.csv

Poi:
    python3 src/infer.py data/samples_to_predict.csv

Output:
    outputs/prediction_batch.csv

Con colonne aggiunte:
- predicted_department
- predicted_sentiment
- timestamp

## 5. Interfaccia Streamlit
Avvio:
    streamlit run app/streamlit_app.py

Funzionalità:
- Predizione singolare (textarea)
- Predizione batch caricando un CSV
- Download del CSV arricchito

## Dettagli Tecnici
- Preprocessing:
    - Lowercase
    - Rimozione punteggiatura
    - Normalizzazione spazi
    - Concatenazione titolo + corpo

- Rappresentazione test:
    - TF-IDF word bigrams (ngram_range(1,2))
    - min_df=2 per ridurre il rumore
    - Pipeline sklearn integrata

- Modelli ML
    Task:                   Modello:                    Motivazione
    Department              Linear SVM (LinearSVC)      Robusto e molto efficace su testo
    Sentiment               Logistic regression         Veloce, probabilistico, ottimo su binario

## Limiti del progetto
- Dataset sintetico e quindi vocabolario limitato
- Nessuna gestione di sarcasmo/ironia
- Assegnazione a singolo reparto, di conseguenza recensioni multi-reparto non gestite
- Manca explainability (es. LIME/SHAP)

Questi limiti sono discussi anche nell'elaborato della project work.

## Possibili estensioni
- Modelli transformer leggeri (BERT italiano)
- Addestramento su dataset reali (anonimizzati)
- Multi-label (es. recensioni multi-reparto)
- Confidence score e gestione incertezza
- Dashboard più evoluta

## Licenza
Uso accademico / didattico

## Autore
Jonatan Palomba

Tema n. 5 Machine Learning per Processi Aziendali

PW 20 Smistamento recensioni hotel e analisi sentimento con Machine Learning

A.A. 2025
