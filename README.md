🚀 Loan Approval Machine Learning API

Live API Endpoint:
🔗 https://loan-approval-ml-pipeline.onrender.com

A fully containerized, production-ready Machine Learning (ML) pipeline that predicts loan approval status based on applicant financial and demographic data. This project demonstrates an end-to-end ML Ops workflow, integrating model training, evaluation, deployment, and cloud hosting with automated CI/CD.

📌 Project Overview

This API uses a Random Forest model trained on a real-world styled loan dataset to predict whether a loan application should be Approved or Rejected. It includes a full ML lifecycle:

Data preprocessing

Feature engineering

Model training & evaluation

Artifact versioning

FastAPI deployment with Docker

CI/CD pipeline & cloud hosting on Render

📊 Architecture Diagram
flowchart TD
    A[Raw Loan Data (CSV)] --> B[Preprocessing Pipeline<br/>(Imputation, Encoding, Scaling)]
    B --> C[Random Forest Model Training]
    C --> D[Artifacts Saved<br/>model_v1.pkl & preprocessor_v1.pkl]
    D --> E[FastAPI Prediction Endpoint /api/predict]
    E --> F[Docker Container]
    F --> G[CI/CD via GitHub Actions]
    G --> H[Render Cloud Deployment]
    H --> I[Public API Access]

🔥 Key Features

✅ Automated ML Preprocessing (handles numeric & categorical features)

✅ Model Training Pipeline with Versioned Artifacts

✅ Random Forest Classification Model

✅ FastAPI-based Prediction Endpoint

✅ Dockerized for Portability

✅ CI/CD Integration via GitHub Actions

✅ Live Deployment on Render Cloud

✅ Swagger UI for easy testing

🧠 Tech Stack
Component	Technology
Model Training	Scikit-learn, Pandas
API Framework	FastAPI
Packaging	Docker
CI/CD	GitHub Actions
Cloud Hosting	Render
Language	Python 3.13


📂 Project Structure
ml-pipeline-deployment/
├── app/
│   ├── api/
│   │   └── predict_route.py
│   ├── core/
│   │   └── logging.py
│   ├── pipeline/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── preprocessing.py
│   └── main.py
├── artifacts/
│   ├── model_v1.pkl
│   └── preprocessor_v1.pkl
├── Dockerfile
├── requirements.txt
└── README.md

⚙️ How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/Ptheju/ml-pipeline-deployment.git
cd ml-pipeline-deployment

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run FastAPI App
uvicorn app.main:app --reload


Visit: http://localhost:8000/docs

🐳 Run with Docker
docker build -t loan-ml-api .
docker run -p 8000:8000 loan-ml-api

📨 Example Prediction Request
Endpoint: POST /api/predict

Input JSON:

{
  "Income": 65000,
  "Credit_Score": 720,
  "Loan_Amount": 200000,
  "DTI_Ratio": 0.28,
  "Employment_Status": "Employed"
}


Response:

{
  "prediction": "Approved"
}

📈 Model Performance (v1)
Metric	Score
Accuracy	0.9981
Precision	0.9920
Recall	0.9960
F1-Score	0.9940
ROC-AUC	0.9999

✅ Excellent performance, ready for production deployment.

🚀 CI/CD Pipeline

GitHub Actions automatically builds Docker images and pushes to Docker Hub

Render auto-deploys from the Dockerfile on each push to main

🌐 Deployment

Hosted on Render using Docker

Exposed on port 8000

Public API URL is always available and live

🔗 Production URL:
https://loan-approval-ml-pipeline.onrender.com

🧭 Future Enhancements

✅ Add text-based NLP features from customer requests

✅ Introduce model version comparison dashboard

🔄 Support retraining pipeline with new data

📊 Add monitoring and logging dashboards

👨‍🔬 Author

Praveen Kumar V
GitHub: Ptheju
