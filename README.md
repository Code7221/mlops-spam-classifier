# MLOps Spam Classifier

## Project Overview
This project implements a Spam Detection Machine Learning model using TF-IDF and Logistic Regression.  
The model is exposed as a REST API using Flask and containerized using Docker.

---

## Technologies Used
- Python
- Scikit-learn
- Pandas
- Flask
- Docker
- GitHub

---

## Model Details
- Text Vectorization: TF-IDF
- Algorithm: Logistic Regression
- Dataset: SMS Spam Collection Dataset

---

## How to Run Locally

1. Install dependencies:
   pip install -r requirements.txt

2. Train model:
   python train.py

3. Run Flask app:
   python app.py

---

## Run Using Docker

Build Docker image:
docker build -t spam-classifier .

Run container:
docker run -p 5000:5000 spam-classifier

---

## API Endpoint

POST request to:
http://localhost:5000/predict

Example JSON:
{
  "message": "Free entry in 2 a wkly comp to win cash!"
}

---

## Links

GitHub Repository:
https://github.com/Code7221/mlops-spam-classifier.git

Docker Hub Image:
https://hub.docker.com/repository/docker/code7221/spam-classifier/general
