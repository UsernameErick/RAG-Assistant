# MLOps Iris Classification

A practical MLOps project covering the main stages of the ML model lifecycle: experiment tracking, model registry, versioning, model promotion, and API serving.

## Tech Stack

* Python
* Scikit-learn
* MLflow 3.15.1
* FastAPI
* Pandas
* Docker
* Uvicorn

## Project Overview

The project uses the Iris dataset to train and compare different ML models while tracking experiments with MLflow.

The main model is **Logistic Regression**. Different `C` values were tested using 5-fold cross-validation.

Best result:

```text
C = 10
max_iter = 200
Accuracy = 1.0000
CV Accuracy = 0.9667
```

Random Forest was also tested, but did not outperform Logistic Regression in the performed experiments.

## MLflow

MLflow is used for:

* experiment tracking;
* parameter and metric logging;
* model artifacts;
* Model Registry;
* model versioning;
* model promotion.

The registered model is:

```text
iris-classifier
```

The application uses the `champion` alias instead of a fixed model version:

```python
models:/iris-classifier@champion
```

### Automatic Promotion

`register_model.py` registers a selected model and compares its `cv_accuracy` with the current champion.

If the new model performs better, it becomes the new `champion`. Otherwise, the existing champion remains unchanged.

This allows the API to always use the currently approved model without changing its code.

## FastAPI

FastAPI provides a simple REST API for predictions.

Main endpoint:

```text
POST /predict
```

The model is loaded directly from MLflow Model Registry using the `champion` alias.

## Docker

The FastAPI application was also containerized with Docker.

The container was successfully configured to communicate with the MLflow Server. However, loading the existing model artifact from Docker was limited by the current MLflow setup: the experiment stores artifacts on the local Windows filesystem (`file:///C:/.../mlruns/1`), which is not directly accessible from the Linux container.

For a production setup, model artifacts should be stored in a container-accessible object storage such as S3 or MinIO.

## Project Structure

```text
MLOps_experiment_tracking/
├── src/
│   ├── train.py
│   ├── register_model.py
│   ├── promote_model.py
│   └── main.py
├── Dockerfile
├── requirements.txt
└── README.md
```

`promote_model.py` is an earlier version of the promotion logic and is kept for reference. The current logic is implemented in `register_model.py`.
All results are saved locally (mlruns, mlflow.db) and not pushed to Github.

## Screenshots
![FastAPI Requests Demonstration](screenshots/fastapi_request.png)
![MLFlow Runs Comparison](screenshots/mlflow_comparison.png)

## What I Learned

* MLflow Experiment Tracking
* Cross-validation
* Model Registry and versioning
* Model aliases
* Automatic model promotion
* FastAPI model serving
* Docker containerization
* Basic MLflow deployment architecture

## Conclusion

This project demonstrates a complete practical workflow from ML experimentation to model registration, promotion, and API serving, while also exploring the challenges of running MLflow-based applications in Docker.

## Author Erick