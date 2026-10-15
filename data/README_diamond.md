# Diamonds Price Prediction

Machine learning regression project for predicting diamond prices from their physical and quality characteristics.

## Model

Three regression models were compared:

* CatBoostRegressor
* RandomForestRegressor
* XGBRegressor

**Best model:** CatBoostRegressor

Final metrics on the test set:

* **MAE:** 269.64
* **RMSE:** 520.62
* **R²:** 0.983

Features:

* `carat`
* `cut`
* `color`
* `clarity`
* `depth`
* `table`
* `x`
* `y`
* `z`

Categorical features were handled directly by CatBoost.

SHAP was used to analyze individual predictions and feature contributions.

## API

The trained CatBoost model is served through **FastAPI**.

### Prediction endpoint

`POST /predict`

Example response:

```json
{
  "predicted_price": 2765.53
}
```

Interactive API documentation is available at `/docs`.

## Docker

The API is packaged as a Docker image.

Docker Hub image:

`summersun11/diamonds-api:latest`

## CI/CD

GitHub Actions automatically:

1. Builds the Docker image after a push to `master`.
2. Pushes the image to Docker Hub.
3. Triggers a Render deployment through a deploy hook.

## Deployment

The API is deployed as a Docker container on Render.

Deployment flow:

`GitHub → GitHub Actions → Docker Hub → Render → FastAPI`

## Screenshots

### Model results

![SHAP results](screenshots/Figure_5.png)

### Prediction example

![Prediction example](screenshots/Figure_6.png)

### Deployed API

![Deployed API](screenshots/response.png)

## Tools
* Pandas  
* CatBoost, RandomForest, XGBoost  
* Scikit-Learn  
* Numpy  
* Matplotlib  
* SHAP
* Docker
* GitHub Actions
* Render Cloud
* FastAPI

## Dataset Source
The dataset comes with Seaborn library.

## Author Erick