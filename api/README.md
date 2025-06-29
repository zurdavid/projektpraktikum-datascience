# Fraud-Detection REST API

## Install and run with Docker

To run the REST API with Docker, you can use the provided Dockerfile. This will build the image and run the container with the necessary configurations.

### Build the Docker Image

```bash
docker build -t fraud-detection-api .
```

### Run the Docker Container

```bash
docker run -d --name fraud-detection-api -p 8080:8080 fraud-detection-api
```

## Configuration

The REST API is configurable via the `config.toml` file in the root directory. The configuration options include:

Necesary settings:

- paths to the trained models and encoders
```toml
[model]
classifier_path = "models/xgb_fraud_classifier.joblib"
regressor_path = "models/xgb_damage_regressor.joblib"
encoder_path =  "models/encoder.json"
```

- values for the cost function
```toml
[costfunction]
cost_false_positive = 10.0
gain_true_positive = 5.0
```

- Wether discounts on certain categories should be handlet as fraud in every case.
```toml
[discounts]
enable_excluded_categories = fals
```

- If `enable_excluded_categories` is set to `true`, the categories, which are not eligible for discounts, can be defined per store as follows:

```toml
[[stores]]
id = "3fffea06-686f-42bd-8362-818af86b48a9"
categories_excluded_from_discount = [
  "PERSONAL_CARE",
  "LONG_SHELF_LIFE",
  "FROZEN_GOODS",
  "BEVERAGES",
  "ALCOHOL",
  "SNACKS",
  "HOUSEHOLD",
  "TOBACCO"
]

[[stores]]
id = "46e6da32-f4b0-40f3-ada7-fc6ca81ed85d"
categories_excluded_from_discount = [
  "PERSONAL_CARE",
  "LONG_SHELF_LIFE",
  "FROZEN_GOODS",
  "BEVERAGES",
  "ALCOHOL",
  "SNACKS",
  "HOUSEHOLD",
  "TOBACCO"
]
```


## Installation for Development

Ideally, the package manager [uv](https://docs.astral.sh/uv) is used to install the dependencies. Then it is sufficient to run the following command:

```bash
uv sync
```

It is also possible to install the dependencies manually. For this, use the `requirements.txt` file in the root folder. The app was developed using python version 3.13 and was not tested with other versions.

```bash
pip install -r requirements.txt
```

## Tests

Run the tests with:

```bash
python -m pytest
```

### Run server

The app can be run in development-mode as follows.

```bash
uvicorn --host 0.0.0.0 --port 8080 --reload app.main:app --log-level debug
```

## Test API calls

In the `scripts` folder, there are some scripts to test the API calls.

- `skeleton.py`
- `make_api_call.py` should be run from root folder
