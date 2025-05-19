# Churn Prediction API

A lightweight **FastAPI** micro-service that exposes a pre-trained Random Forest model for customer-churn inference.  It ships with automated tests and a ready-to-use Swagger UI.

---

## 📋 Features

* **FastAPI** backend with automatic OpenAPI/Swagger documentation (`/docs`).
* `GET /` – welcome banner.
* `GET /health` – container-friendly health probe.
* `POST /predict` – JSON input → predicted class **and** churn probability.
* Configurable model path (defaults to `model.pkl`).
* Basic `logging` already wired.
* Pytest test-suite (`tests/test_main.py`).

---

## 🚀 Quick Start

### 1 · Clone & create a virtual env

```bash
$ git clone https://github.com/your-org/churn-prediction-api.git
$ cd churn-prediction-api
$ python -m venv .venv
$ source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2 · Install requirements

```bash
$ pip install -r requirements.txt
```

### 3 · Add the trained model

Place the `model.pkl` file generated during training at:

```
└── churn-prediction-api/
    └── model.pkl   ← here
```

…or change **`MODEL_PATH`** in `main.py` to match your location.

### 4 · Run the service

```bash
$ uvicorn main:app --reload
```

Visit **[http://localhost:8000/docs](http://localhost:8000/docs)** for an interactive Swagger UI.

---

## 🔌 API Reference

| Method | Endpoint   | Description                                                        |
| ------ | ---------- | ------------------------------------------------------------------ |
| `GET`  | `/`        | Welcome message with a hint to `/docs`.                            |
| `GET`  | `/health`  | Returns `{"status": "Healthy"}` if the service & model are loaded. |
| `POST` | `/predict` | Accepts customer attributes and returns churn class & probability. |

### `POST /predict`

#### Request Body

```json
{
  "CreditScore": 600,
  "Geography": "France",
  "Gender": "Female",
  "Age": 40,
  "Tenure": 5,
  "Balance": 50000.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 70000.0
}
```

#### Successful Response `200 OK`

```json
{
  "prediction": 0,
  "probability_of_churn": 0.1245
}
```

*`prediction`* is **0** → won’t churn, **1** → will churn.
*`probability_of_churn`* is the Random Forest positive-class probability, rounded to 4 dp.

#### Error Responses

| Code                        | When                                         |
| --------------------------- | -------------------------------------------- |
| `422 Unprocessable Entity`  | Validation failure (e.g., wrong data types). |
| `500 Internal Server Error` | Unexpected errors (model missing, etc.).     |

---

## 🧪 Running Tests

The repo includes a minimal pytest suite:

```bash
$ pytest -q
```

All three endpoints are exercised; the invalid-payload case must return 422.

---

## 🗂️ Project Layout

```
.
├── main.py            # FastAPI application
├── tests/
│   └── test_main.py   # Endpoint unit tests
├── requirements.txt   # Pinned dependencies
└── README.md          # You are here
```

---

## ⚙️ Configuration

* **Python ≥ 3.10** recommended.
* Adjust log verbosity via the `LOGLEVEL` environment variable if desired.
* For containerisation, copy the repo & `model.pkl` and expose port `8000`.

---

## 📑 License

Specify your license here (e.g., MIT, Apache-2.0).

---

## 🙋‍♂️ Contributing

PRs are welcome!  Please run `black`, `isort`, and the test-suite before submitting.

---

> *Happy predicting!*
