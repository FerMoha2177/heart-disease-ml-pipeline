# Heart Disease ML Pipeline

A machine learning pipeline for heart disease prediction using medallion architecture (Bronze, Silver, Gold layers) with MongoDB and FastAPI.

## Prerequisites

- Python >= 3.10.12
- MongoDB
- FastAPI
- Motor
- Pydantic
- Scikit-learn
- XGBoost
- Joblib
- PyMongo
- Python-dotenv
- Matplotlib
- Seaborn
- Jupyter
- Loguru
- Pytest
- Httpx
- IPython

## Project Structure

```
heart-disease-ml-pipeline/
├── api/                    # FastAPI application
├── notebooks/              # Jupyter notebooks for ML pipeline
├── models/                 # Saved ML models
├── data/                   # Dataset storage
├── src/                    # Shared utilities
├── tests/                  # Test files
├── docs/                   # Documentation
├── deployment/             # Deployment configuration
└── scripts/                # Utility scripts
```

## Quick Start

1. **Clone and setup environment**
   ```bash
      git clone https://github.com/FerMoha2177/heart-disease-ml-pipeline.git
      cd heart-disease-ml-pipeline
      python -m venv venv
      source venv/bin/activate  # Windows: venv\Scripts\activate
      pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB connection string
   ```

3. **Run the API**
   ```bash
   cd api
   python main.py
   ```

4. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/api/v1/health

## Development Workflow

1. **Data Pipeline**: Use notebooks in order (01-05)
2. **API Development**: Implement endpoints in `api/routes/`
3. **Testing**: Run tests with `pytest tests/`
4. **Deployment**: Deploy to Render using deployment configs

## API Endpoints

- `GET /api/v1/health` - Basic health check
- `GET /api/v1/db-health` - Database health check
- `POST /api/v1/predict` - Heart disease prediction

```python
# User sends this JSON to /predict
{
    "age": 55,
    "sex": 1,
    "cp": 0,
    "trestbps": 140,
    "chol": 200,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

# API returns:
{
    "prediction": 1,
    "probability": 0.75,
    "confidence": "high",
    "model_version": "1.0.0"
}
```


# Medallion Architecture

## **Bronze Layer**: Raw CSV data stored in MongoDB

   - Purpose: Raw CSV data storage
   - Processing: No preprocessing - direct JSON storage of CSV records
   - Collection: healthcare.heart_disease_bronze
   - Records: 19,320 raw patient records


## **Silver Layer**: Cleaned and validated data

   - Purpose: Cleaned and validated data
   - Processing:
      - MNAR justifications for dropping:
         - ca: number of major vessels colored by fluoroscopy missing 66%, 
         - thal: thalassemia missing 52%
         - Ran Chi-square test for missingness correlation with target variable
      - Missing value imputation (median for numeric, mode for categorical)
      - Impossible zero imputation (trestbps, chol, thalach)
      - Data type validation
   - Collection: healthcare.heart_disease_silver
   - Records: 19,320 cleaned patient records

## **Gold Layer**: Feature-engineered, model-ready data

   - Purpose: Feature-engineered, model-ready data
   - Processing:
      - Binary label encoding (sex, fbs, exang)
      - One-hot encoding (cp, restecg, slope)
      - Min-max scaling (all numerical features)
      - Feature ordering
      - Feature selection
      - Convergent feature selection (16 final features)
   - Collection: healthcare.heart_disease_gold
   - Records: 19,320 cleaned patient records
## End-to-End Pipeline

```CSV Data → Bronze → Silver → Gold → Model Training → .pkl file → API → Predictions```

# Model Development & Justification

## Models Evaluated
Trained and compared 5 classification models using 80/20 train-test split:

| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| RandomForest       | 0.7989   | 0.8037    | 0.8431 | 0.8230   | 0.9078  |
| SVM                | 0.8152   | 0.8148    | 0.8627 | 0.8381   | 0.8916  |
| LogisticRegression | 0.8043   | 0.8056    | 0.8529 | 0.8286   | 0.8969  |
| XGBoost            | 0.8098   | 0.8190    | 0.8431 | 0.8309   | 0.8736  |
| DecisionTree       | 0.7772   | 0.8081    | 0.7843 | 0.7960   | 0.8117  |

### Final Model Choice: Random Forest
**Justification:** Selected Random Forest based on highest ROC-AUC (0.9078), which is critical for medical screening applications where we need to distinguish between positive and negative cases across all probability thresholds.

#### Key Advantages:
- Best ROC-AUC performance (90.78%)
- High recall (84.31%) - important for catching true heart disease cases
- Robust to overfitting
- Handles feature interactions well
- Provides feature importance scores

## Feature Selection
- **Method:** Convergent selection using SelectKBest + RandomForest feature importance
- **Original features:** 19 (after one-hot encoding)
- **Final features:** 16 (removed 3 least important)
- **Protected features:** age, sex (always included for medical relevance)

# API Documentation

## Base URL
- Local: http://localhost:8000
- Deployed: https://your-app.onrender.com (Update with actual URL)

## Endpoints

### 1. Health Check
- **GET /api/v1/health**
- **Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-25T20:00:00.000000",
  "message": "Heart Disease Prediction API is running"
}
```

### 2. Heart Disease Prediction
- **POST /api/v1/predict**
- **Content-Type:** application/json
- **Request Body:**
```json
{
  "age": 55,
  "sex": 1,
  "cp": 3,
  "trestbps": 130,
  "chol": 250,
  "fbs": 0,
  "restecg": 0,
  "thalach": 140,
  "exang": 1,
  "oldpeak": 1.5,
  "slope": 1,
  "ca": 1,
  "thal": 0
}
```
- **Response:**
```json
{
  "prediction": 1,
  "probability": 0.7287551044386565,
  "confidence": "medium",
  "risk_factors": [
    "advanced_age",
    "high_cholesterol", 
    "high_blood_pressure",
    "male_gender",
    "exercise_induced_angina",
    "asymptomatic_chest_pain"
  ],
  "timestamp": "2025-07-25T20:00:00.000000",
  "model_version": "1.0.0"
}
```

## Feature Descriptions
- **age:** Patient age (years)
- **sex:** Gender (1=male, 0=female)
- **cp:** Chest pain type (0=typical angina, 1=atypical angina, 2=non-anginal, 3=asymptomatic)
- **trestbps:** Resting blood pressure (mm Hg)
- **chol:** Serum cholesterol (mg/dl)
- **fbs:** Fasting blood sugar > 120 mg/dl (1=true, 0=false)
- **restecg:** Resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)
- **thalach:** Maximum heart rate achieved
- **exang:** Exercise induced angina (1=yes, 0=no)
- **oldpeak:** ST depression induced by exercise
- **slope:** Slope of peak exercise ST segment (0=upsloping, 1=flat, 2=downsloping)
- **ca:** Number of major vessels colored by fluoroscopy (0-3)
- **thal:** Thalassemia (0=normal, 1=fixed defect, 2=reversible defect)

# Testing with Postman

## Setup Instructions
1. Open Postman
2. Create new request
3. Set method to POST for /predict or GET for /health
4. Set URL to your API endpoint
5. For /predict:
   - Go to Body tab
   - Select "raw" and "JSON"
   - Paste example request body
6. Send request

## Test Cases
**Low Risk Patient:**
```json
{
  "age": 25, "sex": 0, "cp": 0, "trestbps": 110, "chol": 180,
  "fbs": 0, "restecg": 0, "thalach": 190, "exang": 0,
  "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 0
}
```
**High Risk Patient:**
```json
{
  "age": 70, "sex": 1, "cp": 3, "trestbps": 180, "chol": 350,
  "fbs": 1, "restecg": 1, "thalach": 100, "exang": 1,
  "oldpeak": 4.0, "slope": 2, "ca": 3, "thal": 2
}
```

# Project Structure
```
heart-disease-ml-pipeline/
├── api/                    # FastAPI application
│   ├── routes/            # API endpoints
│   ├── services/          # Business logic
│   └── models/            # Pydantic schemas
├── notebooks/             # Jupyter notebooks (01-05)
├── models/                # Saved ML models and artifacts
├── src/                   # Shared utilities
├── screenshots/           # MongoDB and Postman screenshots
└── requirements.txt       # Dependencies
```

# Deployment
Deployed on Render at: [Update with actual URL]

# Future Improvements
- Enhanced preprocessing for categorical feature encoding
- Model ensembling for improved accuracy
- Real-time model monitoring and drift detection
- Extended feature validation and error handling

# Technical Notes
- **Framework:** FastAPI with uvicorn
- **Database:** MongoDB Atlas
- **ML Framework:** scikit-learn
- **Model Format:** joblib (.pkl)
- **Python Version:** 3.10+

## Contributing

1. Follow the project structure
2. Add tests for new features
3. Update documentation
4. Ensure all health checks pass

## License

MIT License
