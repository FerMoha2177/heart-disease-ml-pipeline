from setuptools import setup, find_packages

setup(
    name="heart-disease-ml-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "pydantic==2.5.0",
        "python-multipart==0.0.6",
        "scikit-learn==1.3.2",
        "xgboost==2.0.1",
        "pandas==2.1.4",
        "numpy==1.24.3",
        "joblib==1.3.2",
        "pymongo==4.6.0",
        "motor==3.3.2",
        "python-dotenv==1.0.0",
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
        "pytest==7.4.3",
        "pytest-asyncio==0.21.1",
        "httpx==0.25.2",
        "jupyter==1.0.0",
        "ipykernel==6.27.1",
        "loguru==0.7.2",
    ],
    entry_points={
        "console_scripts": [
            "heart-disease-predictor=api.main:main",
        ],
    },
)