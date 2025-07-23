# Heart Disease ML Pipeline

A machine learning pipeline for heart disease prediction using medallion architecture (Bronze, Silver, Gold layers) with MongoDB and FastAPI.

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
   cd heart-disease-ml-pipeline
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
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

## Medallion Architecture

- **Bronze Layer**: Raw CSV data stored in MongoDB
- **Silver Layer**: Cleaned and validated data
- **Gold Layer**: Feature-engineered, model-ready data

## Contributing

1. Follow the project structure
2. Add tests for new features
3. Update documentation
4. Ensure all health checks pass

## License

MIT License
