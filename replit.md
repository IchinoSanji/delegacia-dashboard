# Delegacia 5.0 — Painel de Risco Operacional

## Overview

Delegacia 5.0 is a police operational risk assessment dashboard built with Streamlit. The system predicts high-risk situations for police officers (involving firearms or explosives) by analyzing historical incident data. It focuses on identifying WHERE (neighborhood), WHEN (day/time), and WHY (key risk factors) dangerous situations are likely to occur, enabling proactive resource allocation and officer safety planning.

The application uses machine learning (Logistic Regression and Random Forest) to classify incidents as high-risk or low-risk, with emphasis on operational metrics like F1@20% to maximize real-world utility for law enforcement decision-making.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Stack**: Streamlit web framework for interactive dashboards

**Core Design Decisions**:
- **Single-page application** with wide layout for comprehensive data visualization
- **Interactive visualizations** using Plotly for dynamic charts and Folium for mapping
- **Real-time analysis** allowing users to explore risk patterns across temporal and spatial dimensions

**Rationale**: Streamlit was chosen for rapid prototyping and deployment of ML-powered dashboards without complex frontend development. The framework's native Python integration allows seamless connection between data processing, modeling, and visualization layers.

### Backend Architecture

**Core Components**:

1. **Data Processing Layer** (`utils/data_processor.py`)
   - Handles CSV ingestion and initial data cleaning
   - Temporal feature engineering (hour, day of week, month extraction)
   - Manages datetime conversions and validation
   - **Design pattern**: Class-based processor for encapsulation and reusability

2. **Machine Learning Layer** (`utils/ml_models.py`)
   - Implements supervised learning pipeline for risk classification
   - Uses scikit-learn's `Pipeline` and `ColumnTransformer` for reproducible preprocessing
   - Supports multiple algorithms (LogisticRegression, RandomForestClassifier)
   - **Design decision**: Separate numeric and categorical feature handling with OneHotEncoder and StandardScaler
   - **Rationale**: Ensures data leakage prevention through proper train/test splitting and transform isolation

3. **Risk Calculation Engine** (`utils/risk_calculator.py`)
   - Implements weighted risk scoring algorithm
   - Multi-factor risk assessment (weapon type: 30%, crime type: 25%, suspect profile: 20%, location history: 15%, temporal factors: 10%)
   - **Design pattern**: Rule-based scoring with configurable weights
   - **Rationale**: Provides interpretable risk scores alongside ML predictions for operational transparency

4. **Unsupervised Analysis Module** (`utils/unsupervised_analysis.py`)
   - Pattern discovery through clustering (KMeans, DBSCAN)
   - Anomaly detection using Isolation Forest
   - Dimensionality reduction with PCA for visualization
   - **Purpose**: Discovers hidden risk patterns not captured by supervised labels

5. **Visualization Layer** (`utils/visualizations.py`)
   - Risk heatmaps (time vs. day of week)
   - Geospatial mapping of incidents
   - Performance metrics visualization (confusion matrix, ROC curves)
   - **Technology**: Plotly for interactive charts, Matplotlib/Seaborn for static plots

**Key Architectural Decisions**:

- **Temporal data splitting**: Prevents data leakage by respecting chronological order
- **Fixed random_state (42)**: Ensures reproducibility across model training sessions
- **Modular utility structure**: Each component (data processing, ML, risk calculation, visualization) is independently testable and maintainable
- **Warning suppression**: Strategic use in production code to reduce noise while maintaining error handling

### Data Layer

**Primary Data Source**: CSV file (`dataset_ocorrencias_delegacia_5.csv`)

**Required Schema**:
- `data_ocorrencia`: Timestamp of incident
- `bairro`: Neighborhood/district
- `tipo_crime`: Crime type
- `arma_utilizada`: Weapon used (critical for high-risk classification)

**Optional Fields**:
- Geolocation: `latitude`, `longitude`
- Incident details: `quantidade_vitimas`, `quantidade_suspeitos`
- Suspect profile: `idade_suspeito`, `sexo_suspeito`
- Administrative: `orgao_responsavel`, `status_investigacao`

**Target Variable**: Binary classification where high-risk = incidents involving "Arma de Fogo" (firearms) or "Explosivos" (explosives)

**Data Processing Strategy**:
- DateTime parsing with error handling (`errors='coerce'`)
- Feature engineering: extraction of temporal components (hour, day, month, day of week)
- Handling missing values through imputation in preprocessing pipeline

### Model Training and Evaluation

**Evaluation Metrics Focus**:
- **F1@20%**: Primary metric for operational gain measurement
- Precision, Recall, ROC-AUC for comprehensive performance assessment
- **Rationale**: F1@20% balances precision/recall while focusing on top 20% riskiest predictions, maximizing resource allocation efficiency

**Model Selection**:
- **Baseline**: Logistic Regression for interpretability
- **Advanced**: Random Forest for capturing non-linear patterns
- **Comparison approach**: Both models evaluated on same metrics for informed selection

**Preprocessing Pipeline**:
- Categorical encoding via OneHotEncoder
- Numeric scaling via StandardScaler
- Unified through ColumnTransformer for maintainability

**Pros of Architecture**:
- Reproducible results through fixed random states
- Modular design allows easy model swapping
- Interpretable risk factors support operational decision-making

**Cons**:
- CSV-based storage limits scalability
- Single-server Streamlit deployment may face performance issues with large datasets
- Lack of automated model retraining pipeline

## External Dependencies

### Python Libraries

**Core ML/Data Stack**:
- `pandas`: Data manipulation and CSV processing
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning algorithms, preprocessing, and evaluation metrics

**Visualization**:
- `matplotlib`: Static plotting for analysis
- `plotly`: Interactive charts (implied by visualization module)
- `seaborn`: Statistical visualizations (implied by visualization module)
- `folium`: Interactive mapping for geospatial analysis
- `streamlit-folium`: Folium integration with Streamlit

**Web Framework**:
- `streamlit`: Primary web application framework

### Data Dependencies

**Input Data**: Local CSV file (`dataset_ocorrencias_delegacia_5.csv`)
- No external database currently configured
- Data must conform to predefined schema with required columns
- **Limitation**: No real-time data ingestion; relies on periodic CSV updates

### Third-Party Integrations

**Mapping Services**:
- Folium for map rendering (may use OpenStreetMap tiles by default)
- Conditional import with fallback (`HAS_FOLIUM` flag) for graceful degradation

**Future Integration Opportunities**:
- Database systems (PostgreSQL/MySQL) for scalable data storage
- Real-time incident reporting APIs
- Authentication services for multi-user access control
- Cloud deployment platforms (AWS/GCP/Azure) for production hosting

### Development and Deployment

**Version Control**: Git-based (implied by repository structure)

**Deployment Options**:
- Local execution: `streamlit run app_risco_operacional.py`
- Cloud platforms: Streamlit Cloud, Replit, or containerized deployment (Docker)

**Reproducibility Requirements**:
- `requirements.txt` for dependency management
- Fixed random seeds (RANDOM_STATE = 42) across all modules
- Deterministic train/test splitting for consistent model evaluation