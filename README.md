# NBA Player Points Prediction & Betting Line Analysis

End-to-end NBA analytics system that:
- trains/loads a scikit-learn regression model to predict player point totals
- stores games + predictions + results in a Django/PostgreSQL backend
- optionally compares model predictions to sportsbook player props (via Odds API)

---

## Tech Stack
- Python, scikit-learn (model training/inference)
- Django (API + persistence)
- PostgreSQL (relational storage)
- nba_api (players and league data)
- Odds API (sportsbook lines / props)

---

## Prerequisites
- Python 3.10+ (3.11 recommended)
- PostgreSQL 14+ (local install or Docker)
- Git

---

## Project Structure
```   
nba-player-points-prediction/
├── core/       # Django models and management commands
├── src/        # Feature engineering and prediction scripts
├── models/     # Saved ML model artifacts
├── data/       # Raw and processed CSV datasets
├── manage.py
├── requirements.txt
├── .env.example
├── README.md
└── .gitignore
```
---

## 1) Clone the repo
```bash
git clone https://github.com/collin0613/nba-player-points-prediction
cd nba-player-points-prediction
```

## 2) Create and Activate a Virtual Environment
macOS / Linux / WSL
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3) Install Dependencies
```bash
pip install -r requirements.txt
```

## 4) Obtain Required Keys
Odds API Key: Create an account with the Odds API provider used in this project and generate an API key.

Generate a Django secret key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(50))"
```

## 5) Configure Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```

Populate .env with real values. Structure is as follows:
```
ODDS_API_KEY=ODDS_API_KEY
DJANGO_SECRET_KEY=DJANGO_SECRET_KEY
DB_NAME=DB_NAME
DB_USER=DB_USER
DB_PASSWORD=DB_PASSWORD
DB_HOST=DB_HOST
DB_PORT=DB_PORT
```

## 6) Create the PostgreSQL Database
Using psql:
```bash
psql -U postgres
```
```sql
CREATE USER nba_user WITH PASSWORD 'your_password';
CREATE DATABASE nba_db OWNER nba_user;
GRANT ALL PRIVILEGES ON DATABASE nba_db TO nba_user;
\q
```
Ensure your .env database configuration matches the database you created.

## 7) Initialize the Database Schema
Run:
```bash
python manage.py makemigrations
python manage.py migrate
```

Create an admin user:
```bash
python manage.py createsuperuser
```

## 8) Start the Django Development Server
```bash
python manage.py runserver
```

The server will be available at: 
```
http://127.0.0.1:8000/
```
Admin panel: 
```
http://127.0.0.1:8000/admin
```

## 9) Run Model Predictions
A pretrained regression model is included at: models/ridge_points_model_2024-25.joblib

Run the prediction script:
```bash
python src/predict.py
```

This loads the trained model pipeline and generates player point predictions using recent performance and matchup features.
It also backfills actual_points scored in past predicted matchups with null values.

## 10) Compare Predictions to Sportsbook Betting Lines
With predictions exported to a CSV file in the root, compare model outputs to sportsbook props using the Django management command:
```bash
python manage.py compare_predictions_to_props \
  --csv predictions_next_games_2025-26.csv \
  --bookmakers DraftKings
```

Flags:
- --csv: path to the predictions CSV file
- --bookmakers: sportsbook name to compare against