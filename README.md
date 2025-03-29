# CrisisWatch AI

## Overview
This project aims to build an AI-driven crisis monitoring dashboard that detects high-risk mental health signals from social media (Reddit) using NLP (DistilBERT, VADER) and geospatial analytics (Folium, GeoPy). 

The system will:

•	Classify posts into risk levels (High, Moderate, Low) using a fine-tuned DistilBERT model.

•	Map high-risk clusters in real-time to guide public health interventions.

•	Track at-risk users based on behavior patterns (post frequency, sentiment trends).

•	Provide an interactive dashboard (Dash/Plotly) for crisis response teams.

## Features
- Extracts posts from Reddit using the Reddit API
- Classifies posts into risk categories (High, Moderate, Low) using an AI model
- Stores data in a database
- Provides a geospatial heatmap for crisis trends
- Interactive dashboard for analysing
   - Risk Analysis
   - Sentiment Analysis
   - Geographic Analysis
   - User Behaviour Analysis
   - Engagement Analysis

## Prerequisites
1. **Create a Reddit API Key**:
   - Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
   - Click **Create an App**
   - Select **script** as the app type
   - Note down the `client_id` and `client_secret`

2. **Set Up Environment Variables**:
   - Change the `.env` file in the project directory.
   - Add the following content:
     ```ini
     CLIENT_ID=your_reddit_client_id
     CLIENT_SECRET=your_reddit_client_secret
     USER_AGENT=your_app_name
     USER_NAME=your_username
     PASSWORD=your_password
     ```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/gokulan006/HumanAI-Crisis-Tracker.git
   cd HumanAI-Crisis-Tracker
   ```
2. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -m nltk.downloader stopwords
   
   ```

## Usage
1. **Run the main application**:
   ```sh
   python app.py
   ```
2. **View CrisisWatch AI**:
   - Open `http://localhost:5000/` in a browser.
   - Press `View dashboard` in the site.

## File Structure
```
.
├── app.py                    # Main application file
├── dash_app.py               # Dashboard application
├── reddit_posts.db           # SQLite database for storing posts
├── reddit_posts.csv          # Raw extracted posts
├── reddit_extracted_posts.csv # Processed posts
├── requirements.txt          # Dependencies
├── templates/                # HTML templates for web application
├── risk_model_package/       # AI model package
└── .env                      # Environment variables (change to your reddit API)
```


## License
This project is licensed under the MIT License.

