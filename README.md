# Suicide Analysis Project

## Overview
This project is an AI-powered behavioral analysis tool designed to detect and analyze mental health crises such as suicide risk, substance use, and depression through Reddit discussions. The project includes a real-time dashboard with geospatial crisis trend analysis.

## Features
- Extracts posts from Reddit using the Reddit API
- Classifies posts into risk categories (High, Moderate, Low) using an AI model
- Stores data in a database
- Provides a geospatial heatmap for crisis trends
- Interactive dashboard for visualization

## Prerequisites
1. **Create a Reddit API Key**:
   - Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
   - Click **Create an App**
   - Select **script** as the app type
   - Note down the `client_id` and `client_secret`

2. **Set Up Environment Variables**:
   - Create a `.env` file in the project directory.
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
   git clone https://github.com/yourusername/suicide_analysis.git
   cd suicide_analysis
   ```
2. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. **Run the main application**:
   ```sh
   python app.py
   ```
2. **Run the Dashboard** (optional):
   ```sh
   python dash_app.py
   ```
3. **View Geospatial Heatmap**:
   - Open `geospatial_heatmap.html` in a browser.

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
└── .env                      # Environment variables (not included in repo)
```

## Contributing
Feel free to submit issues or pull requests to enhance this project.

## License
This project is licensed under the MIT License.

