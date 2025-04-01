from flask import Flask, jsonify, request, render_template
import praw
import pandas as pd
import re
import os
from nltk.corpus import stopwords
import nltk
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from geopy.geocoders import Nominatim
from flask_sqlalchemy import SQLAlchemy
import folium
from folium.plugins import HeatMap
from sqlalchemy import create_engine
from dash_app import create_dashboard
from dotenv import load_dotenv
from transformers import DistilBertTokenizerFast, TFDistilBertModel
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
 
import warnings
warnings.filterwarnings('ignore')

# Load NLP model for location extraction

nlp = spacy.load("en_core_web_sm")

# Download stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Initialize geocoder
geolocator = Nominatim(user_agent="crisis_mapping")


 
#  Initialize the app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'reddit_posts.db') + '?check_same_thread=False'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Risk labels
risk_labels = ['High Risk', 'Low Risk', 'Moderate Risk']

# Loading the finetuned distilbert model
def load_model_components():
    """Load only tokenizer and model"""
    base_dir = Path("risk_model_package")
    
    # 1. Load tokenizer (with fallback)
    tokenizer_dir = base_dir / "tokenizer"
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            str(tokenizer_dir),
            local_files_only=True
        )
    except Exception:
        # Fallback to direct file loading
        tokenizer = DistilBertTokenizerFast(
            vocab_file=str(tokenizer_dir/"vocab.txt"),
            tokenizer_config_file=str(tokenizer_dir/"tokenizer_config.json")
        )
    
    # 2. Load TensorFlow model
    model = tf.keras.models.load_model(
        str(base_dir/"model"),
        custom_objects={
            "TFDistilBertModel": TFDistilBertModel,
            "Attention": tf.keras.layers.Attention
        }
    )
    
    return tokenizer, model

# Initialize components
tokenizer, model = load_model_components()

# Define the RedditPost model
class RedditPost(db.Model):
    __tablename__ = "reddit"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    subreddit = db.Column(db.String(100))
    post_id = db.Column(db.String(50), unique=True, nullable=False)
    username= db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime)
    title = db.Column(db.Text)
    content = db.Column(db.Text)
    upvotes = db.Column(db.Integer)
    comments = db.Column(db.Integer)
    url = db.Column(db.Text)
    sentiment = db.Column(db.String(50))
    risk_level = db.Column(db.String(50))

class HighRiskLocation(db.Model):
    __tablename__ = "high_risk_countries"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), nullable=False)
    risk_level = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.Float(), nullable=False)
    longitude = db.Column(db.Float(), nullable=False)
class UserBehaviorTrend(db.Model):
    __tablename__ = "user_behaviors"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    postid = db.Column(db.String(50), unique=True)
    username = db.Column(db.String(100), nullable=False)
    post_count = db.Column(db.Integer, default=0)
    high_risk_count = db.Column(db.Integer, default=0)
    sentiment_trend = db.Column(db.Float, default=0.0)   
    last_post_timestamp = db.Column(db.DateTime, nullable=False)
    
class UserBehaviorHistory(db.Model):
    __tablename__ = "user_behaviors_history"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), nullable=False)
    post_count = db.Column(db.Integer, default=0)
    high_risk_count = db.Column(db.Integer, default=0)
    sentiment_trend = db.Column(db.Float, default=0.0)
    timestamp = db.Column(db.DateTime, nullable=False)

# Create the database and table
with app.app_context():
    db.create_all()

dash_app = create_dashboard(app, db)


load_dotenv() 

# Set up Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    check_for_async=False,
    verify=False
)

# Load Sentiment Analyzer
analyser = SentimentIntensityAnalyzer()



risk_labels=['High Risk','Low Risk','Moderate Risk']
# Define subreddits and keywords
subreddits = [
    # Mental Health
    "mentalhealth",          
    "depression_help",

    # Suicidal
    "offmychest",
    "suicidewatch",  

    # Substance Use    
    "Drugs",         
    "addiction" 
       
]
 
crisis_lexicon = {
    "mental_health": {
        "explicit": [
            "depression", "anxiety", "hopeless", "overwhelmed", "panic attack", "social anxiety", "PTSD",
            "flashbacks", "nightmares", "dissociation", "intrusive thoughts", "trauma", "worthless", "numb",
            "empty", "self-hate", "isolated", "exhausted", "insomnia", "brain fog", "mood swings", "crying",
            "irritable", "emotional pain", "stress", "burnout", "mental breakdown", "disconnected", "lonely",
            "fear", "can’t breathe", "racing thoughts", "uncontrollable thoughts", "guilt", "shame",
            "unmotivated", "feeling lost", "no energy", "feeling stuck", "lack of focus", "can’t function",
            "losing interest", "fatigue", "restless", "self-doubt", "low self-esteem", "can’t concentrate",
            "overthinking", "avoiding people", "heart racing", "bipolar disorder", "manic episode", "depressive episode",
            "psychosis", "schizophrenia", "dissociative identity disorder", "borderline personality disorder", "OCD",
            "eating disorder", "self-harm", "cutting", "burning", "scratching", "hair pulling", "skin picking"
        ],
        "coded": [
            "low spoons", "heavy chest", "stuck in my head", "faking smiles", "pretending to be okay",
            "can’t mask anymore", "floating away", "trapped in my mind", "checked out", "shutting down",
            "mind is dark", "crawling skin", "no escape", "lost in a loop", "echo chamber", "NPC",
            "ghost mode", "just background noise", "permanent headache", "seeing static", "grey days",
            "can’t wake up", "everything is too much", "drowning in thoughts", "the void", "static noise in my head",
            "too tired to exist", "mind racing nonstop", "watching from the outside", "just surviving",
            "in my bubble", "white noise brain", "time slipping away", "everything feels fake", "no feeling left",
            "pretending to care", "nobody notices", "losing myself", "shadow of myself"
        ]
    },
    "suicide": {
        "explicit": [
            "suicide", "end my life", "kill myself", "no reason to live", "want to die", "take my life",
            "self-harm", "overdose", "jump off", "hang myself", "poison myself", "gun to my head",
            "rope", "final decision", "permanent solution", "giving up", "goodbye message", "can’t keep going",
            "done fighting", "no more pain", "too much to handle", "empty inside", "tired of everything",
            "can’t escape", "no future", "last chance", "no point anymore", "nobody would care", "the pain won",
            "losing control", "final thoughts", "goodbye world", "writing a note", "checking out",
            "sleep forever", "want to disappear", "can’t go on", "ready to leave", "the end is near",
            "don’t wake up", "fading away", "unbearable pain", "crushed inside", "nothing left to give",
            "nothing left to lose", "life is pointless", "done with everything", "no second chances",
            "only one way out"
        ],
        "coded": [
            "KMS", "unalive", "permanent nap", "going to sleep forever", "catching the bus", "rope game",
            "taking the exit", "no respawn", "checking out early", "logging off for good", "closing my book",
            "gone fishing", "see you on the other side", "one way trip", "empty chair soon", "no more reruns",
            "lights out", "fading signal", "CTRL + ALT + DELETE", "end scene", "GG", "long ride", "no tomorrow",
            "last sunrise", "countdown started", "signing off", "done pretending", "no more next time",
            "too tired to restart", "can’t respawn this time", "silent mode forever", "taking my final bow",
            "no more chapters left", "final logout", "going ghost", "shutting down for good",
            "see you in another life", "just a memory soon", "yeet", "CTB", "SH", "final yeet", "taking the L",
            "logging off", "permanent sleep", "no respawn", "final exit", "see you on the other side", "one way trip",
            "lights out", "fading signal", "CTRL + ALT + DELETE", "end scene", "GG", "long ride", "no tomorrow",
            "last sunrise", "countdown started", "signing off", "done pretending", "no more next time",
            "too tired to restart", "can’t respawn this time", "silent mode forever", "taking my final bow",
            "no more chapters left", "final logout", "going ghost", "shutting down for good",
            "see you in another life", "just a memory soon"
        ]
    },
    "substance_use": {
        "explicit": [
            "alcoholic", "drunk", "high", "cocaine", "heroin", "meth", "pills", "painkillers", "Xanax",
            "fentanyl", "addiction", "rehab", "withdrawal", "overdose", "OD", "drugged out", "binge drinking",
            "blackout", "opiates", "stimulants", "narcotics", "prescription abuse", "substance abuse",
            "dealer", "relapse", "detox", "cold turkey", "mixing drugs", "taking too much", "need another hit",
            "pushing limits", "can’t quit", "always craving", "hooked", "strung out", "need a fix",
            "chasing the high", "spiraling", "downward spiral", "numb the pain", "out of control",
            "can’t stop", "slurring speech", "losing grip", "body shutting down", "no more control",
            "shaky hands", "nightly drinking", "losing memory", "fentanyl", "methamphetamine", "crack cocaine",
            "LSD", "ecstasy", "ketamine", "mushrooms", "spice", "synthetic opioids", "bath salts", "GHB",
            "roofies", "date rape drugs", "inhalants", "nitrous oxide"
        ],
        "coded": [
            "snow", "smack", "molly", "lean", "benzos", "bars", "percs", "dope", "zaza", "nod", "candy flipping",
            "skittles", "chasing dragons", "plug", "gassed", "geeked", "wired", "faded", "blow", "black tar",
            "speedball", "cloud 9", "laced", "ripped", "popped a bean", "shot up", "420", "bump",
            "plug hit me up", "one more round", "sippin’", "cooking up", "jib", "Xannies", "tweak",
            "on a bender", "too deep", "red eyes"
        ]
    }
}

country_mapping = {
    'US': 'United States', 'U.S.': 'United States', 'USA': 'United States', 'U.S': 'United States',
    'IN': 'India', 'CA': 'Canada', 'Italy': 'Italy', 'UK': 'United Kingdom', 'U.K.': 'United Kingdom',
    'Philippines': 'Philippines', 'France': 'France', 'Spain': 'Spain', 'Mexico': 'Mexico',
    'Germany': 'Germany', 'Berlin': 'Germany', 'China': 'China', 'Denmark': 'Denmark',
    'Turkey': 'Turkey', 'Sweden': 'Sweden', 'Iraq': 'Iraq', 'Netherlands': 'Netherlands',
    'Puerto Rico': 'Puerto Rico', 'NZ': 'New Zealand', 'Norway': 'Norway'
}

keywords = (
        crisis_lexicon["mental_health"]["explicit"] +
        crisis_lexicon["mental_health"]["coded"] +
        crisis_lexicon["suicide"]["explicit"] +
        crisis_lexicon["suicide"]["coded"] +
        crisis_lexicon["substance_use"]["explicit"] +
        crisis_lexicon["substance_use"]["coded"]
)

# Function to extract posts
def extract_reddit_posts(limit=500):
     
    posts_data = []


    try:
        for subreddit_name in subreddits:
            print(f"Fetching posts from r/{subreddit_name}...")
            subreddit = reddit.subreddit(subreddit_name)

            for post in subreddit.new(limit=limit):  # Fetch latest posts
                 
                if any(keyword.lower() in post.title.lower() + post.selftext.lower() for keyword in keywords):
                    posts_data.append([
                        subreddit_name,   
                        post.id,
                        str(post.author),
                        post.created_utc,
                        post.title,
                        post.selftext,
                        post.score,
                        post.num_comments,
                        post.url
                    ])

        # Convert to DataFrame
        columns = ["Subreddit", "Post ID","Username", "Timestamp", "Title", "Content", "Upvotes", "Comments", "URL"]
        df = pd.DataFrame(posts_data, columns=columns)
        # save to csv file
        df.to_csv("reddit_extracted_posts.csv", index=False)
        return df
    except Exception as e:
        print(f"An error occurred: {e}")

def clean_text(text):
    """ Cleans text by removing special characters, stopwords, and emojis """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)                                      # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)                                  # Remove special characters
    text = re.sub(r'\d+', '', text)                                          # Remove numbers
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # Remove stopwords
    return text

# Function to analyze sentiment
def sentiment_scores(sentence):
    if isinstance(sentence, float):
        sentence = str(sentence)
    sentiment_dict = analyser.polarity_scores(sentence)
    
    if sentiment_dict['compound'] >= 0.10:
        return "Positive"
    elif sentiment_dict['compound'] <= -0.10:
        return "Negative"
    else:
        return "Neutral"

# Function to predict risk level
def predict_text(text):
    """Predict using DistilBERT + BiLSTM + CNN"""
    try:
        # Tokenize
        inputs = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        )
        
        # Predict
        probs = model.predict({
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        })
        
        # Get class
        predicted_class = np.argmax(probs, axis=-1)[0]
        return risk_labels[predicted_class]   
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error"
    
# Function to extract location
def extract_location(text):
    if isinstance(text, str):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "GPE":   
                return ent.text
    return None

# Function to get coordinates
def geocode_location(place):
    try:
        location = geolocator.geocode(place)
        return (location.latitude, location.longitude) if location else None
    except:
        return None

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard/")
def dashboard():
    return render_template("dashboard.html")

import time
from sqlalchemy.exc import OperationalError

def update_user_behavior(postid,username, sentiment, risk_level, timestamp, retries=5, delay=1):
    with app.app_context():
        timestamp=pd.to_datetime(timestamp,unit='s')
        try:   
            user = UserBehaviorTrend.query.filter_by(username=str(username)).first()
            check=UserBehaviorTrend.query.filter_by(postid=postid).first()
            if user and not check:
                user.post_count += 1
                if risk_level == "High Risk":
                    user.high_risk_count += 1
                sentiment=str(sentiment)
                sentiment_score = analyser.polarity_scores(sentiment)["compound"]
                user.sentiment_trend = (user.sentiment_trend + sentiment_score) / 2
                user.last_post_timestamp = timestamp

                # Add historical data to UserBehaviorHistory
                history_entry = UserBehaviorHistory(
                    username=str(username),
                    post_count=user.post_count,
                    high_risk_count=user.high_risk_count,
                    sentiment_trend=user.sentiment_trend,
                    timestamp=timestamp
                )
                db.session.add(history_entry)

            elif not check:
                sentiment_score = analyser.polarity_scores(sentiment)["compound"]
                user = UserBehaviorTrend(
                    username=str(username),
                    post_count=1,
                    high_risk_count=1 if risk_level == "High Risk" else 0,
                    sentiment_trend=sentiment_score,
                    last_post_timestamp=timestamp
                )
                db.session.add(user)
                history_entry = UserBehaviorHistory(
                    username=str(username),
                    post_count=1,
                    high_risk_count=1 if risk_level == "High Risk" else 0,
                    sentiment_trend=sentiment_score,
                    timestamp=timestamp
                )
                db.session.add(history_entry)

            db.session.commit()
            db.session.close()  #  Close session to avoid locking
            return   

         
        except KeyboardInterrupt  as e:
            print(f"KeyboardInterrupt: {e}")
            return

        print("Database is still locked after multiple attempts. Skipping update.")

# storing the data in table 
def store_df_in_sql(df):
    with app.app_context():
        for _, row in df.iterrows():
            existing_post = RedditPost.query.filter_by(post_id=row["Post ID"]).first()
            if not existing_post:   
                post = RedditPost(
                    subreddit=row["Subreddit"],
                    post_id=row["Post ID"],
                    username=str(row['Username']),
                    timestamp=pd.to_datetime(row["Timestamp"], unit='s'),
                    title=row["Title"],
                    content=row["Content"],
                    upvotes=row["Upvotes"],
                    comments=row["Comments"],
                    url=row["URL"],
                    sentiment=row["Sentiment"],
                    risk_level=row["Risk Level"]
                )
                db.session.add(post)
                  

        db.session.commit()
 

# Fetch the data from the high_risk_countries table for Locations                
def fetch_data():
    with app.app_context():
        engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
        with engine.connect() as conn:
            query = "SELECT * FROM high_risk_countries"
            df = pd.read_sql(query, conn)   
            
            return df
                  
def store_high_risk_locations(df_clone):
    with app.app_context():
        for _, row in df_clone.iterrows():
            existing_entry = HighRiskLocation.query.filter_by(username=row['Username']).first()
            if not existing_entry:
                entry = HighRiskLocation(
                    
                    username=row['Username'],
                    risk_level=row['Risk Level'],
                    timestamp=pd.to_datetime(row['Timestamp'],unit='s'),
                    location=row['location'],
                    latitude=row['Latitude'],
                    longitude=row['Longitude']
                )
                db.session.add(entry)

        db.session.commit()
 
def get_coordinates(df_clone):
    
    df_clone['location'] = df_clone['Content'].apply(lambda x: extract_location(str(x)))
    df_clone = df_clone.dropna()
    df_clone['location'] = df_clone['location'].map(country_mapping)
    
    valid_countries = list(set(country_mapping.values()))
    df_clone.loc[~df_clone['location'].isin(valid_countries), 'location'] = 'Unknown'
    
    df_clone['location'] = df_clone['location'].str.title()
    df_clone=df_clone[df_clone['location']!="Unknown"]
    print(df_clone.info())
    df_clone['Coordinates'] = df_clone['location'].apply(lambda x: geocode_location(x))
    df_clone = df_clone.dropna()
    
    if df_clone.empty:
        print("No valid coordinates found.")
        return None
    print("Coordinations completed")
    print(df_clone.info())
    df_clone['Latitude'] = df_clone['Coordinates'].apply(lambda x: x[0])
    df_clone['Longitude'] = df_clone['Coordinates'].apply(lambda x: x[1])

    return df_clone
           
  
@app.route('/analyze/', methods=['GET'])
def analyze_posts():
    global df
    df = extract_reddit_posts()
    df=df.dropna()

    df["Cleaned Content"] = df["Content"].apply(clean_text)
    
    df["Cleaned Title"] = df["Title"].apply(clean_text)
     
    df = df.dropna()
    df["Sentiment"] = df["Cleaned Content"].apply(sentiment_scores)
    df["Risk Level"] = df["Cleaned Content"].apply(predict_text)
    print(df['Risk Level'].value_counts())
    store_df_in_sql(df)
    print("Lets start analyzing the users")
    for i in range(0, len(df)):
        time.sleep(1)   
        if(i%100==0):
            print(i)
        if(df.iloc[i]['Post ID']!=None):
            update_user_behavior(
                postid=df.iloc[i]['Post ID'],
                username=df.iloc[i]["Username"],
                sentiment=df.iloc[i]["Cleaned Content"],
                risk_level=df.iloc[i]["Risk Level"],
                timestamp=df.iloc[i]["Timestamp"]
            )

    print("Lets get the coordinates of the location")        
    df_clone=get_coordinates(df)
    print("lets store the coordinates of the location")
    store_high_risk_locations(df_clone)

    return jsonify({"message": "Data stored in SQL successfully!", "rows": len(df)})

if __name__ == "__main__":
    app.run(host="0.0.0.0" ,debug=True, port=int(os.environ.get("PORT", 5000)))
