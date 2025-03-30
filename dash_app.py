import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import folium
from folium.plugins import HeatMap
from sqlalchemy.exc import SQLAlchemyError
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Modern theme configuration
MODERN_THEME = {
    'background': '#f8f9fa',
    'text': '#343a40',
    'primary': '#4e73df',
    'secondary': '#1cc88a',
    'accent': '#f6c23e',
    'dark': '#5a5c69',
    'light': '#ffffff',
    'danger': '#e74a3b',
    'font': 'Nunito'
}

# Start date variable
start_time = '2025-03-16'
start_time = pd.to_datetime(start_time)

def create_dashboard(flask_app, db):
    dash_app = dash.Dash(
        server=flask_app,
        name="Dashboard",
        url_base_pathname="/dashboard/",
        external_stylesheets=[
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
            f'https://fonts.googleapis.com/css2?family={MODERN_THEME["font"]}:wght@400;600;700&display=swap',
            {
                'href': 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
                'rel': 'stylesheet'
            }
        ]
    )

    # Custom CSS for modern look
    dash_app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    font-family: ''' + MODERN_THEME['font'] + ''', sans-serif;
                    background-color: ''' + MODERN_THEME['background'] + ''';
                    color: ''' + MODERN_THEME['text'] + ''';
                }
                .card {
                    border: none;
                    border-radius: 0.5rem;
                    box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
                    margin-bottom: 1.5rem;
                    transition: transform 0.3s, box-shadow 0.3s;
                }
                .card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 0.5rem 1.5rem rgba(58, 59, 69, 0.2);
                }
                .card-header {
                    background-color: ''' + MODERN_THEME['primary'] + ''';
                    color: white;
                    border-radius: 0.5rem 0.5rem 0 0 !important;
                    font-weight: 700;
                    padding: 1rem 1.5rem;
                }
                .card-body {
                    padding: 1.5rem;
                }
                .tab-content {
                    padding: 1.5rem;
                    background-color: white;
                    border-radius: 0 0 0.5rem 0.5rem;
                    box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.1);
                }
                .nav-tabs {
                    border-bottom: none;
                    margin-bottom: 0;
                }
                .nav-tabs .nav-link {
                    color: ''' + MODERN_THEME['dark'] + ''';
                    font-weight: 600;
                    border: none;
                    padding: 1rem 1.5rem;
                    margin-right: 0.5rem;
                }
                .nav-tabs .nav-link.active {
                    color: ''' + MODERN_THEME['primary'] + ''';
                    border-bottom: 3px solid ''' + MODERN_THEME['primary'] + ''';
                    background-color: transparent;
                }
                .nav-tabs .nav-link:hover {
                    border-color: transparent;
                }
                .loading-spinner {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 300px;
                }
                .spinner {
                    width: 3rem;
                    height: 3rem;
                    border: 0.25rem solid rgba(78, 115, 223, 0.3);
                    border-radius: 50%;
                    border-top-color: ''' + MODERN_THEME['primary'] + ''';
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
                .date-picker-container {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.1);
                    margin-bottom: 1.5rem;
                }
                .dashboard-header {
                    background: linear-gradient(135deg, ''' + MODERN_THEME['primary'] + ''' 0%, ''' + MODERN_THEME['secondary'] + ''' 100%);
                    color: white;
                    padding: 2rem;
                    margin-bottom: 2rem;
                    border-radius: 0.5rem;
                }
                .graph-container {
                    background: white;
                    border-radius: 0.5rem;
                    padding: 1rem;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.1);
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''

    # Import Tables
    from app import RedditPost, HighRiskLocation, UserBehaviorHistory

    # ====================== DATA FETCHING FUNCTIONS ======================
    def fetch_data(start_date=start_time, end_date=None):
        try:
            with flask_app.app_context():
                query = db.session.query(RedditPost)
                if start_date:
                    query = query.filter(RedditPost.timestamp >= start_date)
                if end_date:
                    query = query.filter(RedditPost.timestamp <= end_date)
                return pd.read_sql(query.statement, db.engine)
        except SQLAlchemyError as e:
            print(f"Database error: {e}")
            return pd.DataFrame()

    def fetch_high_risk_locations(start_date=start_time, end_date=None):
        try:
            with flask_app.app_context():
                query = db.session.query(
                    HighRiskLocation.timestamp,
                    HighRiskLocation.location,
                    HighRiskLocation.risk_level,
                    HighRiskLocation.latitude,
                    HighRiskLocation.longitude
                ).all()
                
                df = pd.DataFrame(query, columns=['timestamp', 'location', 'risk_level', 'latitude', 'longitude'])
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d')
                if start_date:
                    df = df[df['timestamp'] >= start_date]
                if end_date:
                    df = df[df['timestamp'] <= end_date]
                return df
        except SQLAlchemyError as e:
            print(f"Database error: {e}")
            return pd.DataFrame(columns=['timestamp', 'location', 'risk_level', 'latitude', 'longitude'])

    def fetch_user_behavior(start_date=start_time, end_date=None):
        try:
            with flask_app.app_context():
                query = db.session.query(UserBehaviorHistory)
                if start_date:
                    query = query.filter(UserBehaviorHistory.timestamp >= start_date)
                if end_date:
                    query = query.filter(UserBehaviorHistory.timestamp <= end_date)
                return pd.read_sql(query.statement, db.engine)
        except SQLAlchemyError as e:
            print(f"Database error: {e}")
            return pd.DataFrame()

    # ====================== CHART GENERATION FUNCTIONS ======================
    def create_risk_analysis_charts(df):
        charts = []
        
        if df.empty:
            return [html.Div(
                html.P("No data available", className='text-center text-muted'),
                className='card'
            )]
        
        # 1. Risk Level Distribution (Pie)
        if 'risk_level' in df.columns:
            risk_fig = px.pie(
                df, 
                names='risk_level', 
                title='Risk Level Distribution',
                color_discrete_sequence=[MODERN_THEME['primary'], MODERN_THEME['secondary'], MODERN_THEME['accent']]
            )
            risk_fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20)
            )
            charts.append(html.Div([
                html.Div('Risk Level Distribution', className='card-header'),
                html.Div(dcc.Graph(figure=risk_fig), className='card-body')
            ], className='card'))
        
        # 2. Risk Level Over Time (Line)
        if 'timestamp' in df.columns and 'risk_level' in df.columns:
            try:
                risk_over_time = df.groupby([df['timestamp'].dt.date, 'risk_level']).size().unstack().fillna(0)
                fig = px.line(
                    risk_over_time,
                    title='Risk Level Trends Over Time',
                    color_discrete_sequence=[MODERN_THEME['primary'], MODERN_THEME['secondary'], MODERN_THEME['accent']]
                )
                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font_color=MODERN_THEME['text'],
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend_title_text='Risk Level'
                )
                charts.append(html.Div([
                    html.Div('Risk Level Trends Over Time', className='card-header'),
                    html.Div(dcc.Graph(figure=fig), className='card-body')
                ], className='card'))
            except Exception as e:
                print(f"Error creating risk over time chart: {e}")
        
        # 3. Posts Over Time (Area)
        if 'timestamp' in df.columns:
            try:
                posts_over_time = df.groupby(df['timestamp'].dt.date).size()
                fig = px.area(
                    posts_over_time,
                    title='Posts Over Time',
                    color_discrete_sequence=[MODERN_THEME['primary']]
                )
                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font_color=MODERN_THEME['text'],
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False,
                    yaxis_title="Number of Posts"
                )
                charts.append(html.Div([
                    html.Div('Posts Over Time', className='card-header'),
                    html.Div(dcc.Graph(figure=fig), className='card-body')
                ], className='card'))
            except Exception as e:
                print(f"Error creating posts over time chart: {e}")
        
        # 4. Risk Level by Hour (Bar)
        if 'timestamp' in df.columns and 'risk_level' in df.columns:
            try:
                df['hour'] = df['timestamp'].dt.hour
                risk_by_hour = df.groupby(['hour', 'risk_level']).size().unstack().fillna(0)
                fig = px.bar(
                    risk_by_hour,
                    title='Risk Level by Hour of Day',
                    color_discrete_sequence=[MODERN_THEME['primary'], MODERN_THEME['secondary'], MODERN_THEME['accent']]
                )
                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font_color=MODERN_THEME['text'],
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title="Hour of Day",
                    yaxis_title="Number of Posts",
                    legend_title_text='Risk Level'
                )
                charts.append(html.Div([
                    html.Div('Risk Level by Hour of Day', className='card-header'),
                    html.Div(dcc.Graph(figure=fig), className='card-body')
                ], className='card'))
            except Exception as e:
                print(f"Error creating risk by hour chart: {e}")
        
        # 5. Upvotes Distribution (Box)
        if 'upvotes' in df.columns and 'risk_level' in df.columns:
            fig = px.box(
                df,
                x='risk_level',
                y='upvotes',
                title='Upvotes Distribution by Risk Level',
                color='risk_level',
                color_discrete_sequence=[MODERN_THEME['primary'], MODERN_THEME['secondary'], MODERN_THEME['accent']]
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                xaxis_title="Risk Level",
                yaxis_title="Upvotes"
            )
            charts.append(html.Div([
                html.Div('Upvotes Distribution by Risk Level', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        return charts if charts else [html.Div(
            html.P("No valid data available for charts", className='text-center text-muted'),
            className='card'
        )]

    def create_sentiment_analysis_charts(df):
        charts = []
        
        if df.empty:
            return [html.Div(
                html.P("No data available", className='text-center text-muted'),
                className='card'
            )]
        
        # 1. Sentiment Distribution (Pie)
        if 'sentiment' in df.columns:
            sentiment_fig = px.pie(
                df,
                names='sentiment',
                title='Sentiment Distribution',
                color_discrete_sequence=[MODERN_THEME['primary'], MODERN_THEME['secondary'], MODERN_THEME['accent']]
            )
            sentiment_fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20)
            )
            charts.append(html.Div([
                html.Div('Sentiment Distribution', className='card-header'),
                html.Div(dcc.Graph(figure=sentiment_fig), className='card-body')
            ], className='card'))
        
        # 2. Sentiment Over Time (Line)
        if 'timestamp' in df.columns and 'sentiment' in df.columns:
            try:
                sentiment_over_time = df.groupby([df['timestamp'].dt.date, 'sentiment']).size().unstack().fillna(0)
                fig = px.line(
                    sentiment_over_time,
                    title='Sentiment Trends Over Time',
                    color_discrete_sequence=[MODERN_THEME['primary'], MODERN_THEME['secondary'], MODERN_THEME['accent']]
                )
                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font_color=MODERN_THEME['text'],
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend_title_text='Sentiment'
                )
                charts.append(html.Div([
                    html.Div('Sentiment Trends Over Time', className='card-header'),
                    html.Div(dcc.Graph(figure=fig), className='card-body')
                ], className='card'))
            except Exception as e:
                print(f"Error creating sentiment over time chart: {e}")
        
        # 3. Sentiment vs Risk Level (Violin)
        if 'sentiment' in df.columns and 'risk_level' in df.columns:
            fig = px.violin(
                df,
                x='sentiment',
                y='upvotes',
                color='risk_level',
                box=True,
                title='Sentiment vs Engagement by Risk',
                color_discrete_sequence=[MODERN_THEME['primary'], MODERN_THEME['secondary'], MODERN_THEME['accent']]
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Sentiment",
                yaxis_title="Upvotes",
                legend_title_text='Risk Level'
            )
            charts.append(html.Div([
                html.Div('Sentiment vs Engagement by Risk', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        # 4. Sentiment Word Cloud
        df=df.dropna()
        if 'content' in df.columns:
            try:
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='Blues'
                ).generate(' '.join(df['content']))

                img = BytesIO()
                wordcloud.to_image().save(img, format='PNG')
                img_str = 'data:image/png;base64,' + base64.b64encode(img.getvalue()).decode()

                charts.append(html.Div([
                    html.Div('Sentiment Word Cloud', className='card-header'),
                    html.Div(
                        html.Img(src=img_str, style={'width': '100%', 'border-radius': '0.25rem'}),
                        className='card-body'
                    )
                ], className='card'))
            except Exception as e:
                print(f"Error creating word cloud: {e}")
        
        # 5. Top Sentiment Keywords
        if 'content' in df.columns:
            try:
                vectorizer = CountVectorizer(max_features=10, stop_words='english')
                word_counts = vectorizer.fit_transform(df['content'])
                sorted_word_counts = np.argsort(-word_counts.sum(axis=0).A1)
                keywords = vectorizer.get_feature_names_out()[sorted_word_counts]
                counts = word_counts.sum(axis=0).A1[sorted_word_counts]

                fig = go.Figure(go.Bar(
                    x=keywords,
                    y=counts,
                    marker_color=MODERN_THEME['primary'],
                    text=counts,
                    textposition='auto'
                ))
                fig.update_layout(
                    title='Top Sentiment Keywords',
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font_color=MODERN_THEME['text'],
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title="Keywords",
                    yaxis_title="Frequency",
                    xaxis={'categoryorder': 'total descending'}
                )
                charts.append(html.Div([
                    html.Div('Top Sentiment Keywords', className='card-header'),
                    html.Div(dcc.Graph(figure=fig), className='card-body')
                ], className='card'))
            except Exception as e:
                print(f"Error creating keywords chart: {e}")
        
        return charts if charts else [html.Div(
            html.P("No valid data available for charts", className='text-center text-muted'),
            className='card'
        )]

    def create_geographic_analysis_charts(df):
        charts = []
        
        if df.empty:
            return [html.Div(
                html.P("No data available", className='text-center text-muted'),
                className='card'
            )]
        
        # 1. High Risk Locations Map
        if 'latitude' in df.columns and 'longitude' in df.columns:
            try:
                m = folium.Map(
                    location=[df['latitude'].mean(), df['longitude'].mean()],
                    zoom_start=2,
                    tiles='cartodbpositron'
                )
                
                heat_data = [[row['latitude'], row['longitude']] for index, row in df.iterrows()]
                HeatMap(heat_data).add_to(m)
                
                m.save('geospatial_heatmap.html')
                
                charts.append(html.Div([
                    html.Div('Geospatial Heatmap of High-Risk Locations', className='card-header'),
                    html.Div(
                        html.Iframe(
                            srcDoc=open('geospatial_heatmap.html', 'r').read(),
                            width='100%',
                            height='500',
                            style={'border': 'none', 'border-radius': '0.25rem'}
                        ),
                        className='card-body'
                    )
                ], className='card'))
            except Exception as e:
                print(f"Error creating geospatial heatmap: {e}")
        
        # 2. Location Distribution (Bar)
        if 'location' in df.columns:
            location_counts = df['location'].value_counts().head(10)
            fig = px.bar(
                location_counts,
                title='Top 10 High-Risk Locations',
                color_discrete_sequence=[MODERN_THEME['primary']]
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Location",
                yaxis_title="Count",
                showlegend=False
            )
            charts.append(html.Div([
                html.Div('Top 10 High-Risk Locations', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        # 3. Risk Density by Country (Choropleth)
        if 'location' in df.columns:
            country_counts = df['location'].value_counts().reset_index()
            country_counts.columns = ['country', 'count']
            fig = px.choropleth(
                country_counts,
                locations='country',
                locationmode='country names',
                color='count',
                title='Risk Density by Country',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                geo=dict(bgcolor='rgba(0,0,0,0)')
            )
            charts.append(html.Div([
                html.Div('Risk Density by Country', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        # 4. Geographic Risk Over Time
        if 'timestamp' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date

                fig = px.scatter_geo(
                    df,
                    lat='latitude',
                    lon='longitude',
                    color='risk_level',
                    animation_frame='date',
                    title='Geographic Risk Over Time',
                    color_discrete_sequence=[MODERN_THEME['primary'], MODERN_THEME['secondary'], MODERN_THEME['accent']]
                )
                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font_color=MODERN_THEME['text'],
                    margin=dict(l=20, r=20, t=40, b=20),
                    geo=dict(bgcolor='rgba(0,0,0,0)')
                )
                charts.append(html.Div([
                    html.Div('Geographic Risk Over Time', className='card-header'),
                    html.Div(dcc.Graph(figure=fig), className='card-body')
                ], className='card'))
            except Exception as e:
                print(f"Error creating animated geo chart: {e}")
        
        # 5. Location vs Risk Level
        if 'location' in df.columns and 'risk_level' in df.columns:
             
            # create pivot table for location vs risk level
            location_risk = df.groupby(['location', 'risk_level']).size().unstack().fillna(0)
            location_risk['total']=location_risk.sum(axis=1)
            location_risk = location_risk.sort_values('total', ascending=False).drop(columns='total')
            fig = px.bar(
                location_risk,
                title='Risk Level by Location',
                color_discrete_sequence=[MODERN_THEME['primary'], MODERN_THEME['secondary'], MODERN_THEME['accent']]
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Location",
                yaxis_title="Count",
                legend_title_text='Risk Level',
                xaxis={'categoryorder':'total descending'}
            )
            charts.append(html.Div([
                html.Div('Risk Level by Location', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        return charts if charts else [html.Div(
            html.P("No valid data available for charts", className='text-center text-muted'),
            className='card'
        )]

    def create_user_behavior_charts(df):
        charts = []
        
        if df.empty:
            return [html.Div(
                html.P("No data available", className='text-center text-muted'),
                className='card'
            )]
        
        # 1. Top High-Risk Users (Bar)
        if 'username' in df.columns:
            df = df[~df['username'].isin(['nan', 'None', '[deleted]'])]
            top_high_risk = df.groupby('username')['high_risk_count'].sum().nlargest(10)
            fig = px.bar(
                top_high_risk,
                title='Top 10 High-Risk Users',
                labels={'value': 'High-Risk Posts', 'username': 'User'},
                color_discrete_sequence=[MODERN_THEME['primary']],
                text_auto=True
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Username",
                yaxis_title="High-Risk Count",
                xaxis={'categoryorder': 'total descending'},
                showlegend=False
            )
            charts.append(html.Div([
                html.Div('Top 10 High-Risk Users', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        # 2. User Activity Over Time (Line)
        if 'timestamp' in df.columns:
            activity = df.groupby(df['timestamp'].dt.date).size()
            fig = px.line(
                activity,
                title='User Activity Over Time',
                color_discrete_sequence=[MODERN_THEME['primary']]
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Date",
                yaxis_title="Activity Count",
                showlegend=False
            )
            charts.append(html.Div([
                html.Div('User Activity Over Time', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        # 3. User Post Distribution (Histogram)
        if 'post_count' in df.columns:
            fig = px.histogram(
                df,
                x='post_count',
                title='User Post Distribution',
                color_discrete_sequence=[MODERN_THEME['primary']]
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Post Count",
                yaxis_title="Number of Users",
                showlegend=False
            )
            charts.append(html.Div([
                html.Div('User Post Distribution', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        # 4. User Behavior Patterns (Scatter)
        if 'post_count' in df.columns and 'high_risk_count' in df.columns:
            fig = px.scatter(
                df,
                x='post_count',
                y='high_risk_count',
                title='User Behavior Patterns',
                color_discrete_sequence=[MODERN_THEME['primary']]
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Total Posts",
                yaxis_title="High Risk Posts",
                showlegend=False
            )
            charts.append(html.Div([
                html.Div('User Behavior Patterns', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        # 5. User Sentiment Distribution (Box)
        if 'sentiment_trend' in df.columns:
            fig = px.box(
                df,
                y='sentiment_trend',
                title='User Sentiment Trend Distribution',
                color_discrete_sequence=[MODERN_THEME['primary']]
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_title="Sentiment Trend",
                showlegend=False
            )
            charts.append(html.Div([
                html.Div('User Sentiment Trend Distribution', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        return charts if charts else [html.Div(
            html.P("No valid data available for charts", className='text-center text-muted'),
            className='card'
        )]

    def create_engagement_analysis_charts(df):
        charts = []
        
        if df.empty:
            return [html.Div(
                html.P("No data available", className='text-center text-muted'),
                className='card'
            )]
        
        # 1. Engagement Over Time (Line)
        if 'timestamp' in df.columns and 'upvotes' in df.columns and 'comments' in df.columns:
            df['engagement'] = df['upvotes'] + df['comments']
            engagement = df.groupby(df['timestamp'].dt.date)['engagement'].mean()
            fig = px.line(
                engagement,
                title='Average Engagement Over Time',
                color_discrete_sequence=[MODERN_THEME['primary']]
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Date",
                yaxis_title="Average Engagement",
                showlegend=False
            )
            charts.append(html.Div([
                html.Div('Average Engagement Over Time', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        # 2. Top Engaging Posts (Bar)
        if 'engagement' in df.columns:
            top_posts = df.nlargest(10, 'engagement')[['post_id', 'engagement']]
            fig = px.bar(
                top_posts,
                x='post_id',
                y='engagement',
                title='Top 10 Engaging Posts',
                color_discrete_sequence=[MODERN_THEME['primary']]
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Post ID",
                yaxis_title="Engagement Score",
                showlegend=False
            )
            charts.append(html.Div([
                html.Div('Top 10 Engaging Posts', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        # 3. Engagement by Risk Level (Box)
        if 'engagement' in df.columns and 'risk_level' in df.columns:
            fig = px.box(
                df,
                x='risk_level',
                y='engagement',
                title='Engagement Distribution by Risk Level',
                color='risk_level',
                color_discrete_sequence=[MODERN_THEME['primary'], MODERN_THEME['secondary'], MODERN_THEME['accent']]
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Risk Level",
                yaxis_title="Engagement Score",
                showlegend=False
            )
            charts.append(html.Div([
                html.Div('Engagement Distribution by Risk Level', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        # 4. Engagement vs Sentiment (Violin)
        if 'engagement' in df.columns and 'sentiment' in df.columns:
            fig = px.violin(
                df,
                x='sentiment',
                y='engagement',
                box=True,
                title='Engagement Distribution by Sentiment',
                color_discrete_sequence=[MODERN_THEME['primary'], MODERN_THEME['secondary'], MODERN_THEME['accent']]
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Sentiment",
                yaxis_title="Engagement Score",
                showlegend=False
            )
            charts.append(html.Div([
                html.Div('Engagement Distribution by Sentiment', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        # 5. Top Engaging Users (Bar)
        if 'username' in df.columns and 'engagement' in df.columns:
            df = df[df['username'] != 'None']
            top_users = df.groupby('username')['engagement'].sum().nlargest(10)
            fig = px.bar(
                top_users,
                title='Top 10 Engaging Users',
                color_discrete_sequence=[MODERN_THEME['primary']]
            )
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color=MODERN_THEME['text'],
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Username",
                yaxis_title="Total Engagement",
                showlegend=False
            )
            charts.append(html.Div([
                html.Div('Top 10 Engaging Users', className='card-header'),
                html.Div(dcc.Graph(figure=fig), className='card-body')
            ], className='card'))
        
        return charts if charts else [html.Div(
            html.P("No valid data available for charts", className='text-center text-muted'),
            className='card'
        )]

    # ====================== DASHBOARD LAYOUT ======================
    dash_app.layout = html.Div([
        # Dashboard Header
        html.Div([
            html.Div([
                html.H1("CrisisWatch AI Dashboard", 
                       style={'color': 'white', 'fontWeight': 'bold', 'marginBottom': '0.5rem'}),
                html.P("Real-time monitoring of mental health risks through advanced NLP and geospatial analytics", 
                      style={'color': 'white', 'opacity': '0.8', 'marginBottom': '0'})
            ], className='text-center')
        ], className='dashboard-header'),
        
        # Main Content Container
        html.Div([
            # Date Range Selector with loading state
            html.Div([
                html.Div([
                    html.H5("Select Date Range", className='mb-3', style={'color': MODERN_THEME['primary']}),
                    dcc.DatePickerRange(
                        id='date-range',
                        min_date_allowed=datetime.date(2025, 1, 1),
                        max_date_allowed=datetime.date.today(),
                        start_date=datetime.date(2025, 3, 17),
                        end_date=datetime.date.today(),
                        display_format='YYYY-MM-DD',
                        className='mb-3'
                    ),
                    dcc.Loading(
                        id="loading-date-range",
                        type="default",
                        children=html.Div(id="loading-output-1")
                    )
                ], className='date-picker-container')
            ], className='row mb-4'),
            
            # Tabs with loading states
            dcc.Tabs([
                dcc.Tab(
                    label='Risk Analysis',
                    children=[
                        dcc.Loading(
                            id="loading-risk",
                            type="default",
                            children=html.Div(id='risk-analysis-charts', className='row')
                        )
                    ],
                    className='py-2'
                ),
                
                dcc.Tab(
                    label='Sentiment Analysis',
                    children=[
                        dcc.Loading(
                            id="loading-sentiment",
                            type="default",
                            children=html.Div(id='sentiment-analysis-charts', className='row')
                        )
                    ],
                    className='py-2'
                ),
                
                dcc.Tab(
                    label='Geographic Analysis',
                    children=[
                        dcc.Loading(
                            id="loading-geo",
                            type="default",
                            children=html.Div(id='geographic-analysis-charts', className='row')
                        )
                    ],
                    className='py-2'
                ),
                
                dcc.Tab(
                    label='User Behavior',
                    children=[
                        dcc.Loading(
                            id="loading-user",
                            type="default",
                            children=html.Div(id='user-behavior-charts', className='row')
                        )
                    ],
                    className='py-2'
                ),
                
                dcc.Tab(
                    label='Engagement Analysis',
                    children=[
                        dcc.Loading(
                            id="loading-engagement",
                            type="default",
                            children=html.Div(id='engagement-analysis-charts', className='row')
                        )
                    ],
                    className='py-2'
                )
            ], className='mb-4'),
            
            # Auto-Refresh Component
            dcc.Interval(
                id='interval-component',
                interval=600*1000,  # 10 minutes
                n_intervals=0
            )
        ], className='container')
    ])

    # ====================== CALLBACKS ======================
    @dash_app.callback(
        [Output('risk-analysis-charts', 'children'),
         Output('sentiment-analysis-charts', 'children'),
         Output('geographic-analysis-charts', 'children'),
         Output('user-behavior-charts', 'children'),
         Output('engagement-analysis-charts', 'children')],
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date'),
         Input('interval-component', 'n_intervals')]
    )
    def update_dashboard(start_date, end_date, n):
        # Fetch data with date filters
        posts_df = fetch_data(start_date, end_date)
        locations_df = fetch_high_risk_locations(start_date, end_date)
        user_behavior_df = fetch_user_behavior(start_date, end_date)
        
        # Generate all charts
        risk_charts = create_risk_analysis_charts(posts_df)
        sentiment_charts = create_sentiment_analysis_charts(posts_df)
        geo_charts = create_geographic_analysis_charts(locations_df)
        user_charts = create_user_behavior_charts(user_behavior_df)
        engagement_charts = create_engagement_analysis_charts(posts_df)
        
        return (risk_charts, sentiment_charts, geo_charts, 
                user_charts, engagement_charts)

    return dash_app
    
