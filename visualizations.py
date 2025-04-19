import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Try to download stopwords if needed
try:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    logger.warning(f"Failed to download NLTK stopwords: {e}")
    STOPWORDS = set()

# Add common Twitter terms and disaster-related terms to stopwords
TWITTER_STOPWORDS = {
    'rt', 'amp', 'http', 'https', 'co', 't.co', 'twitter', 'tweet',
    'retweet', 'disaster', 'emergency', 'breaking', 'news', 'update',
    'updates', 'reported', 'reports', 'just', 'says', 'via', 'today',
    'watch', 'watching', 'video', 'photo', 'photos', 'pictures', 'pic',
    'pics', 'live', 'happening', 'now', 'breaking'
}

# Combine stopwords
STOPWORDS = STOPWORDS.union(TWITTER_STOPWORDS)

def create_sentiment_chart(df):
    """
    Create a time-based sentiment analysis chart.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data with sentiment
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if df.empty or 'created_at' not in df.columns or 'sentiment' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for sentiment analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Group by hour and sentiment
    df_copy = df.copy()
    df_copy['hour'] = df_copy['created_at'].dt.floor('H')
    
    sentiment_counts = df_copy.groupby(['hour', 'sentiment']).size().reset_index(name='count')
    
    # Pivot the data for plotting
    pivot_df = sentiment_counts.pivot(index='hour', columns='sentiment', values='count').fillna(0)
    pivot_df = pivot_df.reset_index()
    
    # Ensure all sentiment categories exist
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment not in pivot_df.columns:
            pivot_df[sentiment] = 0
    
    # Create the stacked area chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pivot_df['hour'],
        y=pivot_df['positive'],
        mode='lines',
        stackgroup='one',
        name='Positive',
        line=dict(width=0.5, color='rgba(0, 128, 0, 0.8)'),
        fillcolor='rgba(0, 128, 0, 0.4)'
    ))
    
    fig.add_trace(go.Scatter(
        x=pivot_df['hour'],
        y=pivot_df['neutral'],
        mode='lines',
        stackgroup='one',
        name='Neutral',
        line=dict(width=0.5, color='rgba(128, 128, 128, 0.8)'),
        fillcolor='rgba(128, 128, 128, 0.4)'
    ))
    
    fig.add_trace(go.Scatter(
        x=pivot_df['hour'],
        y=pivot_df['negative'],
        mode='lines',
        stackgroup='one',
        name='Negative',
        line=dict(width=0.5, color='rgba(255, 0, 0, 0.8)'),
        fillcolor='rgba(255, 0, 0, 0.4)'
    ))
    
    # Update layout
    fig.update_layout(
        title='Sentiment Analysis Over Time',
        xaxis_title='Time',
        yaxis_title='Tweet Count',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_tweet_volume_chart(df):
    """
    Create a chart showing tweet volume over time.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if df.empty or 'created_at' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for tweet volume analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Group by hour
    df_copy = df.copy()
    df_copy['hour'] = df_copy['created_at'].dt.floor('H')
    
    volume_counts = df_copy.groupby('hour').size().reset_index(name='count')
    
    # Create the volume chart
    fig = px.line(
        volume_counts, 
        x='hour', 
        y='count',
        labels={'hour': 'Time', 'count': 'Tweet Count'},
        title='Tweet Volume Over Time'
    )
    
    # Add markers
    fig.update_traces(mode='lines+markers', marker=dict(size=8))
    
    # Update layout
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Tweet Count',
        hovermode='x unified'
    )
    
    return fig

def create_word_cloud(df, column='text', max_words=100):
    """
    Create a word cloud visualization from tweet text.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data
        column (str): Column containing text to analyze
        max_words (int): Maximum number of words to include
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure with word cloud
    """
    if df.empty or column not in df.columns:
        # Return empty figure with message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available for word cloud generation',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return fig
    
    # Combine all text
    text = ' '.join(df[column].dropna().astype(str).tolist())
    
    if not text.strip():
        # Return empty figure if no text
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No text content available for word cloud',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return fig
    
    # Define color map for word cloud (blue gradient)
    colors = [(0.12, 0.47, 0.71), (0.2, 0.6, 0.86), (0.4, 0.7, 0.96)]
    cmap = LinearSegmentedColormap.from_list('TwitterBlue', colors)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=max_words,
        stopwords=STOPWORDS,
        colormap=cmap,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def create_location_map(df):
    """
    Create a map visualization of tweet locations.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data with location info
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with map
    """
    # Check if we have location data
    if df.empty or 'location' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No location data available for mapping",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # For this example, we'll create a simplified map with random points
    # In a real application, you would geocode the locations
    
    # Filter to tweets with location
    location_df = df[df['location'].notna() & (df['location'] != '')].copy()
    
    if location_df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No location data available for mapping",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Generate random coordinates for demonstration
    # In a real app, you would use geocoding
    import random
    
    location_df['lat'] = [random.uniform(25, 50) for _ in range(len(location_df))]
    location_df['lon'] = [random.uniform(-125, -70) for _ in range(len(location_df))]
    
    # Set marker colors based on sentiment
    colors = {
        'positive': 'green',
        'neutral': 'gray',
        'negative': 'red'
    }
    
    location_df['color'] = location_df['sentiment'].map(colors)
    
    # Create map
    fig = px.scatter_mapbox(
        location_df,
        lat='lat',
        lon='lon',
        hover_name='username',
        hover_data=['text', 'sentiment', 'created_at'],
        color='sentiment',
        color_discrete_map=colors,
        zoom=3,
        height=600,
        mapbox_style="carto-positron"
    )
    
    # Update layout
    fig.update_layout(
        title='Tweet Locations',
        margin={"r":0,"t":50,"l":0,"b":0},
        legend_title_text='Sentiment'
    )
    
    return fig

def create_impact_chart(df):
    """
    Create a chart showing disaster impact levels from tweets.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data with impact analysis
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if df.empty or 'disaster_impact' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No impact data available for analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Count impact levels
    impact_counts = df['disaster_impact'].value_counts().reset_index()
    impact_counts.columns = ['impact', 'count']
    
    # Define order and colors
    impact_order = ['severe', 'moderate', 'minor', 'unknown']
    impact_colors = {
        'severe': 'rgba(255, 0, 0, 0.7)',
        'moderate': 'rgba(255, 165, 0, 0.7)',
        'minor': 'rgba(255, 255, 0, 0.7)',
        'unknown': 'rgba(128, 128, 128, 0.7)'
    }
    
    # Filter and sort by impact level
    impact_counts = impact_counts[impact_counts['impact'].isin(impact_order)]
    impact_counts['impact'] = pd.Categorical(impact_counts['impact'], categories=impact_order, ordered=True)
    impact_counts = impact_counts.sort_values('impact')
    
    # Create the bar chart
    fig = px.bar(
        impact_counts,
        x='impact',
        y='count',
        color='impact',
        color_discrete_map=impact_colors,
        title='Disaster Impact Levels from Tweets',
        labels={'impact': 'Impact Level', 'count': 'Tweet Count'}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Impact Level',
        yaxis_title='Tweet Count',
        showlegend=False
    )
    
    return fig
