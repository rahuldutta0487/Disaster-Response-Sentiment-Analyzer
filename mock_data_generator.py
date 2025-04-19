"""
This module provides mock data generation for disaster-related tweets.
It's used when Twitter API access is limited or unavailable.
"""

import pandas as pd
import random
import uuid
from datetime import datetime, timedelta
import numpy as np
from sentiment_analyzer import analyze_sentiment, analyze_disaster_impact

# Sample usernames
USERNAMES = [
    "DisasterAlert", "WeatherWatcher", "StormChaser", "EmergencyInfo", "SafetyFirst",
    "CrisisResponse", "WeatherChannel", "DisasterRelief", "EmergencyUpdate", "NewsFeed",
    "WeatherUpdates", "StormTracker", "DisasterMonitor", "EmergencyServices", "FirstResponder",
    "ReliefWorker", "WeatherForecast", "DisasterRecovery", "EmergencyNotice", "SafetyTips"
]

# Sample locations
LOCATIONS = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ",
    "Philadelphia, PA", "San Antonio, TX", "San Diego, CA", "Dallas, TX", "San Jose, CA",
    "Miami, FL", "Seattle, WA", "Boston, MA", "Detroit, MI", "Denver, CO",
    "Atlanta, GA", "New Orleans, LA", "San Francisco, CA", "Austin, TX", "Portland, OR"
]

# Sample coordinates (lat, lon) for major US cities
COORDINATES = {
    "New York, NY": (40.7128, -74.0060),
    "Los Angeles, CA": (34.0522, -118.2437),
    "Chicago, IL": (41.8781, -87.6298),
    "Houston, TX": (29.7604, -95.3698),
    "Phoenix, AZ": (33.4484, -112.0740),
    "Philadelphia, PA": (39.9526, -75.1652),
    "San Antonio, TX": (29.4241, -98.4936),
    "San Diego, CA": (32.7157, -117.1611),
    "Dallas, TX": (32.7767, -96.7970),
    "San Jose, CA": (37.3382, -121.8863),
    "Miami, FL": (25.7617, -80.1918),
    "Seattle, WA": (47.6062, -122.3321),
    "Boston, MA": (42.3601, -71.0589),
    "Detroit, MI": (42.3314, -83.0458),
    "Denver, CO": (39.7392, -104.9903),
    "Atlanta, GA": (33.7490, -84.3880),
    "New Orleans, LA": (29.9511, -90.0715),
    "San Francisco, CA": (37.7749, -122.4194),
    "Austin, TX": (30.2672, -97.7431),
    "Portland, OR": (45.5051, -122.6750)
}

# Dictionary for tweet templates by disaster type
TWEET_TEMPLATES = {
    "Hurricane": [
        "Hurricane {name} is approaching the coast of {location}. Stay safe everyone! #Hurricane #StaySafe",
        "Wind speeds of {wind_speed} mph reported as Hurricane {name} makes landfall in {location}. #WeatherAlert",
        "Evacuations underway in {location} as Hurricane {name} strengthens to Category {category}. #Hurricane",
        "Storm surge expected to reach {surge_height} feet in {location} due to Hurricane {name}. #StormSurge",
        "Hurricane {name} has been downgraded to a tropical storm near {location}. Stay cautious. #HurricaneRecovery",
        "Flooding reported in {location} after Hurricane {name} passed through. #FloodWarning",
        "Power outages affecting {outage_count} homes in {location} due to Hurricane {name}. #PowerOutage"
    ],
    
    "Earthquake": [
        "Magnitude {magnitude} earthquake reported in {location}. #Earthquake #Breaking",
        "Aftershocks continue in {location} following yesterday's {magnitude} earthquake. #Aftershock",
        "Building damage reported in {location} after the {magnitude} earthquake. #EarthquakeDamage",
        "Rescue teams searching for survivors in {location} after the devastating earthquake. #RescueEfforts",
        "Tsunami warning issued for coastal {location} following offshore earthquake. #TsunamiWarning",
        "Earthquake of {magnitude} magnitude felt across {location}. No major damage reported. #Earthquake",
        "Seismic activity continues in {location} region. Experts monitoring closely. #SeismicActivity"
    ],
    
    "Flood": [
        "Flash flood warning for {location}. Seek higher ground immediately! #FlashFlood",
        "River levels rising rapidly in {location}. Flood stage expected by {time}. #FloodWarning",
        "Evacuation orders issued for low-lying areas in {location} due to flooding. #Evacuation",
        "Roads closed in {location} due to severe flooding. Avoid travel if possible. #RoadClosure",
        "Flood waters receding in {location}, but damage assessment still ongoing. #FloodRecovery",
        "Emergency shelters open in {location} for those displaced by flooding. #EmergencyShelter",
        "Levees at risk of breaching in {location} as flood waters continue to rise. #FloodDanger"
    ],
    
    "Wildfire": [
        "Wildfire spreading rapidly near {location}. Evacuation orders in place. #Wildfire",
        "Fire crews battling {acres} acre wildfire in {location}. Containment at {containment}%. #FireAlert",
        "Smoke advisory issued for {location} due to nearby wildfire. Air quality poor. #SmokeAdvisory",
        "High winds complicating firefighting efforts in {location} wildfire. #FireDanger",
        "Wildfire in {location} now {containment}% contained. Crews making progress. #FireUpdate",
        "New evacuation orders for {location} as wildfire changes direction. #FireEvacuation",
        "Fire danger remains extreme in {location} due to hot, dry conditions. #FireSeason"
    ],
    
    "Tornado": [
        "Tornado warning issued for {location}. Seek shelter immediately! #TornadoWarning",
        "Tornado touched down in {location}. Damage reported. Stay clear of the area. #Tornado",
        "Funnel cloud spotted near {location}. Take cover now! #TornadoAlert",
        "Tornado damage assessment underway in {location}. Several buildings destroyed. #TornadoDamage",
        "Storm system producing multiple tornadoes moving through {location}. #SevereWeather",
        "Tornado sirens activated in {location}. Move to interior room immediately. #TornadoSiren",
        "All clear given for {location} after tornado warning expires. Stay alert for updates. #WeatherUpdate"
    ],
    
    "Tsunami": [
        "Tsunami warning issued for {location} following offshore earthquake. #Tsunami",
        "Wave height of {wave_height} meters reported in {location} tsunami. #TsunamiAlert",
        "Coastal evacuations underway in {location} due to tsunami threat. #CoastalEvacuation",
        "Tsunami waves expected to reach {location} by {time}. Move to higher ground. #TsunamiWarning",
        "Tsunami warning cancelled for {location}. All clear given. #TsunamiUpdate",
        "Tsunami damage reported along {location} coastline. Emergency teams responding. #TsunamiDamage",
        "Small tsunami waves observed in {location}. Monitoring continues. #TsunamiWatch"
    ]
}

# For mixing in the content, to have a variety of tweets
GENERAL_TWEETS = [
    "Emergency response teams deployed to {location} for {disaster} relief. #EmergencyResponse",
    "Latest update on the {disaster} in {location}: {status}. Stay tuned for more information. #DisasterUpdate",
    "Resources available for those affected by the {disaster} in {location}. Visit {website} for details. #DisasterRelief",
    "Our thoughts are with everyone affected by the {disaster} in {location}. #StaySafe",
    "Volunteers needed for {disaster} recovery efforts in {location}. #HelpNeeded",
    "Weather conditions improving in {location} following the {disaster}. #WeatherUpdate",
    "Road closures in effect around {location} due to {disaster}. Check local traffic updates. #TrafficAlert",
    "Schools closed in {location} tomorrow due to {disaster}. #SchoolClosure",
    "Remember to check on elderly neighbors during this {disaster} in {location}. #CommunitySupport",
    "Donation center for {disaster} victims open at {location} community center. #Donations"
]

# Hurricane names
HURRICANE_NAMES = ["Alex", "Bonnie", "Colin", "Danielle", "Earl", "Fiona", "Gaston", "Hermine", "Ian", "Julia", "Karl", "Lisa", "Martin", "Nicole", "Owen", "Paula", "Richard", "Shary", "Tobias", "Virginie", "Walter"]

# Hashtags by disaster type
HASHTAGS = {
    "Hurricane": ["Hurricane", "StormAlert", "WeatherWarning", "Evacuation", "StormSurge", "HurricaneSeason", "StormPrep", "FloodWatch"],
    "Earthquake": ["Earthquake", "Quake", "Seismic", "TsunamiWarning", "Aftershock", "EarthquakeSafety", "QuakeDamage", "SeismicActivity"],
    "Flood": ["Flood", "FloodWarning", "HighWater", "FlashFlood", "RisingWater", "FloodSafety", "RiverWatch", "EvacuationOrder"],
    "Wildfire": ["Wildfire", "FireDanger", "FireWarning", "ForestFire", "EvacuationAlert", "FireSeason", "SmokeAdvisory", "FireSafety"],
    "Tornado": ["Tornado", "TornadoWarning", "SevereWeather", "TakeCover", "StormChasers", "TwisterAlert", "TornadoSeason", "FunnelCloud"],
    "Tsunami": ["Tsunami", "TsunamiWarning", "CoastalEvacuation", "TsunamiAlert", "OceanSurge", "WaveHeight", "TsunamiDanger", "SeaLevelRise"]
}

# Status updates for general tweets
STATUS_UPDATES = [
    "emergency response ongoing", "situation stabilizing", "damage assessment in progress", 
    "evacuations continuing", "relief efforts underway", "conditions worsening", 
    "recovery beginning", "emergency teams on scene", "shelters at capacity", 
    "volunteers needed", "conditions improving"
]

def generate_mock_tweet(disaster_type, time_range=None):
    """Generate a mock tweet based on disaster type and time range."""
    # Set time range (default to last 7 days)
    if time_range is None:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
    else:
        start_time, end_time = time_range
        
    # Create mock tweet data
    tweet_time = random.uniform(start_time.timestamp(), end_time.timestamp())
    tweet_datetime = datetime.fromtimestamp(tweet_time)
    
    # Select a template
    if random.random() < 0.7:  # 70% chance of disaster-specific tweet
        if disaster_type == "All":
            selected_type = random.choice(list(TWEET_TEMPLATES.keys()))
            templates = TWEET_TEMPLATES[selected_type]
            hashtag_list = HASHTAGS[selected_type]
        else:
            templates = TWEET_TEMPLATES[disaster_type]
            hashtag_list = HASHTAGS[disaster_type]
    else:  # 30% chance of general tweet
        templates = GENERAL_TWEETS
        if disaster_type == "All":
            selected_type = random.choice(list(TWEET_TEMPLATES.keys()))
            hashtag_list = HASHTAGS[selected_type]
        else:
            hashtag_list = HASHTAGS[disaster_type]
    
    template = random.choice(templates)
    
    # Select a location and get coordinates
    location = random.choice(LOCATIONS)
    lat, lon = COORDINATES.get(location, (0, 0))
    
    # Add some randomness to coordinates (within ~5 miles)
    lat += random.uniform(-0.07, 0.07)
    lon += random.uniform(-0.07, 0.07)
    
    # Format the template with relevant info
    tweet_text = template.format(
        location=location,
        disaster=disaster_type if disaster_type != "All" else random.choice(list(TWEET_TEMPLATES.keys())),
        name=random.choice(HURRICANE_NAMES),
        wind_speed=random.randint(75, 180),
        category=random.randint(1, 5),
        surge_height=random.randint(3, 20),
        magnitude=round(random.uniform(4.0, 8.5), 1),
        time=tweet_datetime.strftime("%H:%M"),
        acres=random.randint(500, 50000),
        containment=random.randint(0, 100),
        wave_height=random.randint(1, 10),
        outage_count=f"{random.randint(1, 100)},000",
        status=random.choice(STATUS_UPDATES),
        website="www.disasterrelief.org"
    )
    
    # Add hashtags (1-3 random ones)
    hashtag_count = random.randint(1, 3)
    selected_hashtags = random.sample(hashtag_list, min(hashtag_count, len(hashtag_list)))
    hashtag_text = " " + " ".join([f"#{tag}" for tag in selected_hashtags])
    tweet_text += hashtag_text
    
    # Clean text for analysis
    clean_text = tweet_text.replace("#", " ")
    
    # Generate sentiment
    sentiment, sentiment_score = analyze_sentiment(clean_text)
    
    # Generate impact level
    impact_level = analyze_disaster_impact(clean_text)
    
    # Generate user info
    username = random.choice(USERNAMES)
    display_name = username
    if random.random() < 0.5:  # 50% chance of having a real name
        display_name = f"{random.choice(['John', 'Jane', 'David', 'Sarah', 'Michael', 'Emily', 'Robert', 'Lisa', 'Thomas', 'Mary'])} {random.choice(['Smith', 'Jones', 'Johnson', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor', 'Anderson'])}"
    
    # Generate tweet metrics
    retweet_count = int(np.random.exponential(10))
    like_count = int(np.random.exponential(25))
    reply_count = int(np.random.exponential(5))
    
    # Create hashtags and mentions arrays
    hashtags = selected_hashtags
    mentions = []
    if random.random() < 0.3:  # 30% chance of mentioning someone
        mention_count = random.randint(1, 2)
        mentions = random.sample(USERNAMES, mention_count)
    
    # Create mock tweet object
    tweet = {
        "id": str(uuid.uuid4()),
        "text": tweet_text,
        "clean_text": clean_text,
        "created_at": tweet_datetime,
        "username": username,
        "display_name": display_name,
        "location": location,
        "retweet_count": retweet_count,
        "like_count": like_count,
        "reply_count": reply_count,
        "hashtags": hashtags,
        "mentions": mentions,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "disaster_impact": impact_level,
        "disaster_type": disaster_type if disaster_type != "All" else random.choice(list(TWEET_TEMPLATES.keys())),
        "lat": lat,
        "lon": lon
    }
    
    return tweet

def generate_mock_tweets(count=100, disaster_type="All", time_range=None):
    """Generate a list of mock tweets for testing."""
    tweets = []
    for _ in range(count):
        tweet = generate_mock_tweet(disaster_type, time_range)
        tweets.append(tweet)
    
    # Convert to DataFrame
    df = pd.DataFrame(tweets)
    return df

def get_mock_tweet_trends(df):
    """Generate mock trends from the DataFrame of tweets."""
    tweets_count = len(df)
    trends = {
        "hashtags": {},
        "mentions": {},
        "terms": {},
        "phrases": [],
        "emerging_topics": []
    }
    
    # Process hashtags
    if "hashtags" in df.columns:
        all_hashtags = []
        for hashtags_list in df["hashtags"]:
            if isinstance(hashtags_list, list):
                all_hashtags.extend(hashtags_list)
        
        # Count occurrences
        for hashtag in all_hashtags:
            if hashtag in trends["hashtags"]:
                trends["hashtags"][hashtag] += 1
            else:
                trends["hashtags"][hashtag] = 1
    
    # Process mentions
    if "mentions" in df.columns:
        all_mentions = []
        for mentions_list in df["mentions"]:
            if isinstance(mentions_list, list):
                all_mentions.extend(mentions_list)
        
        # Count occurrences
        for mention in all_mentions:
            if mention in trends["mentions"]:
                trends["mentions"][mention] += 1
            else:
                trends["mentions"][mention] = 1
    
    # Generate mock terms
    common_terms = ["emergency", "disaster", "relief", "help", "evacuation", "damage", 
                    "warning", "alert", "safety", "shelter", "recovery", "response", 
                    "crisis", "impact", "flood", "fire", "storm", "earthquake", "tornado"]
    
    # Assign random counts to terms
    for term in common_terms:
        trends["terms"][term] = random.randint(1, max(1, tweets_count // 3))
    
    # Generate mock phrases
    phrases = ["stay safe", "emergency response", "evacuation order", "rescue teams", 
               "disaster relief", "immediate evacuation", "take shelter", "flash flood", 
               "weather update", "road closed"]
    
    # Add random phrase counts
    for phrase in phrases:
        trends["phrases"].append({"phrase": phrase, "count": random.randint(1, max(1, tweets_count // 5))})
    
    # Generate mock emerging topics
    emerging_topics = ["power outage", "shelter locations", "road closures", "volunteer coordination", 
                       "donation centers", "emergency contacts", "medical assistance", "pet rescue"]
    
    # Add random emerging topic scores
    for topic in emerging_topics:
        trends["emerging_topics"].append({"topic": topic, "score": random.random() * 10})
    
    return trends