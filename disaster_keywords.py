"""
This module contains predefined keywords related to different types of disasters.
These keywords are used for filtering tweets and searching for relevant content.
"""

def get_disaster_keywords(disaster_type=None):
    """
    Get a list of keywords related to a specific disaster type or all disasters.
    This function returns a limited set of keywords to prevent exceeding Twitter API limits.
    
    Args:
        disaster_type (str, optional): Type of disaster or 'All' for all types
        
    Returns:
        list: List of keywords related to the specified disaster type
    """
    # Define high-priority keywords for different disaster types (limited set)
    disaster_keywords = {
        "Hurricane": [
            "hurricane", "tropical storm", "cyclone", "typhoon", "storm surge"
        ],
        
        "Earthquake": [
            "earthquake", "quake", "tremor", "seismic", "aftershock"
        ],
        
        "Flood": [
            "flood", "flooding", "flash flood", "flood warning", "rising water"
        ],
        
        "Wildfire": [
            "wildfire", "forest fire", "brush fire", "fire spread", "fire evacuation"
        ],
        
        "Tornado": [
            "tornado", "twister", "tornado warning", "funnel cloud", "severe storm"
        ],
        
        "Tsunami": [
            "tsunami", "tidal wave", "tsunami warning", "coastal evacuation", "wave height"
        ],
        
        "General": [
            "disaster", "emergency", "evacuation", "rescue", "crisis"
        ]
    }
    
    # If disaster_type is None or 'All', return a limited set from all disaster types
    if disaster_type is None or disaster_type == "All":
        # Take top 2 keywords from each category for a reasonable query size
        limited_keywords = []
        for category, keywords in disaster_keywords.items():
            if category != "General":  # Skip general when getting All keywords
                limited_keywords.extend(keywords[:2])
        
        return limited_keywords
    
    # If the specified disaster type exists, return its keywords
    if disaster_type in disaster_keywords:
        # Return specific disaster keywords plus general keywords,
        # but limit total to avoid Twitter API search limits
        disaster_specific = disaster_keywords[disaster_type]
        general = disaster_keywords["General"][:3]  # Limit to top 3 general terms
        
        return disaster_specific + general
    
    # If disaster type is not recognized, return general keywords
    return disaster_keywords["General"]

def get_disaster_types():
    """
    Get a list of available disaster types.
    
    Returns:
        list: List of available disaster types
    """
    return ["Hurricane", "Earthquake", "Flood", "Wildfire", "Tornado", "Tsunami", "General"]

def get_impact_keywords():
    """
    Get keywords that indicate different levels of disaster impact.
    
    Returns:
        dict: Dictionary with impact levels as keys and keywords as values
    """
    impact_keywords = {
        "Severe": [
            "catastrophic", "devastating", "fatal", "death", "killed", "casualties", 
            "destroyed", "emergency", "evacuate", "evacuation", "crisis", "danger", "severe", 
            "tragedy", "disaster", "critical", "massive damage", "deadly", "fatalities"
        ],
        
        "Moderate": [
            "damage", "injured", "wounded", "affected", "impact", "hit", "threat",
            "loss", "moderate", "concern", "worried", "warning", "displacement",
            "disruption", "power outage", "destruction", "property damage"
        ],
        
        "Minor": [
            "minor", "small", "limited", "contained", "controlled", "restored",
            "recovery", "stable", "manageable", "relief", "minimal", "slight",
            "improving", "under control", "returning to normal"
        ]
    }
    
    return impact_keywords
