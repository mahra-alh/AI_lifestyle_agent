"""
Dubai area reference file for the Lifestyle AI Agent.

Purpose:
- Validate the user's Dubai area.
- Convert area names into coordinates.
- Help calculate distance between the user and activities.
- Support aliases like "Marina" -> "dubai_marina".

Important:
These coordinates are approximate area-center points, not exact user addresses.
For a real production app, use Google Maps API, Geoapify, or OpenStreetMap geocoding.
"""

DUBAI_AREAS = {
    "dubai_marina": {
        "display_name": "Dubai Marina",
        "latitude": 25.08525,
        "longitude": 55.14646,
        "aliases": ["dubai marina", "marina"],
    },

    "jbr": {
        "display_name": "Jumeirah Beach Residence",
        "latitude": 25.08000,
        "longitude": 55.13500,
        "aliases": ["jbr", "jumeirah beach residence"],
    },

    "jlt": {
        "display_name": "Jumeirah Lake Towers",
        "latitude": 25.06900,
        "longitude": 55.14100,
        "aliases": ["jlt", "jumeirah lake towers"],
    },

    "business_bay": {
        "display_name": "Business Bay",
        "latitude": 25.184242,
        "longitude": 55.272430,
        "aliases": ["business bay"],
    },

    "downtown_dubai": {
        "display_name": "Downtown Dubai",
        "latitude": 25.19408,
        "longitude": 55.27810,
        "aliases": ["downtown dubai", "downtown", "burj khalifa area"],
    },

    "difc": {
        "display_name": "DIFC",
        "latitude": 25.21300,
        "longitude": 55.28100,
        "aliases": ["difc", "dubai international financial centre"],
    },

    "deira": {
        "display_name": "Deira",
        "latitude": 25.266666,
        "longitude": 55.316666,
        "aliases": ["deira"],
    },

    "bur_dubai": {
        "display_name": "Bur Dubai",
        "latitude": 25.25800,
        "longitude": 55.29700,
        "aliases": ["bur dubai"],
    },

    "al_barsha": {
        "display_name": "Al Barsha",
        "latitude": 25.11200,
        "longitude": 55.20000,
        "aliases": ["al barsha", "barsha"],
    },

    "barsha_heights": {
        "display_name": "Barsha Heights",
        "latitude": 25.09600,
        "longitude": 55.17500,
        "aliases": ["barsha heights", "tecom"],
    },

    "jvc": {
        "display_name": "Jumeirah Village Circle",
        "latitude": 25.06000,
        "longitude": 55.21000,
        "aliases": ["jvc", "jumeirah village circle"],
    },

    "jvt": {
        "display_name": "Jumeirah Village Triangle",
        "latitude": 25.04500,
        "longitude": 55.18500,
        "aliases": ["jvt", "jumeirah village triangle"],
    },

    "dubai_hills": {
        "display_name": "Dubai Hills",
        "latitude": 25.11000,
        "longitude": 55.24700,
        "aliases": ["dubai hills", "dubai hills estate"],
    },

    "mirdif": {
        "display_name": "Mirdif",
        "latitude": 25.220530,
        "longitude": 55.419472,
        "aliases": ["mirdif"],
    },

    "silicon_oasis": {
        "display_name": "Dubai Silicon Oasis",
        "latitude": 25.12500,
        "longitude": 55.38100,
        "aliases": ["silicon oasis", "dubai silicon oasis", "dso"],
    },

    "dubai_internet_city": {
        "display_name": "Dubai Internet City",
        "latitude": 25.09500,
        "longitude": 55.16000,
        "aliases": ["dubai internet city", "internet city", "dic"],
    },

    "dubai_media_city": {
        "display_name": "Dubai Media City",
        "latitude": 25.09400,
        "longitude": 55.15200,
        "aliases": ["dubai media city", "media city", "dmc"],
    },

    "palm_jumeirah": {
        "display_name": "Palm Jumeirah",
        "latitude": 25.11200,
        "longitude": 55.13900,
        "aliases": ["palm jumeirah", "the palm"],
    },

    "al_quoz": {
        "display_name": "Al Quoz",
        "latitude": 25.134415,
        "longitude": 55.245258,
        "aliases": ["al quoz"],
    },

    "karama": {
        "display_name": "Karama",
        "latitude": 25.24800,
        "longitude": 55.30500,
        "aliases": ["karama", "al karama"],
    },

    "satwa": {
        "display_name": "Al Satwa",
        "latitude": 25.22200,
        "longitude": 55.27500,
        "aliases": ["satwa", "al satwa"],
    },

    "jumeirah": {
        "display_name": "Jumeirah",
        "latitude": 25.20800,
        "longitude": 55.25500,
        "aliases": ["jumeirah"],
    },

    "umm_suqeim": {
        "display_name": "Umm Suqeim",
        "latitude": 25.14100,
        "longitude": 55.19300,
        "aliases": ["umm suqeim", "um suqeim"],
    },

    "motor_city": {
        "display_name": "Motor City",
        "latitude": 25.04600,
        "longitude": 55.23900,
        "aliases": ["motor city", "dubai motor city"],
    },

    "sports_city": {
        "display_name": "Dubai Sports City",
        "latitude": 25.03800,
        "longitude": 55.22200,
        "aliases": ["sports city", "dubai sports city"],
    },

    "dubai_south": {
        "display_name": "Dubai South",
        "latitude": 24.92000,
        "longitude": 55.09000,
        "aliases": ["dubai south", "dxb south"],
    },

    "discovery_gardens": {
        "display_name": "Discovery Gardens",
        "latitude": 25.04000,
        "longitude": 55.13500,
        "aliases": ["discovery gardens"],
    },

    "the_greens": {
        "display_name": "The Greens",
        "latitude": 25.09100,
        "longitude": 55.17100,
        "aliases": ["the greens", "greens"],
    },

    "the_views": {
        "display_name": "The Views",
        "latitude": 25.08700,
        "longitude": 55.16800,
        "aliases": ["the views", "views"],
    },
}
# Convert user input into clean lowercase text.
def normalize_area_text(value: str) -> str:
    return value.strip().lower()


def get_area_by_user_input(user_input: str) -> dict | None:
    """
    Find Dubai area by official name or alias.
    Example:
    - 'Marina' -> Dubai Marina
    - 'JVC' -> Jumeirah Village Circle
    - 'Downtown' -> Downtown Dubai
    """

    cleaned_input = normalize_area_text(user_input)

    for area_id, area_data in DUBAI_AREAS.items():
        aliases = area_data["aliases"]

        if cleaned_input in aliases:
            return {
                "area_id": area_id,
                **area_data
            }

    return None

from math import radians, sin, cos, sqrt, atan2

# Calculate distance between two lat/lon points using Haversine formula.
def calculate_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:

    earth_radius_km = 6371

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return earth_radius_km * c