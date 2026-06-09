from dotenv import load_dotenv
load_dotenv()

import os
print(os.environ.get("WEATHER_API_KEY"))
print(os.environ.get("VISUAL_CROSSING_API_KEY"))