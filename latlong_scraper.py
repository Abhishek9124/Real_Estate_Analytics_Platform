import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

# --- Configuration ---
# Use Nominatim (OpenStreetMap) for geocoding, as it's free and open-source.
# You must provide a unique user agent string.
GEO_LOCATOR = Nominatim(user_agent="gurgaon_sector_scraper_v1")

# Use a rate limiter to ensure we don't hit the API too fast (Nominatim limit is 1 req/sec)
# This pauses the script for at least 1 second between each geocoding call.
geocode_with_delay = RateLimiter(GEO_LOCATOR.geocode, min_delay_seconds=1.5)

# Initialize data storage
data_list = []

print("Starting coordinate fetching using Nominatim Geocoding API...")
print("-" * 50)

# Iterate over sectors and fetch coordinates
# Using a smaller range (1-5) for a quick test run. Change to (1, 116) for all.
for sector_num in range(1, 116): # Change range(1, 6) back to range(1, 116) for all sectors
    sector_name = f"Sector {sector_num}, Gurgaon, India"
    
    # CRITICAL: Print the sector being processed
    print(f"Processing: {sector_name}")
    
    try:
        # Use the rate-limited geocoder
        location = geocode_with_delay(sector_name, timeout=10) 
        
        if location:
            lat = location.latitude
            lon = location.longitude
            print(f"  -> Found: Lat={lat}, Lon={lon}")
            data_list.append({"Sector": f"Sector {sector_num}", "Latitude": lat, "Longitude": lon})
        else:
            print(f"  -> **Could not find coordinates for {sector_name}.**")
            data_list.append({"Sector": f"Sector {sector_num}", "Latitude": None, "Longitude": None})
            
    except Exception as e:
        print(f"  -> An error occurred while geocoding {sector_name}: {e}")
        data_list.append({"Sector": f"Sector {sector_num}", "Latitude": None, "Longitude": None})
        # Wait a bit longer if there was an error
        time.sleep(5) 

# Create DataFrame from the list
df = pd.DataFrame(data_list)

# Save DataFrame
FILE_NAME = "gurgaon_sectors_coordinates_geocoded.csv"
df.to_csv(FILE_NAME, index=False)
print("-" * 50)
print(f"✅ Data collection complete. Data saved to **{FILE_NAME}**")