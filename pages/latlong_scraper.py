import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

GEO_LOCATOR = Nominatim(user_agent="gurgaon_sector_scraper_v1")

geocode_with_delay = RateLimiter(GEO_LOCATOR.geocode, min_delay_seconds=1.5)

data_list = []

print("Starting coordinate fetching using Nominatim Geocoding API...")
print("-" * 50)


for sector_num in range(1, 116):
    sector_name = f"Sector {sector_num}, Gurgaon, India"
    
    print(f"Processing: {sector_name}")
    
    try:
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
        time.sleep(5) 

df = pd.DataFrame(data_list)

FILE_NAME = "gurgaon_sectors_coordinates_geocoded.csv"
df.to_csv(FILE_NAME, index=False)
print("-" * 50)
print(f"✅ Data collection complete. Data saved to **{FILE_NAME}**")