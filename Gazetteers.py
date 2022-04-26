from geopy.geocoders import GeoNames #GeoNames Gazetteer
import numpy as np
import matplotlib as mat

#GeoNames access
geo = GeoNames(username= "me_toponymboy")

# Marine Regions Data
"https://geo.vliz.be/geoserver/MarineRegions/wfs"


#Code

# get location information
latitude, longitude = 23.765328, 90.358641 #enter your desired value
location = geo.reverse(query=(latitude, longitude), exactly_one=False, timeout=5)
# get location name
location_name = location[0]
# print location name
print(location_name)