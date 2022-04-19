import geocoder as geo #Geonames gazetteer
import numpy as np
import matplotlib as mat
import nltk as nlp

import geopandas as gpd


# Link to european gazetteer: https://www.mapsforeurope.org/access-data
#It is to large to share on Github

# Or as WFS
# Your Token
# ImV1cm9nZW9ncmFwaGljc19yZWdpc3RlcmVkXzE2MjIi.FTmabg.tMcw7-tbRuKigswHCQWgYuhkRiI
# Open Gazetteer - WFS
# https://www.mapsforeurope.org/api/v2/maps/external/wfs/open-gazetteer?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetCapabilities&token=ImV1cm9nZW9ncmFwaGljc19yZWdpc3RlcmVkXzE2MjIi.FTmabg.tMcw7-tbRuKigswHCQWgYuhkRiI


data = gpd.read_file("/Users/chaualala/Desktop/UZH/MSc Geographie/2. Semester/GEO877 - Spatial Algorithms/GEO877/open-gazetteer-gpkg/data/open_regional_gazetteer_2021.gpkg")
data.head()
