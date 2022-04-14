import geocoder as geo #Geonames gazetteer
import numpy as np
import matplotlib as mat
import nltk as nlp

import geopandas as gpd


data = gpd.read_file("/Users/chaualala/Desktop/UZH/MSc Geographie/2. Semester/GEO877 - Spatial Algorithms/GEO877/open-gazetteer-gpkg/data/open_regional_gazetteer_2021.gpkg")
data.head()
