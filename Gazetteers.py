import numpy as np
filename = '/Users/chaualala/Desktop/allCountries.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=str)
print(data)