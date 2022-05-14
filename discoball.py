import numpy as np
import json
from PIL import Image
from tqdm import tqdm
#import plotly.graph_objects as go

class Point():
    
    def __init__(self, lon=None, lat=None):

        self.lon = lon
        self.lat = lat
        
    def __repr__(self):

        return f'Point(x = {self.lon}, lat = {self.lat})'


    def distance(self, other):
        """Return Haversine distance between two points"""

        r = 6371
        phi1 = np.radians(self.lat) # latitudes
        phi2 = np.radians(other.lat)
        lam1 = np.radians(self.lon) # longitudes
        lam2 = np.radians(other.lon)

        d = 2 * r * np.arcsin(np.sqrt(np.sin((phi2 - phi1) / 2)**2 +
            np.cos(phi1) * np.cos(phi2) * np.sin((lam2 - lam1) / 2)**2))

        return d


class DiscoBall():

    def __init__(self, res=100):

        # Earth parameters
        self.sphere_radius = 6371
        self.sphere_area = 4 * np.pi * self.sphere_radius**2
        self.sphere_circumference = 2 * np.pi * self.sphere_radius

        # Resolution parameters
        self.resolution_km = res
        self.resolution_deg = 360 / (self.sphere_circumference / res)

        # Subdivision parameteres
        self.n_lat = (180 / self.resolution_deg) - ((180 / self.resolution_deg) % 2)
        self.n_lon = 2 * self.n_lat

        # Global cell height in degrees
        self.cell_height = 180 / self.n_lat

        # Raster elements
        self.key = []       # Geometry key
        self.bucket = []    # Cell-wise point container
        self.layers = {}    # Raster layers (cell values)

        # Container for border cells
        self.ocean_border_cells = []

        # Initialize raster
        self._compute_raster_geometry()

        # Cell area statistics
        self.area_mean, self.area_sd, self.k = self._statisics()
        

    def __repr__(self):

        return f'''Disco Ball Raster:

Number of cells:        {self.k}
Number of rows:         {len(self.key)}
Mean area of cells:     {self.area_mean:.2f} km2
SD of area of cells:    {self.area_sd:.2f} km2'''


    def _statisics(self):
        """Returns cell area statistics"""

        m = 0
        s = 0
        k = 0

        for row in self.key:
            for cell in range(row['cell_quantity']):
                area = row['cell_area']
                oldm = m
                olds = s
                k += 1
                m = oldm + (area - oldm) / k
                s = olds + (area - oldm) * (area - m)

        s = np.sqrt(s / (k - 1))

        return m, s, k


    def _compute_cell_area(self, ll, ur):
        """Returns the area of a cell [km2] given the lower-left and upper-right coordinate on a sphere"""

        # Maths on surface integral: https://www.sharetechnote.com/html/Calculus_Integration_Surface.html

        d_theta = abs(np.radians(ll.lon) - np.radians(ur.lon))    # longitudinal arc length
        d_phi = abs(np.radians(ll.lat) - np.radians(ur.lat))      # latitudinal arc length

        phi = np.radians(((ll.lat + ur.lat) / 2) * (-1) + 90)     # average latitudinal angle of cell (north pole = 0°; south pole = 180°)

        area = np.sin(phi) * d_phi * d_theta * self.sphere_radius**2

        return area


    def _compute_raster_geometry(self):
        """Computes raster geometry based on resolution and creates a corresponding empty raster bucket and geometry key"""

        # Equatorial cell for size reference 
        equatorial_cell_width = 360 / self.n_lon
        equatorial_cell_area = self._compute_cell_area(Point(0, 0), Point(equatorial_cell_width, self.cell_height))

        start_lls = []

        # Establish raster rows
        for i in np.linspace(0, 90, int(self.n_lat / 2), endpoint=False):
            start_lls.append(Point(0, i))                            # Norhtern hemisphere
            start_lls.insert(0, Point(0, - i - self.cell_height))    # Southern hemisphere

        cell_index = 0

        # Create row-specific geometry keys for equal area cells
        for ll in start_lls:

            ur = Point(ll.lon + equatorial_cell_width, ll.lat + self.cell_height)
            row_area = self._compute_cell_area(ll, ur) * self.n_lon    # Area of a horizontal band around the earth

            cell_quantity = int(row_area // equatorial_cell_area)
            cell_width = 360 / cell_quantity
            cell_area = row_area / cell_quantity

            start_index = cell_index
            cell_index += cell_quantity

            self.key.append({'start_index':start_index, 'cell_width':cell_width, 'cell_area':cell_area, 'cell_quantity':cell_quantity})
        self.bucket = [[] for i in range(cell_index)]


    def _compute_point_cell_location(self, point):
        """Find cell index for point location"""

        # Map [-180:180] to [0:360] by substracting the negative longitudes from 360°
        if point.lon < 0:
            point.lon = 360 + point.lon

        if point.lon == 0 and point.lat == 90:    # TODO: Handle this exception more elegantly
            cell_index = self.k - 1

        else:
            # Map [-90:90] to [0:180] by adding 90°
            row_index = int((point.lat + 90) // self.cell_height)
            cell_index = int(self.key[row_index]['start_index'] + (point.lon // self.key[row_index]['cell_width']))

        return cell_index


    def add_point_to_raster(self, point, cell_index=None):
        """Add point to raster; optionally cell index already computed"""

        # Don't compute the cell location again if it is already given as an argument
        if cell_index == None:
            cell_index = self._compute_point_cell_location(point)

        self.bucket[cell_index].append(point)


    def _compute_cell_row(self, cell_index):
        """Returns row index given a cell index"""

        # Compute a first row estimate based on product of average cell area and cell index
        # The product represents an estimate of the area filled up to the row we are looking for
        row = int(((((cell_index * self.area_mean) / self.sphere_area) / 2) * 360) // self.cell_height)

        m = 1
        n = 1

        # Find exact row by checking rows above and below the estimate, until certain (usually 0 to 4 iterations)
        while not (cell_index >= self.key[row]['start_index'] and cell_index < (self.key + [{'start_index':self.k}])[row + 1]['start_index']):
            if not (row + (m * n) < 0 or row + (m * n) > len(self.key) -1):
                row += (m * n)
                m += 1     # Increase search range by 1
                n *= -1    # Oscillate sign (to look above and below the row estimate)
            else:
                m -= 1     # If raster border reached, don't look beyond the border

        return row


    def _compute_cell_centroid(self, cell_index, row=None):
        """Returns the centroid (point) of a cell given a cell index"""

        # Don't compute the row again if it is already given as an argument
        if row == None:
            row = self._compute_cell_row(cell_index)

        # Total width of cells on the left of target cell plus half a cell to reach the middle of the cell
        centroid_lon = ((cell_index - self.key[row]['start_index']) * self.key[row]['cell_width']) + (self.key[row]['cell_width'] / 2)
        # Total width of cells below the target cell plus half a cell to reach the middle of the cell
        centroid_lat = row * self.cell_height + (self.cell_height / 2) - 90

        # Create a point entity for the centroid
        centroid = Point(centroid_lon, centroid_lat)

        return centroid


    def _compute_cell_window(self, cell_index, window_size, row=None, centroid=None):
        """Returns a list of cell indices of a size-by-size window around a give cell index"""

        # Don't compute the row again if it is already given as an argument
        if row == None:
            row = self._compute_cell_row(cell_index)

        # Don't compute the centroid again if it is already given as an argument
        if centroid == None:
            centroid = self._compute_cell_centroid(cell_index, row=row)

        window = []

        reach = int(window_size // 2)    # Numbers of cells left/right/above/below a cell

        # Check if a window can be drawn (raster edge issue)
        if (row - reach) < 0 or (row + reach + 1) > len(self.key):
            raise IndexError('Entire window must be inside of raster')

        # Iterate through window vertically
        for i in range(row - reach, row + reach + 1):
            center_cell = int((centroid.lon // self.key[i]['cell_width']) + self.key[i]['start_index'])

            # Iterate through window horizontally
            for j in range(center_cell - reach, center_cell + reach + 1):
                if j < self.key[i]['start_index']:
                    j += self.key[i]['cell_quantity']
                if j < self.k:
                    window.append(j)

        window = list(set(window))    # Make window elements unique

        return window


    def _compute_cell_point_density(self, cell_index, search_radius, window_size):
        """Returns point density given a cell index, a search radius and a corresponding window size"""

        # Making sure row and centroid are computed only once for each cell
        row = self._compute_cell_row(cell_index)
        centroid = self._compute_cell_centroid(cell_index, row=row)
        window = self._compute_cell_window(cell_index, window_size, row=row, centroid=centroid)

        s = 0

        # Iterate through all the points in the cells in the window
        # Compute density according to equation provided by ArcGIS
        # (https://doc.arcgis.com/en/insights/latest/analyze/calculate-density.htm)
        for cell_index in window:
            for point in self.bucket[cell_index]:
                distance = centroid.distance(point)
                if distance <= search_radius:
                    a = (1 - (distance / search_radius)**2)**2
                    b = a * 3 / np.pi

                    s += b

        rho = (1 / search_radius**2) * s

        return rho


    def compute_point_density_layer(self, search_radius):
        """Creates a point density raster in the form of the main raster"""

        cell_size = self.sphere_circumference / self.n_lon    # ~ resolution

        reach = int(search_radius // cell_size)

        # Find the maximum reach required for specified search radius
        if search_radius % cell_size != 0:
            reach += 1

        window_size = 2 * reach + 1

        # Find first and last cell whose windows are entirely contained by the raster
        start_cell_index = self.key[reach]['start_index']
        stop_cell_index = self.key[len(self.key) - reach]['start_index']

        density_layer = []

        for cell_index in range(start_cell_index):
            density_layer.append(0)

        # Compute point density for relevant cells
        for cell_index in tqdm(range(start_cell_index, stop_cell_index)):    # Use tqdm module for progress bar
            density = self._compute_cell_point_density(cell_index, search_radius, window_size)
            density_layer.append(density)

        for cell_index in range(stop_cell_index, self.k):
            density_layer.append(0)

        # Save layer to dictionary
        self.layers['density'] = density_layer


    def classify_layer(self, layer, n_quantiles, bins=None):
        """Classify layer using quantiles"""

        layer_raw = np.array(self.layers[layer])

        # Use user-specified bins if given as argument
        if bins == None:
            bins = []

            layer_nozeros = np.sort(layer_raw[layer_raw != 0])

            # Find the values of the first and last cell of each bin
            for i in range(1, n_quantiles):
                index = (len(layer_nozeros) // n_quantiles) * i
                value = layer_nozeros[index]
                bins.append(value)

            bins.insert(0, layer_nozeros[0])    # lowest value
            bins.append(layer_nozeros[-1])      # highest value

        else:
            bins = bins

        # Classify into bins
        layer_classified = np.digitize(layer_raw, bins)

        # Save layer to dictionary
        self.layers[layer + '_classified'] = layer_classified

        # Return bins to use in further classifications
        return bins


    def save_layer(self, layername, filepath):
        """Save layer to text file"""

        with open(filepath, 'w') as f:
            for item in self.layers[layername]:
                f.write("%s\n" % item)

        print(f'Layer saved: {filepath}')


    def load_layer(self, layername, filepath):
        """Load layer from text file"""

        layer = []

        with open(filepath, 'r') as f:
            for line in f:
                value = line[:-1]
                if value == 'None':
                    layer.append(None)
                else:
                    layer.append(float(value))

        self.layers[layername] = layer

        print(f'Layer loaded: {filepath}')


    def load_boundaries(self, filepath):
        """Load points for border and extract the corresponding raster cells"""

        f = open(filepath, 'r')
        data = json.load(f)
        f.close()

        for i in data['features']:
            coordinates = i['geometry']['coordinates']
            point = Point(coordinates[0], coordinates[1])
            cell = self._compute_point_cell_location(point)
            self.ocean_border_cells.append(cell)

        self.ocean_border_cells = set(self.ocean_border_cells)

        print(f'Boundaries loaded: {filepath}')


    def create_image_from_layer(self, layername, filepath):
        """Projects layer into 2D and saves it as PNG file"""

        # Color palette according to Acheson et al. 2017 (for 5 classes only!)
        color_key = {0: (255,255,255), 1: (252,208,163), 2: (253,174,107), 3: (241,105,19), 4: (215,72,1), 5: (141,45,3), 6: (141,45,3)}

        # Equatorial cell as reference for pixel size
        equatorial_row = int(len(self.key) / 2)
        equatorial_row_key = self.key[equatorial_row]
        equatorial_start_index = equatorial_row_key['start_index']
        equatorial_cell_width = equatorial_row_key['cell_width']
        equatorial_cell_quantity = equatorial_row_key['cell_quantity']

        centroid_lon_list = []

        # List of lon centeroids of all equatorial cells
        for cell in range(equatorial_start_index, equatorial_start_index + equatorial_cell_quantity):
            equatorial_centroid = self._compute_cell_centroid(cell, row=equatorial_row)
            centroid_lon_list.append(equatorial_centroid.lon)

        # Move left edge of image to Pacific Ocean (180°) instead of Europe (0°) 
        centroid_lon_list = centroid_lon_list[len(centroid_lon_list) // 2:] + centroid_lon_list[:len(centroid_lon_list) // 2]

        image_raster = []

        # Iterate through each row and create lat centroids
        for i, row in enumerate(self.key):
            centroid_lat = (i * self.cell_height) + (self.cell_height / 2) - 90

            image_row = []

            # Iterate through each equatorial lon centroid and map it onto the raster
            for j, centroid_lon in enumerate(centroid_lon_list):
                centroid = Point(centroid_lon, centroid_lat)
                cell_index = self._compute_point_cell_location(centroid)
                pixel_value = self.layers[layername][cell_index]
                if cell_index in self.ocean_border_cells:
                    rgb = (0,0,0)    # black
                else:
                    rgb = color_key[pixel_value]
                image_row.append(rgb)

            image_raster.insert(0, image_row)

        # Compile image
        image_array = np.array(image_raster, dtype=np.uint8)
        image = Image.fromarray(image_array)
        image.save(filepath)

        print(f'Image saved: {filepath}')


    # def display_density_layer(self, height=500):
    #     """Visualize layer and points using plotly (for development purposes)"""

    #     if self.resolution_km < 500:
    #         raise Exception('Resoltion too high for visualization')

    #     data = []

    #     for i, row in enumerate(self.key):
    #         for j in range(row['cell_quantity']):

    #             cell_index = row['start_index'] + j

    #             if self.layers['density_scaled'][cell_index] != None:
    #                 opacity = self.layers['density_scaled'][cell_index]
    #                 color = 'rgba(168, 50, 50, 1)'
    #             else:
    #                 opacity = 0
    #                 color = 'rgba(0, 0, 0, 0)'

    #             left = row['cell_width'] * j
    #             right = row['cell_width'] * (j + 1)
    #             bottom = self.cell_height * i - 90
    #             top = self.cell_height * (i + 1) - 90

    #             lat = [bottom, top, top, bottom, bottom]
    #             lon = [left, left, right, right, left]

    #             data.append(go.Scattergeo(lat = lat, lon = lon,
    #                 mode = 'lines', line = dict(width = 1, color='white'), fill='toself', fillcolor=color, opacity=opacity))

    #     fig = go.Figure(data=data)

    #     fig.update_geos(projection_type="orthographic")
    #     fig.update_layout(height=height, margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)

    #     fig.show()

    #     print(self)
