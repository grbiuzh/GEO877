import numpy as np
import plotly.graph_objects as go
from PIL import Image
from sklearn.preprocessing import quantile_transform

class Point():
    
    def __init__(self, x=None, y=None):

        self.x = x
        self.y = y
        
    def __repr__(self):

        return f'Point(x = {self.x}, y = {self.y})'


    def distance(self, other):
        """Return Haversine distance between two points"""

        r = 6371
        phi1 = np.radians(self.y) # latitudes
        phi2 = np.radians(other.y)
        lam1 = np.radians(self.x) # longitudes
        lam2 = np.radians(other.x)

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

        # Initialize raster
        self._compute_raster_geometry()

        # Cell area statistics
        self.area_mean, self.area_sd, self.k = self._statisics()
        

    def __repr__(self):

        return f'''Disco Ball Raster:

Number of cells:        {self.k}
Number of rows:         {len(self.key)}
Mean area of cells:     {int(self.area_mean)} km2
SD of area of cells:    {int(self.area_sd)} km2'''


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

        d_theta = abs(np.radians(ll.x) - np.radians(ur.x))    # longitudinal arc length
        d_phi = abs(np.radians(ll.y) - np.radians(ur.y))      # latitudinal arc length

        phi = np.radians(((ll.y + ur.y) / 2) * (-1) + 90)     # average latitudinal angle of cell (north pole = 0°; south pole = 180°)

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

            ur = Point(ll.x + equatorial_cell_width, ll.y + self.cell_height)
            row_area = self._compute_cell_area(ll, ur) * self.n_lon

            cell_quantity = int(row_area // equatorial_cell_area)
            cell_width = 360 / cell_quantity
            cell_area = row_area / cell_quantity

            for cell in range(cell_quantity):
                self.bucket.append([])
                cell_index += 1

            start_index = cell_index - cell_quantity

            self.key.append({'start_index':start_index, 'cell_width':cell_width, 'cell_area':cell_area, 'cell_quantity':cell_quantity})


    def _compute_point_cell_location(self, point):
        """Find cell index for point location"""

        row_index = int((point.y + 90) // self.cell_height)
        cell_index = int(self.key[row_index]['start_index'] + (point.x // self.key[row_index]['cell_width']))

        return cell_index


    def add_point_to_raster(self, point, cell_index=None):
        """Add point to raster; optionally cell index already computed"""

        # Don't compute the cell location again if it is already given as an argument
        if cell_index == None:
            cell_index = self._compute_point_cell_location(point)

        self.bucket[cell_index].append(point)


    def _compute_cell_row(self, cell_index):
        """Returns row index given a cell index"""

        # Compute first row estimate based on product of average cell area and cell index
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

        centroid_x = ((cell_index - self.key[row]['start_index']) * self.key[row]['cell_width']) + (self.key[row]['cell_width'] / 2)
        centroid_y = row * self.cell_height + (self.cell_height / 2) - 90

        centroid = Point(centroid_x, centroid_y)

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

        # Check if a window can be drawn (border issue)
        if (row - reach) < 0 or (row + reach + 1) > len(self.key):
            raise IndexError('Entire window must be inside of raster')

        # Iterate through window vertically
        for i in range(row - reach, row + reach + 1):
            center_cell = int((centroid.x // self.key[i]['cell_width']) + self.key[i]['start_index'])

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
        points = []

        # Create a list of all points relevant for the density computation
        for cell_index in window:
            for point in self.bucket[cell_index]:
                points.append(point)

        s = 0

        # Compute density according to equation provided by ArcGIS
        # (https://doc.arcgis.com/en/insights/latest/analyze/calculate-density.htm)
        for point in points:
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

        start_cells = []     # Placeholders (NoneType)
        middle_cells = []    # Cells with value
        end_cells = []       # Placeholders (NoneType)

        for cell_index in range(start_cell_index):
            start_cells.append(None)

        # Compute point density for relevant cells
        for cell_index in range(start_cell_index, stop_cell_index):
            density = self._compute_cell_point_density(cell_index, search_radius, window_size)
            middle_cells.append(density)

        for cell_index in range(stop_cell_index, self.k):
            end_cells.append(None)

        # Compile layer
        density_layer = start_cells + middle_cells + end_cells

        # Scale cells with values
        middle_cells = quantile_transform(np.array(middle_cells).reshape(-1, 1), n_quantiles=256, random_state=0, copy=True).reshape(1, -1).tolist()[0]

        # Compile layer
        density_layer_scaled = start_cells + middle_cells + end_cells

        # Save layers to dictionary
        self.layers['density'] = density_layer
        self.layers['density_scaled'] = density_layer_scaled


    def create_image_from_layer(self, layer):
        """Projects layer into 2 dimensional space and saves it as PNG file"""

        # Equatorial cell as reference for pixel size
        equatorial_row = int(len(self.key) / 2)
        equatorial_row_key = self.key[equatorial_row]
        equatorial_start_index = equatorial_row_key['start_index']
        equatorial_cell_width = equatorial_row_key['cell_width']
        equatorial_cell_quantity = equatorial_row_key['cell_quantity']

        centroid_x_list = []

        # List of x centeroids of all equatorial cells
        for cell in range(equatorial_start_index, equatorial_start_index + equatorial_cell_quantity):
            equatorial_centroid = self._compute_cell_centroid(cell, row=equatorial_row)
            centroid_x_list.append(equatorial_centroid.x)

        image_raster = []

        # Iterate through each row and create y centroids
        for i, row in enumerate(self.key):
            centroid_y = (i * self.cell_height) + (self.cell_height / 2) - 90

            image_row = []

            # Iterate through each equatorial x centroid and map it onto the raster
            for j, centroid_x in enumerate(centroid_x_list):
                centroid = Point(centroid_x, centroid_y)
                cell_index = self._compute_point_cell_location(centroid)
                pixel_value = self.layers[layer][cell_index]
                if pixel_value == None:
                    color_value = 0
                else:
                    color_value = int(pixel_value * 255)
                image_row.append((color_value, color_value, color_value))

            image_raster.insert(0, image_row)

        # Compile image
        image_array = np.array(image_raster, dtype=np.uint8)
        image = Image.fromarray(image_array)
        image.save(layer + '.png')


    def display_density_layer(self, height=500):
        """Visualize layer and points using plotly (for development purposes)"""

        if self.resolution_km < 500:
            raise Exception('Resoltion too high for visualization')

        data = []

        for i, row in enumerate(self.key):
            for j in range(row['cell_quantity']):

                cell_index = row['start_index'] + j

                if self.layers['density_scaled'][cell_index] != None:
                    opacity = self.layers['density_scaled'][cell_index]
                    color = 'rgba(168, 50, 50, 1)'
                else:
                    opacity = 0
                    color = 'rgba(0, 0, 0, 0)'

                left = row['cell_width'] * j
                right = row['cell_width'] * (j + 1)
                bottom = self.cell_height * i - 90
                top = self.cell_height * (i + 1) - 90

                lat = [bottom, top, top, bottom, bottom]
                lon = [left, left, right, right, left]

                data.append(go.Scattergeo(lat = lat, lon = lon,
                    mode = 'lines', line = dict(width = 1, color='white'), fill='toself', fillcolor=color, opacity=opacity))

        fig = go.Figure(data=data)

        fig.update_geos(projection_type="orthographic")
        fig.update_layout(height=height, margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)

        fig.show()

        print(self)
