import numpy as np
import json
from PIL import Image
from tqdm import tqdm

class Point():
    '''A class to represent a point feature using its coordinates

    Args:
        lon (float): Longitude of point feature
        lat (float): Latitude of point feature

    Attributes:
        lon (float): Longitude of point feature
        lat (float): Latitude of point feature
    '''
    
    def __init__(self, lon=None, lat=None):

        self.lon = lon
        self.lat = lat
        
    def __repr__(self):

        return f'Point(lon = {self.lon}, lat = {self.lat})'


    def distance(self, other):
        '''Calculate distance to another Point object using the Haversine formula
        
        Args:
            other (Point): Point object to which to compute the distance

        Returns:
            distance (float): Distance to the other Point object
        
        '''

        r = 6371                       # Earth radius

        phi1 = np.radians(self.lat)    # Latitudes
        phi2 = np.radians(other.lat)

        lam1 = np.radians(self.lon)    # Longitudes
        lam2 = np.radians(other.lon)

        # Haversine formula
        distance = 2 * r * np.arcsin(np.sqrt(np.sin((phi2 - phi1) / 2)**2 +
            np.cos(phi1) * np.cos(phi2) * np.sin((lam2 - lam1) / 2)**2))

        return distance


class DiscoBall():
    '''A class to create a spherical raster with equal area cells

    Args:
        res (int/float): Resolution of raster, defined as pixel size (km2)

    Attributes:
        sphere_radius (int): Radius of the earth
        sphere_area (float): Total surface area of the earth
        sphere_circumference (float): Max circumference of the earth (Equator)
        resolution_km (int/float): Resolution of raster, defined as pixel size (km2)
        resolution_deg (int/float): Resolution of raster, defined as pixel size (degrees)
        n_lat (int): Number of latitudinal subdivisions of raster (number of rows)
        n_lon (int): Number of longitudinal subdivisions of raster (at the Equator)
        cell_height (float): Equal latitudinal size (height) of all raster cells (degrees)
        key (list): A list containing a decoding key (dict) for each row of the raster (from south to north)
        bucket (list): A list containing a container (list) for each cell of the raster
        layers (dict): A dict storing raster layers (list of floats)
        ocean_border_cells (list): A list of all raster pixel indices representing a border to be visualised in the final image
        area_mean (float): Mean of the cell areas in the raster
        area_sd (float): Standard Deviation of the cell areas in the raster
        k (int): Number of cells in the raster
    '''

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
        self.key = []       # Row-wise geometry key
        self.bucket = []    # Cell-wise point container
        self.layers = {}    # Raster layers (cell values)

        # Container for border cells
        self.ocean_border_cells = []

        # Initialise raster (formats key & bucket according to resolution)
        self._compute_raster_geometry()

        # Cell area statistics
        self.area_mean, self.area_sd, self.k = self._statistics()
        

    def __repr__(self):

        return f'''Disco Ball Raster:

Number of cells:        {self.k}
Number of rows:         {len(self.key)}
Mean area of cells:     {self.area_mean:.2f} km2
SD of area of cells:    {self.area_sd:.2f} km2'''


    def _statistics(self):
        '''Compute statistics for cell areas of raster

        (Based on the code of Ross Purves: Welford method)

        Returns:
            m (float): Mean of the cell areas in the raster
            s (float): Standard deviation of the cell areas in the raster
            k (int): Number of cells in the raster
        '''

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
        '''Calculate area of a "rectangular" patch on a sphere given the lower-left and upper-right coordinate

        (Function created based on the maths on surface integrals provided by ShareTechnote:
        https://www.sharetechnote.com/html/Calculus_Integration_Surface.html)

        Args:
            ll (Point): Lower-left coordinate as Point object
            ur (Point): Upper-right coordinate as Point object

        Returns:
            area (float): Area spanned by the two Point objects [km2]
        '''

        d_theta = abs(np.radians(ll.lon) - np.radians(ur.lon))    # Longitudinal arc length
        d_phi = abs(np.radians(ll.lat) - np.radians(ur.lat))      # Latitudinal arc length

        phi = np.radians(((ll.lat + ur.lat) / 2) * (-1) + 90)     # Centered latitudinal angle of cell (only here: north pole = 0°; south pole = 180°)

        area = np.sin(phi) * d_phi * d_theta * self.sphere_radius**2

        return area


    def _compute_raster_geometry(self):
        '''Computes raster geometry, stores it row-wise in geometry keys and adds correct amount of containers to the bucket'''

        # "ll" means lower-left coordinate (as Point object)
        # "ur" means upper-right coordinate (as Point object)

        # Equatorial cell for size reference 
        equatorial_cell_width = 360 / self.n_lon    # Width in degrees
        equatorial_cell_area = self._compute_cell_area(Point(0, 0), Point(equatorial_cell_width, self.cell_height))

        start_lls = []    # List of first lower-left coordinates on each row (at 0° Longitude)

        # Establish raster rows
        # (Use append & insert to preserve order from south to north)
        for i in np.linspace(0, 90, int(self.n_lat / 2), endpoint=False):
            start_lls.append(Point(0, i))                            # Northern hemisphere
            start_lls.insert(0, Point(0, - i - self.cell_height))    # Southern hemisphere

        cell_index = 0

        # Create row-specific geometry keys for equal area cells
        for ll in start_lls:

            # Establish a upper-right coordinate based on Equatorial cell width to compute the total area of the row afterwards
            ur = Point(ll.lon + equatorial_cell_width, ll.lat + self.cell_height)
            row_area = self._compute_cell_area(ll, ur) * self.n_lon    # Area of a horizontal band around the earth

            cell_quantity = int(row_area // equatorial_cell_area)    # Number of whole Equatorial cells that fit into that row
            cell_width = 360 / cell_quantity                         # Row-specific cell width [deg]
            cell_area = row_area / cell_quantity                     # Row-specific cell area ~ Equatorial cell area

            start_index = cell_index
            cell_index += cell_quantity

            # Create dict entry for each row
            self.key.append({'start_index':start_index, 'cell_width':cell_width, 'cell_area':cell_area, 'cell_quantity':cell_quantity})

        # Create correct number of containers (= total number of cells)    
        self.bucket = [[] for i in range(cell_index)]


    def _compute_point_cell_location(self, point):
        '''Calculate cell index corresponding to the location of a given Point object

        Args:
            point (Point): Point for which the raster location (cell index) is calculated

        Returns:
            cell_index (int): Cell index corresponding to location of Point object
        '''

        # Map [-180:180] to [0:360] by substracting the negative longitudes from 360°
        if point.lon < 0:
            point.lon = 360 + point.lon

        if point.lon == 0 and point.lat == 90:    # TODO: Investigate this exception
            cell_index = self.k - 1

        else:
            # Map [-90:90] to [0:180] by adding 90°
            row_index = int((point.lat + 90) // self.cell_height)
            cell_index = int(self.key[row_index]['start_index'] + (point.lon // self.key[row_index]['cell_width']))

        return cell_index


    def add_point_to_raster(self, point, cell_index=None):
        '''Add a Point object to the raster data structure

        Args:
            point (Point): Point to be added to the raster (bucket)
            cell_index (int): Overwrite the cell index manually (optional)
        '''

        # Don't compute the cell index if one is given as an argument
        if cell_index == None:
            cell_index = self._compute_point_cell_location(point)

        # Append the point to the correct cell (list) in the bucket (list of lists) by indexing it 
        self.bucket[cell_index].append(point)


    def add_multiple_points_to_raster(self, filepath, key=None):
        '''Add multiple Point objects to the raster data structure, given a json file of point coordinates

        Args:
            filepath (str): System path of json file containing the point coordinates
            key (str): If the json file is structured, a key is necessary to access the correct data
        '''

        with open(filepath, 'r') as f:
            data = json.load(f)
            if key != None:
                data = data[key]                     # Access data subset if necessary
            for point in data:
                point = Point(point[1], point[0])    # Create Point objects from coordinate tuples
                self.add_point_to_raster(point)

        print(f'{len(data)} points added to the raster')


    def _compute_cell_row(self, cell_index):
        '''Calculate the row index of a cell given its cell index

        (This approach first creates an estimate of the row using some geometric calculations
        and then finds the exact one by iterating through the neighbouring rows until
        the condition describing the correct row is satisfied)

        Args:
            cell_index (int): The cell index of the cell, for which the row is computed

        Returns:
            row (int): The row index of the cell
        '''

        # First row estimate based on product of average cell area and cell index
        # The product represents an estimate of the area filled up to the row we are looking for
        row = int(((((cell_index * self.area_mean) / self.sphere_area) / 2) * 360) // self.cell_height)    # TODO: Optimise initial estimate

        m = 1    # Iteration parameter: distance (search range)
        n = 1    # Iteration parameter: sign (above/below)

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
        '''Calculate the centroid (Point object) of a cell given its cell index

        Args:
            cell_index (int): The cell index of the cell, for which the centroid is computed
            row (int): Overwrite the row index manually (optional)

        Returns:
            centroid (Point): A Point object representing the centroid of the cell
        '''

        # Don't compute the row index if one is given as an argument
        if row == None:
            row = self._compute_cell_row(cell_index)

        # Total width of cells on the left of target cell plus half a cell to reach the middle of the cell
        centroid_lon = ((cell_index - self.key[row]['start_index']) * self.key[row]['cell_width']) + (self.key[row]['cell_width'] / 2)

        # Total height of cells below the target cell plus half a cell to reach the middle of the cell
        centroid_lat = row * self.cell_height + (self.cell_height / 2) - 90

        # Create a Point object for the centroid
        centroid = Point(centroid_lon, centroid_lat)

        return centroid


    def _compute_cell_window(self, cell_index, window_size, row=None, centroid=None):
        '''Determine the set of cells which make up the window (neighbourhood) around a cell

        (Due to the spherical nature of the raster, the cells are not aligned,
        making the establishment of a window around a cell more complex.
        Thus, this function first determines the centroid of the cell for which the
        neighbourhood is to be determined. This centroid is then projected along
        the vertical row-range of the window (bottom to top). With that, for each
        row of the window a cell is determined which represents the center of
        that window row. Starting from that row center a defined number of
        cells on the left and right of it then make up the cells belonging to
        that row of the window. This approach does not create a pretty window
        with straight edges. However, it still provides the needed subset of
        cells needed for the aggregation of the points in a defined search
        radius around a cell, which is why this window approximation more than
        suffices for the purpose of this algorithm.)

        Args:
            cell_index (int): The cell index of the cell, for which the window is computed
            row (int): Overwrite the row index manually (optional)
            centroid (int): Overwrite the centroid manually (optional)

        Returns:
            window (list): A list of cell indices making up the window around the cell

        Raises:
            IndexError: If the window cannot be fully contained by the raster
        '''

        # Don't compute the row if one is given as an argument
        if row == None:
            row = self._compute_cell_row(cell_index)

        # Don't compute the centroid if one is given as an argument
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
                if j < self.key[i]['start_index']:    # Case where cell doesn't belong to the same row anymore (beyond Greenwich)
                    j += self.key[i]['cell_quantity']
                if j < self.k:
                    window.append(j)

        window = list(set(window))    # Make window elements unique (needed for windows near poles)

        return window


    def _compute_cell_point_density(self, cell_index, search_radius, window_size):
        '''Calculate the point density for a cell, given its index and a specified search radius (and a corresponding window size)

        (The point density was calculated using the formula provided in the ArcGIS documentation:
        https://doc.arcgis.com/en/insights/latest/analyze/calculate-density.htm)

        Args:
            cell_index (int): The cell index of the cell, for which the point density is computed
            search_radius (float): The search radius used for the point density calculation
            window_size (int): The size of the window, derived from the search radius in an earlier step

        Returns:
            rho (float): Density value [features per km2] for the cell
        '''

        # Ensuring each computation is only executed once
        row = self._compute_cell_row(cell_index)
        centroid = self._compute_cell_centroid(cell_index, row=row)
        window = self._compute_cell_window(cell_index, window_size, row=row, centroid=centroid)

        s = 0

        # Iterate through all the points in the cells in the window
        # Compute density according to equation provided by ArcGIS
        for cell_index in window:
            for point in self.bucket[cell_index]:
                distance = centroid.distance(point)
                if distance <= search_radius:    # Drop points farther away than the search radius
                    a = (1 - (distance / search_radius)**2)**2
                    b = a * 3 / np.pi

                    s += b

        rho = (1 / search_radius**2) * s

        return rho


    def compute_point_density_layer(self, search_radius):
        '''Calculate the point density for the whole raster and create a density layer (list of floats)

        (This function iterates through the raster using a moving
        window and computes the point density for each cell.)

        Args:
            search_radius (float): The search radius used for the point density calculation and also for determining the window size
        '''

        cell_size = self.sphere_circumference / self.n_lon    # The exact cell size (~ resolution)

        reach = int(search_radius // cell_size)    # Numbers of cells left/right/above/below a cell

        # Find the maximum reach required for specified search radius
        # Ensuring the reach covers the whole search radius (needed due to shifted nature of window)
        if search_radius % cell_size != 0:
            reach += 1

        window_size = 2 * reach + 1    # Cells on the left and right of cell, plus cell itself

        # Find first and last cell whose windows are entirely contained by the raster
        start_cell_index = self.key[reach]['start_index']                   # Bottom border
        stop_cell_index = self.key[len(self.key) - reach]['start_index']    # Top border

        density_layer = []

        # Bottom border zero padding
        for cell_index in range(start_cell_index):
            density_layer.append(0)

        # Compute point density for relevant cells
        for cell_index in tqdm(range(start_cell_index, stop_cell_index)):    # Use tqdm module to display a progress bar
            density = self._compute_cell_point_density(cell_index, search_radius, window_size)
            density_layer.append(density)

        # Top border zero padding
        for cell_index in range(stop_cell_index, self.k):
            density_layer.append(0)

        # Save layer to dictionary
        self.layers['density'] = density_layer


    def classify_layer(self, layer, n_quantiles, bins=None):
        '''Classify a layer using quantile classification and store it as a new layer

        (This function classifies a layer into n classes of equal numbers of cell values.
        A new layer with the suffix "_classified" is created and added to the layer.
        Since we sometimes want to use a predefined classification scheme we can optionally
        define bin values, to be used instead of calculating new quantiles. If we do that,
        however, the classification ceiling needs to be adjusted if the layer contains a
        value larger than that.)

        Args:
            layer (str): Name (str) of layer (list) to be classified
            n_quantiles (int): Number of quantiles (number of classes)
            bins (list): Bin values to overwrite the classification (optional)

        Returns:
            bins (list): Bin values used for the applied classification
        '''

        layer_raw = np.array(self.layers[layer])    # Create a numpy array to use numpy array methods

        # Use user-specified bins if given as argument
        if bins == None:
            bins = []

            layer_nozeros = np.sort(layer_raw[layer_raw != 0])    # Omit zero values for quantile calculation

            # Find the values of the first and last cell of each bin
            for i in range(1, n_quantiles):
                index = (len(layer_nozeros) // n_quantiles) * i
                value = layer_nozeros[index]
                bins.append(value)

            bins.insert(0, layer_nozeros[0])    # lowest value
            bins.append(layer_nozeros[-1])      # highest value

        else:
            bins = bins    # Use predefined bins if available

            if max(self.layers[layer]) > bins[-1]:    # Check if highest value covered by predefined bins
                bins[-1] = max(self.layers[layer])    # Adjust if needed

        # Classify into bins
        layer_classified = np.digitize(layer_raw, bins)

        # Save layer to dictionary
        self.layers[layer + '_classified'] = layer_classified

        # Return bins to use in other classifications
        return bins


    def save_layer(self, layername, filepath):
        '''Save layer as text file in specified path

        Args:
            layername (str): Name (str) of layer (list) to be saved
            filepath (str): System path of layer file to be saved (.txt)
        '''

        with open(filepath, 'w') as f:
            for item in self.layers[layername]:
                f.write("%s\n" % item)

        print(f'Layer saved: {filepath}')


    def load_layer(self, layername, filepath):
        '''Load layer text file from specified path

        Args:
            layername (str): Name (str) of layer (list) to be loaded
            filepath (str): System path of layer file to be loaded (.txt)
        '''

        layer = []

        with open(filepath, 'r') as f:
            for line in f:
                value = line[:-1]
                if value == 'None':    # TODO: Check if necessary
                    layer.append(None)
                else:
                    layer.append(float(value))

        self.layers[layername] = layer

        print(f'Layer loaded: {filepath}')


    def load_boundaries(self, filepath):
        '''Load border point data to define set of cells to be marked as boundary

        (Border points can be generated using GIS software, by generating points
        along polyline features. The point coordinates can subsequently be exported
        as GeoJSON and loaded into this class. This is the easiest way, as the
        existing code structure can be used to handle the point data.)

        Args:
            filepath (str): System path of point data (.geojson)
        '''

        f = open(filepath, 'r')
        data = json.load(f)
        f.close()

        # Iterate through all points and note down the cell it is located in
        for i in data['features']:
            coordinates = i['geometry']['coordinates']
            point = Point(coordinates[0], coordinates[1])
            cell = self._compute_point_cell_location(point)
            self.ocean_border_cells.append(cell)

        self.ocean_border_cells = set(self.ocean_border_cells)    # Make cells unique and also faster for look-ups

        print(f'Boundaries loaded: {filepath}')


    def create_image_from_layer(self, layername, filepath):
        '''Create a PNG image from a classified layer

        (This function projects the 3D layer onto 2D and defines a color value for
        each pixel according to the classification done in a earlier step. Using
        these color values, a PNG image is generated using the Pillow (PIL) module.

        The projection is done by taking the equator pixels as reference. A centroid
        of every equatorial pixel is therefore created. These centroids can then be shifted
        vertically onto every row of the raster, creating a new 2D raster with equal
        numbers of pixels in each row. To assign a value to each 2D pixel, the value is
        looked up in the 3D layer by first finding the corresponding 3D cell index of
        the new 2D pixels (centroid coordinates) and then indexing it in the 3D layer.
        The closer a 2D pixel is to the north or south pole, the higher the chance is,
        that multiple 2D pixels share a value in the 3D raster, creating a distortion
        in the 2D image.)

        Args:
            layername (str): Name (str) of layer (list) to be saved as image
            filepath (str): System path of image file to be saved (.png)
        '''

        # Color palette according to Acheson et al. 2017 (for 5 classes only!; extend if needed)
        # Sixth class was added because the density value equal to the classification ceiling is classed as n+1 (however: class 6 = class 5)
        color_key = {0: (255,255,255), 1: (252,208,163), 2: (253,174,107), 3: (241,105,19), 4: (215,72,1), 5: (141,45,3), 6: (141,45,3)}

        # Equatorial cell as reference for pixel size of final image
        equatorial_row = int(len(self.key) / 2)
        equatorial_row_key = self.key[equatorial_row]
        equatorial_start_index = equatorial_row_key['start_index']
        equatorial_cell_width = equatorial_row_key['cell_width']
        equatorial_cell_quantity = equatorial_row_key['cell_quantity']

        centroid_lon_list = []

        # Store longitudes of centeroids of all equatorial cells in a list
        for cell in range(equatorial_start_index, equatorial_start_index + equatorial_cell_quantity):
            equatorial_centroid = self._compute_cell_centroid(cell, row=equatorial_row)
            centroid_lon_list.append(equatorial_centroid.lon)

        # Move left edge of image to Pacific Ocean (180°) instead of Europe (0°) 
        centroid_lon_list = centroid_lon_list[len(centroid_lon_list) // 2:] + centroid_lon_list[:len(centroid_lon_list) // 2]

        image_raster = []

        # Iterate through each row and create centroid latitudes
        for i, row in enumerate(self.key):
            centroid_lat = (i * self.cell_height) + (self.cell_height / 2) - 90

            image_row = []

            # Iterate through each equatorial longitude centroid and map it onto the raster
            for j, centroid_lon in enumerate(centroid_lon_list):
                centroid = Point(centroid_lon, centroid_lat)                # Centroid of 2D pixel
                cell_index = self._compute_point_cell_location(centroid)    # Cell index in 3D raster
                pixel_value = self.layers[layername][cell_index]            # Pixel value from 3D layer

                # If pixel represents the boundary, it is assigned the color black
                if cell_index in self.ocean_border_cells:
                    rgb = (0,0,0)    # black
                else:
                    rgb = color_key[pixel_value]                            # Color value of 2D pixel based on value from 3D layer
                image_row.append(rgb)

            image_raster.insert(0, image_row)

        # Compile image
        image_array = np.array(image_raster, dtype=np.uint8)
        image = Image.fromarray(image_array)
        image.save(filepath)

        print(f'Image saved: {filepath}')
        