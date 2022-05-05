import numpy as np
import plotly.graph_objects as go

class Point():
    
    def __init__(self, x=None, y=None):

        self.x = x
        self.y = y
        
    def __repr__(self):

        return f'Point(x = {self.x}, y = {self.y})'


    def distance(self, other):

        r = 6371
        phi1 = np.radians(self.y) # latitudes
        phi2 = np.radians(other.y)
        lam1 = np.radians(self.x) # longitudes
        lam2 = np.radians(other.x)

        d = 2 * r * np.arcsin(np.sqrt(np.sin((phi2 - phi1) / 2)**2 +
            np.cos(phi1) * np.cos(phi2) * np.sin((lam2 - lam1) / 2)**2))

        return d        


class DiscoBall():

    def __init__(self, res=10):

        self.sphere_radius = 6371
        self.sphere_area = 4 * np.pi * 6371**2
        self.sphere_circumference = 2 * np.pi * 6371

        self.n_lat = (180 / res) - ((180 / res) % 2)
        self.n_lon = 2 * self.n_lat

        self.cell_height = 180 / self.n_lat

        self.key = []
        self.bucket = []

        self.layer_density = []

        self._compute_raster()

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
        d_phi = abs(np.radians(ll.y) - np.radians(ur.y))    # latitudinal arc length

        phi = np.radians(((ll.y + ur.y) / 2) * (-1) + 90)    # average latitudinal angle of cell (north pole = 0°; south pole = 180°)

        area = np.sin(phi) * d_phi * d_theta * self.sphere_radius**2

        return area


    def _compute_raster(self):
        """Computes raster geometry based on resolution and creates a corresponding empty raster and geometry key"""

        equatorial_cell_width = 360 / self.n_lon
        equatorial_cell_area = self._compute_cell_area(Point(0, 0), Point(equatorial_cell_width, self.cell_height))

        row_lls = []

        for i in np.linspace(0, 90, int(self.n_lat / 2), endpoint=False):
            row_lls.append(Point(0, i))
            row_lls.insert(0, Point(0, - i - self.cell_height))

        cell_index = 0

        for ll in row_lls:

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

        if cell_index == None:
            cell_index = self._compute_point_cell_location(point)

        self.bucket[cell_index].append(point)


    def _compute_row(self, cell_index):
        """Returns row index given a cell index"""

        row = int(((((cell_index * self.area_mean) / self.sphere_area) / 2) * 360) // self.cell_height)

        m = 1
        n = 1

        while not (cell_index >= self.key[row]['start_index'] and cell_index < (self.key + [{'start_index':self.k}])[row + 1]['start_index']):
            if not (row + (m * n) < 0 or row + (m * n) > len(self.key) -1):
                row += (m * n)
                m += 1
                n *= -1
            else:
                m -= 1

        return row


    def _compute_centroid(self, cell_index, row=None):
        """Returns the centroid (point) of a cell given a cell index"""

        if row == None:
            row = self._compute_row(cell_index)

        centroid_x = ((cell_index - self.key[row]['start_index']) * self.key[row]['cell_width']) + (self.key[row]['cell_width'] / 2)
        centroid_y = row * self.cell_height + (self.cell_height / 2) - 90

        centroid = Point(centroid_x, centroid_y)

        return centroid


    def _compute_window(self, cell_index, window_size, row=None, centroid=None):
        """Returns a list of cell indices of a size-by-size window around a give cell index"""

        if row == None:
            row = self._compute_row(cell_index)

        if centroid == None:
            centroid = self._compute_centroid(cell_index, row=row)

        window = []

        reach = int(window_size // 2)

        if (row - reach) < 0 or (row + reach + 1) > len(self.key):
            raise IndexError('Entire window must be inside of raster')

        for i in range(row - reach, row + reach + 1):
            center_cell = int((centroid.x // self.key[i]['cell_width']) + self.key[i]['start_index'])
            for j in range(center_cell - reach, center_cell + reach + 1):
                if j < self.key[i]['start_index']:
                    j += self.key[i]['cell_quantity']
                if j < self.k:
                    window.append(j)

        window = list(set(window))

        return window


    def _compute_point_density(self, cell_index, search_radius, window_size):
        """Returns point density given a cell index, a search radius and a corresponding window size"""

        # Density Equation: https://doc.arcgis.com/en/insights/latest/analyze/calculate-density.htm

        row = self._compute_row(cell_index)
        centroid = self._compute_centroid(cell_index, row=row)
        window = self._compute_window(cell_index, window_size, row=row, centroid=centroid)
        points = []

        for cell_index in window:
            for point in self.bucket[cell_index]:
                points.append(point)

        s = 0

        for point in points:
            distance = centroid.distance(point)
            if distance <= search_radius:
                a = (1 - (distance / search_radius)**2)**2
                b = a * 3 / np.pi

                s += b

        rho = (1 / search_radius**2) * s

        return rho


    def compute_point_density_layer(self, search_radius, normalize=True):
        """Creates a point density raster in the form of the main raster"""

        cell_size = self.sphere_circumference / self.n_lon

        reach = int(search_radius // cell_size)

        if search_radius % cell_size != 0:
            reach += 1

        window_size = 2 * reach + 1

        start_cell_index = self.key[reach]['start_index']
        stop_cell_index = self.key[len(self.key) - reach]['start_index']

        start_cells = []
        middle_cells = []
        end_cells = []

        for cell_index in range(start_cell_index):
            start_cells.append(None)

        for cell_index in range(start_cell_index, stop_cell_index):
            density = self._compute_point_density(cell_index, search_radius, window_size)
            middle_cells.append(density)

        for cell_index in range(stop_cell_index, self.k):
            end_cells.append(None)

        if normalize:
            middle_cells = np.array(middle_cells)
            middle_cells = middle_cells / middle_cells.max()
            middle_cells = middle_cells.tolist()

        self.layer_density = start_cells + middle_cells + end_cells


    def display_density_layer(self, height=500):
        """Visualize layer and points using plotly (for development purposes; don't use if resolution too high)"""

        data = []

        for i, row in enumerate(self.key):
            for j in range(row['cell_quantity']):

                cell_index = row['start_index'] + j

                if len(self.bucket[cell_index]) != 0:

                    for point in self.bucket[cell_index]:
                        data.append(go.Scattergeo(lat = [point.y], lon = [point.x],
                            mode = 'markers', marker = dict(size = 5, color='black')))

                if self.layer_density[cell_index] != None:
                    opacity = self.layer_density[cell_index]
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
