import numpy
import plotly.graph_objects as go

class Point():
    
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        
    def __repr__(self):
        return f'Point(x = {self.x}, y = {self.y})'


class DiscoBall():

    def __init__(self, res=10):

        self.sphere_radius = 6371
        self.sphere_area = 4 * numpy.pi * 6371**2

        self.n_lat = (180 / res) - ((180 / res) % 2)
        self.n_lon = 2 * self.n_lat

        self.cell_height = 180 / self.n_lat

        self.raster = []
        self.key = []

        self._compute_cells()

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

        s = numpy.sqrt(s / (k - 1))

        return m, s, k


    def _compute_cell_area(self, ll, ur):
        """Returns the area of a cell [km2] given the lower-left and upper-right coordinate on a sphere"""

        # Maths on surface integral: https://www.sharetechnote.com/html/Calculus_Integration_Surface.html

        d_theta = abs(numpy.radians(ll.x) - numpy.radians(ur.x))    # longitudinal arc length
        d_phi = abs(numpy.radians(ll.y) - numpy.radians(ur.y))    # latitudinal arc length

        phi = numpy.radians(((ll.y + ur.y) / 2) * (-1) + 90)    # average latitudinal angle of cell (north pole = 0°; south pole = 180°)

        area = numpy.sin(phi) * d_phi * d_theta * self.sphere_radius**2

        return area


    def _compute_cells(self):
        """Computes raster geometry based on resolution and creates a corresponding empty raster and geometry key"""

        equatorial_cell_width = 360 / self.n_lon
        equatorial_cell_area = self._compute_cell_area(Point(0, 0), Point(equatorial_cell_width, self.cell_height))

        row_lls = []

        for i in numpy.linspace(0, 90, int(self.n_lat / 2), endpoint=False):
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
                self.raster.append([])
                cell_index += 1

            start_index = cell_index - cell_quantity

            self.key.append({'start_index':start_index, 'cell_width':cell_width, 'cell_area':cell_area, 'cell_quantity':cell_quantity})


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
                n *= -1

        return row


    def _compute_centroid(self, cell_index):
        """Returns the centroid (point) of a cell given a cell index"""

        row = self._compute_row(cell_index)

        center_x = ((cell_index - self.key[row]['start_index']) * self.key[row]['cell_width']) + (self.key[row]['cell_width'] / 2)
        center_y = row * self.cell_height + (self.cell_height / 2) - 90

        centroid = Point(center_x, center_y)

        return centroid


    def _compute_window(self, cell_index, size):
        """Returns a list of cell indices of a size-by-size window around a give cell index"""

        window = []

        reach = int(size // 2)

        row = self._compute_row(cell_index)

        centroid = self._compute_centroid(cell_index)

        center_x, center_y = centroid.x, centroid.y

        if (row - reach) < 0 or (row + reach + 1) > len(self.key):
            raise IndexError('Entire window must be inside of raster')

        for i in range(row - reach, row + reach + 1):
            center_cell = int((center_x // self.key[i]['cell_width']) + self.key[i]['start_index'])
            for j in range(center_cell - reach, center_cell + reach + 1):
                if j < self.key[i]['start_index']:
                    j += self.key[i]['cell_quantity']
                window.append(j)

        return window


    def add_point(self, point):
        """Find cell for point location and add point to raster accordingly"""

        row_index = int((point.y + 90) // self.cell_height)
        cell_index = int(self.key[row_index]['start_index'] + (point.x // self.key[row_index]['cell_width']))

        self.raster[cell_index].append(point)


    def display(self, height=500):
        """Visualize raster and containing points using plotly (for development purposes; don't use if resolution too high)"""

        data = []

        window_elements = []

        for i, cell in enumerate(self.raster):
            if len(cell) != 0:
                window_elements += self._compute_window(i, 5)

        for i, row in enumerate(self.key):
            for j in range(row['cell_quantity']):
                cell_index = row['start_index'] + j
                if len(self.raster[cell_index]) == 0:
                    if cell_index in window_elements:
                        color = 'green'
                    else:
                        color = 'grey'
                else:
                    color = 'red'
                    for point in self.raster[cell_index]:
                        data.append(go.Scattergeo(lat = [point.y], lon = [point.x],
                            mode = 'markers', marker = dict(size = 5, color='black')))

                left = row['cell_width'] * j
                right = row['cell_width'] * (j + 1)
                bottom = self.cell_height * i - 90
                top = self.cell_height * (i + 1) - 90

                lat = [bottom, top, top, bottom, bottom]
                lon = [left, left, right, right, left]

                data.append(go.Scattergeo(lat = lat, lon = lon,
                    mode = 'lines', line = dict(width = 1, color=color), fill='toself'))

        fig = go.Figure(data=data)

        fig.update_geos(projection_type="orthographic")
        fig.update_layout(height=height, margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)

        fig.show()

        print(self)
