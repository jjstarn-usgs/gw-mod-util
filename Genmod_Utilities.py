import pysal as ps
import numpy as np
import pandas as pd
import os
import ast
from shutil import copyfile
import gdal
gdal.UseExceptions()
import ogr
import osr


class SourceProcessing(object):
    """
    Geospatial functions for use with rectangular grid finite-difference models. To properly
    initialize, you must use either read_raster or create_model_grid.
    """
    
    def __init__(self, nodata=-9999):
        self.nodata = nodata
        
    def read_raster(self, src_pth):
        assert os.path.exists(src_pth), 'raster source does not exist'
        self.src = gdal.Open(src_pth)
        band = self.src.GetRasterBand(1)
        
        self.nrow = int(self.src.RasterYSize)
        self.ncol = int(self.src.RasterXSize)
        self.gt = self.src.GetGeoTransform()
        self.output_raster_prj = self.src.GetProjection()
        self.old_array = band.ReadAsArray()  
        
        src = None
            
    def create_model_grid(self, theta, origin, LX, LY, nrow, ncol, output_raster_proj):
        '''theta is rotation from positive x axis in radians
        origin is a tuple of projected coordinates of the upper left corner
        LX and LY are grid cell dimensions'''
        self.theta = theta
        self.origin = origin      
        A = LX * np.cos(theta)
        B = LY * np.sin(theta)
        D = LX * np.sin(theta)
        E = LY * -np.cos(theta)
        
        self.nrow = int(nrow)
        self.ncol = int(ncol)
        self.gt = [origin[0], A, B, origin[1], D, E]
        self.output_raster_prj = output_raster_proj
        self.old_array = np.zeros((self.nrow, self.ncol))
        
    def _make_transforms(self):
        assert len(self.gt) == 6, 'geotransformation list must exist and have 6 elements'
        # format the geotransformation list into an affine transformation matrix))
        forward_transform = np.array(self.gt).reshape(2, -1)
        # add a row to get homogeneous coodinates (offsets are in the first column)
        self.forward_transform = np.vstack((forward_transform, [1, 0, 0]))
        # invert the forward transform
        self.reverse_transform = np.linalg.inv(self.forward_transform)
        
    def prj_coords_to_array_coords(self, x, y):
        _make_transforms()
        # reverse transform the real-world coordinate to pixel coordinates (row, column)
        assert x.shape[0] == y.shape[0], 'x and y have to have the same dimensions'
        ones = np.ones(x.shape[0])
        wpts = np.column_stack((x, y, ones))
        wpp = self.reverse_transform.dot(wpts.T)
        return wpp[1:,].T
        
    def array_coords_to_prj_coords(self, r, c):
        _make_transforms()
       # reverse transform cell-center coordinates to projected coordinates
        assert r.shape[0] == c.shape[0], 'r and c have to have the same dimensions'
        ones = np.ones(r.shape[0])
        wpts = np.column_stack((ones, c, r))
        dat = self.forward_transform.dot(wpts.T).T
        return dat[:, :-1]

    def process_raster_data(self, src, method, conversion=1.0):
        '''
        Takes a raster data source (ESRI grid, GeoTiff, .IMG and many other formats)
        and returns a numpy array. Arrangment of pixels is given as input and may 
        correspond to a MODFLOW grid.
        
        src : string
            complete path to raster data source
        method : string
            gdal method for interpolation. Choices are:
                gdal.GRA_NearestNeighbour 
                    Nearest neighbour (select on one input pixel)
                gdal.GRA_Bilinear
                    Bilinear (2x2 kernel)
                gdal.GRA_Cubic
                    Cubic Convolution Approximation (4x4 kernel)
                gdal.GRA_CubicSpline
                    Cubic B-Spline Approximation (4x4 kernel)
                gdal.GRA_Lanczos
                    Lanczos windowed sinc interpolation (6x6 kernel)
                gdal.GRA_Average
                    Average (computes the average of all non-NODATA contributing pixels)
                gdal.GRA_Mode
                    Mode (selects the value which appears most often of all the sampled points)
                gdal.GRA_Max
                    Max (selects maximum of all non-NODATA contributing pixels)
                gdal.GRA_Min
                    Min (selects minimum of all non-NODATA contributing pixels)
                gdal.GRA_Med
                    Med (selects median of all non-NODATA contributing pixels)
                gdal.GRA_Q1
                    Q1 (selects first quartile of all non-NODATA contributing pixels)
                gdal.GRA_Q3
                    Q3 (selects third quartile of all non-NODATA contributing pixels)

        conversion : float
            factor to be applied to raw data values to change units

        returns:
        2D array of raster data source projected onto model grid. 
        Returns a zero array with the correct shape if the source does not exist.
        '''
        if os.path.exists(src):
            rast = gdal.Open(src)
            dest = self._make_grid()
            
            gdal.ReprojectImage(rast, dest, rast.GetProjection(), self.output_raster_prj, method)
 
            grid = dest.GetRasterBand(1).ReadAsArray()
            grid = grid * conversion

            dest = None
            rast = None
 
        else:
            grid = np.ones((self.nrow, self.ncol)) * self.nodata
            print('Data not processed for\n{}\n Check that the file exists and path is correct'.format(src))
            
        self.new_array = grid

    def process_vector_data(self, src, attribute):
        '''
        Takes a vector data source (ESRI shapefile) and returns a numpy array.
        Arrangement of pixels is given as input and may correspond to a MODFLOW grid.

        src : complete path to vector data source
        attribute : field in data table to assign to rasterized pixels
        
        returns:
        2D array of vector data source projected onto model grid.
        Returns a zero array with the correct shape if the source does not exist.
        '''
        if os.path.exists(src):
            datasource = ogr.Open(src)
            layer = datasource.GetLayer()

            dest = self._make_grid()
            args = 'ATTRIBUTE={}'.format(attribute)
            gdal.RasterizeLayer(dest, [1], layer, options = [args])

            grid = dest.GetRasterBand(1).ReadAsArray()

            src = None
            dst = None      
            
        else:
            grid = np.ones((self.nrow, self.ncol)) * self.nodata
            print('Data not processed for\n{}\n Check that the file exists and path is correct'.format(src))
            
        self.new_array = grid

    def write_raster(self, dst_file):
        '''
        Writes numpy array to a GeoTiff file.
        
        dst_file : name of file to write
        data : 2D numpy array
        NCOL, NROW : number of rows and columns. These may coincide with a MODFLOW grid.
        gt : 6-element geotransform list [C, A, B, F, E, D]. Gives the coordinates of one pixel
            (the upper left pixel). If there is no rotation, B=D=0. If cells are square, A=-E.   
            Letter designations come from the original documentation.
            
            C = x coordinate in map units of the upper left corner of the upper left pixel
            A = distance from C along x axis to upper right pixel corner of the upper left pixel
            B = distance from C along x axis to lower left pixel corner of the upper left pixel,
            F = y coordinate in map units of the upper left corner of the upper left pixel
            E = distance from C along y axis to lower left pixel corner of the upper left pixel
            D = distance from C along y axis to upper right pixel corner of the upper left pixel
            
        proj : projection of the GeoTiff
        nodata : value to use as missing data in the GeoTiff
        '''
        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create(dst_file, self.ncol, self.nrow, 1, gdal.GDT_Float32)
        dst.SetGeoTransform(self.gt)
        dst.SetProjection(self.output_raster_prj)
        band = dst.GetRasterBand(1)
        band.SetNoDataValue(self.nodata)
        band.WriteArray(self.new_array)
        dst = None

    def _make_grid(self):  
        '''
        Creates a blank raster image in memory.
            
        NCOL, NROW : number of rows and columns. These may coincide with a MODFLOW grid.
        gt : 6-element geotransform list [C, A, B, F, E, D]. Gives the coordinates of one pixel
            (the upper left pixel). If there is no rotation, B=D=0. If cells are square, A=-E.   
            Letter designations come from the original documentation.
            
            C = x coordinate in map units of the upper left corner of the upper left pixel
            A = distance from C along x axis to upper right pixel corner of the upper left pixel
            B = distance from C along x axis to lower left pixel corner of the upper left pixel,
            F = y coordinate in map units of the upper left corner of the upper left pixel
            E = distance from C along y axis to lower left pixel corner of the upper left pixel
            D = distance from C along y axis to upper right pixel corner of the upper left pixel
            
        shapeproj : projection of the GeoTiff
        nodata : value to use as missing data in the GeoTiff
        '''
        mem_drv = gdal.GetDriverByName('MEM')
        grid_ras = mem_drv.Create('', self.ncol, self.nrow, 1, gdal.GDT_Float32)
        grid_ras.SetGeoTransform(self.gt)
        grid_ras.SetProjection(self.output_raster_prj)
        band = grid_ras.GetRasterBand(1)
        band.SetNoDataValue(self.nodata)
        self.new_array = np.zeros((self.nrow, self.ncol))
        band.WriteArray(self.new_array)
        return grid_ras
 
    def make_clockwise(self, coords):
        '''
        Function to determine direction of vertices of a polygon (clockwise or CCW).
        Probably not needed, but here just in case. 
        
        coords : array with dim (n, 2)
                n is number of vertices in the polygon. The last vertex is the same 
                as the first to close the polygon. The first column is x and the second is y.
        '''
        # if the points are counterclockwise, reverse them
        x1 = coords[:-1, 0]
        x2 = coords[1:, 0]
        y1 = coords[:-1, 1]
        y2 = coords[1:, 1]
        ccw = np.sum((x2 - x1) * (y2 + y1)) < 0
        if ccw:
            coords = np.flipud(coords)
            print('yup, coordinates are ccw')
            print("let's change them to CW")
        return coords

    # test data for make_clockwise

    # print('clockwise')
    # x = np.array([1, 1, 2, 2, 1])
    # y = np.array([1, 2, 2, 1, 1])
    # coords = np.array(zip(x, y))
    # c = make_clockwise(coords)
    # print( c)
    # print('\n')
    # print('CCW')
    # x = np.array([1, 2, 2, 1, 1])
    # y = np.array([1, 1, 2, 2, 1])
    # coords = np.array(zip(x, y))
    # c = make_clockwise(coords)
    # print( c)


    def dbf2df(self, dbf_path, index=None, cols=False, incl_index=False):
        '''
        Read a dbf file as a pandas.DataFrame, optionally selecting the index
        variable and which columns are to be loaded.

        __author__  = "Dani Arribas-Bel <darribas@asu.edu> "
        ...

        Arguments
        ---------
        dbf_path    : str
                      Path to the DBF file to be read
        index       : str
                      Name of the column to be used as the index of the DataFrame
        cols        : list
                      List with the names of the columns to be read into the
                      DataFrame. Defaults to False, which reads the whole dbf
        incl_index  : Boolean
                      If True index is included in the DataFrame as a
                      column too. Defaults to False

        Returns
        -------
        df          : DataFrame
                      pandas.DataFrame object created
        '''
        db = ps.open(dbf_path)
        if cols:
            if incl_index:
                cols.append(index)
            vars_to_read = cols
        else:
            vars_to_read = db.header
        data = dict([(var, db.by_col(var)) for var in vars_to_read])
        if index:
            index = db.by_col(index)
            db.close()
            return pd.DataFrame(data, index=index)
        else:
            db.close()
            return pd.DataFrame(data)

