# gw-mod-util
Utilities to support the mapping of groundwater age

These utilities

* Genmod_Utilities contains a class (SourceProcessing) that supports creation of raster images from finite difference model grids, changing from model row/column coordinates to real world coordinate systems. Can be tested with GIS_test.ipynb. The class contains GDAL functions (which do not rely on ArcPy).  You can use the class to create and plot a new raster image (i.e. model grid) from existing vector (e.g. shapefiles) or raster (e.g. GeoTiff or Arc grid) data. The source data is either vectorized (vector data) or reprojected and resampled (raster data).  You can specify the size, location, orientation, and pixel size of the output raster or you can read an existing raster to use as a template.  The Class also has functions to transform between model (row, column) and real-world coordinates. It can be imported into any Python script or Notebook. 
* fit_parametric_distributions.py contains functions for reading MODPATH particle endpoint files and fitting various age disitrbutions to particle ages.
* RTD_util cotains a class (RTD_util) that is for creating particle input files with particle placement in proportion to cell volume or cell flux; also handles transient particle placement.
* xgb_util contains a class (Model) that encapsulates the workflow for using XGBoost, inlcuding grid search and cross validation.

