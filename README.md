# gw-mod-util
Utilities to support the mapping of groundwater age

These utilities

* Genmod_Utilities contains a class (SourceProcessing) that supports creation of raster images from finite difference model grids, changing from model row/column coordinates to real world coordinate systems. Can be tested with GIS_test.ipynb
* fit_parametric_distributions.py contains functions for reading MODPATH particle endpoint files and fitting various age disitrbutions to particle ages.
* RTD_util cotains a class (RTD_util) that is for creating particle input files with particle placement in proportion to cell volume or cell flux; also handles transient particle placement.
* xgb_util contains a class (Model) that encapsulates the workflow for using XGBoost, inlcuding grid search and cross validation.
