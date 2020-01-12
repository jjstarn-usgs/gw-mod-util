import pandas as pd
import numpy as np
import os
import sys


class ReadVar(object):

    def __init__(self, target, homes = ['../Models'], data_dir = '../Figures', list_print=True):
        self.target = target
        self.homes = homes
        self.data_dir = data_dir
        self.list_print = list_print
        
        self._read_variables()

    def _print_results(self, tdf, label):
        print('{} file has {} rows amd {} columns\n'.format(label, *tdf.shape))
        dl = list(tdf.columns)
        for a,b,c,d,e in zip(dl[::5],dl[1::5],dl[2::5],dl[3::5],dl[4::5]):
            print ('{:<20} {:20} {:<20} {:<20} {:<}'.format(a,b,c,d,e))
        print()
    
    def _read_variables(self):      
        # self.pred_var = ['top', 'fraction_2_bottom', 
               # 'DSD1', 'LP1', 'DSD2',
               # 'LP2', 'DSD3', 'LP3', 'DSD4', 'LP4', 'DSD5', 'LP5', 'DSD6', 'LP6',
               # 'DSD7', 'LP7', 'DSD8', 'LP8', 'DSD9', 'LP9', 'CrseStratSed',
               # 'median_dtw', 'median_meters', 'median_Reitz', 
               # 'median_StreamDensity', 'NEAR_DIST', 'tau', 'Terrane_1A',
               # 'Terrane_1B', 'Terrane_1C', 'Terrane_1D', 'Terrane_1E', 'Terrane_1F',
               # 'Terrane_1G', 'Terrane_2A', 'Terrane_2B', 'Terrane_2C', 'Terrane_2D',
               # 'Terrane_2E', 'Terrane_3A', 'Terrane_3B', 'Terrane_3C', 'Terrane_4B']
        self.pred_var = ['top', 'fraction_2_bottom', 
               'DSD1', 'LP1', 'DSD2',
               'LP2', 'DSD3', 'LP3', 'DSD4', 'LP4', 'DSD5', 'LP5', 'DSD6', 'LP6',
               'DSD7', 'LP7', 'DSD8', 'LP8', 'DSD9', 'LP9', 'CrseStratSed',
               'median_meters', 'median_Reitz', 
               'tau', 'Terrane_1A',
               'Terrane_1B', 'Terrane_1C', 'Terrane_1D', 'Terrane_1E', 'Terrane_1F',
               'Terrane_1G', 'Terrane_2A', 'Terrane_2B', 'Terrane_2C', 'Terrane_2D',
               'Terrane_2E', 'Terrane_3A', 'Terrane_3B', 'Terrane_3C', 'Terrane_4B']
        
        src = os.path.join(self.data_dir, 'all_data_no_tau_limit.csv')
        # src = os.path.join(self.fig_dir, 'all_data.csv')
        df = pd.read_csv(src, index_col=[0])
    
        df.replace(np.inf, np.nan, inplace=True)
        # remove rows that have no predicted value
        df = df.loc[~df[self.target].isna()]
        # make a row index to select rows with no missing values
        self.index = df.loc[:, self.pred_var].notna().all(axis=1)

        self.data = df.loc[self.index, :]
        self.features = df.loc[self.index, self.pred_var]
        self.labels = df.loc[self.index, [self.target]]
        
        if self.list_print:
            self._print_results(self.data, 'Data')
            self._print_results(self.features, 'Features')
            self._print_results(self.labels, 'Labels')