import pandas as pd
import numpy as np
import os
import sys


class ReadVar(object):

    def __init__(self, target, data_dir = '../model output', list_print=True):
        self.target = target
        self.data_dir = data_dir
        self.list_print = list_print
        
        self._read_variables()

    def _print_results(self, tdf, label):
        print('{} file has {} rows and {} columns\n'.format(label, *tdf.shape))
        dl = list(tdf.columns)
        for a,b,c,d,e in zip(dl[::5],dl[1::5],dl[2::5],dl[3::5],dl[4::5]):
            print ('{:<20} {:20} {:<20} {:<20} {:<}'.format(a,b,c,d,e))
        print()
    
    def _read_variables(self):      
        self.pred_var = ['fraction_2_center',
               'DSD1', 'LP1', 'DSD2','LP2', 'DSD3', 'LP3', 'DSD4', 'LP4', 'DSD5', 'LP5', 
               'DSD6', 'LP6', 'DSD7', 'LP7', 'DSD8', 'LP8', 'DSD9', 'LP9', 
               'CrseStratSed', 'median_meters', 'median_Reitz', 'tau', 
               'Terrane_1A', 'Terrane_1B', 'Terrane_1C', 'Terrane_1D', 'Terrane_1E', 'Terrane_1F',
               'Terrane_1G', 'Terrane_2A', 'Terrane_2B', 'Terrane_2C', 'Terrane_2D',
               'Terrane_2E', 'Terrane_3A', 'Terrane_3B', 'Terrane_3C', 'Terrane_4B']
        
        src = os.path.join(self.data_dir, 'all_data_no_tau_limit.csv')
        df = pd.read_csv(src, index_col='model')
        df.columns = [item.lower() for item in df.columns]
        self.pred_var = [item.lower() for item in self.pred_var]
        
        # models eliminated because of poor mass balance (abs(mass balance error) > 2% or > 1%)
        delete_models = ['Pas13_MinnesotaArrowhead', 'Pas06_UpperMilk', 'LittleNemaha'] # > 2%
        # delete_models = ['Pas13_MinnesotaArrowhead','Pas06_UpperMilk','LittleNemaha', 'Pas04_FlatheadLake', 'Poultney'] # > 1%
        df.drop(index=delete_models, inplace=True)
        
        all_var = self.pred_var.copy()
        all_var.append(self.target)
        
        if 'fit' in self.target:
            df = df.iloc[np.where(df.err < 0.3)].copy()
        
        index = np.isfinite(df[all_var]).all(axis=1)
        self.data = df[index].copy()
        self.features = self.data[self.pred_var].copy()
        self.labels = self.data[[self.target]].copy()
        
        if self.list_print:
            self._print_results(self.data, 'Data')
            self._print_results(self.features, 'Features')
            self._print_results(self.labels, 'Labels')