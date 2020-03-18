import os
import pandas as pd
import flopy as fp
import datetime as dt
import numpy as np
import scipy.stats as ss
import scipy.optimize as so

# import warnings
# warnings.filterwarnings("ignore", message="converting a masked element to nan")

class RTD_util(object):
    '''Class to perform various functions from setting up an age-based backtracking MODPATH simulation
    and analyzing the results'''
    
    def __init__(self, ml, weight_label, group):
        # import various variables from the MODFLOW model
        self.ml = ml
        self.model_ws = ml.model_ws
        self.oc = ml.get_package('OC')
        self.bas = ml.get_package('BAS6')
        self.dis = ml.get_package('DIS')
        self.upw = ml.get_package('UPW')
        self.namefile = ml.namefile
        self.l, self.r, self.c = np.indices((self.dis.nlay, self.dis.nrow, self.dis.ncol))
        self.prng = np.random.RandomState(9591029)
        self.delr = self.dis.delr
        self.delc = self.dis.delc
        self.nlay = self.dis.nlay
        self.nrow = self.dis.nrow
        self.ncol = self.dis.ncol
        self.bot = self.dis.getbotm()
        self.top = self.dis.gettop()
        
        self.hnoflo = self.bas.hnoflo
        self.hdry = self.upw.hdry
        self.ibound = np.asarray(self.bas.ibound.get_value())
        self.hk = np.asarray(self.upw.hk.get_value())
        self.vka = np.asarray(self.upw.vka.get_value())

        self.weight_label = weight_label
        self.group = group
        self.mpname = '{}_{}_{}'.format(self.ml.name, self.weight_label, self.group)
        
        self._len_mult()
        
        # Create dictionary of multipliers for converting model time units to days
        time_dict = dict()
        time_dict[0] = 1.0 # undefined assumes days
        time_dict[1] = 24 * 60 * 60
        time_dict[2] = 24 * 60
        time_dict[3] = 24
        time_dict[4] = 1.0
        time_dict[5] = 1.0
        self.time_dict = time_dict
    
    def _get_output_dfs(self):
        # Make dataframes of budget information
        bud_file_unit = np.unique(self.oc.get_budgetunit())
        assert len(bud_file_unit == 1), 'More than one budget file is used'
        bud_file_name = self.ml.get_output(unit=bud_file_unit[0])
        src = os.path.join(self.model_ws, bud_file_name)
        self.bud_obj = fp.utils.CellBudgetFile(src)
        all_bud_df = pd.DataFrame(self.bud_obj.recordarray)
        # convert to zero base
        all_bud_df['kper'] -= 1
        all_bud_df['kstp'] -= 1
        self.all_bud_df = all_bud_df
        
        head_file_name = self.ml.get_output(unit=self.oc.iuhead)
        src = os.path.join(self.model_ws, head_file_name)
        self.hd_obj = fp.utils.HeadFile(src)       

    def _get_kstpkper(self, mf_start_date_str = '01/01/1900', mp_release_date_str = '01/01/2018' ):   
        # Use calendar release date and MODFLOW start date to pick out head and budget
        # items from transient model output
        self._get_output_dfs()
        
        # convert string representation of dates into Python datetime objects
        self.mf_start_date = dt.datetime.strptime(mf_start_date_str , '%m/%d/%Y')
        self.mp_release_date = dt.datetime.strptime(mp_release_date_str , '%m/%d/%Y')
    
        # check to make sure they are valid
        assert self.mf_start_date < self.mp_release_date, 'The particle release date has \
        to be after the start of the MODFLOW simulation'
    
        # group by period and step
        kdf = self.all_bud_df.groupby(['kper', 'kstp']).median()
        kdf = kdf[['pertim', 'totim']]
    
        # make a datetime series for timesteps starting with 0
        # totim is elapsed time in simulation time
        units = self.dis.itmuni_dict[self.dis.itmuni][0]
        if units == 'd':
            units = 'D'
        end_date = self.mf_start_date + pd.to_timedelta(np.append(0, kdf.totim), unit=units)
        end_date = end_date.map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
        kdf.loc[:, 'start_date'] = end_date[0:-1]
        kdf.loc[:, 'end_date'] = end_date[1:]
    
        # make a datetime series for timesteps starting with 0
        # totim is elapsed time in simulation time
        # reformat the dates to get rid of seconds
        end_date = self.mf_start_date + pd.to_timedelta(np.append(0, kdf.totim), unit=units)
        kdf.loc[:, 'start_date'] = end_date[0:-1].map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
        kdf.loc[:, 'end_date'] = end_date[1:].map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
    
        # reference time and date are set to the end of the last stress period
        self.ref_time = kdf.totim.max()
        self.ref_date = end_date.max()
    
        # release time is calculated in tracking time (for particle release) and 
        # in simulation time (for identifying head and budget components)
        self.release_time_trk = np.abs((self.ref_date - self.mp_release_date).days)
        self.release_time_sim = (self.mp_release_date - self.mf_start_date).days
    
        # find the latest group index that includes the release date
        idx = (kdf.totim >= self.release_time_sim).idxmax()
        kdf.loc[idx, 'particle_release'] = True
    
        # switch period and step 
        self.kstpkper = (idx[1], idx[0])
    
        assert self.ref_date > self.mp_release_date, 'The reference date has \
        to be after the particle release'
    
    def get_heads(self):
        # Get the highest non-dry head in the 2D representation of the MODFLOW model
        # in each vertical stack of cells
        self._get_kstpkper()
        heads = self.hd_obj.get_data(kstpkper=self.kstpkper)
        hd = heads.copy()
        hd[np.isclose(self.bas.hnoflo, hd)] = np.nan
        hd[np.isclose(self.upw.hdry, hd, atol=10)] = np.nan
        self.hd = hd
    
    def get_watertable(self):
        # Get the highest non-dry head in the 2D representation of the MODFLOW model
        # in each vertical stack of cells
        self.get_heads()
        hin = np.argmax(np.isfinite(self.hd), axis=0)    
        self.water_table =  np.squeeze(self.hd[hin, self.r[0,:,:], self.c[0,:,:]])
    
    def make_particle_array(self, parts_per_cell):  
        # Given the number of desired particles per cell, return an array in the 
        # format of MODPATH starting location information
        if not hasattr(self, 'release_time_trk'):
            self._get_kstpkper()
            
        self.parts_per_cell = parts_per_cell
        lg = self.l.ravel()
        rg = self.r.ravel()
        cg = self.c.ravel()
        label = parts_per_cell
    
        lrep = np.repeat( lg, parts_per_cell.ravel() )
        rrep = np.repeat( rg, parts_per_cell.ravel() )
        crep = np.repeat( cg, parts_per_cell.ravel() )
        label = np.repeat( label, parts_per_cell.ravel() )
    
        self.num_parts = lrep.shape[0]
    
        # generate random relative coordinates within a cell in 3D
        cell_coords = self.prng.rand( self.num_parts, 3 )
    
        grp = 1
    
        particles = np.zeros( ( self.num_parts, 11 ) )
        particles[:, 0] = np.arange( 1, self.num_parts + 1 )
        particles[:, 1] = grp
        particles[:, 2] = 1
        particles[:, 3] = lrep + 1
        particles[:, 4] = rrep + 1
        particles[:, 5] = crep + 1
        particles[:, 6:9] = cell_coords
        particles[:, 9] = self.release_time_trk
        particles[:, 10] = label
        
        return particles
        
    def make_arbitrary_particle_array(self, seqnum, label, parts_per_cell=1000):
        # Given the number of desired particles per cell, return an array in the 
        # format of MODPATH starting location information
        if not hasattr(self, 'release_time_trk'):
            self._get_kstpkper()
            
        self.parts_per_cell = parts_per_cell
        lg = self.l.ravel()[seqnum]
        rg = self.r.ravel()[seqnum]
        cg = self.c.ravel()[seqnum]
    
        lrep = np.repeat( lg, parts_per_cell)
        rrep = np.repeat( rg, parts_per_cell)
        crep = np.repeat( cg, parts_per_cell)
        label = np.repeat( label, parts_per_cell)
    
        self.num_parts = lrep.shape[0]
    
        # generate random relative coordinates within a cell in 3D
        cell_coords = self.prng.rand( self.num_parts, 3 )
    
        grp = 1
    
        particles = np.zeros( ( self.num_parts, 11 ) )
        particles[:, 0] = np.arange( 1, self.num_parts + 1 )
        particles[:, 1] = grp
        particles[:, 2] = 1
        particles[:, 3] = lrep + 1
        particles[:, 4] = rrep + 1
        particles[:, 5] = crep + 1
        particles[:, 6:9] = cell_coords
        particles[:, 9] = self.release_time_trk
        particles[:, 10] = label
        
        return particles
    
    def write_starting_locations_file(self, particles):
        # Given a particle starting array, write a MODPATH starting location file with 
        # header information
        line = '{:5d}\n{:5d}\n'.format(1, 1)
        line = line + 'group_{}\n'.format(1)
        npart = particles.shape[0]
        line = line + '{:6d}'.format(npart)
        dst_pth = os.path.join(self.model_ws, '{}_{}_{}.loc'.format(self.ml.name, self.weight_label, self.group))
        form = '%6d %6d %3d %3d %3d %3d %12.9f %12.9f %12.9f %12.9e %15d'
        np.savetxt(dst_pth, particles, delimiter=' ', fmt=form, header=line, comments='')
        
    def run_MODPATH(self, por, mp_exe_name):
        # Run backtracking MODPATH simulation using a starting locations file
        # prepare Modpath files   
        SimulationType = 1              # 1 endpoint; 2 pathline; 3 timeseries
        TrackingDirection = 2           # 1 forward; 2 backward
        WeakSinkOption = 1              # 1 pass; 2 stop
        WeakSourceOption = 1            # 1 pass; 2 stop
        ReferemceTimeOption = 1         # 1 time value; 2 stress period, time step, relative offset
        StopOption = 2                  # 1 stop with simulation 2; extend if steady state 3; specify time
        ParticleGenerationOption = 2    # 1 automatic; 2 external file
        TimePointOption = 1             # 1 none; 2 number at fixed intervals; 3 array
        BudgetOutputOption = 3          # 1 none; 2 summary; 3 list of cells; 4 trace mode
        ZoneArrayOption = 1             # 1 none; 2 read zone array(s) 
        RetardationOption = 1           # 1 none; 2 read array(s) 
        AdvectiveObservationsOption = 1 # 1 none; 2 saved for all time pts 3; saved for final time pt

        options = [SimulationType, TrackingDirection, WeakSinkOption, WeakSourceOption, ReferemceTimeOption, 
                   StopOption, ParticleGenerationOption, TimePointOption, BudgetOutputOption, ZoneArrayOption, 
                   RetardationOption, AdvectiveObservationsOption]

        mpnf = '{}_{}_{}.mpnam'.format(self.ml.name, self.weight_label, self.group)
        mplf = '{}_{}_{}.mplst'.format(self.ml.name, self.weight_label, self.group)

        mp = fp.modpath.Modpath(modelname=self.mpname, modflowmodel=self.ml, dis_file=self.dis.file_name[0], exe_name=mp_exe_name,
                                model_ws=self.model_ws, simfile_ext='mpsim', dis_unit=self.dis.unit_number[0])

        mpsim = fp.modpath.ModpathSim(mp, mp_name_file=mpnf, 
                                      mp_list_file=mplf, 
                                      option_flags=options,
                                      ref_time=self.ref_time,
                                      cell_bd_ct=0, 
        #                               bud_loc=bud_chk_dict[group].loc[:, ('Grid', 'Layer', 'Row', 'Column')].values.tolist(),
                                      extension='mpsim')

        mpbas = fp.modpath.ModpathBas(mp, hnoflo=self.bas.hnoflo, hdry=self.upw.hdry, 
                                      def_face_ct=1, bud_label=['RECHARGE'], def_iface=[6], 
                                      laytyp=self.upw.laytyp.get_value(), ibound=self.bas.ibound.array, 
                                      prsity=por, prsityCB=0.20)    

        mp.write_input()
        success, msg = mp.run_model(silent=True, report=False)

        #     delete starting locations to save space--this information is now in the endpoint file
        if success:
            dst_pth = os.path.join(self.model_ws, '{}_{}_{}.loc'.format(self.ml.name, self.weight_label, self.group))
            os.remove(dst_pth)
            
    def modify_endpoint_file(self, ep_data_, write=False):  
        if not hasattr(self, 'water_table'):
            self.get_watertable()
        ep_data_ = ep_data_.copy()
        # Clean up and enhance an MODPATH endpoint file
        # set the Z coordinate for particles that end in dry cells to the 
        # head of the nearest non-dry cell below the dry cell.
        ind = np.isclose(ep_data_.loc[:, 'Final Global Z'], self.upw.hdry, atol=100)
        ep_data_.loc[:, 'Final Global Z'] = np.where(ind, self.water_table[ep_data_.loc[:, 'Final Row'] - 1, 
                                            ep_data_.loc[:, 'Final Column']-1], ep_data_.loc[:, 'Final Global Z'])

        # eliminate particles that start in dry cells
        ind = np.isclose(ep_data_.loc[:, 'Initial Global Z'], self.upw.hdry, rtol=0.99999)
        self.ep_data = ep_data_.loc[~ind, :]

        # calculate approximate linear path distances
        x_dist = ep_data_.loc[:, 'Final Global X'] - ep_data_.loc[:, 'Initial Global X']
        y_dist = ep_data_.loc[:, 'Final Global Y'] - ep_data_.loc[:, 'Initial Global Y']
        z_dist = ep_data_.loc[:, 'Final Global Z'] - ep_data_.loc[:, 'Initial Global Z']
        ep_data_.loc[:, 'xy_path_len'] = np.sqrt(x_dist**2 + y_dist**2)
        ep_data_.loc[:, 'xyz_path_len'] = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)

        mendpoint_file = '{}_mod.{}'.format(self.mpname, 'mpend')
        mendpoint_file = os.path.join(self.model_ws, mendpoint_file)
        if write:
            ep_data_.to_csv(mendpoint_file)
            endpoint_file = '{}.{}'.format(self.mpname, 'mpend')
            endpoint_file = os.path.join(self.model_ws, endpoint_file)
            if os.path.exists(endpoint_file):
                os.remove(endpoint_file)
        self.ep_data = ep_data_

    def get_budget(self, text):
        # Get the MODFLOW budget file for the time period specified by the MODPATH release date
        # and the MODFLOW start date. 
        self._get_kstpkper()
        budget = self.bud_obj.get_data(kstpkper=self.kstpkper, text=text, full3D=True)[0]
        self.budget = budget
        
    def _len_mult(self):
        # the database values are in feet; if the model is in meters, 
        # provide a multiplier to convert database values to match the model
        lenuni_dict = {0: 'undefined units', 1: 'feet', 2: 'meters', 3: 'centimeters'}
        self.len_unit = lenuni_dict[self.dis.lenuni]
        if self.len_unit == 'meters':
            self.len_mult = 0.3048006096012192
        elif self.len_unit == 'feet':
            self.len_mult = 1.0
        else:
            print('unknown length units')
            self.len_mult = 1.0

    def read_endpoints(self, endpoint_file):
        # read MODPATH 6 endpoint file
        # count the number of header lines
        i = 0
        with open(endpoint_file) as f:
            while True:
                line = f.readline()
                i += 1
                if 'END HEADER' in line:
                    break
                elif not line:
                    break

        # columns names from MP6 docs 
        cols = ['Particle ID', 'Particle Group', 'Status', 'Initial Time', 'Final Time', 'Initial Grid', 
                'Initial Layer', 'Initial Row', 'Initial Column', 'Initial Cell Face', 'Initial Zone', 
                'Initial Local X', 'Initial Local Y', 'Initial Local Z', 'Initial Global X', 'Initial Global Y', 
                'Initial Global Z', 'Final Grid', 'Final Layer', 'Final Row', 'Final Column', 'Final Cell Face', 
                'Final Zone', 'Final Local X', 'Final Local Y', 'Final Local Z', 'Final Global X', 
                'Final Global Y', 'Final Global Z', 'Label']            

        # read the endpoint data
        ep_data = pd.read_table(endpoint_file, names=cols, header=None, skiprows=i, delim_whitespace=True)

        # select only 'Normally Terminated' particles; status code = 2
        ep_data = ep_data.loc[ep_data.Status == 2, :]

        tmp = ep_data[['Initial Layer', 'Initial Row', 'Initial Column']] - 1
        ep_data['initial_node_num'] = self.dis.get_node(tmp.values.tolist())

        tmp = ep_data[['Final Layer', 'Final Row', 'Final Column']] - 1
        ep_data['final_node_num'] = self.dis.get_node(tmp.values.tolist())
        
        # calculate particle travel time in years
        ep_data['rt'] = (ep_data['Final Time'] - ep_data['Initial Time']) / self.time_dict[self.dis.itmuni] / 365.25
        ep_data.set_index('initial_node_num', drop=True, inplace=True)
        return ep_data   
        
    def _distfit(self, t, sh_e, sc_e):
        first = t.min()
        _cdf = self.dist.cdf(t, sh_e, first, sc_e)
        return  _cdf

    def _explicit(self, t, sh_e, sc_e, sh_l, sc_l, fy):
        first = t.min()
        _cdf_e = self.dist.cdf(t, sh_e, first, sc_e)
        _cdf_l = self.dist.cdf(t, sh_l, first, sc_l)
        return fy * _cdf_e + (1 - fy) * _cdf_l

    def _fit(self, lprt, ly, func, components, bnds):
        for dist in self.dist_list:
            self.dist = dist
            lab = '{}_{}'.format(components, dist.name)
            # print('fitting {}'.format(lab))
            try:
                # with np.errstate(divide='ignore',invalid='ignore'):
                up1, cov = so.curve_fit(func, lprt, ly, bounds = bnds, method='trf')
                e1_cdf = func(lprt, *up1)
                e1 = ly - e1_cdf
                sse1 = e1.T.dot(e1)
            except Exception as e: 
                print(lab, e)
                sse1 = np.inf

            try:
                # with np.errstate(divide='ignore',invalid='ignore'):
                up2, cov = so.curve_fit(func, lprt, ly, bounds = bnds, method='dogbox')
                e2_cdf = func(lprt, *up2)
                e2 = ly - e2_cdf
                sse2 = e2.T.dot(e2)
            except Exception as e: 
                print(lab, e)
                sse2 = np.inf

            if sse1 < sse2:
                up = up1
                e_cdf = e1_cdf
                e = np.sqrt(sse1 / self.s)
                meth = 'trf'
            elif sse1 > sse2:
                up = up2
                e_cdf = e2_cdf
                e = np.sqrt(sse2 / self.s)
                meth = 'dogbox'
            else:
                up = np.zeros((5))
                e_cdf = np.nan
                e = np.nan
                meth = 'none'   
                
            return up, e_cdf, e, meth, lab

    def fit_dists(self, ly, lprt, dist_list, fit_one=True, fit_two=False):
        # fit a list of distributions in either one or two components
        # return a dictionary of results
        self.dist_list = dist_list
        param_dict = {}
        cdf_dict = {}
        error_dict = {}
        tt_dict = {}

        # dist_list = [ss.invgauss, ss.gamma, ss.weibull_min] 
        first = lprt.min()
        s = ly.shape[0]
        tt_dict['rt'] = lprt
        tt_dict['rt_cdf'] = ly
        self.s = ly.shape[0]

        if fit_one:
            bnds = (0, [+np.inf, +np.inf])
            func = self._distfit
            
            up, e_cdf, e, meth, lab = self._fit(lprt, ly, func, 'one', bnds)
            
            up = up[0:2]
            up = np.insert(up, 1, first)
            param_dict[lab] = up
            error_dict[lab] = e
            cdf_dict[lab] = e_cdf

        if fit_two:
            bnds = (0, [+np.inf, +np.inf, +np.inf, +np.inf, 1.0]) 
            func = self._explicit
            
            ep, e_cdf, e, meth, lab = self._fit(lprt, ly, func, 'two', bnds)
            
            ep = np.insert(ep, 1, first)
            ep = np.insert(ep, 4, first)
            param_dict[lab] = ep
            error_dict[lab] = e
            cdf_dict[lab] = e_cdf

        return {'cdf' : cdf_dict, 'par' : param_dict, 'err' : error_dict, 'tt' : tt_dict}


        
