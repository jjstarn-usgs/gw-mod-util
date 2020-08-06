import os, sys
import shutil
import numpy as np
import numpy as np
import flopy as fp
import pandas as pd
import subprocess
import pickle

# from model_specs import *
# from gen_mod_dict import *

class model_area(object):
    '''Create parent model for parallel runs'''
    def __init__(self, proj_dir, model_dict, scenario_name, scenario_dict, rock_dict, K_dict, mfpth, e_tol=1.,
                 use_existing=False):
        self.proj_dir = proj_dir
        self.scenario_dict = scenario_dict
        self.rock_dict = rock_dict
        self.mfpth = mfpth
        self.K_dict = K_dict
        self.scenario_name = scenario_name

        self.run_num = np.nan
        self.ms = model_dict
        self.md = model_dict['ws']
        self.e_tol = e_tol
        self.hydro_wt = 1.
        self.model_ws = '.'
        self.stage = None
        self.batch_num = None           

        self.nam_file = '{}.nam'.format(self.md)
        if use_existing:
            os.chdir(scenario_name)
            # zone_file = 'zone_array.npz'
            # zones = np.load(zone_file)
            # self.zones = zones['zone']
            mf = fp.modflow.Modflow.load(self.nam_file, version='mfnwt', exe_name=mfpth, check=False,
                                         verbose=False, model_ws=self.model_ws, load_only=['BAS6', 'UPW', 'NWT', 'OC'])

            mf = fp.modflow.Modflow.load(self.nam_file, version='mfnwt', exe_name=mfpth, check=False, verbose=False)
            infil_list = [item for item in mf.get_name_file_entries().split('\n') if 'DATA' not in item]
            copy_list = [item.split()[2] for item in infil_list if len(item) != 0]
            copy_list.append(self.nam_file)
            self.copy_list = copy_list

            bas = mf.get_package('BAS6')
            self.hnoflo = bas.hnoflo

            upw = mf.get_package('UPW')
            self.hdry = upw.hdry
            self.Kh = np.array(upw.hk.get_value())

            nwt = mf.get_package('NWT')
            self.tol0 = nwt.headtol

            oc = mf.get_package('OC')
            self.head_file_pth = os.path.join(self.model_ws, oc.parent.output_fnames[1])

            self.mf_lst_pth = mf.lst.fn_path
            self.mf_nam = mf.namefile

            mf.external_path = 'arrays'

            model_file = 'model_grid.csv'
            model_grid = pd.read_csv(model_file, na_values=[self.hnoflo, self.hdry])
            self.obs_type = model_grid.obs_type
            self.land_surface = model_grid.top
            # w = model_grid.dist2str / model_grid.dist2str.max()
            # self.w = 1 - w
            self.w = 1

            self.num_hydro = self.obs_type.value_counts()['hydro']
            self.num_topo = self.obs_type.value_counts()['topo']
            self.num_cells = self.num_hydro + self.num_topo

            import pickle
            dst = 'K_Dict.pickle'
            with open(dst, 'rb') as f:
                K_zone_list = pickle.load(f)

            self.K_zone_list = K_zone_list

            self.copy_list=[]
            for filetype in ['bas','dis','drn','nam','nwt','oc','rch','upw']:
                self.copy_list.append(self.md + '.' + filetype)
    def create_model(self):
        md = self.md
        ms = self.ms
        K_dict = self.K_dict
        rock_riv_dict = self.rock_dict
        scenario = self.scenario_dict
        hnoflo = -9999
        hdry = -8888
        mfpth = self.mfpth


        geo_ws = self.ms['ws']
        model_ws = geo_ws
        array_pth = os.path.join(model_ws, 'arrays')

        # try:
        #     shutil.rmtree(array_pth)
        # except:
        #     pass
        #
        # try:
        #     shutil.rmtree(model_ws)
        # except:
        #     pass

        # os.makedirs(model_ws)

        head_file_name = '{}.hds'.format(md)
        head_file_pth = os.path.join(head_file_name)

        # Replace entries from the default K_dict with the model specific K values from model_dict if they exist.

        # In[ ]:

        for key, value in K_dict.items():
            if key in ms['K_Dict'].keys():
                K_dict[key] = ms['K_Dict'][key]

        # Replace entries from the default rock_riv_dict with the model specific values from model_dict if they exist. rock_riv_dict has various attributes of bedrock and stream geometry.

        # In[ ]:

        for key, value in rock_riv_dict.items():
            if key in ms.keys():
                rock_riv_dict[key] = ms[key]

        # Assign values to variables used in this notebook using rock_riv_dict

        # In[ ]:

        min_thk = rock_riv_dict['min_thk']
        stream_width = rock_riv_dict['stream_width']
        stream_bed_thk = rock_riv_dict['stream_bed_thk']
        river_depth = rock_riv_dict['river_depth']
        bedrock_thk = rock_riv_dict['bedrock_thk']
        stream_bed_kadjust = rock_riv_dict['stream_bed_kadjust']

        # ## Read the information for a model domain processed using Notebook 1

        # Read the model_grid data frame from a csv file. Extract grid dimensions and ibound array.

        # In[ ]:

        model_file = 'model_grid.csv'
        model_grid = pd.read_csv(model_file, index_col='node_num', na_values=['nan', hnoflo])

        NROW = model_grid.row.max() + 1
        NCOL = model_grid.col.max() + 1
        num_cells = NROW * NCOL

        ibound = model_grid.ibound.values.reshape(NROW, NCOL)
        inactive = (ibound == 0)

        # ## Translate geologic information into hydrologic properties

        # In[ ]:

        surf_zone_dict = {}
        bedrock_zone_dict = {}
        K_zone_list = []
        # Get surficial zones from model_grid.csv
        surf_zones = model_grid.coarse_flag.unique().astype(int)

        surf_zones = np.sort(surf_zones)
        nzone = 0
        # put lakes first- then surficial zones - then bedrock zones
        K_zone_list.append(K_dict['K_lakes'])

        qa = np.zeros((NROW, NCOL), dtype=np.int32)
        br = np.zeros((NROW, NCOL), dtype=np.int32)

        # assign surficial zones and K
        for szone in surf_zones:
            nzone += 1
            surf_zone_dict[szone] = nzone
            K_zone_list.append(K_dict['K_surf'][szone])
            qa[model_grid.coarse_flag.values.reshape(NROW, NCOL) == szone] = nzone
        # assign bedrock zones and K
        if not (scenario['use_bedrock_zones']):
            model_grid.bedrock_flag = np.zeros(NROW * NCOL).astype(int)
        bedrock_zones = model_grid.bedrock_flag.unique().astype(int)
        bedrock_zones = np.sort(bedrock_zones)
        for bzone in bedrock_zones:
            nzone += 1
            bedrock_zone_dict[bzone] = nzone
            kb = K_dict['K_bedrock']
            K_zone_list.append(kb[bzone])
            br[model_grid.bedrock_flag.values.reshape(NROW, NCOL) == bzone] = nzone

        import pickle
        dst = 'K_Dict.pickle'
        pickle.dump(K_zone_list, open(dst, 'wb'))
        print(K_zone_list)
        self.K_zone_list = K_zone_list

        # This version replaces Soller's Surfmat with the Quaternary Atlas. Look-up table for coarse deposits (zone = 1) from Dick Yager's new_unit.  All other categories are lumped with fine deposits (zone = 0).
        # * alluvium = 1
        # * ice contact = 9
        # * lacustrine coarse = 11
        # * outwash = 17

        # Create a dictionary that maps the K_dict from gen_mod_dict to zone numbers (key=zone number, value=entry in K_dict).  Make sure these correspond with the correct units. If you're using the defaults, it is correct.

        # In[ ]:

        # zone_dict = {0 : 'K_fine', 1 : 'K_coarse', 2 : 'K_lakes', 3 : 'K_bedrock'}


        # Perform the mapping from zone number to K to create the Kh1d array.

        # In[ ]:

        zones1d = np.zeros((NROW, NCOL), dtype=np.int32)

        zones1d = qa.copy()

        la = model_grid.lake.values.reshape(NROW, NCOL)

        zones1d[la == 1] = 0
        Kh1d = np.take(K_zone_list, zones1d)
        model_grid['K0'] = Kh1d.ravel()
        model_grid['kzone'] = zones1d.ravel()
        # ## Process boundary condition information
        stream_cond_dict = {}
        # Create a dictionary of stream information for the drain or river package.
        # River package input also needs the elevation of the river bed. Don't use both packages. The choice is made by commenting/uncommenting sections of the modflow function. Replace segment_len (segment length) with the conductance. The river package has not been tested.

        # In[ ]:

        drn_flag = (model_grid.stage != np.nan) & (model_grid.ibound == 1)
        drn_data = model_grid.loc[drn_flag, ['lay', 'row', 'col', 'stage', 'segment_len', 'K0','kzone']]
        drn_data.columns = ['k', 'i', 'j', 'stage', 'segment_len', 'K0','kzone']
        drn_data['cond_fact'] = stream_bed_kadjust*drn_data.segment_len * stream_width / stream_bed_thk
        #drn_data['K1'] = np.take(K_zone_list, drn_data['kzone'])
        self.drn_data = drn_data.copy()
        drn_data.loc[drn_data.K0 == K_dict['K_lakes'], 'K0'] = 100
        dcond = drn_data.K0 * drn_data['cond_fact']
        drn_data['segment_len'] = dcond
        drn_data.rename(columns={'segment_len': 'cond'}, inplace=True)
        drn_data.to_csv('drndata.txt')
        drn_data.drop(['K0','kzone','cond_fact'], axis=1, inplace=True)
        drn_data.dropna(axis='index', inplace=True)
        drn_data.insert(drn_data.shape[1], 'iface', 6)
        drn_recarray = drn_data.to_records(index=False)
        drn_dict = {0: drn_recarray}

        # In[ ]:

        riv_flag = (model_grid.stage != np.nan) & (model_grid.ibound == 1)
        riv_data = model_grid.loc[riv_flag, ['lay', 'row', 'col', 'stage', 'segment_len',
                                             'reach_intermit', 'K0','kzone']]
        riv_data.columns = ['k', 'i', 'j', 'stage', 'segment_len', 'rbot', 'K0','kzone']
        riv_data[['rbot']] = riv_data.stage - river_depth
        riv_data['cond_fact'] = stream_bed_kadjust*riv_data.segment_len * stream_width / stream_bed_thk
        self.riv_data = riv_data.copy()

        riv_data.loc[riv_data.K0 == K_dict['K_lakes'], 'K0'] = 100
        rcond = riv_data.K0 * riv_data['cond_fact']
        riv_data['segment_len'] = rcond
        riv_data.rename(columns={'segment_len': 'rcond'}, inplace=True)
        riv_data.drop(['K0','kzone','cond_fact'], axis=1, inplace=True)
        riv_data.dropna(axis='index', inplace=True)
        riv_data.insert(riv_data.shape[1], 'iface', 6)
        riv_recarray = riv_data.to_records(index=False)
        riv_dict = {0: riv_recarray}

        # ### Create 1-layer model to get initial top-of-aquifer on which to drape subsequent layering

        # Get starting heads from top elevations. The top is defined as the model-cell-mean NED elevation except in streams, where it is interpolated between MaxElevSmo and MinElevSmo in the NHD (called 'stage' in model_grid). Make them a little higher than land so that drains don't accidentally go dry too soon.

        # In[ ]:
        # get bottom value
        top = model_grid.top.values.reshape(NROW, NCOL)
        strt = top * 1.05

        # Modify the bedrock surface, ensuring that it is always at least min_thk below the top elevation. This calculation will be revisited for the multi-layer case.

        # In[ ]:

        # TODO Put this back in when we convert quaternary thickness to bedrock elevation
        # if('bedrock_el' in model_grid):
        if (False):
            bedrock = model_grid.bedrock_el.values.reshape(NROW, NCOL)
            thk = top - bedrock
            print('using bedrock elevation')
        else:
            thk = model_grid.soller_thk.values.reshape(NROW, NCOL)
            bedrock = top - thk
            print('using quaternary thickness')
        thk[thk < min_thk] = min_thk
        bot = top - thk
        model_grid['bot']  = bot.reshape(NROW*NCOL)

        # Create a dictionary of information for the general-head boundary package.
        # Similar to the above cell. Not tested.

        # In[ ]:
        model_grid.loc[model_grid.coast_ele == -999, 'coast_ele'] = 0
        #don't use coastele if less than bottom
        model_grid.loc[model_grid.coast_ele < model_grid.bot, 'coast_ele'] = 0
        self.ghb_data = None
        if model_grid.coast_ele.sum() > 0:
            ghb_flag = model_grid.coast_ele >0
            ghb_data = model_grid.loc[ghb_flag, ['lay', 'row', 'col', 'coast_ele', 'segment_len', 'K0','kzone']]
            ghb_data.columns = ['k', 'i', 'j', 'stage', 'segment_len', 'K0','kzone']
            ghb_data['cond_fact'] = stream_bed_kadjust*scenario['L'] * scenario['L'] / stream_bed_thk
            self.ghb_data = ghb_data.copy()
            ghb_data.loc[ghb_data.K0 == K_dict['K_lakes'], 'K0'] = 100
            gcond = ghb_data.K0 * ghb_data['cond_fact']
            ghb_data['segment_len'] = gcond
            ghb_data.rename(columns={'segment_len': 'cond'}, inplace=True)
            ghb_data.drop(['K0','kzone','cond_fact'], axis=1, inplace=True)
            ghb_data.dropna(axis='index', inplace=True)
            ghb_data.insert(ghb_data.shape[1], 'iface', 6)
            ghb_recarray = ghb_data.to_records(index=False)
            ghb_dict = {0: ghb_recarray}

        if scenario['GHB'] and model_grid.ghb.sum() > 0:
            ghb_flag = model_grid.ghb == 1
            ghb_data = model_grid.loc[ghb_flag, ['lay', 'row', 'col', 'top', 'segment_len', 'K0','kzone']]
            ghb_data.columns = ['k', 'i', 'j', 'stage', 'segment_len', 'K0','kzone']
            ghb_data['cond_fact'] = stream_bed_kadjust*scenario['L'] * scenario['L'] / stream_bed_thk
            self.ghb_data = ghb_data.copy()
            ghb_data.loc[ghb_data.K0 == K_dict['K_lakes'], 'K0'] = 100
            gcond = ghb_data.K0 * ghb_data['cond_fact']
            ghb_data['segment_len'] = gcond
            ghb_data.rename(columns={'segment_len': 'cond'}, inplace=True)
            ghb_data.drop(['K0','kzone','cond_fact'], axis=1, inplace=True)
            ghb_data.dropna(axis='index', inplace=True)
            ghb_data.insert(ghb_data.shape[1], 'iface', 6)
            ghb_recarray = ghb_data.to_records(index=False)
            ghb_dict = {0: ghb_recarray}


        # ## Create recharge array

        # This version replaces the Wolock/Yager recharge grid with the GWRP SWB grid.

        # In[ ]:

        if (scenario['use_alt_recharge']):
            rech = model_grid.alt_recharge.values.reshape(NROW, NCOL) * scenario['recharge_multiplier']
        else:
            rech = model_grid.recharge.values.reshape(NROW, NCOL) * scenario['recharge_multiplier']
        print(rech.sum())

        # Replace rech array with
        # * calculate total recharge for the model domain
        # * calculate areas of fine and coarse deposits
        # * apportion recharge according to the ratio specified in gen_mod_dict.py
        # * write the values to an array

        # In[ ]:

        ## rech = model_grid.recharge.values.reshape(NROW, NCOL) * scenario['recharge_multiplier']

        # In[ ]:

        if (scenario['coarse_fine_rch_adj'] != 1):
            rech_ma = np.ma.MaskedArray(rech, mask=inactive)
            coarse_ma = np.ma.MaskedArray(qa > 1, mask=inactive)
            fine_ma = np.ma.MaskedArray(qa == 1, mask=inactive)

            total_rech = rech_ma.sum()
            Af = fine_ma.sum()
            Ac = coarse_ma.sum()
            Rf = total_rech / (scenario['coarse_fine_rch_adj'] * Ac + Af)
            Rc = scenario['coarse_fine_rch_adj'] * Rf

            rech = np.zeros_like(rech)
            rech[qa > 1] = Rc
            rech[qa == 1] = Rf

        # ## Define a function to create and run MODFLOW

        # In[ ]:

        def modflow(md, mfpth, model_ws, nlay=1, top=top, strt=strt, nrow=NROW, ncol=NCOL, botm=bedrock,
                    ibound=ibound, hk=Kh1d, rech=rech, stream_dict=drn_dict, delr=scenario['L'],
                    delc=scenario['L'], hnoflo=hnoflo, hdry=hdry, iphdry=1, vani=K_dict['K_vani'],laytyp=1):

            strt_dir = os.getcwd()
            # os.chdir(model_ws)

            ml = fp.modflow.Modflow(modelname=md, exe_name=mfpth, version='mfnwt',
                                    external_path='arrays')

            # add packages (DIS has to come before either BAS or the flow package)
            dis = fp.modflow.ModflowDis(ml, nlay=nlay, nrow=NROW, ncol=NCOL, nper=1, delr=delr, delc=delc,
                                        laycbd=0, top=top, botm=botm, perlen=1.E+05, nstp=1, tsmult=1,
                                        steady=True, itmuni=4, lenuni=2, extension='dis',
                                        unitnumber=11)

            bas = fp.modflow.ModflowBas(ml, ibound=ibound, strt=strt, ifrefm=True,
                                        ixsec=False, ichflg=False, stoper=None, hnoflo=hnoflo, extension='bas',
                                        unitnumber=13)

            upw = fp.modflow.ModflowUpw(ml, laytyp=laytyp, layavg=0, chani=1.0, layvka=1, laywet=0, ipakcb=53,
                                        hdry=hdry, iphdry=iphdry, hk=hk, hani=1.0, vka=vani, ss=1e-05,
                                        sy=0.15, vkcb=0.0, noparcheck=False, extension='upw',
                                        unitnumber=31)

            rch = fp.modflow.ModflowRch(ml, nrchop=3, ipakcb=53, rech=rech, irch=1,
                                        extension='rch', unitnumber=19)

            drn = fp.modflow.ModflowDrn(ml, ipakcb=53, stress_period_data=drn_dict,
                                        dtype=drn_dict[0].dtype,
                                        extension='drn', unitnumber=21, options=['NOPRINT', 'AUX IFACE'])

            riv = fp.modflow.ModflowRiv(ml, ipakcb=53, stress_period_data=riv_dict,
                                        dtype=riv_dict[0].dtype,
                                        extension='riv', unitnumber=18, options=['NOPRINT', 'AUX IFACE'])

            if (scenario['GHB'] and model_grid.ghb.sum() > 0) or model_grid.coast_ele.sum() > 0:
                ghb = fp.modflow.ModflowGhb(ml, ipakcb=53, stress_period_data=ghb_dict,
                                            dtype=ghb_dict[0].dtype,
                                            extension='ghb', unitnumber=23, options=['NOPRINT', 'AUX IFACE'])

            oc = fp.modflow.ModflowOc(ml, ihedfm=0, iddnfm=0, chedfm=None, cddnfm=None, cboufm=None,
                                      compact=True, stress_period_data={(0, 0): ['save head', 'save budget']},
                                      extension=['oc', 'hds', 'ddn', 'cbc'], unitnumber=[14, 51, 52, 53])

            #     nwt = fp.modflow.ModflowNwt(ml, headtol=0.0001, fluxtol=500, maxiterout=1000,
            #                                 thickfact=1e-05, linmeth=2, iprnwt=1, ibotav=0, options='COMPLEX')

            nwt = fp.modflow.ModflowNwt(ml, headtol=0.0001, fluxtol=500, maxiterout=100, thickfact=1e-04,
                                        linmeth=2, iprnwt=1, ibotav=1, options='SPECIFIED', dbdtheta=0.70,
                                        dbdkappa=0.1, dbdgamma=0.2, momfact=0.001, backflag=1,
                                        maxbackiter=50, backtol=1.1, backreduce=0.2, iacl=2, norder=1,
                                        level=8, north=2, iredsys=1, rrctols=0.0, idroptol=1, epsrn=1.0E-3,
                                        hclosexmd=1.0e-3, mxiterxmd=100)

            ml.write_input()
            ml.remove_package('RIV')

            ml.write_input()
            success, output = ml.run_model(silent=True)
            # os.chdir(strt_dir)
            if success:
                print("    Your {:0d} layer model ran successfully".format(nlay))
            else:
                print("    Your {:0d} layer model didn't work".format(nlay))

            return ml

        # ## Run 1-layer MODFLOW

        # Use the function to run MODFLOW for 1 layer to getting approximate top-of-aquifer elevation

        # In[ ]:

        modflow(md, mfpth, model_ws, nlay=1, top=top, strt=strt, nrow=NROW, ncol=NCOL, botm=bot, ibound=ibound,
                hk=Kh1d, rech=rech, stream_dict=drn_dict, iphdry=0)

        # Read the head file and calculate new layer top (wt) and bottom (bot) elevations based on the estimated
        # water table (wt) being the top of the top layer. Divide the surficial layer into NLAY equally thick layers between wt and the bedrock surface elevation (as computed using minimum surficial thickness).

        # In[ ]:

        hdobj = fp.utils.HeadFile(head_file_pth)
        heads1 = hdobj.get_data(kstpkper=(0, 0))
        heads1[heads1 == hnoflo] = np.nan
        heads1[heads1 <= hdry] = np.nan
        heads1 = heads1[0, :, :]
        hdobj = None

        # ## Create layering using the scenario in gen_mod_dict

        # Make new model with (possibly) multiple layers. If there are dry cells in the 1 layer model, they are converted to NaN (not a number). The minimum function in the first line returns NaN if the element of either input arrays is NaN.  In that case, replace NaN in modeltop with the top elevation. The process is similar to the 1 layer case. Thickness is estimated based on modeltop and bedrock and is constrained to be at least min_thk (set in gen_mod_dict.py). This thickness is divided into num_surf_layers number of layers. The cumulative thickness of these layers is the distance from the top of the model to the bottom of the layers. This 3D array of distances (the same for each layer) is subtracted from modeltop.

        # In[ ]:

        modeltop = np.minimum(heads1, top)
        nan = np.isnan(heads1)
        modeltop[nan] = top[nan]
        thk = modeltop - bedrock
        thk[thk < min_thk] = min_thk

        NLAY = scenario['num_surf_layers']
        lay_extrude = np.ones((NLAY, NROW, NCOL))
        lay_thk = lay_extrude * thk / NLAY
        bot = modeltop - np.cumsum(lay_thk, axis=0)

        # Using the estimated water table as the new top-of-aquifer elevations sometimes leads to the situation, in usually a very small number of cells, that the drain elevation is below the bottom of the cell.  The following procedure resets the bottom elevation to one meter below the drain elevation if that is the case.

        # In[ ]:

        stg = model_grid.stage.fillna(1.E+30, inplace=False)
        tmpdrn = (lay_extrude * stg.values.reshape(NROW, NCOL)).ravel()
        tmpbot = bot.ravel()
        index = np.less(tmpdrn, tmpbot)
        tmpbot[index] = tmpdrn[index] - 1.0
        bot = tmpbot.reshape(NLAY, NROW, NCOL)

        # * If add_bedrock = True in gen_mod_dict.py, add a layer to the bottom and increment NLAY by 1.
        # * Assign the new bottom-most layer an elevation equal to the elevation of the bottom of the lowest surficial layer minus bedrock_thk, which is specified in rock_riv_dict (in gen_mod_dict.py).
        # * Concatenate the new bottom-of-bedrock-layer to the bottom of the surficial bottom array.
        # * Compute the vertical midpoint of each cell. Make an array (bedrock_index) that is True if the bedrock surface is higher than the midpoint and False if it is not.
        # * lay_extrude replaces the old lay_extrude to account for the new bedrock layer. It is not used in this cell, but is used later to extrude other arrays.

        # In[ ]:

        sol_thk = model_grid.soller_thk.values.reshape(NROW, NCOL)
        tmp = top - sol_thk
        bedrock_4_K = bedrock.copy()
        bedrock_4_K[bedrock > top] = tmp[bedrock > top]

        if scenario['add_bedrock']:
            NLAY = scenario['num_surf_layers'] + 1
            lay_extrude = np.ones((NLAY, NROW, NCOL))
            bed_bot = bot[-1:, :, :] - bedrock_thk
            bot = np.concatenate((bot, bed_bot), axis=0)

            mids = bot + thk / NLAY / 2
            bedrock_index = mids < bedrock_4_K
            bedrock_index[-1:, :, :] = True

        # if(add_bedrock2):
        #         NLAY = NLAY + 1
        #         lay_extrude = np.ones((NLAY, NROW, NCOL))
        #         br2_bot = (min_bedrock_elev - 10)*np.ones((1,NROW, NCOL))
        #         bot = np.concatenate((bot, br2_bot), axis=0)
        #         bedrock_index = np.concatenate((bedrock_index, bedrock_index[-1:,:,:]), axis=0)

        elif not scenario['add_bedrock']:
            print('    no bedrock')
            pass

        else:
            print('    add_bedrock variable needs to True or False')

        # Extrude all arrays to NLAY number of layers. Create a top-of-aquifer elevation (fake_top) that is higher (20% in this case) than the simulated 1-layer water table because in doing this approximation, some stream elevations end up higher than top_of_aquifer and thus do not operate as drains. The fake_top shouldn't affect model computations if it is set high enough because the model uses convertible (confined or unconfined) layers.

        # In[ ]:

        fake_top = (modeltop * 1.2).astype(np.float32)
        strt = (lay_extrude * modeltop * 1.05).astype(np.float32)
        ibound = (lay_extrude * ibound).astype(np.int16)

        # Perform the mapping from zone number to K to create the Kh3d array.

        # In[ ]:

        zones3d = np.zeros((NLAY, NROW, NCOL), dtype=np.int32)

        qa3d = (lay_extrude * qa).astype(np.int32)
        br3d = (lay_extrude * br).astype(np.int32)
        zones3d = qa3d
        if scenario['add_bedrock']:
            zones3d[bedrock_index] = br3d[bedrock_index]

        la = model_grid.lake.values.reshape(NROW, NCOL)
        zones3d[0, la == 1] = 0
        Kh3d = np.take(K_zone_list, zones3d)

        self.Kh = Kh3d
        self.zones = zones3d
        # if (scenario['add_bedrock2']):
        #     Kh3d[-1::] = Kh3d[-1::] / 10


        # Run MODFLOW again using the new layer definitions.  The difference from the first run is that the top-of-aquifer elevation is the 1-layer water table rather than land surface, and of course, the number of surficial layers and/or the presence of a bedrock layer is different.

        # In[ ]:

        rech[bedrock_index[0]] = 0

        # In[ ]:

        Kvani3d = K_dict['K_vani'] * np.ones((NLAY, NROW, NCOL), dtype=np.float32)

        if scenario['use_confining']:
            aq_depth_df = model_grid[['ibound', 'aq_depth']]
            aq_depth_active = aq_depth_df[aq_depth_df.ibound == 1]
            mean_aq_depth = aq_depth_active[aq_depth_active.aq_depth != -99].aq_depth.mean()
            print('Replacing missing aquifer depths with mean value: ' + str(mean_aq_depth))
            model_grid.loc[model_grid.aq_depth == -99, 'aq_depth'] = mean_aq_depth
            model_grid.loc[model_grid.aq_depth < 1, 'aq_depth'] = 1
            Kvani3d[1, :, :] = K_dict['K_vani2'] * model_grid.aq_depth.values.reshape(NROW, NCOL)
            print('Using confining layer')

        # In[ ]:

        mf = modflow(md, mfpth, model_ws, nlay=NLAY, top=fake_top, strt=strt, nrow=NROW, ncol=NCOL,
                botm=bot, ibound=ibound, hk=Kh3d, rech=rech, stream_dict=drn_dict,
                hnoflo=hnoflo, hdry=hdry, iphdry=1,laytyp=[1,1,1,0])

        # Read the new head array

        # In[ ]:

        def test_model(mf_lst_pth, bad_run=10.0):
            # check that the model has a reasonable mass balance
            with open(mf_lst_pth) as fn:
                tmp = fn.readlines()
            line = [line.split() for line in tmp if 'PERCENT DISCREPANCY =' in line]
            perc_disc = float(line[0][3])
            if abs(perc_disc) > bad_run:
                print('Percent discrepancy for this run is bad and equals {}'.format(perc_disc))
                return False, perc_disc
            else:
                print('Percent discrepancy for this run is okay and equals {}'.format(perc_disc))
                return True, perc_disc

        # In[ ]:

        good_run, pct_discrep = test_model(md + '.list')

        # In[ ]:

        hdobj = fp.utils.HeadFile(head_file_pth)
        heads = hdobj.get_data()
        hdobj = None

        # Make a 2D array of the heads in the highest active cells and call it the water_table

        # In[ ]:

        heads[heads == hnoflo] = np.nan
        heads[heads <= hdry] = np.nan
        hin = np.argmax(np.isfinite(heads), axis=0)
        row, col = np.indices((hin.shape))
        water_table = heads[hin, row, col]

        water_table_ma = np.ma.MaskedArray(water_table, inactive)


        # Save the heads and K from the upper-most layer to model_grid.csv

        # In[ ]:

        model_grid['pre_cal_heads'] = water_table_ma.ravel()
        model_grid['pre_cal_K'] = Kh3d[0, :, :].ravel()

        if scenario['add_bedrock']:
            model_grid['thk'] = model_grid.top - bot[-1, :, :].ravel() + bedrock_thk
        else:
            model_grid['thk'] = model_grid.top - bot[-1, :, :].ravel()

        model_grid['thkR'] = model_grid.thk / model_grid.recharge

        model_grid.to_csv('model_grid.csv')

        # Save zone array for use in calibration.

        # In[ ]:

        zone_file = 'zone_array.npz'
        np.savez(zone_file, zone=zones3d)

        # Plot a cross-section to see what the layers look like.  Change row_to_plot to see other rows.  Columns could be easily added.

        # In[ ]:

        def calc_error(top, head, obs_type):
            # an offset of 1 is used to eliminate counting heads that
            # are within 1 m of their target as errors.
            # count topo and hydro errors
            t = top < (head - scenario['error_tol'])
            h = top > (head + scenario['error_tol'])

            tmp_df = pd.DataFrame({'head': head, 'ot': obs_type, 't': t, 'h': h})

            tmp = tmp_df.groupby('ot').sum()
            h_e_ = tmp.loc['hydro', 'h']
            t_e_ = tmp.loc['topo', 't']
            result = np.array([h_e_, t_e_])
            return result

        # In[ ]:

        hydro, topo = calc_error(model_grid.top, water_table.ravel(), model_grid.obs_type)
        num_hydro = model_grid.obs_type.value_counts()['hydro']
        num_topo = model_grid.obs_type.value_counts()['topo']
        num_cells = num_hydro + num_topo
        hydro = hydro / num_hydro
        topo = topo / num_topo

        with open('top_hyd_error.txt','w') as f:
            f.write('topo error: {}'.format(topo))
            f.write('hydro error: {}'.format(hydro))
        infil_list = [item for item in mf.get_name_file_entries().split('\n') if 'DATA' not in item]
        copy_list = [item.split()[2] for item in infil_list if len(item) != 0]
        copy_list.append(self.nam_file)
        self.copy_list = copy_list

        bas = mf.get_package('BAS6')
        self.hnoflo = bas.hnoflo

        upw = mf.get_package('UPW')
        self.hdry = upw.hdry
        self.K = np.array(upw.hk.get_value())

        nwt = mf.get_package('NWT')
        self.tol0 = nwt.headtol

        oc = mf.get_package('OC')
#        self.head_file_pth = os.path.join(self.model_ws, oc.file_name[1])
        self.head_file_pth = md + '.hds'
        self.mf_lst_pth = mf.lst.fn_path
        self.mf_nam = mf.namefile

        mf.external_path = 'arrays'

        # model_file = 'model_grid.csv'
        # model_grid = pd.read_csv(model_file, na_values=[self.hnoflo, self.hdry])
        self.obs_type = model_grid.obs_type
        self.land_surface = model_grid.top
        # w = model_grid.dist2str / model_grid.dist2str.max()
        # self.w = 1 - w

        self.num_hydro = self.obs_type.value_counts()['hydro']
        self.num_topo = self.obs_type.value_counts()['topo']
        self.num_cells = self.num_hydro + self.num_topo

    def make_par_grid(self, k_ar_f, k_ar_c, k_ar_b, k_ar_va):
        Kf, Kc, Kb, KVa = np.meshgrid( k_ar_f, k_ar_c, k_ar_b, k_ar_va )
        Kf, Kc, Kb, KVa = Kf.ravel(), Kc.ravel(), Kb.ravel(), KVa.ravel()
        index = Kf < Kc
        num_par = index.sum()
        par_grid = np.zeros((num_par, 4))
        par_grid[:, 0:4] = np.array((Kf[index], Kc[index], Kb[index], KVa[index])).T
        self.par_grid = par_grid
                
    def change_nwt_tol(self, tol=0.001):
        # Re-write nwt package with head tol = 0.001 for quicker convergence.
        mf = fp.modflow.Modflow.load(self.nam_file, version='mfnwt', exe_name=self.mfpth, check=False,
              verbose=False, model_ws=self.model_ws, load_only=['NWT'])
        nwt = mf.get_package('NWT')
        nwt.headtol = tol
        mf.write_input()
        
    def write_K(self, K):
        nl = K.shape[0]
        for i in range(nl):
            name = os.path.join(self.model_ws, 'arrays', 'hk_layer_{}.ref'.format(i + 1))
            np.savetxt(name, K[i, :, :], fmt='%15.6E', delimiter='')  

    def write_V(self, V):
        nl = V.shape[0]
        for i in range(nl):
            name = os.path.join(self.model_ws, 'arrays', 'vani{}.ref'.format(i + 1))
            np.savetxt(name, V[i, :, :], fmt='%15.6E', delimiter='')

    def __tofile(self, f, data,names,fmt_string):
        # Write the recarray (data) to the file (or file handle) f
        assert isinstance(data, np.recarray), "MfList.__tofile() data arg " + \
                                              "not a recarray"

        # Add one to the kij indices
        lnames = [name.lower() for name in names]
        # --make copy of data for multiple calls
        d = np.recarray.copy(data)
        for idx in ['k', 'i', 'j', 'node']:
            if idx in lnames:
                d[idx] += 1
        np.savetxt(f, d, fmt=fmt_string, delimiter='')

    def write_Cond(self, drn_data,K_zone_list,array_file,names,fmt_string):
        dd = drn_data.copy()
        dd['K0'] = np.take(K_zone_list, dd['kzone'])

        dcond = dd['K0']*dd['cond_fact']
        dd['segment_len'] = dcond
        dd.rename(columns={'segment_len': 'cond'}, inplace=True)
        dd.drop(['K0', 'kzone', 'cond_fact'], axis=1, inplace=True)
        dd.dropna(axis='index', inplace=True)
        dd.insert(dd.shape[1], 'iface', 6)
        drn_recarray = dd.to_records(index=False)
        self.__tofile(array_file,drn_recarray,names,fmt_string)

    def write_new_arrays(self, pars):
        K = self.Kh.copy()
        K_zone_dict2 = self.K_zone_list.copy()
        K_zone_dict2[1] = pars[0]  # fine K
        K_zone_dict2[2] = pars[1]  # coarse K
        for i in range(3, len(self.K_zone_list)):
            K_zone_dict2[i] = pars[2] * self.K_zone_list[i]
        K = np.take(K_zone_dict2, self.zones)
        self.write_K(K)

        # V = pars[3] * base_vani
        # if (conf_method == 2):
        #     V[0, :, :] = fixed_vani
        #     V[2, :, :] = fixed_vani
        #     V[3, :, :] = fixed_vani
        V = np.ones(K.shape) * pars[3]
        self.write_V(V)

        self.write_Cond(self.drn_data,K_zone_dict2,'arrays/DRN_0000.dat',['k','i','j','stage','cond','iface'],' %9d %9d %9d %15.7E %15.7E %9d')
        self.write_Cond(self.riv_data,K_zone_dict2,'arrays/RIV_0000.dat',['k','i','j','stage','cond', 'rbot','iface'],' %9d %9d %9d %15.7E %15.7E %15.7E %9d')
        if not (self.ghb_data is None):
            self.write_Cond(self.ghb_data,K_zone_dict2,'arrays/GHB_0000.dat',['k','i','j','stage','cond','iface'],' %9d %9d %9d %15.7E %15.7E %9d')

    def run_model(self, batch_file):
        '''Function to run model'''
        #self.change_nwt_tol()
        try:
            subprocess.run(batch_file, **{'cwd': self.model_ws})
            bad_run = 10.0
            # check that the model has a reasonable mass balance
            list_file = os.path.join(self.model_ws, self.md + '.list')
            with open(list_file, 'r') as fn:
                tmp = fn.readlines()
            line = [line.split() for line in tmp if 'PERCENT DISCREPANCY =' in line]
            self.percent_d = np.abs(float(line[0][3]))
        except Exception as e:
            self.percent_d = 999
            print(e)
        
    def get_heads(self):
        if(self.percent_d == 999):
            return
        hedf = fp.utils.HeadFile(os.path.join(self.model_ws, self.head_file_pth))
        heads = hedf.get_data()
        # eliminate unrealistic heads that sometimes occur in isolated cells
        heads[heads > 1.E+29] = np.nan
        heads[heads == self.hnoflo] = np.nan
        heads[heads == self.hdry] = np.nan
        hin = np.argmax(np.isfinite(heads), axis=0)
        row, col = np.indices((hin.shape))
        h = heads[hin, row, col]
        self.heads = h.ravel()
        
    def ob_func(self, runtime):
        if self.percent_d == 999:
            self.h_e_ =self.num_hydro
            self.t_e_ = self.num_topo
        else:
            # count topo and hydro errors
            err_tol = 1.0 # meters
            t = self.land_surface < (self.heads - err_tol)
            h = self.land_surface > (self.heads + err_tol)
            tmp_df = pd.DataFrame({'ot':self.obs_type, 't':t, 'h':h})

            tmp_gb = tmp_df.groupby('ot').sum()
            self.h_e_ = tmp_gb.loc['hydro', 'h']
            self.t_e_ = tmp_gb.loc['topo', 't']
        
        result = pd.Series()
        result['model'] = self.md
        result['run_num'] = self.run_num
        result['num_cells'] = self.num_cells
        result['num_hydro'] = self.num_hydro
        result['num_hydro_err'] = self.h_e_
        result['rate_hydro_err'] = self.h_e_ / self.num_hydro
        result['num_topo'] = self.num_topo
        result['num_topo_err'] = self.t_e_
        result['rate_topo_err'] = self.t_e_ / self.num_topo
        result['rate_diff'] = np.abs(result.rate_topo_err - result.rate_hydro_err)
        result['rate_sum'] = result.rate_topo_err + result.rate_hydro_err
        result['percent_d'] = self.percent_d
        result['pars'] = np.array(self.par_grid[self.run_num], dtype=np.float32())
        result['Kf'] = self.par_grid[self.run_num, 0]
        result['Kc'] = self.par_grid[self.run_num, 1]
        result['Kb'] = self.par_grid[self.run_num, 2]
        result['vani'] = self.par_grid[self.run_num, 3]
        
        result['run_time'] = runtime
        result = pd.DataFrame(result).T

        phi_file = os.path.join(self.model_ws, 'results.csv')
        if not os.path.exists(phi_file):
            ofp = open(phi_file, 'w')
            result.to_csv(ofp)
        else:
            ofp = open(phi_file, 'a')
            result.to_csv(ofp, header=False)
        ofp.close()
    
    def new_k_arr(self, k, par_ar, num_k):
        index = np.arange(new_num_k)[par_ar == k]
        lo = par_ar[index - 1]
        hi = par_ar[index + 1]
        return np.logspace(np.log10(lo), np.log10(hi), num_k)

    def update_starting_heads(self):
        hedf = fp.utils.HeadFile(os.path.normpath(self.head_file_pth.replace('\\','/')))
        heads = hedf.get_data()
        nl = heads.shape[0]
        for i in range(nl):
            name = os.path.join(self.model_ws, 'arrays', 'strt_layer_{}.ref'.format(i + 1))
            np.savetxt(name, heads[i, :, :], fmt='%15.6E', delimiter='')

    def test_converge(self):
        # check that the model has a reasonable mass balance
        with open(os.path.normpath(self.mf_lst_pth.replace('\\','/'))) as fn:
            tmp = fn.readlines()
        converge = [0 for line in tmp if 'FAILED TO MEET SOLVER CONVERGENCE CRITERIA' in line]
        return converge ==[]

    def set_iphdry(self,value_to_set):
        # mf = fp.modflow.Modflow.load(self.mf_nam, model_ws=self.model_ws, exe_name=self.mfpth, load_only=['UPW'],
        #                             version='mfnwt',check=False )
        # upw_file = mf.get_package('UPW')
        # upw_file.iphdry = value_to_set
        # upw_file.write_file(True)
        with open(self.md + '.upw') as fn:
            tmp = fn.readlines()
        tmp[1]= tmp[1][:-2] + str(value_to_set) + '\n'
        with open(self.md + '.upw','w') as fn:
            fn.writelines(tmp)

