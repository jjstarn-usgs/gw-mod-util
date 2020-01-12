import os

# The directories in the following list will be searched for models
homes = ['../../IWAS_DRB_BigFiles/subprojects/siteGeneral']
fig_dir = '../Figures'

for h in homes:
    assert os.path.exists(h), 'path does not exist'

mfpth = 'C:/Workspace/IWAS_DRB_BigFiles/executables/MODFLOW-NWT_1.0.9/bin/MODFLOW-NWT_64.exe'
assert os.path.exists(mfpth), 'path does not exist'

mp_exe_name = 'C:/Workspace/IWAS_DRB_BigFiles/executables/modpath.6_0/bin/mp6x64.exe' 
assert os.path.exists(mp_exe_name), 'path does not exist'

mf_start_date_str = '01/01/1900' 
mp_release_date_str = '01/01/2018' 

# The zone number will be encoded in the MODPATH endpoint file under the "Label" variable.
# The RTD for all zones will be calculated if use_all_zones is True.
# You can also calculate RTDs groups of zones.
use_all_zones = True
use_groups_of_zones = False
number_of_particles_per_group = 4.0E+06

# If zones are to be grouped, put the zone numbers in a list inside a tuple.
# You can have more than one group.
# Zones can be part of more than one group and not all zones have to be used,
# e.g. zones_to_group = ([4, 5, 6], [41]).  If only one group is used, put a comma 
# after it, e.g. zones_to_group = ([41],)
zones_to_group = ([0, 1, 2], [3])

num_cells2budchk = 10

# weighting scheme, either 'flow' or 'volume'
# regardless which weighting scheme is used, the notebook will output
# a file representing cell-by-cell flux per particle that can be used as a weight
# when fitting distributions to get flux weighted travel times
# weight_scheme = 'flow'
weight_scheme = 'volume'

por = 0.20
scenario = 'bedrock_cal_wt_1.00'
