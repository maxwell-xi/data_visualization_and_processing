import numpy as np
from pyrec.mpl_helpers import ComplexVectorPlotter
cplt = ComplexVectorPlotter()
import pyrec.numpy_ext as npx

def plot_field_at_slice(field, grid_1, grid_2, field_name='H', grid_1_name='x', grid_2_name='y', db_scale=True):
    grid_1b, grid_2b = npx.meshgrid(grid_1, grid_2)
    
    if db_scale == True:
        cplt.contourf_quantities(grid_1b, grid_2b, 
                                 [field[0], field[1], field[2], field[3]], 
                                 titles = ['$|%s_{x}|$'%field_name, '$|%s_{y}|$'%field_name, '$|%s_{z}|$'%field_name, '$|%s_{tot}|$'%field_name],                        
                                 levels_min = [-40,]*4,
                                 xlabel='%s [mm]'%grid_1_name, ylabel='%s [mm]'%grid_2_name,  
                                 plot_rows_as_cols=True,
                                 compute_quantities = ['mag_db', 'mag_db', 'mag_db', 'mag_db'],
                                 tight_subplots = True,
                                 plot_colorbar = [False, False, False, True],
                                 db_ref = [np.nanmax(field[3]), np.nanmax(field[3]), np.nanmax(field[3]), np.nanmax(field[3])],
                                 colorbar_label = [None, None, None, 'dB (ref={:.3f} A/m)'.format(np.nanmax(field[3]))],
                                 figsize = (16,4))
    else:
        cplt.contourf_quantities(grid_1b, grid_2b, 
                                 [field[0], field[1], field[2], field[3]], 
                                 titles = ['$|%s_{x}|$'%field_name, '$|%s_{y}|$'%field_name, '$|%s_{z}|$'%field_name, '$|%s_{tot}|$'%field_name],                        
                                 levels_min = [0, 0, 0, 0],
                                 xlabel='%s [mm]'%grid_1_name, ylabel='%s [mm]'%grid_2_name,  
                                 plot_rows_as_cols=True,                                 
                                 tight_subplots = True,
                                 plot_colorbar = [False, False, False, True],                                 
                                 figsize = (16,4))

def temp_func():
	print('test')