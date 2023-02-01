import numpy as np
from data_visualization_and_processing.complex_field_plotter import ComplexVectorPlotter
cplt = ComplexVectorPlotter()


def plot_field_at_slice(field, grid_1, grid_2, field_name='H', grid_1_name='x', grid_2_name='y', db_scale=True, db_min=-40):
    grid_1b, grid_2b = np.meshgrid(grid_1, grid_2, indexing='ij')
    
    if db_scale == True:
        cplt.contourf_quantities(grid_1b, grid_2b, 
                                 [field[0], field[1], field[2], field[3]], 
                                 titles = ['$|%s_{x}|$'%field_name, '$|%s_{y}|$'%field_name, '$|%s_{z}|$'%field_name, '$|%s_{tot}|$'%field_name],                        
                                 levels_min = [db_min,]*4,
                                 xlabel='%s [mm]'%grid_1_name, ylabel='%s [mm]'%grid_2_name,  
                                 plot_rows_as_cols=True,
                                 compute_quantities = ['mag_db', 'mag_db', 'mag_db', 'mag_db'],
                                 tight_subplots = True,
                                 plot_colorbar = [False, False, False, True],
                                 db_ref = [np.nanmax(field[3]), np.nanmax(field[3]), np.nanmax(field[3]), np.nanmax(field[3])],
                                 colorbar_label = [None, None, None, 'dB (ref={:.3g} A/m)'.format(np.nanmax(field[3]))],
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
