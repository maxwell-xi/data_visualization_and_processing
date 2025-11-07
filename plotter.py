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
                                 colorbar_label = [None, None, None, 'dB (ref={:.3g} A/m)'.format(np.nanmax(field[3]))] if field_name == 'H' else [None, None, None, 'dB (ref={:.3g} V/m)'.format(np.nanmax(field[3]))] if field_name == 'E' else [None, None, None, 'dB (ref={:.3g})'.format(np.nanmax(field[3]))],
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


def plot_2trace_with_dual_y_axes(x1, y1, x2, y2, x_label, y1_label, y2_label, x_lim=[None, None], y1_lim=[None, None], y2_lim=[None, None], show_legend=True, show_grid='both', figsize=[12, 4]):
    '''
    draw 2 traces in a plot with dual y axes at two sides
    '''
    fig, ax1 = plt.subplots(figsize=(figsize[0], figsize[1]))
    ax2 = ax1.twinx()

    trace1 = ax1.plot(x1, y1, c='C0', label=y1_label)
    trace2 = ax2.plot(x2, y2, c='C1', label=y2_label)

    ax1.set_xlabel(x_label)
    ax1.set_xlim(x_lim)

    ax1.set_ylabel(y1_label, color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    #ax1.grid(axis='y')
    ax1.set_ylim(y1_lim)

    ax2.set_ylabel(y2_label, color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    #ax2.grid(False)
    ax2.set_ylim(y2_lim)

    if show_legend == True:
        traces = trace1 + trace2
        labels = [t.get_label() for t in traces]
        plt.legend(traces, labels)

    if show_grid == 'x':
        ax1.grid(axis = 'x')
    elif show_grid == 'y':
        ax1.grid(axis = 'y')
    elif show_grid == 'none':
        ax1.grid(False)
    else:
        ax1.grid(True)
    ax2.grid(False)     