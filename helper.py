import numpy as np

def extract_field_at_slice(field, slice_normal, slice_index):
    field_x = field[0]
    field_y = field[1]
    field_z = field[2]
    field_tot = field[3]
    
    if slice_normal == 'x':   # yz slice        
        field_x_slice   =   field_x[slice_index,:,:]
        field_y_slice   =   field_y[slice_index,:,:]
        field_z_slice   =   field_z[slice_index,:,:]
        field_tot_slice = field_tot[slice_index,:,:]
    elif slice_normal == 'y':         
        field_x_slice   =   field_x[:,slice_index,:]
        field_y_slice   =   field_y[:,slice_index,:]
        field_z_slice   =   field_z[:,slice_index,:]
        field_tot_slice = field_tot[:,slice_index,:]
    else:
        field_x_slice   =   field_x[:,:,slice_index]
        field_y_slice   =   field_y[:,:,slice_index]
        field_z_slice   =   field_z[:,:,slice_index]
        field_tot_slice = field_tot[:,:,slice_index]
        
    field_slice = np.array([field_x_slice, field_y_slice, field_z_slice, field_tot_slice])
    
    return field_slice

def extract_field_at_line(field, line_orientation, line_index_1, line_index_2):
    field_x = field[0]
    field_y = field[1]
    field_z = field[2]
    field_tot = field[3]
    
    if line_orientation == 'x':
        field_x_line   =   field_x[:,line_index_1,line_index_2]
        field_y_line   =   field_y[:,line_index_1,line_index_2]
        field_z_line   =   field_z[:,line_index_1,line_index_2]
        field_tot_line = field_tot[:,line_index_1,line_index_2]
    elif line_orientation == 'y':
        field_x_line   =   field_x[line_index_1,:,line_index_2]
        field_y_line   =   field_y[line_index_1,:,line_index_2]
        field_z_line   =   field_z[line_index_1,:,line_index_2]
        field_tot_line = field_tot[line_index_1,:,line_index_2]
    else:
        field_x_line   =   field_x[line_index_1,line_index_2,:]
        field_y_line   =   field_y[line_index_1,line_index_2,:]
        field_z_line   =   field_z[line_index_1,line_index_2,:]
        field_tot_line = field_tot[line_index_1,line_index_2,:]
    
    field_line = np.array([field_x_line, field_y_line, field_z_line, field_tot_line])
    
    return field_line 