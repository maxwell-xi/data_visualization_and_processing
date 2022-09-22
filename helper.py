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
	
def x_component_avg(field, avg_square_sidelength_pts, ix, iy, iz):
    iy_start = iy-int(avg_square_sidelength_pts/2)
    iy_end = iy+int(avg_square_sidelength_pts/2)
    iz_start = iz-int(avg_square_sidelength_pts/2)
    iz_end = iz+int(avg_square_sidelength_pts/2)
    
    if iy_start < 0:
        iy_start = 0
        print('iy_start falls out of the dataset! iy_start is forced to be 0.')
    
    if iy_end > field.shape[2]:
        iy_end = field.shape[2]
        print('iy_end falls out of the dataset! iy_end is set to be the largest y index.')
        
    
    if iz_start < 0:
        iz_start = 0
        print('iz_start falls out of the dataset! iz_start is forced to be 0.')
        
        
    if iz_end > field.shape[3]:
        iz_end = field.shape[3]
        print('iz_end falls out of the dataset! iz_end is set to be the largest z index.')        
    
    
    field_slice = extract_field_at_slice(field, 'x', ix)
    field_sample = field_slice[0][iy_start:iy_end+1, iz_start:iz_end+1]
    field_x_avg = np.sum(field_sample)/field_sample.size
    
    return field_x_avg


def y_component_avg(field, avg_square_sidelength_pts, ix, iy, iz):
    ix_start = ix-int(avg_square_sidelength_pts/2)
    ix_end = ix+int(avg_square_sidelength_pts/2)
    iz_start = iz-int(avg_square_sidelength_pts/2)
    iz_end = iz+int(avg_square_sidelength_pts/2)
    
    if ix_start < 0:
        ix_start = 0
        print('ix_start falls out of the dataset! ix_start is forced to be 0.')
    
    if ix_end > field.shape[1]:
        ix_end = field.shape[1]
        print('ix_end falls out of the dataset! ix_end is set to be the largest x index.')
        
    
    if iz_start < 0:
        iz_start = 0
        print('iz_start falls out of the dataset! iz_start is forced to be 0.')
        
        
    if iz_end > field.shape[3]:
        iz_end = field.shape[3]
        print('iz_end falls out of the dataset! iz_end is set to be the largest z index.')    
    
    field_slice = extract_field_at_slice(field, 'y', iy)
    field_sample = field_slice[1][ix_start:ix_end+1, iz_start:iz_end+1]
    field_y_avg = np.sum(field_sample)/field_sample.size
    
    return field_y_avg

def z_component_avg(field, avg_square_sidelength_pts, ix, iy, iz):
    ix_start = ix-int(avg_square_sidelength_pts/2)
    ix_end = ix+int(avg_square_sidelength_pts/2)
    iy_start = iy-int(avg_square_sidelength_pts/2)
    iy_end = iy+int(avg_square_sidelength_pts/2)
    
    if ix_start < 0:
        ix_start = 0
        print('ix_start falls out of the dataset! ix_start is forced to be 0.')
    
    if ix_end > field.shape[1]:
        ix_end = field.shape[1]
        print('ix_end falls out of the dataset! ix_end is set to be the largest x index.')
        
    
    if iy_start < 0:
        iy_start = 0
        print('iy_start falls out of the dataset! iy_start is forced to be 0.')
        
        
    if iy_end > field.shape[2]:
        iy_end = field.shape[2]
        print('iy_end falls out of the dataset! iy_end is set to be the largest y index.')  
        
    field_slice = extract_field_at_slice(field, 'z', iz)
    field_sample = field_slice[2][ix_start:ix_end+1, iy_start:iy_end+1]
    field_z_avg = np.sum(field_sample)/field_sample.size
    
    return field_z_avg
