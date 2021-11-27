import scipy.io
import numpy as np

def import_field_from_s4l_in_mat(filename, print_grid=0):
    input_data = scipy.io.loadmat(filename)

    grid_x = input_data['Axis0'][0,:]
    grid_y = input_data['Axis1'][0,:]
    grid_z = input_data['Axis2'][0,:]
    grid = [grid_x, grid_y, grid_z]
    
    field_x = input_data['Snapshot0'][:,0]
    field_y = input_data['Snapshot0'][:,1]
    field_z = input_data['Snapshot0'][:,2]
    
    # F-order should be used in the reshaping, since the first index (for X coordiante) changes the fastest, 
    # and the last index (for Z coordinate) changes the slowest 
    field_x_reshaped = field_x.reshape(grid_x.shape[0], grid_y.shape[0], grid_z.shape[0], order='F') 
    field_y_reshaped = field_y.reshape(grid_x.shape[0], grid_y.shape[0], grid_z.shape[0], order='F')
    field_z_reshaped = field_z.reshape(grid_x.shape[0], grid_y.shape[0], grid_z.shape[0], order='F')    
    
    field_tot = np.sqrt(np.square(field_x_reshaped) + np.square(field_y_reshaped) + np.square(field_z_reshaped))      
    output_field = [field_x_reshaped, field_y_reshaped, field_z_reshaped, field_tot]
    
    if print_grid == 1:
        print('X grid')
        print('Min and max coordinates [mm]: {}, {}; Step [mm]: {}; Number of points: {}'.
              format(1e3*grid_x[0], 1e3*grid_x[-1], 1e3*(grid_x[1]-grid_x[0]), grid_x.shape[0]))
        print('\nY grid')
        print('Min and max coordinates [mm]: {}, {}; Step [mm]: {}; Number of points: {}'.
              format(1e3*grid_y[0], 1e3*grid_y[-1], 1e3*(grid_y[1]-grid_y[0]), grid_y.shape[0]))
        print('\nZ grid')
        print('Min and max coordinates [mm]: {}, {}; Step [mm]: {}; Number of points: {}'.
              format(1e3*grid_z[0], 1e3*grid_z[-1], 1e3*(grid_z[1]-grid_z[0]), grid_z.shape[0]))    
    
    return grid, output_field
	

def import_field_from_dasy_in_mat(filename, grid_step_mm=5, print_grid=0):
    input_data = scipy.io.loadmat(filename)

    grid_x = input_data['x_coord']
    grid_y = input_data['y_coord']
    grid_z = input_data['z_coord']
    field_x =  input_data['Value_of_X_Comp_0s']
    field_y =  input_data['Value_of_Y_Comp_0s']
    field_z =  input_data['Value_of_Z_Comp_0s']
    field_tot =  input_data['Value_of_Total_0s']   
   
    # F-order should be used in the reshaping, since the first index (for X coordiante) changes the fastest, 
    # and the last index (for Z coordinate) changes the slowest
    grid_x_max = np.max(grid_x)
    grid_x_min = np.min(grid_x)
    grid_x_pts = int(np.round((grid_x_max - grid_x_min) / (grid_step_mm*1e-3)) + 1)
    
    grid_y_max = np.max(grid_y)
    grid_y_min = np.min(grid_y)
    grid_y_pts = int(np.round((grid_y_max - grid_y_min) / (grid_step_mm*1e-3)) + 1)
    
    grid_z_max = np.max(grid_z)
    grid_z_min = np.min(grid_z)
    grid_z_pts = int(np.round((grid_z_max - grid_z_min) / (grid_step_mm*1e-3)) + 1)
    
    grid_x_reshaped = grid_x.reshape(grid_x_pts, grid_y_pts, grid_z_pts, order='F')
    grid_y_reshaped = grid_y.reshape(grid_x_pts, grid_y_pts, grid_z_pts, order='F')
    grid_z_reshaped = grid_z.reshape(grid_x_pts, grid_y_pts, grid_z_pts, order='F')
    grid_x_extracted = grid_x_reshaped[:,:,0][:,0]  # x and y grids vary slightly on different z slices
    grid_y_extracted = grid_y_reshaped[:,:,0][0,:]
    grid_z_extracted = grid_z_reshaped[0,0,:]
    grid = [grid_x_extracted, grid_y_extracted, grid_z_extracted]
    
    field_x_reshaped = field_x.reshape(grid_x_pts, grid_y_pts, grid_z_pts, order='F') 
    field_y_reshaped = field_y.reshape(grid_x_pts, grid_y_pts, grid_z_pts, order='F')
    field_z_reshaped = field_z.reshape(grid_x_pts, grid_y_pts, grid_z_pts, order='F')
    field_tot_reshaped = field_tot.reshape(grid_x_pts, grid_y_pts, grid_z_pts, order='F')    
    output_field = np.array([field_x_reshaped, field_y_reshaped, field_z_reshaped, field_tot_reshaped])
   
    if print_grid == 1:
        print('X grid')
        print('Min and max coordinates [mm]: {:.1f}, {:.1f}; Step [mm]: {}; Number of points: {}'.
              format(1e3*np.min(grid_x_extracted), 1e3*np.max(grid_x_extracted), grid_step_mm, grid_x_pts))
        print('\nY grid')
        print('Min and max coordinates [mm]: {:.1f}, {:.1f}; Step [mm]: {}; Number of points: {}'.
              format(1e3*np.min(grid_y_extracted), 1e3*np.max(grid_y_extracted), grid_step_mm, grid_y_pts))
        print('\nZ grid')
        print('Min and max coordinates [mm]: {:.1f}, {:.1f}; Step [mm]: {}; Number of points: {}'.
              format(1e3*np.min(grid_z_extracted), 1e3*np.max(grid_z_extracted), grid_step_mm, grid_z_pts))
        
    return grid, output_field