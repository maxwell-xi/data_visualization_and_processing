import scipy.io
import numpy as np

# modules needed by the func import_field_from_dasy_in_cache
#from pyrec.supermash_cachefile_loader import SuperMashCachefileLoader
#import pyrec.mathvec3 as mv
import h5py

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
	

def import_field_from_dasy_in_cache(filename, distance_mm, print_grid=True):
	# extract the meas grid
	f = h5py.File(filename, 'r')
	grid_points = f['gridcache']['mapentry_0_']['grid']['_Object']['_Points']
	grid_shape = f['gridcache']['mapentry_0_']['grid']['_Object']['_Dimensions']
	grid_points_x = grid_points[:,0].reshape(grid_shape[0], grid_shape[1], order='F') # reshape operation cannot be made in some cases!
	grid_points_y = grid_points[:,1].reshape(grid_shape[0], grid_shape[1], order='F')
	grid_points_z = grid_points[:,2].reshape(grid_shape[0], grid_shape[1], order='F')
	grid_x = grid_points_x[:,0]
	grid_y = grid_points_y[0,:]
	grid_z = grid_points_z[:,0]
	grid_0 = [grid_x, grid_y] # meas grid whose center corresponds to the DUT reference point
	
	
	# extract E, H, and S, as well as their grids. Note that S has a grid different from E and H. 
	result = SuperMashCachefileLoader([filename])
	e_meas, loc_e = result.extract_fieldslice('E', distance_mm*1e-3)  # distance tolerance -0.09 mm ~ +0.1 mm
	h_meas, loc_h = result.extract_fieldslice('H', distance_mm*1e-3)
	s_meas, loc_s = result.extract_fieldslice('S', distance_mm*1e-3)
	
	e_meas = e_meas / np.sqrt(2) # convert to rms value
	h_meas = h_meas / np.sqrt(2)
  
	grid_e_x = loc_e[0,:,0]; grid_e_y = loc_e[1,0,:]; grid_e_z = loc_e[2,0,:]; grid_e = [grid_e_x, grid_e_y]
	grid_h_x = loc_h[0,:,0]; grid_h_y = loc_h[1,0,:]; grid_h_z = loc_h[2,0,:]; grid_h = [grid_h_x, grid_h_y] # H-field grid, same as E-field grid
	grid_s_x = loc_s[0,:,0]; grid_s_y = loc_s[1,0,:]; grid_s_z = loc_s[2,0,:]; grid_s = [grid_s_x, grid_s_y] # S-field grid, different from E-field grid

	# calcu the interested quantities
	#e_tot_rms = np.real(mv.mag3(e_meas / np.sqrt(2)))
	#h_tot_rms = np.real(mv.mag3(h_meas / np.sqrt(2)))
	#s_z_real = np.real(s_meas[2,:,:])
	#s_tot_real = mv.mag3(np.real(s_meas))
	#s_tot_mod = np.real(mv.mag3(s_meas))
	
	if print_grid == True:
		print('Meas grid')
		print('Numbers of x-, y-axis grid lines: {}, {}'.format(grid_shape[0], grid_shape[1]))
		print('Min and max x-axis coordinates [mm]: {:.2f}, {:.2f}; Grid step [mm]: {:.3f}'.format(np.min(grid_x)*1e3, np.max(grid_x)*1e3, 1e3*(np.max(grid_x)-np.min(grid_x))/(grid_shape[0]-1)))
		print('Min and max y-axis coordinates [mm]: {:.2f}, {:.2f}; Grid step [mm]: {:.3f}'.format(np.min(grid_y)*1e3, np.max(grid_y)*1e3, 1e3*(np.max(grid_y)-np.min(grid_y))/(grid_shape[1]-1)))
		if np.min(grid_z) == np.max(grid_z):
			print('z-axis coordinate [mm]: {:.2f}'.format(np.min(grid_z)*1e3))
		else:
			print('Min and max z-axis coordinates [mm]: {:.2f}, {:.2f}'.format(np.min(grid_z)*1e3, np.max(grid_z)*1e3))
		
		print('\nE-field grid')
		print('Numbers of x-, y-axis grid lines: {}, {}'.format(loc_e.shape[1], loc_e.shape[2]))
		print('Min and max x-axis coordinates [mm]: {:.2f}, {:.2f}; Grid step [mm]: {:.3f}'.format(np.min(grid_e_x)*1e3, np.max(grid_e_x)*1e3, 1e3*(np.max(grid_e_x)-np.min(grid_e_x))/(loc_e.shape[1]-1)))
		print('Min and max y-axis coordinates [mm]: {:.2f}, {:.2f}; Grid step [mm]: {:.3f}'.format(np.min(grid_e_y)*1e3, np.max(grid_e_y)*1e3, 1e3*(np.max(grid_e_y)-np.min(grid_e_y))/(loc_e.shape[2]-1)))
		if np.min(grid_e_z) == np.max(grid_e_z):
			print('z-axis coordinate [mm]: {:.2f}'.format(np.min(grid_e_z)*1e3))
		else:
			print('Min and max z-axis coordinates [mm]: {:.2f}, {:.2f}'.format(np.min(grid_e_z)*1e3, np.max(grid_e_z)*1e3))
		
		print('\nH-field grid')
		print('Numbers of x-, y-axis grid lines: {}, {}'.format(loc_h.shape[1], loc_h.shape[2]))
		print('Min and max x-axis coordinates [mm]: {:.2f}, {:.2f}; Grid step [mm]: {:.3f}'.format(np.min(grid_h_x)*1e3, np.max(grid_h_x)*1e3, 1e3*(np.max(grid_h_x)-np.min(grid_h_x))/(loc_h.shape[1]-1)))
		print('Min and max y-axis coordinates [mm]: {:.2f}, {:.2f}; Grid step [mm]: {:.3f}'.format(np.min(grid_h_y)*1e3, np.max(grid_h_y)*1e3, 1e3*(np.max(grid_h_y)-np.min(grid_h_y))/(loc_h.shape[2]-1)))
		if np.min(grid_h_z) == np.max(grid_h_z):
			print('z-axis coordinate [mm]: {:.2f}'.format(np.min(grid_h_z)*1e3))
		else:
			print('Min and max z-axis coordinates [mm]: {:.2f}, {:.2f}'.format(np.min(grid_h_z)*1e3, np.max(grid_h_z)*1e3))
		
		print('\nS-field grid')
		print('Numbers of x-, y-axis grid lines: {}, {}'.format(loc_s.shape[1], loc_s.shape[2]))
		print('Min and max x-axis coordinates [mm]: {:.2f}, {:.2f}; Grid step [mm]: {:.3f}'.format(np.min(grid_s_x)*1e3, np.max(grid_s_x)*1e3, 1e3*(np.max(grid_s_x)-np.min(grid_s_x))/(loc_s.shape[1]-1)))
		print('Min and max y-axis coordinates [mm]: {:.2f}, {:.2f}; Grid step [mm]: {:.3f}'.format(np.min(grid_s_y)*1e3, np.max(grid_s_y)*1e3, 1e3*(np.max(grid_s_y)-np.min(grid_s_y))/(loc_s.shape[2]-1)))
		if np.min(grid_s_z) == np.max(grid_s_z):
			print('z-axis coordinate [mm]: {:.2f}'.format(np.min(grid_s_z)*1e3))
		else:
			print('Min and max z-axis coordinates [mm]: {:.2f}, {:.2f}'.format(np.min(grid_s_z)*1e3, np.max(grid_s_z)*1e3))
	
	return grid_0, grid_e, grid_h, grid_s, e_meas, h_meas, s_meas
    

def import_field_from_hfss_in_fld(filename, print_grid=0):
    f = open(filename, 'r')
    lines = f.readlines()
    
    if print_grid == 1:
        print(lines[:2])
        
    x = []
    y = []
    z = []
    field = []
    
    for line in lines[2:]:
        x.append(float(line.split()[0]))
        y.append(float(line.split()[1]))
        z.append(float(line.split()[2]))
        field.append(float(line.split()[3]))
    f.close()
    
    grid_x = np.unique(x)
    grid_y = np.unique(y)
    grid_z = np.unique(z)
    grid = [grid_x, grid_y, grid_z]
    
    field_reshaped = np.array(field).reshape(len(grid_x), len(grid_y), len(grid_z), order='C')
    
    return grid, field_reshaped  