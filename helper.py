import numpy as np
from sigfig import round
import glob
import os
import zipfile


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
    
    if iy_end > np.array(field).shape[2]:
        iy_end = np.array(field).shape[2]
        print('iy_end falls out of the dataset! iy_end is set to be the largest y index.')
        
    
    if iz_start < 0:
        iz_start = 0
        print('iz_start falls out of the dataset! iz_start is forced to be 0.')
        
        
    if iz_end > np.array(field).shape[3]:
        iz_end = np.array(field).shape[3]
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
    
    if ix_end > np.array(field).shape[1]:
        ix_end = np.array(field).shape[1]
        print('ix_end falls out of the dataset! ix_end is set to be the largest x index.')
        
    
    if iz_start < 0:
        iz_start = 0
        print('iz_start falls out of the dataset! iz_start is forced to be 0.')
        
        
    if iz_end > np.array(field).shape[3]:
        iz_end = np.array(field).shape[3]
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
    
    if ix_end > np.array(field).shape[1]:
        ix_end = np.array(field).shape[1]
        print('ix_end falls out of the dataset! ix_end is set to be the largest x index.')
        
    
    if iy_start < 0:
        iy_start = 0
        print('iy_start falls out of the dataset! iy_start is forced to be 0.')
        
        
    if iy_end > np.array(field).shape[2]:
        iy_end = np.array(field).shape[2]
        print('iy_end falls out of the dataset! iy_end is set to be the largest y index.')  
        
    field_slice = extract_field_at_slice(field, 'z', iz)
    field_sample = field_slice[2][ix_start:ix_end+1, iy_start:iy_end+1]
    field_z_avg = np.sum(field_sample)/field_sample.size
    
    return field_z_avg

	
def remove_artifacts_from_2d_field_distribution(field, artifact_index):
    artifact_index = artifact_index.tolist() # array-to-list conversion necessary to make "if a in b" check work
    field_corrected = field.copy()
    
    i_max = field.shape[0] - 1
    j_max = field.shape[1] - 1
    
    for artifact in artifact_index:
        i = artifact[0]
        j = artifact[1] 
        
        if i == 0: # left boundary
            if j == 0: # left bottom corner                
                print('Artifact happens at left bottom corner, thus cannot be estimated!')
            elif j == j_max: # left top corner
                print('Artifact happens at left top corner, thus cannot be estimated!')
            else:
                if (([i, j+1] in artifact_index) or ([i, j-1] in artifact_index)):
                    print('Artifact happens at left boundary with adjacent artifact(s), thus cannot be estimated!')
                else:                    
                    field_corrected[i, j] = np.mean([field[i, j+1], field[i, j-1]])
        elif i == i_max: # right boundary
            if j == 0: # right bottom corner
                print('Artifact happens at right bottom corner, thus cannot be estimated!')
            elif j == j_max: # right top corner
                print('Artifact happens at right top corner, thus cannot be estimated!')
            else:
                if (([i, j+1] in artifact_index) or ([i, j-1] in artifact_index)):
                    print('Artifact happens at right boundary with adjacent artifact(s), thus cannot be estimated!')
                else:                    
                    field_corrected[i, j] = np.mean([field[i, j+1], field[i, j-1]])
        else:
            if j == 0: # bottom boundary except two end points
                if (([i+1, j] in artifact_index) or ([i-1, j] in artifact_index)):
                    print('Artifact happens at bottom boundary with adjacent artifact(s), thus cannot be estimated!')
                else:
                    field_corrected[i, j] = np.mean([field[i+1, j], field[i-1, j]])
            elif j == j_max: # top boundary except two end points
                if (([i+1, j] in artifact_index) or ([i-1, j] in artifact_index)):
                    print('Artifact happens at top boundary with adjacent artifact(s), thus cannot be estimated!')
                else:
                    field_corrected[i, j] = np.mean([field[i+1, j], field[i-1, j]])
            else:
                if (([i, j+1] in artifact_index) or ([i, j-1] in artifact_index)):
                    if (([i+1, j] in artifact_index) or ([i-1, j] in artifact_index)):
                        print('Artifact happens within boundary with multiple adjacent artifacts, thus cannot be estimated!')
                    else:
                        field_corrected[i, j] = np.mean([field[i+1, j], field[i-1, j]])
                elif (([i+1, j] in artifact_index) or ([i-1, j] in artifact_index)):
                    if (([i, j+1] in artifact_index) or ([i, j-1] in artifact_index)):
                        print('Artifact happens within boundary with multiple adjacent artifacts, thus cannot be estimated!')
                    else:
                        field_corrected[i, j] = np.mean([field[i, j+1], field[i, j-1]])
                else:                
                    field_corrected[i, j] = np.mean([field[i, j+1], field[i, j-1], field[i-1, j], field[i+1, j]])
        
    return field_corrected
   
  
# https://stackoverflow.com/questions/17930473/how-to-make-my-pylab-poly1dfit-pass-through-zero
def fit_poly_with_fixed_low_order_coeff(x, y, n=3, low_order_coeff=[1, 1]):
    a = x[:, np.newaxis] ** np.arange(len(low_order_coeff), n+1)
    coeff = np.linalg.lstsq(a, y)[0]
    return np.concatenate((low_order_coeff, coeff))

# input - df: a Dataframe, chunkSize: the chunk size
# output - a list of DataFrame
# purpose - splits the DataFrame into smaller chunks
def split_dataframe(df, chunk_size = 100): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def extract_percentile_envelope(df, variable_name_x, variable_name_y, chunk_size, percentile_value):
    if len(df) < 3*chunk_size:
        print('Too large chunk size compared to the data set scale!')
    
    df_chunks = split_dataframe(df.sort_values(by=variable_name_x), chunk_size)
    
    if (len(df_chunks[-1]) < 10) or (len(df_chunks[-1]) < 0.2*chunk_size):
        print('Too few data points in the last chunk!')      
    
    chunk_x = [np.mean(df[variable_name_x]) for df in df_chunks]
    chunk_y = [np.percentile(df[variable_name_y], percentile_value) for df in df_chunks]
    
    return chunk_x, chunk_y

def round_it(value, significant_fig_num):
    if np.isnan(value) or np.isinf(value) or (value == 0):
        value_rounded = value
    elif int(np.log10(np.abs(value)) + 1) > 3:
        value_rounded = np.round(value)
    else:
        value_rounded = round(value, sigfigs=significant_fig_num)
    
    return value_rounded
    
def determine_field_decay_radius(field, scan_step_mm, decay_threshold_db):
    '''
    field: should be a 2D array describing the field distribution at a given plane
    scan_step_mm: the uniform step of the grid in which the field values exist
    decay_threshold_db: the decay threshold (in dB) relative to the peak value at a given plane    
    '''
    field_max = np.max(field)
    above_threshold = field >= 10**(decay_threshold_db/20)*field_max
    num_above_threshold = np.count_nonzero(above_threshold)
    radius_mm = np.sqrt(num_above_threshold*scan_step_mm**2 / np.pi)
    
    return radius_mm

# find files whose names meet the specified pattern under the specified directory
def get_files(file_dir, file_pattern):
    """
    Examples
    --------
    get_files(file_dir, '*.csv') # returns all csv files in the target directory
    get_files(file_dir, 'project_*') # returns all files whose names start with "project_" in the target directory
    """
    full_pattern = os.path.join(file_dir, file_pattern)
    files = glob.glob(full_pattern)
    files.sort()
    
    return files

# extract compressed files whose names start with the specified string to the specified directory
# if out_path does not exist, it will be created first 
def extract_zip_files(zip_files, file_header, out_dir):
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file) as archive:
            for file in archive.namelist():
                if file.startswith(file_header):
                    archive.extract(file, out_dir)

def extract_param_value_from_file_name(file_name, param_name):
    param_string = file_name.split('\\')[-1] # extract the file name
    end_index = param_string.rfind('.') # determine the end index of the actual file name (i.e., without the file extension)
    param_string = param_string[:end_index] # remove the file extension
    param_string_2 = param_string.split('_') # split the actual file name into multiple sections based on deliminator '_'
    param_name_len = len(param_name)

    for s in param_string_2:
        if param_name in s:
            start_index = s.find(param_name) + param_name_len # determine the start index of the param value
            param_value = float(s[start_index:])
    
    return param_value 