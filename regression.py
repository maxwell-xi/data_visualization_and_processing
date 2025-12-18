import numpy as np

# total sum of squares (SST)
def calc_total_sum_of_squares(y):
    return np.sum((y - np.mean(y)) ** 2)

# residual sum of squares (SSR), also called sum squared regression
def calc_residual_sum_of_squares(y, y_fit):
    return np.sum((y - y_fit) ** 2)

# R-squared, also called coefficient of determination
def calc_r_squared(y, y_fit):
    ss_total = calc_total_sum_of_squares(y)
    ss_residual = calc_residual_sum_of_squares(y, y_fit)
    return 1 - (ss_residual / ss_total)

# adjusted R-squared
def calc_r_squared_adj(y, y_fit, degree):
    r_squared = calc_r_squared(y, y_fit)
    obs_num = len(y)
    return 1 - ((1-r_squared) * (obs_num-1)) / (obs_num - degree -1)

# multiple R, also called correlation coefficient
def calc_multiple_r(y, y_fit):
    r_squared = calc_r_squared(y, y_fit)
    return np.sqrt(r_squared)

# standard error of the regression/estimate (SER), also called residual standard error
def calc_standard_error_of_regression(y, y_fit, degree):
    ss_residual = calc_residual_sum_of_squares(y, y_fit)
    obs_num = len(y)
    return np.sqrt(ss_residual / (obs_num - degree - 1))

# root mean square error (RMSE)
def calc_root_mean_square_error(y, y_fit):
    ss_residual = calc_residual_sum_of_squares(y, y_fit)
    obs_num = len(y)
    return np.sqrt(ss_residual / obs_num)

# calculate Predicted R-squared using LOOCV (leave-one-out cross-validation) method
def calc_r_squared_pred(x, y, degree):
    n = len(x)
    y_pred = np.zeros(n)
    
    for i in range(n):
        # Fit a polynomial excluding the i-th point
        x_train = np.delete(x, i)
        y_train = np.delete(y, i)
        coefficients = np.polyfit(x_train, y_train, degree)
        poly_model = np.poly1d(coefficients)
        
        # Predict the excluded point
        y_pred[i] = poly_model(x[i])
    
    ss_total = calc_total_sum_of_squares(y)
    ss_residual_pred = calc_residual_sum_of_squares(y, y_pred) # calc PRESS (Predicted Residual Error Sum of Squares)
    
    return 1 - (ss_residual_pred / ss_total)

def calc_prediction_error(x, y, model, output_in_db=True):
    y_pred = model(x)
    
    if output_in_db == True:
        error = 20*np.log10(y_pred / y)
    else:
        error = 100 * (y_pred - y) / y
    
    return error

def format_polynomial_latex(poly_model):
    terms = []
    for power, coeff in enumerate(poly_model.coefficients):
        # Skip terms with a zero coefficient
        #if np.isclose(coeff, 0):
        if coeff == 0:
            continue # skip the current iteration and proceed to the next one
            
        if (np.abs(coeff) < 0.1) or (np.abs(coeff) >= 1000):
            # Format coefficient to 3 significant digits with scientific notation for too small and too large values
            coeff_str = f"{coeff:.2e}"
        else:
            # Format coefficient to 3 significant digits without scientific notation
            coeff_str = f"{coeff:.3g}"
        
        if power == len(poly_model.coefficients) - 2:
            term = f"{coeff_str}x"  # For the x term
        elif power < len(poly_model.coefficients) - 2:
            exponent = len(poly_model.coefficients) - power - 1
            term = f"{coeff_str}x^{{{exponent}}}"  # For x^n terms
        else:
            term = coeff_str  # Constant term
        
        # Add term to list, handling sign for non-first terms
        if terms and coeff > 0:
            term = f"+ {term}"
        terms.append(term)
    return " ".join(terms)

# https://stackoverflow.com/questions/17930473/how-to-make-my-pylab-poly1dfit-pass-through-zero
def fit_poly_with_fixed_low_order_coeff(x, y, n=3, low_order_coeff=[1, 1]):
    a = x[:, np.newaxis] ** np.arange(len(low_order_coeff), n+1)
    coeff = np.linalg.lstsq(a, y)[0]
    return np.concatenate((low_order_coeff, coeff))