import numpy as np

# induced E-field averaged over 2x2x2-mm^3 cube as per ICNIRP2010, ICNIRP2020, and FCC
def induced_e_cube(g, h, f): 
    k = 1/((1+6.5e-6*g**5.8)**(1/5.8))
    e_est = k * h/21 *(135*f*1e-6)  # RL=21 A/m (i.e., 27 uT, from ICNIRP2010&2020); BR=135*f_MHz V/m 
    return e_est

# induced E-field averaged over 5-mm line as per IEEE2005 and IEEE2019
def induced_e_line(g, h, f): 
    k = 1/((1+4e-11*g**6.6)**(1/6.6))
    e_est = k * h/163 * (209*f*1e-6) # RL=163 A/m (i.e., 205 uT, from IEEE2005&2019); BR=209*f_MHz V/m
    return e_est

# for current density averaged over 1-cm^2 area as per ICNIRP1998; constant RL used
def induced_j_area(g, h, f): 
    k = 1/((1+4e-3*g**2.9)**(1/2.9))
    j_est = k * h/5 * (2*f*1e-6)  # RL=5 A/m (i.e., 6.25 uT, from ICNIRP1998); BR=2*f_MHz A/m^2
    return j_est

# SAR averaged over 10-g mass as per ICNIRP1998, ICNIRP2020, IEEE2005, and IEEE2019
def sar_10g(g, h, f): 
    k = ( 1/((1+2e-1*g**1.2)**(1/1.2)) )**2
    sar_est = k * (h/5)**2 * (f/100e3)**2 * 2 * 2e-4  # RL=5 A/m (i.e., 6.25 uT, from ICNIRP1998), BR=2 W/kg
    return sar_est

# SAR averaged over 1-g mass as per FCC and HC Code 6 (for head/torso)
def sar_1g(g, h, f): 
    k = ( 1/((1+2.5e-1*g**1.1)**(1/1.1)) )**2
    sar_est = k * (h/1.63)**2 * (f/100e3)**2 * 1.6 * 4.6e-5  # RL=1.63 A/m (i.e., 2.04 uT, from FCC), BR=1.6 W/kg
    return sar_est


# set negative gradients to zero, otherwise the coupling-factor calculation cannot proceed
def negative_gradient_check(g):
    neg_amount = g[g<0].size
    
    if neg_amount !=0:
        print('Number of negative-gradient data points fixed: {}'.format(neg_amount))
        g[np.argwhere(g<0)] = 0
    
    return g