""" Helper functions and tools to assist matplotlib (mpl) plotting

Copyright (c) 2016 Z43, Zurich, Switzerland
"""
import os
import logging
import copy
import numpy as np
from numpy.fft import fftshift
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes

#import numpy_ext as npx
#import pwe as pwe

_LOGGER = logging.getLogger(__name__)

# Defaults See http://matplotlib.org/users/customizing.html
mpl.rcParams['axes.titlesize'] = 'x-large'
mpl.rcParams['axes.labelsize'] = 13#'x-large' # fontsize of the x any y labels
mpl.rcParams['axes.grid'] = True # display grid or not
mpl.rcParams['font.size'] = 11

_DEFAULTS = mpl.rcParams.copy() # our defaults

def build_z43_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    red = np.array([0.0, 0.0, 0.0, 0.5, 0.75, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0])
    green = np.array([0.0, 0.22, 0.1, 0.0, 0.0, 0.0, 0.5, 0.75, 0.94, 1.0, 1.0])
    blue = np.array([0.0, 0.5, 0.75, 0.75, 0.5, 0.0, 0.0, 0.0, 0.25, 0.66, 1.0])

    colors = list(zip(red, green, blue))

    cmap = LinearSegmentedColormap.from_list('z43', colors, N=100)
    cmap.set_under('k')
    cmap.set_bad('grey')
    return cmap

Z43_CMAP = build_z43_cmap()

# Complex Field Plotter ------------------
class ComplexVectorPlotter(object):
    SPATIAL_LABELS = [(r'$x$', r'$y$'), (r'$x/\lambda$', r'$y/\lambda$'),
                      (r'$x{}$', r'$y{}$')]
    SPECTRAL_LABELS = [(r'$k_x$', r'$k_y$'), (r'$k_x/\beta$', r'$k_y/\beta$'),
                       (r'$k_x{}$', r'$k_y{}$')]

    def __init__(self):
        self.mag_levels = np.arange(-30, +1, 1)
        self.phs_levels = np.linspace(-180.0, 180., 10)

        self.fig = None
        self._ckargs = dict(cmap=Z43_CMAP) # contourf key arguments

    @classmethod
    def _check_data(cls, X, Y, data):
        assert X.shape == Y.shape
        assert isinstance(data, (tuple, list))

    @classmethod
    def _assert_equal_dimensions(cls, X, Y, data):
        assert isinstance(X, (tuple, list))
        assert isinstance(Y, (tuple, list))
        assert isinstance(data, (tuple, list))
        for Xi, Yi, Di in list(zip(X, Y, data)):
            assert np.all(Xi.shape == Di.shape)
            assert np.all(Yi.shape == Di.shape)

    @classmethod
    def _extend_grid_tuples(cls, X, Y, data):
        assert isinstance(X, (tuple, list))
        assert isinstance(Y, (tuple, list))
        assert isinstance(data, (tuple, list))
        assert len(X) == len(Y)
        if len(X) == len(data):
            return X, Y
        elif len(X) > 1:
            assert False, "Number of grids does not match number of data entries"
        else:
            return X*len(data), Y*len(data)

    @classmethod
    def _force_tuple(cls, data):
        if isinstance(data, np.ndarray):
            if np.ndim(data) == 2:
                data = [data, ]
            elif np.ndim(data) == 3:
                data = [d for d in data]
            else:
                raise ValueError("must be 2- or 3-dimensional data")
        return data

    @classmethod
    def _force_quantity_tuple(cls, data, length):
        if isinstance(data, (np.ndarray, tuple, list)):
            if cls._num_dims(data) == 2:
                data = [cls._force_tuple_of_len(data, length), ]
            elif cls._num_dims(data) == 3:
                data = [cls._force_tuple_of_len(d, length) for d in data]
            elif cls._num_dims(data) == 4:
                data = [[di for di in d] for d in data]
            else:
                raise ValueError("must be 2, 3 or 4-dimensional data")
        return data

    @classmethod
    def _num_dims(cls, data):
        if isinstance(data, str) or not hasattr(data, '__len__'):
            return 0

        return 1+cls._num_dims(data[0])

    @classmethod
    def _force_4d_tuple(cls, data, ninner=None, nouter=None):
        if isinstance(data, (np.ndarray, tuple, list)):
            if cls._num_dims(data) == 2:
                if ninner is None:
                    ninner = 1
                if nouter is None:
                    nouter = 1
                data = [[data, ]*ninner, ]*nouter
            elif cls._num_dims(data) == 3:
                if len(data) == ninner:
                    if len(data) == nouter:
                        raise ValueError("Ambiguous dimensions for input arrays. Try so specify "
                                         "parameters explicitely for all data and components.")
                    if nouter is None:
                        nouter = 1
                    data = [d for d in data]*nouter
                elif len(data) == nouter:
                    if ninner is None:
                        ninner = 1
                    data = [[d]*ninner for d in data]
                else:
                    if ninner is None and nouter is None:
                        ninner = 1
                        nouter = len(data)
                        data = [[d]*ninner for d in data]
                    else:
                        raise ValueError("3-dimensional data does neither correspond to nouter "
                                         "({}) or ninner ({})".format(nouter, ninner))
            elif cls._num_dims(data) == 4:
                data = [[di for di in d] for d in data]
            else:
                raise ValueError("must be 2, 3 or 4-dimensional data")
        return data

    @classmethod
    def _force_tuple_of_len(cls, data, length):
        data_list = cls._force_tuple(data)
        if len(data_list) == 1 and length > 1:
            return data_list*length
        return data_list

    @classmethod
    def _check_compare_data(cls, X_list, Y_list, data_list,
                            title_list=None, norm_plot_min=None, norm_plot_max=None):
        assert isinstance(data_list, (tuple, list))
        for i in range(len(data_list)-1):
            assert np.ndim(data_list[i]) == np.ndim(data_list[i+1])
        assert len(data_list) == len(X_list) or len(
            X_list) == 1 or isinstance(X_list, np.ndarray)
        assert len(data_list) == len(Y_list) or len(
            Y_list) == 1 or isinstance(Y_list, np.ndarray)
        assert len(X_list) == len(Y_list)
        assert isinstance(X_list, np.ndarray) == isinstance(Y_list, np.ndarray)
        if not isinstance(X_list, (tuple, list)):
            assert np.all(X_list.shape == Y_list.shape)
            for data in data_list:
                if np.ndim(data) == 2:
                    assert np.all(data.shape == X_list.shape)
                elif np.ndim(data) == 3:
                    for d in data:
                        assert np.all(d.shape == X_list.shape)
                else:
                    assert False  # should have 2 or 3 dimensions
        else:
            for (X, Y, data) in zip(X_list, Y_list, data_list):
                assert np.all(X.shape == Y.shape)
                if np.ndim(data) == 3:
                    for d in data:
                        assert np.all(d.shape == X.shape)
                else:
                    assert np.all(data.shape == X.shape)

        # check titles (either one per column or one per plot)
        if title_list is not None:
            assert len(data_list) == len(title_list)
            for (data, titles) in zip(data_list, title_list):
                if np.ndim(data) == 3:
                    assert len(data) == len(
                        titles) or isinstance(titles, str)
                elif np.ndim(data) == 2:
                    assert len(titles) == 1 or isinstance(titles, str)
                else:
                    assert False and "too many dimensions for input data"

        # check min/max of plots
        if norm_plot_min is not None:
            assert (len(data_list) == len(norm_plot_min) or len(data_list[0]) == len(norm_plot_min) or
                    (not isinstance(data_list[0], (tuple, list)) and len(norm_plot_min) == 1))
            for (data, values) in zip(data_list, norm_plot_min):
                if isinstance(values, (tuple, list)):
                    assert len(data) == len(values)
                else:
                    assert len(data) == len(norm_plot_min) or (not isinstance(data, (tuple, list)) and
                                                               np.isscalar(values))
        if norm_plot_max is not None:
            assert (len(data_list) == len(norm_plot_max) or len(data_list[0]) == len(norm_plot_max) or
                    (not isinstance(data_list[0], (tuple, list)) and len(norm_plot_max) == 1))
            for (data, values) in zip(data_list, norm_plot_max):
                if isinstance(values, (tuple, list)):
                    assert len(data) == len(values)
                else:
                    assert len(data) == len(norm_plot_max) or (not isinstance(data, (tuple, list)) and
                                                               np.isscalar(values))

    @classmethod
    def _force_compare_data(cls, X_list, Y_list, data_list, norm_plot_min=None, norm_plot_max=None):
        X_tmp = X_list
        Y_tmp = Y_list

        # determine how many subplots we'll have
        if np.ndim(data_list[0]) == 2:  # 1 row of subplots
            data_all = [np.array(cls._force_tuple(data)) for data in data_list]
        elif np.ndim(data_list[0]) == 3:  # multiple rows of subplots
            data_all = [np.array([d for d in data]) for data in data_list]
        else:
            assert False and 'expects 2d or n x 2d data'

        # make lists of the same length as data_list:
        if isinstance(X_list, np.ndarray):
            X_tmp = [X_list for data in data_all]
            Y_tmp = [Y_list for data in data_all]
        elif len(X_list) == 1:
            X_tmp = [X_list[0] for data in data_all]
            Y_tmp = [Y_list[0] for data in data_all]
        else:
            assert len(X_list) == len(data_all)
        X_all = [None]*len(data_all)
        Y_all = [None]*len(data_all)
        for (i, data) in enumerate(data_all):
            if len(X_tmp[i]) == 1 or isinstance(X_tmp[i], np.ndarray):
                X_all[i] = [X_tmp[i] for d in data]
                Y_all[i] = [Y_tmp[i] for d in data]
            elif len(X_tmp[i]) != len(data):
                assert False  # should be either length 1 or same dimension
            else:
                X_all[i] = [X for X in X_tmp]
                Y_all[i] = [Y for Y in Y_tmp]

        norm_plot_min_all = None
        norm_plot_max_all = None
        if norm_plot_min is not None:
            norm_plot_min_all = [None]*len(data_all)
            if len(data_all[0]) == len(norm_plot_min) and np.isscalar(norm_plot_max[0]):
                for (i, data) in enumerate(data_all):
                    norm_plot_min_all[i] = [v for v in norm_plot_min]
            elif len(data_all) == len(norm_plot_min):
                if np.isscalar(norm_plot_min[0]):
                    for (i, data) in enumerate(data_all):
                        norm_plot_min_all[i] = [norm_plot_min[i]
                                                for j in range(len(data))]
                for (i, data) in enumerate(data_all):
                    norm_plot_min_all[i] = [v for v in norm_plot_min[i]]

        if norm_plot_max is not None:
            norm_plot_max_all = [None]*len(data_all)
            if len(data_all[0]) == len(norm_plot_max) and np.isscalar(norm_plot_max[0]):
                for (i, data) in enumerate(data_all):
                    norm_plot_max_all[i] = [v for v in norm_plot_max]
            elif len(data_all) == len(norm_plot_max):
                if np.isscalar(norm_plot_max[0]):
                    for (i, data) in enumerate(data_all):
                        norm_plot_max_all[i] = [norm_plot_max[i]
                                                for j in range(len(data))]
                else:
                    for (i, data) in enumerate(data_all):
                        norm_plot_max_all[i] = [v for v in norm_plot_max[i]]

        return X_all, Y_all, data_all, norm_plot_min_all, norm_plot_max_all

    @classmethod
    def _get_param_dims(cls, param):
        # basestr, int/float, or np.ndarray parameters supported
        if isinstance(param, (int, float, str, np.ndarray)):
            # 1 parameter
            return 0
        elif any(isinstance(item, (int, float, str, np.ndarray)) for item in param):
            # "list/tuple of params
            return 1
        elif any([all(isinstance(subitem, (int, float, str, np.ndarray)) for subitem in item) for item in param]):
            # "list/tuple of lists/tuples of params"
            return 2
        elif all([all(subitem is None for subitem in item) for item in param]):
            return 2
        else:
            assert False and 'dimensions of parameterlist not supported'

    @classmethod
    def _assign_params_to_subplots(cls, data_all, parameter):
        ''' most parameters to contourf_compare can be given as :
            * one element (applied to all subplots)
            * list/tuple with number of rows (same applied to each row)
            * list/tuple of lists/tuples with same dimensions as data_list (individual for each subplot)
            a single parameter can be a scalar, ndarray or string
            this function expands the parameter accordingly, such that it has the
            same dimension as data_list as described above
        '''
        assert parameter is not None

        # data_all should already be in the format that we expect
        for data in data_all:
            assert np.ndim(data) == 3  # (rows x columns x 2D-data)

        if cls._get_param_dims(parameter) == 0:
            parameters_all = [[parameter for i in range(
                len(data_all[0]))] for j in range(len(data_all))]
        elif cls._get_param_dims(parameter) == 1:
            assert len(parameter) == len(data_all)
            parameters_all = [[parameter[i] for i in range(
                len(data_all[0]))] for j in range(len(data_all))]
        elif cls._get_param_dims(parameter) == 2:
            assert len(parameter) == len(data_all)
            for param, data in zip(parameter, data_all):
                assert len(param) == len(data)
            parameters_all = [[parameter[j][i] for i in range(
                len(data_all[0]))] for j in range(len(data_all))]
        else:
            assert False and 'dimensions of parameterlist not supported'

        return parameters_all

    @classmethod
    def _find_axes(cls, fig):
        if fig is None:
            fig = plt.gcf()
        ax = [a for a in fig.axes if hasattr(a, 'is_first_col')]
        return ax

    @classmethod
    def _subplots(cls, *args, **kargs):
        kargs['facecolor'] = 'white'
        fig, ax = plt.subplots(*args, **kargs)
        try:
            if len(ax.shape) == 1:
                ax = np.resize(ax, (kargs['nrows'], kargs['ncols']))
        except AttributeError: # no shape attribute
            ax = np.resize(ax, (1, 1))
        return fig, ax

    @classmethod
    def _colorbar(cls, h, ax, label, fmt=None, size="10%", pad=0.1, ticks=None, ticklabels=None):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        #  See
        #  http://stackoverflow.com/questions/18266642/multiple-imshow-subplots-each-with-colorbar
        # Create divider for existing axes instance
        divider = make_axes_locatable(ax)
        # Append axes to the right of ax3, with 5% width of ax3
        cax = divider.append_axes("right", size=size, pad=pad)
        cbar = plt.colorbar(h, cax=cax, format=fmt, label=label, ticks=ticks)
        if ticklabels is not None and ticks is not None:
            assert len(ticklabels) == len(ticks)
            cbar.ax.set_yticklabels(ticklabels)  # vertically oriented colorbar

    @classmethod
    def shift_input(cls, *args):
        _args = [fftshift(arg) for arg in args[:-1]]
        _args.append([fftshift(F) for F in args[-1]])
        return _args

    def plotcut(self, t, Fdata, sel=None):
        data = self._force_tuple(Fdata)

        if sel is None:
            sel = np.s_[:]

        self.fig, ax = self._subplots(
            nrows=len(data), ncols=2, figsize=(10, 5), sharex=True, sharey=False)

        for i, d in enumerate(data):
            mag = np.abs(d[sel])
            h = ax[i, 0].plot(t, 20.*np.log10(mag/np.nanmax(mag)))

            ax[i, 0].set_ylabel("Mag dB/max={:3.2e}".format(np.nanmax(mag)))

            phs = np.angle(d[sel])
            h = ax[i, 1].plot(t, phs*180./np.pi)

            ax[i, 1].set_ylabel("Phase deg")

        plt.tight_layout(h_pad=0, w_pad=0)

    def plotcut_spatial(self, t, Fdata, sel=None):
        data = self._force_tuple(Fdata)
        self.plotcut(t, data, sel)

    def plotcut_spectral(self, w, Fdata, sel=None):
        """ fftshifts inputs and plots contourf """
        data = self._force_tuple(Fdata)
        self.plotcut(*self.shift_input(w, data), sel=sel)

    def contourf(self, X, Y, data):
        data = self._force_tuple(data)
        self._check_data(X, Y, data)

        self.fig, ax = self._subplots(
            nrows=len(data), ncols=2, figsize=(15, 15), sharex=True, sharey=True)

        for i, d in enumerate(data):
            mag = np.abs(d)
            h = ax[i, 0].contourf(X, Y, 20.*np.log10(mag/np.nanmax(mag)),
                                  levels=self.mag_levels, extend='min', **self._ckargs)

            self._colorbar(
                h, ax[i, 0], "Mag dB/max={:3.2e}".format(np.nanmax(mag)))

            h = ax[i, 1].contourf(X, Y, np.angle(d)*180./np.pi,
                                  levels=self.phs_levels, **self._ckargs)

            self._colorbar(h, ax[i, 1], "Phase deg")

            for a in ax[i, :]:
                a.set_aspect('equal', 'datalim')
                a.grid(False)

        plt.tight_layout(h_pad=0, w_pad=0)

    def contourf_quantities(self, X, Y, data, **kwargs):
        ''' Conveniently plot different field quantities in subfigures

        Args:
            data : Array or list of shape : (nrows, ncols, Nx, Ny)
            X : Meshgrid with x-coordinates (unit assumed to be wavelength, ij-indexing)
            Y : Meshgrid with y-coordinates (unit assumed to be wavelength, ij-indexing)

        Keyword arguments:
            The following kwargs can be supplied as list of lists with the lengths nrows, ncols,
            or a single values, in which case the same is used for all subplots

            compute_quantities: String(s) indicating computed quantity you would like to plot.
            If not supplied, defaults are used for known quantitites E,H,S, else
                Supported compute_quantities:
                - 'mag_db' (default)
                - 'mag_db_field'
                - 'mag_db_power'
                - 'mag'
                - 'phase'
                - 'phase_deg'
                - 'real'
                - 'imag'
                - 'mag_spectrum'
                - 'mag_db_spectrum'
                - 'phase_spectrum'
                - 'phase_deg_spectrum'
                - '' # (no processing)

            data_quantities: String(s) inicating tha quantity of the data, e.g. 'E', 'H', 'S'

            titles: Titles for the subplots. I not supplied, will be deduced from
                compute_quantities and data_quantities

            db_ref: Reference value to use for db computation (i.e. 10*log10(val/db_ref)).
                Default is max value of the subplot

            levels_min: Minimum value for the contourf plot

            levels_max: Maximum value for the contourf plot

            plot_colorbar: Plot the colorbar for the given subplot. Default: True

            colorbar_label: Label of the colorbar. Default displays db_ref for db, min/max otherwise

            xlabel: Label of the x-axis (default is x/lambda for spatial, k_x/beta for spectral)

            ylabel: Label of the y-axis (default is y/lambda for spatial, k_y/beta for spectral)

            ---

            The following kwargs are global i.e. only one value is accepted.

            warn_about_nans: Warn if NaNs occur in the data to be plotted.
                Default is False

            plot_rows_as_cols: Transpose the whole figure.
                Default is False

            tight_subplots: Squeezes the subplots together to save space, mainly for printing.
                Default is False

            figsize: Tuple (width, height) indicating figure size for plt.subplots.
                Default is 4 per column/row

            wspace_adjust : Scalar value to add to width spacing between subplots if tight_subplots is enabled.
                Default is 0

            hspace_adjust : Scalar value to add to height spacing between subplots if tight_subplots is enabled.
                 Default is 0

            nan_color: matplotlib color which is used to render NaN values
                Default is 'grey'


        Examples:
            minimal:
                contourf_quantities(Xobs_n, Yobs_n, [e_meas_rms_tot, h_meas_rms_tot])

            plot components as rows, different compute quantities:
                contourf_quantities(Xobs_n, Yobs_n,
                         [e_meas_rms, h_meas_rms, s_meas],
                         data_quantities=[['Ex', 'Ey', 'Ez'], ['Hx', 'Hy', 'Hz'], ['Sx', 'Sy', 'Sz']],
                         compute_quantities=[['mag_db',]*3, ['mag_db',]*3, ['real',]*3],
                         plot_rows_as_cols=True)

            normalize to component y/x, plot colorbar only on right side, show -20 to 0 dB:
                db_ref_E = np.abs(e_meas_rms[1]).max()
                db_ref_H = np.abs(h_meas_rms[0]).max()
                contourf_quantities(Xobs_n, Yobs_n,
                             [e_meas_rms, h_meas_rms],
                             data_quantities=[['Ex', 'Ey', 'Ez'],
                                              ['Hx', 'Hy', 'Hz']],
                             compute_quantities=[['mag_db',]*3, ['mag_db',]*3],
                             db_ref = [[db_ref_E, db_ref_E, db_ref_E],
                                       [db_ref_H, db_ref_H, db_ref_H]],
                             plot_colorbar = [[False, False, True],
                                              [False, False, True]],
                             levels_min = -20.,
                             levels_max = 0.)

            use custom quantity Ohm:
                qR = Quantity('R', unit_str='Ohm')
                contourf_quantities2(cplt, Xobs_n, Yobs_n, [R1, R2],
                         data_quantities=[qR, qR],
                         compute_quantities='mag_db')

            compare measurement and simulation H_tot, Sx,Sy,Sz, and export as PDF:
                contourf_quantities(Xobs_n, Yobs_n,
                         [[h_sim_rms_tot, h_meas_rms_tot*scale_meas],
                          [np.real(s_sim[0]), np.real(s_meas[0]*scale_meas**2)],
                          [np.real(s_sim[1]), np.real(s_meas[1]*scale_meas**2)],
                          [np.real(s_sim[2]), np.real(s_meas[2]*scale_meas**2)]],

                         data_quantities=[['Htot', 'Htot'],
                                          ['Sx', 'Sx'],
                                          ['Sy', 'Sy'],
                                          ['Sz', 'Sz']],

                         plot_colorbar=[[False, True],
                                        [False, True],
                                        [False, True],
                                        [False, True]],

                         titles=[['|$H_\mathrm{tot}$| sim.', '|$H_\mathrm{tot}$| rec. from meas.'],
                                 ['$\operatorname{Re}(S_x)$  sim.', '$\operatorname{Re}(S_x)$ rec. from meas.'],
                                 ['$\operatorname{Re}(S_y)$  sim.', '$\operatorname{Re}(S_y)$ rec. from meas.'],
                                 ['$\operatorname{Re}(S_z)$  sim.', '$\operatorname{Re}(S_z)$ rec. from meas.']],

                         db_ref=[[db_ref_H_rms, db_ref_H_rms],
                                 [0,0],
                                 [0,0],
                                 [0,0]],

                         levels_min=[[None, None],
                                    [min_Sx, min_Sx],
                                    [min_Sy, min_Sy],
                                    [min_Sz, min_Sz]],
                         levels_max=[[None, None],
                                    [max_Sx, max_Sx],
                                    [max_Sy, max_Sy],
                                    [max_Sz, max_Sz]],

                         colorbar_label=[['db/ref={} A/m (RMS)'.format(db_ref_H_rms),]*2,
                                         ['W/m$^2$', 'W/m$^2$'],
                                         ['W/m$^2$', 'W/m$^2$'],
                                         ['W/m$^2$', 'W/m$^2$']],

                         tight_subplots=True)

                # export PDF
                from pyrec.mpl_helpers import save_figure
                save_figure('TestFigure_z_{:.2g}_lambda_Htot_Sz'.format(first_measurement_plane_m/wavelength).replace('.', 'p'), fig=cplt.fig, destdir="./")

        '''
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # force the data to always have 4 dimensions (nrows, ncols, Nx, Ny)
        data_all = self._force_4d_tuple(data)
        nrows = len(data_all)
        ncols = len(data_all[0])
        X_all = self._force_4d_tuple(X, ncols, nrows)
        Y_all = self._force_4d_tuple(Y, ncols, nrows)

        def expand_or_assert(arg, argname):
            if isinstance(arg, str) or not hasattr(arg, '__len__'):
                return [[arg, ]*ncols]*nrows
            else:
                if ncols == 1:
                    if len(arg) == nrows and isinstance(arg[0], str) or not hasattr(arg[0], '__len__'):
                        return [[arg[r], ] for r in range(nrows)]
                assert nrows == len(arg) and hasattr(arg[0], '__len__') and ncols == len(
                    arg[0]), "dimensions of parameter '{}' must match data array".format(argname)
                return arg

        # potentially per-plot paramters
        compute_quantities = kwargs.pop('compute_quantities', None)
        data_quantities = kwargs.pop('data_quantities', None)
        titles          = kwargs.pop('titles', None)
        wavelength      = kwargs.pop('wavelength', 1.0)
        db_ref          = kwargs.pop('db_ref', 'max')
        levels_min      = kwargs.pop('levels_min', None)
        levels_max      = kwargs.pop('levels_max', None)
        plot_colorbar   = kwargs.pop('plot_colorbar', None)
        colorbar_label  = kwargs.pop('colorbar_label', None)
        xlabel          = kwargs.pop('xlabel', None)
        ylabel          = kwargs.pop('ylabel', None)

        compute_quantities = expand_or_assert(compute_quantities, 'compute_quantities')
        data_quantities = expand_or_assert(data_quantities, 'data_quantities')
        titles          = expand_or_assert(titles, 'titles')
        wavelength      = expand_or_assert(wavelength, 'wavelength')
        db_ref          = expand_or_assert(db_ref, 'db_ref')
        levels_min      = expand_or_assert(levels_min, 'levels_min')
        levels_max      = expand_or_assert(levels_max, 'levels_max')
        plot_colorbar   = expand_or_assert(plot_colorbar, 'plot_colorbar')
        colorbar_label  = expand_or_assert(colorbar_label, 'colorbar_label')
        xlabel          = expand_or_assert(xlabel, 'xlabel')
        ylabel          = expand_or_assert(ylabel, 'ylabel')

        # global paramters
        tight_subplots     = kwargs.pop('tight_subplots', False)
        warn_about_nans    = kwargs.pop('warn_about_nans', False)
        wspace_adjust      = kwargs.pop('wspace_adjust', 0.0)
        hspace_adjust      = kwargs.pop('hspace_adjust', 0.0)
        nan_color          = kwargs.pop('nan_color', 'grey')

        plot_rows_as_cols  = kwargs.pop('plot_rows_as_cols', False)
        # switch rows/cols if required (after params above are expanded)
        if plot_rows_as_cols:
            nrows, ncols = ncols, nrows

        figsize            = kwargs.pop('figsize', (4.2*ncols, 4*nrows ))

        if kwargs:
            raise TypeError('Got unexpected keyword arguments {}'.format(kwargs))

        all_spectral = np.all([np.all([quantity is not None and 'spectrum' in quantity for quantity in compute_quantities_col]) for compute_quantities_col in compute_quantities])
        all_spatial = np.all([np.all([quantity is not None and 'spectrum' not in quantity for quantity in compute_quantities_col]) for compute_quantities_col in compute_quantities])
        all_spatial = all_spatial or np.all([np.all([quantity is None for quantity in compute_quantities_col]) for compute_quantities_col in compute_quantities])

        share_axes = all_spectral or all_spatial
        if share_axes and tight_subplots:
            self.fig, ax = self._subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
        else:
            self.fig, ax = self._subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=False, sharey=False)

        for ii, (Di_col, X_col, Y_col, data_quantity_col, plot_quant_col, title_col, 
                db_ref_col, plot_colorbar_col) in enumerate(zip(
            data_all, X_all, Y_all, data_quantities, compute_quantities, titles, db_ref, plot_colorbar)):

            for jj, (Di, X, Y, plot_quant, data_quant, title, dbref, cbar) in enumerate(zip(
                Di_col, X_col, Y_col, plot_quant_col, data_quantity_col, title_col, 
                db_ref_col, plot_colorbar_col)):

                i,j = ii,jj
                if plot_rows_as_cols:
                    i,j = jj,ii

                Xq = X
                Yq = Y

                if plot_quant is not None and ('_spectrum' in plot_quant or isinstance(data_quant, SpectralQuantity)):
                    qo = SpectralQuantity(data_quant, eval_opt=plot_quant.replace('_spectrum', ''), db_ref=dbref, levels_min=levels_min[ii][jj], levels_max=levels_max[ii][jj] )
                    Xq, Yq, Dq = qo.compute_from_spatial(X*wavelength[ii][jj], Y*wavelength[ii][jj], Di, wavelength=wavelength[ii][jj], warn_about_nans=warn_about_nans)
                else:
                    # if data_quant is not a string
                    if data_quant is not None and isinstance(data_quant, Quantity):
                        qo = copy.copy(data_quant)
                        if plot_quant is not None:
                            qo.eval_opt = plot_quant 
                        if dbref is not None:
                            qo.db_ref = dbref
                        if levels_min[ii][jj] is not None: 
                            qo.levels_min = levels_min[ii][jj] 
                        if levels_max[ii][jj] is not None:
                            qo.levels_min = levels_max[ii][jj] 
                    else:
                        qo = Quantity(data_quant, eval_opt=plot_quant, db_ref=dbref, levels_min=levels_min[ii][jj], levels_max=levels_max[ii][jj])
                    Dq = qo.compute_quantity(Di, warn_about_nans=warn_about_nans)

                _msg = "Coordinates and data shape mismatch ({} : {} != data : {}). Maybe try to specify X,Y for all components."
                assert np.all(Xq.shape == Dq.shape), _msg.format("X", Xq.shape, Dq.shape)
                assert np.all(Yq.shape == Dq.shape), _msg.format("Y", Yq.shape, Dq.shape)

                h = ax[i,j].contourf(Xq, Yq, Dq, levels=qo.plt_levels(), extend=qo.plt_extend(), **self._ckargs)
                # mark NaNs with defined color
                ax[i,j].set_facecolor(nan_color)

                # only allow to leave out colorbar if they are the same
                can_skip_colorbar = True
                if plot_rows_as_cols:
                    for l in range(ncols-1):
                        for k in range(nrows):
                            if compute_quantities[l][k] is not None and '_db' in compute_quantities[l][k]:
                                can_skip_colorbar = can_skip_colorbar and db_ref[l][k] != 'max' and db_ref[l+1][k] != 'max'
                                can_skip_colorbar = can_skip_colorbar and db_ref[l][k] == db_ref[l+1][k]
                            can_skip_colorbar = can_skip_colorbar and levels_min[l][k] == levels_min[l+1][k]
                            can_skip_colorbar = can_skip_colorbar and levels_max[l][k] == levels_max[l+1][k]
                            can_skip_colorbar = can_skip_colorbar and colorbar_label[l][k] == colorbar_label[l+1][k]
                            can_skip_colorbar = can_skip_colorbar and compute_quantities[l][k] == compute_quantities[l+1][k]
                else:
                    for l in range(nrows):
                        for k in range(ncols-1):
                            if compute_quantities[l][k] is not None and '_db' in compute_quantities[l][k]:
                                can_skip_colorbar = can_skip_colorbar and db_ref[l][k] != 'max' and db_ref[l][k+1] != 'max'
                                can_skip_colorbar = can_skip_colorbar and db_ref[l][k] == db_ref[l][k+1]
                            can_skip_colorbar = can_skip_colorbar and levels_min[l][k] == levels_min[l][k+1]
                            can_skip_colorbar = can_skip_colorbar and levels_max[l][k] == levels_max[l][k+1]
                            can_skip_colorbar = can_skip_colorbar and colorbar_label[l][k] == colorbar_label[l][k+1]
                            can_skip_colorbar = can_skip_colorbar and compute_quantities[l][k] == compute_quantities[l][k+1]

                if cbar is not None:
                    print_colorbar = cbar
                else:
                    if can_skip_colorbar:
                        if tight_subplots and not ax[i,j].is_last_col():
                            print_colorbar = False
                        else:
                            print_colorbar = True
                    else:
                        print_colorbar = True

                if print_colorbar:
                    cbar_label = qo.range_str() if colorbar_label[ii][jj] is None else colorbar_label[ii][jj]
                    self._colorbar(h, ax[i,j], cbar_label, ticks=qo.colorbar_ticks(), ticklabels=qo.colorbar_ticklabels(), size="5%", pad=0.05)
                else:
                    divider = make_axes_locatable(ax[i,j])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cax.axis('off')

                if title is not None:
                    ax[i,j].set_title(title)
                else:
                    ax[i,j].set_title(qo.desc())

                if plot_quant is not None and ('_spectrum' in plot_quant or isinstance(data_quant, SpectralQuantity)):
                    xlabel_str = r'$k_x/\beta$' if xlabel[ii][jj] is None else xlabel[ii][jj]
                    ylabel_str = r'$k_y/\beta$' if ylabel[ii][jj] is None else ylabel[ii][jj]
                    if not all_spectral or ax[i,j].is_first_col():
                        ax[i,j].set_ylabel( ylabel_str )
                    if not all_spectral or ax[i,j].is_last_row():
                        ax[i,j].set_xlabel( xlabel_str )
                    add_visible_region(beta=1, ax=ax[i,j])
                else:
                    xlabel_str = r'$x/\lambda$' if xlabel[ii][jj] is None else xlabel[ii][jj]
                    ylabel_str = r'$y/\lambda$' if ylabel[ii][jj] is None else ylabel[ii][jj]            
                    if not tight_subplots or (not all_spatial or ax[i,j].is_first_col()):
                        ax[i,j].set_ylabel( ylabel_str )
                    if not tight_subplots or (not all_spatial or ax[i,j].is_last_row()):
                        ax[i,j].set_xlabel( xlabel_str )

                ax[i,j].set_aspect('equal', 'datalim')
                ax[i,j].grid(False)

        if tight_subplots:
            plt.tight_layout(h_pad=(-0.5 + hspace_adjust), w_pad=(-0.2 + wspace_adjust))
        else:
            if share_axes:
                plt.tight_layout(h_pad=0.4, w_pad=0.4)
            else:
                plt.tight_layout(h_pad=0.8, w_pad=0.8)
        self.ax = ax


    def contourf_compare(self, X_list, Y_list, data_list, title_list=None, quantity='mag_db', 
                                sharex=True, sharey=True, norm_plot_tocolumn=None, norm_plot_min=None, norm_plot_max=None, 
                                contourf_levels=None, unit_str='', colorbar_labels=None, colorbar_ticks=None, tight_subplots=False, **kargs):
        """ plots the data in the list next to each other
            data_list can be a list of tuples/lists, e.g. with multiple components
            in this case the components will be plotted below each other

            A few example usages:

            - e_field is 3 x 64 x 64 array or list of 3 64x64 arrays [Ex, Ey, Ez]. to plot phase and magnitude:
            cplt.contourf_compare(
                X/lambdas, Y/lambdas, 
                [e_field, e_field],
                [[r'$|E_{}|$'.format(c) for c in ['x', 'y', 'z']], [r'$\angle E_{}$'.format(c) for c in ['x', 'y', 'z']]], 
                quantity=[['mag_db' for i in range(3)], ['phase' for i in range(3)]],
                sharex=False, sharey=False, 
                unit_str = 'V/m' )

            - to compare three columns, normalize to the third column:
            cplt.contourf_compare(
                X/lambdas, Y/lambdas, 
                [e_meas, e_rec, e_sim], 
                [[r'$E_{}$ meas'.format(c) for c in ['x', 'y', 'z']], [r'$E_{}$ rec'.format(c) for c in ['x', 'y', 'z']], [r'$E_{}$ sim'.format(c) for c in ['x', 'y', 'z']]],
                norm_plot_tocolumn=2,
                sharex=False, sharey=False, 
                unit_str = 'V/m' )


        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # \todo SP: check unit_str (if multiple given)
        self._check_compare_data(X_list, Y_list, data_list, title_list, norm_plot_min, norm_plot_max)
        X_all, Y_all, data_all, norm_plot_min_all, norm_plot_max_all = self._force_compare_data(X_list, Y_list, data_list, norm_plot_min, norm_plot_max)

        # expand other params:
        expanded_colorbar_ticks = None if colorbar_ticks is None else self._assign_params_to_subplots(data_all, colorbar_ticks)
        if expanded_colorbar_ticks is not None:
            assert len(expanded_colorbar_ticks) == len(data_all)

        expanded_colorbar_labels = None if colorbar_labels is None else self._assign_params_to_subplots(data_all, colorbar_labels)
        if expanded_colorbar_labels is not None:
            assert len(expanded_colorbar_labels) == len(data_all)

        expanded_contourf_levels = None if contourf_levels is None else self._assign_params_to_subplots(data_all, contourf_levels)
        if expanded_contourf_levels is not None:
            assert len(expanded_contourf_levels) == len(data_all)
        
        expanded_quantities = None if quantity is None else self._assign_params_to_subplots(data_all, quantity)
        if expanded_quantities is not None:
            assert len(expanded_quantities) == len(data_all)

        ncompare = len(data_all)

        if len(data_all[0]) == 1:
            self.fig, ax = self._subplots(nrows=len(data_all[0]), ncols=ncompare, figsize=(ncompare*5, len(data_all[0])*3.75 ), sharex=sharex, sharey=sharey)
        else:
            self.fig, ax = self._subplots(nrows=len(data_all[0]), ncols=ncompare, figsize=(ncompare*5, len(data_all[0])*4 ), sharex=sharex, sharey=sharey)

        kallargs = self._ckargs.copy()
        kallargs.update(kargs) # user supplied-arguments take higher priority

        for j, data in enumerate(data_all): # the columns to compare
            Xj_list = X_all[j]
            Yj_list = Y_all[j]
            if title_list is not None:
                title_str_or_list = title_list[j]
            for i, d in enumerate(data): # loop over the rows (likely components)
                X = Xj_list[i]
                Y = Yj_list[i]

                colorbar_str = unit_str
                if isinstance(unit_str, (tuple, list)):
                    colorbar_str = unit_str[j][i]

                ticks = None
                if expanded_colorbar_ticks is not None:
                    assert len(expanded_colorbar_ticks[j])==len(data)
                    ticks = expanded_colorbar_ticks[j][i]

                label = None
                if expanded_colorbar_labels is not None:
                    assert len(expanded_colorbar_labels[j])==len(data)
                    label = expanded_colorbar_labels[j][i]

                levels = None
                if expanded_contourf_levels is not None:
                    assert len(expanded_contourf_levels[j])==len(data)
                    levels = expanded_contourf_levels[j][i]

                quantity=='mag_db'
                if expanded_quantities is not None:
                    assert len(expanded_quantities[j])==len(data)
                    quantity = expanded_quantities[j][i]

                if quantity=='mag_db':
                    mag = np.abs(d)
                    normval = 1
                    if norm_plot_max is not None:
                        normval=norm_plot_max_all[j][i]
                    elif norm_plot_tocolumn is not None:
                        normval=np.nanmax(np.abs(data_all[norm_plot_tocolumn][i]))
                    else:
                        normval=np.nanmax(mag)
                    assert np.isscalar(normval)
                    
                    # defaults for magnitude plot
                    mag_levels = self.mag_levels if levels is None else levels
                    colorbar_label = "Mag dB/ref={:.3g} {}".format(normval, colorbar_str) if label is None else label

                    # default is extend='min' for magnitude
                    kallargs.update(dict(extend='min'))
                    kallargs.update(kargs) # user supplied-arguments take higher priority

                    h = ax[i,j].contourf(X, Y, 20.*np.log10(mag/normval), levels=mag_levels, **kallargs)
                    
                    if norm_plot_tocolumn is not None:
                        if j==norm_plot_tocolumn:
                            self._colorbar(h, ax[i,j], label=colorbar_label, size="5%", pad=0.05, ticks=ticks)
                        else:
                            divider = make_axes_locatable(ax[i,j])
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            cax.axis('off')
                    else:
                        self._colorbar(h, ax[i,j], label=colorbar_label, ticks=ticks)
                
                elif quantity=='phase':

                    # defaults for phase plots
                    phs_levels = self.phs_levels if levels is None else levels
                    colorbar_label = "Phase (deg)" if label is None else label

                    h = ax[i,j].contourf(X, Y, np.angle(d)*180./np.pi, levels=phs_levels, **kallargs)

                    # for phase always colorbar only on rightmost column
                    norm_plot_tocolumn = len(data_all)-1
                    if j==len(data_all)-1:
                        self._colorbar(h, ax[i,j], label=colorbar_label, size="5%", pad=0.05, ticks=ticks)
                    else:
                        divider = make_axes_locatable(ax[i,j])
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cax.axis('off')

                elif quantity=='real':
                    lower = 0
                    upper = 1

                    if norm_plot_max is not None:
                        upper=norm_plot_max_all[j][i]
                    elif norm_plot_tocolumn is not None:
                        upper=np.nanmax(np.real(data_all[norm_plot_tocolumn][i]))*1.000001
                    else:
                        upper = np.nanmax(np.real(d))*1.000001
                    assert np.isscalar(upper)

                    if norm_plot_min is not None:
                        lower=norm_plot_min_all[j][i]
                    elif norm_plot_tocolumn is not None:
                        lower=np.nanmin(np.real(data_all[norm_plot_tocolumn][i]))
                    else:
                        lower = np.nanmin(np.real(d))
                    assert np.isscalar(lower)

                    # defaults for real plots
                    real_levels = np.linspace(lower, upper, 30) if levels is None else levels
                    colorbar_label = "[{:.3g} .. {:.3g}] {}".format(lower, upper, colorbar_str) if label is None else label

                    if upper > lower: # is not the case if input is all zero
                        h = ax[i,j].contourf(X, Y, np.real(d), vmin=lower, vmax=upper, levels=real_levels, **kallargs)
                    else:
                        h = ax[i,j].contourf(X, Y, np.real(d), **kallargs)

                    if norm_plot_tocolumn is not None:
                        if j==norm_plot_tocolumn:
                            self._colorbar(h, ax[i,j], label=colorbar_label, size="5%", pad=0.05, ticks=ticks)
                        else:
                            divider = make_axes_locatable(ax[i,j])
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            cax.axis('off')
                    else:
                        self._colorbar(h, ax[i,j], label=colorbar_label, ticks=ticks)
                else:
                    assert False # unknown quantity


                # axis labels:
                if not tight_subplots or i==len(data_all[0])-1:
                    ax[i,j].set_xlabel('$x/\lambda$')
                if not tight_subplots or j==0:
                    ax[i,j].set_ylabel('$y/\lambda$')
                
                # remove tick labels if tight_subplots is enabled:
                if tight_subplots and i!=len(data_all[0])-1:
                    current_labels = [item.get_text() for item in ax[i,j].get_xticklabels()]
                    empty_string_labels = ['']*len(current_labels)
                    ax[i,j].set_xticklabels(empty_string_labels)
                if tight_subplots and j!=0:
                    current_labels = [item.get_text() for item in ax[i,j].get_yticklabels()]
                    empty_string_labels = ['']*len(current_labels)
                    ax[i,j].set_yticklabels(empty_string_labels)
                

                if (title_list is not None) and isinstance(title_str_or_list, str) and i==0:
                    ax[i,j].set_title(title_str_or_list)
                elif (title_list is not None) and  isinstance(title_str_or_list, (tuple, list)):
                    ax[i,j].set_title(title_str_or_list[i])
            
                for a in ax[i,:]:
                    a.set_aspect('equal', 'datalim')
                    a.grid(False)
        
        self.ax = ax
        if tight_subplots and (norm_plot_tocolumn is not None):
            plt.subplots_adjust(wspace=-0.35, hspace=0.1)
        else:
            plt.tight_layout(h_pad=1, w_pad=1)

    def contourf_compare_ehs(self, Xobs_n, Yobs_n, 
                            field_list, 
                            title_suffix_list=None, 
                            plot_components_inds=None, 
                            norm_to_component=None, 
                            field_quantity='E',
                            plot_quantity=None,
                            components_subscripts=None,
                            tight_subplots=True,
                            print_peak_values=False,
                            unit_str='',
                            warn_about_nans=True):
        ''' Plot fields next to each other, using reasonable defaults
        
            Expects E-field in V/m (RMS), H-field in A/m (RMS), S in W/m^2
        
        
            contourf_compare_ehs( [Xobs_n, Xobs2_n], [Yobs_n, Yobs2_n], [e_meas_rms, e_meas2_rms], 
                         title_suffix_list=[' measurement 1', ' measurement 2'],
                         norm_to_component=NORM_TO_COMPONENT,
                         field_quantity='E')
        
            Example:
                h_meas : 4 x 20 x 20 field data
                h_sim  : 4 x 24 x 24 field data
            
                Xobs_n, Yobs_n         : 20 x 20 meshgrid of locations ('ij' indexing, normalized to lambda)
                Xobs_n_sim, Yobs_n_sim : 24 x 24 meshgrid of locations ('ij' indexing, normalized to lambda)
            
                title_suffix_list = ['measured', 'simulated']
                norm_to_component = [0, 3]  # column 0, component 3 (e.g. vector total, is computed if only 3 components are supplied)
                plot_quantity='mag_db' # or 'phase'
        
                contourf_compare_ehs([Xobs_n, Xobs_n_sim], [Yobs_n, Yobs_n_sim], [h_meas, h_sim], 
                                     title_suffix_list, 
                                     norm_to_component=norm_to_component,
                                     plot_quantity='mag_db',
                                     field_quantity='H')
        
        '''
    
        #
        # make sure input data is what we expect
        #
        if unit_str == '' and (not plot_quantity=='phase') and (not (field_quantity.upper()=='E' or field_quantity.upper()=='H' or field_quantity.upper()=='S')):
            print("Unknown field quantity, plotting magnitude without units")
        if title_suffix_list is None:
            title_suffix_list = ['' for e in range(len(field_list))]
        assert len(field_list)==len(title_suffix_list)
    
        for i in range(len(field_list)-1):
            assert len(field_list[i])==len(field_list[i+1])
    
        for i in range(len(field_list)):
            for j in range(len(field_list[i])-1):
                assert np.ndim(field_list[i][j])==2
                assert np.ndim(field_list[i][j+1])==2
                assert np.all(field_list[i][j].shape == field_list[i][j+1].shape)
            
        if plot_quantity is None:
            plot_quantity = 'mag_db'
            if field_quantity.upper()=='S':
                plot_quantity = 'real' 
    
        # make a copy so we do not alter original data
        field_list = [np.array(e.copy()) for e in field_list]
    
        # components x,y,z
        num_rows = len(field_list[0])  # 4 to include total
        if plot_components_inds is None:
            plot_components_inds = range(num_rows)
        num_cols = len(field_list)  
        # if 3 components are supplied, compute vector total
        if num_rows==3:
            for i in range(num_cols):
                f_tot = np.sqrt( np.abs(field_list[i][0])**2 + np.abs(field_list[i][1])**2 + np.abs(field_list[i][2])**2)
                if plot_quantity=='real':
                    f_tot = np.sqrt( np.real(field_list[i][0])**2 + np.real(field_list[i][1])**2 + np.real(field_list[i][2])**2)
                field_list[i] = np.concatenate( (field_list[i], np.array([f_tot])) )
        num_rows = len(field_list[0])
        
        if norm_to_component is not None:
            assert len(norm_to_component)==2
            assert norm_to_component[0] < num_cols
            assert norm_to_component[1] < num_rows     
    
        assert len(field_list[i])>=num_rows # in case we want to introduce additional args later
    
        if components_subscripts is not None:
            assert len(components_subscripts)==len(field_list[0])
    
    
        #
        # replace exactly 0 with something small for plotting, and warn if it has NaNs
        #
        has_NaNs = False
        for i in range(len(field_list)):
            mask = field_list[i] == 0    
            field_list[i][mask] = 1e-20
            if np.any(np.isnan(field_list[i])):
                has_NaNs = True
        if has_NaNs and warn_about_nans:
            _LOGGER.warn("input data contains NaNs")
    
        #
        # set min/max of plots
        #
        if plot_quantity=='mag_db':
            #
            # plot min/max
            #
            max_vec = [[np.nanmax(np.abs(field_list[j][plot_components_inds[i]])) for i in range(len(plot_components_inds))] for j in range(num_cols)]
            #for j in range(num_cols):
            #    for i in range(len(plot_components_inds)):
            #        max_vec[j][i] = np.abs(field_list[j][plot_components_inds[i]]).max() # second colum
            #        # ONE SIGNIFICANT DIGIT ACCORDING TO NIELS' REQUEST: - only for norm to tot
            #        #max_vec[0][i] = float('{:.1g}'.format(np.abs(e_meas[i]).max()))
            if norm_to_component is not None:
                #max_vec = [[np.abs(e_sim[3]).max() for i in range(4)] for j in range(3)]
                # ONE SIGNIFICANT DIGIT ACCORDING TO NIELS' REQUEST:
                #print("Normalizing to component index {} (assumed total) of column {}".format(norm_to_component[1], norm_to_component[0]))
                max_vec = [[ float('{:.1g}'.format(np.nanmax(np.abs(field_list[norm_to_component[0]][norm_to_component[1]])))) for i in plot_components_inds] for j in range(num_cols)]
            min_vec = None
        
        elif plot_quantity=='real':        
            max_vec = [[np.nanmax(np.real(field_list[j][plot_components_inds[i]])) for i in range(len(plot_components_inds))] for j in range(num_cols)]
            min_vec = [[np.nanmin(np.real(field_list[j][plot_components_inds[i]])) for i in range(len(plot_components_inds))] for j in range(num_cols)]
            # ONE SIGNIFICANT DIGIT ACCORDING TO NIELS' REQUEST:
            #max_vec = [[ float('{:.1g}'.format(np.real(e_sim[i]).max())) for i in range(num_rows)] for j in range(num_cols)]
            #min_vec = [[ float('{:.1g}'.format(np.real(e_sim[i]).min())) for i in range(num_rows)] for j in range(num_cols)]

            #for j in range(num_cols):
            #    for i in range(len(plot_components_inds)):
            #        max_vec[0][i] = np.real(e_meas[i]).max() # second colum
            #        min_vec[0][i] = np.real(e_meas[i]).min() # second colum

            # for Poynting vector use special scaling:
            # x, y: [-Stot/2, Stot/2]. z: [0, Stot]
            if norm_to_component is not None:
                #print("Normalizing to component index {} (assumed total) of column {}".format(norm_to_component[1], norm_to_component[0]))
                Stot = np.nanmax(np.real(field_list[norm_to_component[0]][norm_to_component[1]]))
            
                for j in range(num_cols):
                    for i in range(len(plot_components_inds)):
                        # x, y:
                        if plot_components_inds[i] in [0, 1]:
                            # ONE SIGNIFICANT DIGIT ACCORDING TO NIELS' REQUEST:
                            max_vec[j][i] = float('{:.1g}'.format( Stot/4))
                            min_vec[j][i] = float('{:.1g}'.format(-Stot/4))

                        # z, tot
                        elif plot_components_inds[i] in [2, 3]:
                            max_vec[j][i] = float('{:.1g}'.format(Stot))
                            min_vec[j][i] = 0
                        
        elif plot_quantity=='phase':
            max_vec = None
            min_vec = None
        else:
            raise ValueError('unknown plot quantity. must be ''real'',  ''mag_db'' or  ''phase''')
        
    
        colorbar_ticks =  [[None]*len(plot_components_inds) for j in range(num_cols)]
        colorbar_labels = [[None]*len(plot_components_inds) for j in range(num_cols)]
        contourf_levels = [[None]*len(plot_components_inds) for j in range(num_cols)]
    
        #
        # colorbar label, ticks, level etc.
        #
        if unit_str == '':
            if field_quantity.upper()=='S':
                unit_str='W/m$^2$'
            elif field_quantity.upper()=='E':
                unit_str='V/m (RMS)'
            elif field_quantity.upper()=='H':
                unit_str='A/m (RMS)'
    
        if plot_quantity=='mag_db':
            # colorbar label
            column_labels = ['dB/ref={{:.2g}} {}'.format(unit_str) for i in range(num_cols)]
            if norm_to_component is not None: # make sure it's printed as Niels' wants it
                column_labels = ['dB/ref={{:1g}} {}'.format(unit_str) for i in range(num_cols)]
            # set for all plots
            for j in range(num_cols): # meas, sim, meas/sim
                label = column_labels[j]
                for i in range(len(plot_components_inds)): # components
                    colorbar_ticks[j][i] = None # should use default
                    colorbar_labels[j][i] = label.format(max_vec[j][i])
                    contourf_levels[j][i] = None # should use default

        elif plot_quantity=='real':
            # colorbar label
            column_labels = ['{}'.format(unit_str) for i in range(num_cols)]
            # set for all plots
            for j in range(num_cols): # meas, sim, meas/sim
                label = column_labels[j]
                for i in range(len(plot_components_inds)): # components
                    colorbar_ticks[j][i] = np.linspace(min_vec[j][i], max_vec[j][i], 7)
                    # ONE SIGNIFICANT DIGIT ACCORDING TO NIELS' REQUEST:
                    if norm_to_component is not None:
                        colorbar_ticks[j][i] = np.array([ float('{:.2g}'.format(e)) for e in colorbar_ticks[j][i]])
                    colorbar_labels[j][i] = label
                    contourf_levels[j][i] = np.linspace(min_vec[j][i], max_vec[j][i], 31)
        
        elif plot_quantity=='phase':        
            column_labels = None
            colorbar_labels = None
            contourf_levels = None
        else:
            raise ValueError('unknown plot quantity. must be ''real'',  ''mag_db'' or  ''phase''')
        
    
        norm_plot_tocolumn = num_cols-1 if norm_to_component is not None else None
    
        # try to guess component labels
        if components_subscripts is None:
            if num_rows==1:
                components_subscripts=['\mathrm{tot}']
            elif num_rows==3:
                components_subscripts=['x', 'y', 'z']
            elif num_rows==4:
                components_subscripts=['x', 'y', 'z', '\mathrm{tot}']
            else:
                components_subscripts=[str(i) for i in range(num_rows)]
        
    
        title_base_str = r'$|{}_{{ {} }}|${}'
        if plot_quantity=='mag_db':
            title_base_str = r'$|{}_{{ {} }}|${}'
        elif plot_quantity=='real':
            title_base_str = r'$\operatorname{{Re}}( {}_{{ {} }} )${}'
        elif plot_quantity=='phase':        
            title_base_str = r'$\angle {}_{{ {} }} ${}'
        else:
            raise ValueError('unknown plot quantity. must be ''real'',  ''mag_db'' or  ''phase''')


        self.contourf_compare( Xobs_n, Yobs_n, 
                                  [[field_list[i][c] for c in plot_components_inds] for i in range(num_cols)],
                                  title_list=[[title_base_str.format(field_quantity.upper(), components_subscripts[c], title_suffix_list[i]) 
                                                 for c in plot_components_inds] for i in range(num_cols)],
                                  colorbar_ticks=colorbar_ticks,
                                  colorbar_labels=colorbar_labels,
                                  contourf_levels=contourf_levels,
                                  quantity=plot_quantity,
                                  sharex=False, sharey=False,
                                  norm_plot_max=max_vec,
                                  norm_plot_min=min_vec,
                                  norm_plot_tocolumn=norm_plot_tocolumn,
                                  tight_subplots=tight_subplots,
                                  extend='both') 

        if print_peak_values:
            header_str =  "Quant | "
            for i in range(num_cols):
                header_str = header_str + "{:16.16}".format(title_suffix_list[i]) + " | "
            print(header_str)

            for c in plot_components_inds:
                peak_str = field_quantity.upper() + "_" + components_subscripts[c].replace(r"\mathrm{tot}", "tot").ljust(3) + " |  "
                for i in range(num_cols):
                    peak_str = peak_str + '{:14.6f}'.format( np.nanmax(np.abs(field_list[i][c])) ) 
                    if i < num_cols-1:
                        peak_str = peak_str + "  |  "
                print(peak_str +  "  | " + unit_str.replace('$', ''))
    
        return self.fig

    def contourf_spatial(self, X, Y, Fdata):
        data = self._force_tuple(Fdata)
        self.contourf(X, Y, data)

    def contourf_spectral(self, Kx, Ky, Fdata):
        """ fftshifts inputs and plots contourf """
        data = self._force_tuple(Fdata)
        self.contourf( *self.shift_input(Kx, Ky, data) )
    
    def add_visible_region(self, beta=1.0, **kargs):
        ax = self._find_axes(self.fig)
        ckargs = dict(color='r', fill=False, linestyle='--')
        ckargs.update(kargs)
        for a in ax:
            _ = a.add_artist(plt.Circle((0, 0), beta, **ckargs))  

    def set_spectral_labels(self, normalized=True):
        self.set_xylabels(*self.SPECTRAL_LABELS[normalized])
    
    def set_spatial_labels(self, normalized=True):
        self.set_xylabels(*self.SPATIAL_LABELS[normalized])
    
    def set_xylabels(self, xlabel, ylabel):
        ax = self._find_axes(self.fig)
        for a in ax:
            if a.is_first_col():
                a.set_ylabel(ylabel)
            if a.is_last_row():
                a.set_xlabel(xlabel) 

    def set_xlabels(self, xlabel):
        ax = self._find_axes(self.fig)
        for a in ax:
            if a.is_last_row():
                a.set_xlabel(xlabel) 




class Quantity(object):
    ''' Create a quantity object, similar to CQuantity in SuperMash.
        Additionally includes information about the evaluation (mag, db, real, phase...)
        
        Constructor requires a string argument, for example
        - E, H, S
        If it refers to a specific component, it can be provided after '_':
        - E_x, E_y, E_z, E_tot

        Optional arguments:
        - db_scale_factor
        - unit_str

        - db_ref (default is 'max', otherwise provide float)

        - eval_opt (possible values:
                'mag_db', 'mag', 'rms_db', 'rms', 'phase_deg', 'phase', 'real', 'imag' )

        TODO: add support for multiple components

    '''

    def __init__(self, quantity_str, **kwargs):
        #unit_str=None, db_scale_factor=None, db_ref=None, eval_opt=None, scale_min=None, scale_max=None, print_nsignificant=3):
        assert isinstance(quantity_str, str) or quantity_str is None
        if quantity_str is None: quantity_str = 'f'
        assert len(quantity_str) > 0

        self.reset_evalopts()
        self.reset_levels()
        self.component = ''

        # some special often encountered cases:
        if len(quantity_str)==2:
            qs = quantity_str[0].lower()
            cs = quantity_str[1].lower()
            if (qs == 'e' or qs == 'h' or qs == 's') and (cs == 'x' or cs == 'y' or cs == 'z'):
                quantity_str = quantity_str[0] + '_' + quantity_str[1]
        if len(quantity_str)==4:
            if quantity_str[1:] == 'tot':
                quantity_str = quantity_str[0] + '_' + quantity_str[1:]

        both = quantity_str.split('_')
        self.symbol = both[0]
        if len(both)>1:
            self.component = both[1]

        self.init_defaults_for_quantity()

        # user-supplied arguments override defaults for E/H/S above:
        self.db_scale_factor    = kwargs.pop('db_scale_factor', self.db_scale_factor)
        self.unit_str            = kwargs.pop('unit_str', self.unit_str)
        self.db_ref                = kwargs.pop('db_ref', 'max')
        self.levels_min            = kwargs.pop('levels_min', None)
        self.levels_max            = kwargs.pop('levels_max', None)
        self.print_nsignificant = kwargs.pop('print_nsignificant', 3)
        self.eval_opt            = kwargs.pop('eval_opt', None)
        self.num_cbar_ticks        = kwargs.pop('num_cbar_ticks', None)

        if kwargs:
            raise TypeError('Got unexpected keyword arguments {}'.format(kwargs))

        self.update_eval_opt()

    @property
    def eval_opt(self):
        return self._eval_opt

    @eval_opt.setter
    def eval_opt(self, value):
        if value is None:
            self.init_defaults_for_quantity()
        else:
            self._eval_opt = value
        self.update_eval_opt()
        self.update_levels()

    @property
    def db_ref(self):
        return self._db_ref

    @db_ref.setter
    def db_ref(self, value):
        if np.iscomplexobj(value): # checks for complex type
            if np.iscomplex(value): # checks for non-zero imaginary part
                raise ValueError('db_ref is {}, must be positve real number'.format(value))
            else:
                value = np.real(value)
        
        #if value <= 0.:  # commented to suppress the error: '<=' not supported between instances of 'str' and 'float'
        #    raise ValueError('db_ref is {}, must be > 0'.format(value))
        self._db_ref = value

    @property
    def levels_min(self):
        return self._levels_min

    @levels_min.setter
    def levels_min(self, value):
        if value is None:
            self._levels_min='min'
        else:
            self._levels_min = value
        self.update_levels()

    @property
    def levels_max(self):
        return self._levels_max

    @levels_max.setter
    def levels_max(self, value):
        if value is None:
            self._levels_max='max'
        else:
            self._levels_max = value
        self.update_levels()

    def reset_evalopts(self):
        self.cached_db_ref = 0
        self.cached_min = np.nan
        self.cached_max = np.nan
        self.levels = None
        self.cbar_ticks = None
        self.cbar_ticklabels = None
        self.level_extend = 'both' # extend the colobar to values lower than min value

    def reset_levels(self):
        self._levels_min = 'min'
        self._levels_max = 'max'

    def init_defaults_for_quantity(self):
        if self.symbol.lower() == 'e':
            self.unit_str = 'V/m'
            self.db_scale_factor = 20.
            self._eval_opt = 'mag_db'

        elif self.symbol.lower() == 'h':
            self.unit_str = 'A/m'
            self.db_scale_factor = 20.
            self._eval_opt = 'mag_db'

        elif self.symbol.lower() == 's':
            self.unit_str = 'W/m$^2$'
            self.db_scale_factor = 10.
            self._eval_opt = 'real'

        else:
            self.unit_str = ''
            self.db_scale_factor = 20.
            self._eval_opt = ''

    def update_eval_opt(self):
        self.reset_evalopts()

        if self.eval_opt == '':
            self.tex_prefix = ''
            self.tex_suffix = ''
            self.desc_suffix = ''
            self.range_desc = ''
            self.level_extend = 'both'

        elif self.eval_opt == 'mag_db':
            self.tex_prefix = '|'
            self.tex_suffix = '|'
            self.desc_suffix = ' (dB)'
            self.range_desc = 'Mag'

        elif self.eval_opt == 'mag_db_power':
            self.tex_prefix = '|'
            self.tex_suffix = '|'
            self.db_scale_factor = 10.
            self.desc_suffix = ' (dB)'
            self.range_desc = 'Mag'            

        elif self.eval_opt == 'mag':
            self.tex_prefix = '|'
            self.tex_suffix = '|'
            self.desc_suffix = ''
            self.range_desc = 'Mag'

        elif self.eval_opt == 'rms':
            self.tex_prefix = r'\operatorname{RMS}('
            self.tex_suffix = ')'
            self.desc_suffix = ''
            self.range_desc = 'RMS'

        elif self.eval_opt == 'rms_db':
            self.tex_prefix = r'\operatorname{RMS}('
            self.tex_suffix = ')'
            self.desc_suffix = ' (dB)'
            self.range_desc = 'RMS'

        elif self.eval_opt == 'phase_deg':
            self.tex_prefix = r'\angle '
            self.tex_suffix = ''
            self.desc_suffix = ''
            self.unit_str = 'deg'
            self.range_desc = 'Phase'
            self.level_extend = 'neither'

        elif self.eval_opt == 'phase':
            self.tex_prefix = r'\angle '
            self.tex_suffix = ''
            self.desc_suffix = ''
            self.unit_str = 'rad'
            self.range_desc = 'Phase'
            self.level_extend = 'neither'

        elif self.eval_opt == 'real':
            self.tex_prefix = r'\operatorname{Re}('
            self.tex_suffix = ')'
            self.desc_suffix = ''
            self.range_desc = ''
            self.level_extend = 'both'

        elif self.eval_opt == 'imag':
            self.tex_prefix = r'\operatorname{Im}('
            self.tex_suffix = ')'
            self.desc_suffix = ''
            self.range_desc = ''
            self.level_extend = 'both'


    def update_levels(self):
        self.cbar_ticklabels = None
        if self.eval_opt == 'mag_db' or self.eval_opt == 'mag_db_power':
            cmin = -30. if self.levels_min=='min' else self.levels_min
            cmax = 0 if self.levels_max=='max' else self.levels_max
            self.levels = np.linspace(cmin, cmax, 31)
            self.cbar_ticks = np.linspace(cmin, cmax, 6)

        elif self.eval_opt == 'rms_db':
            cmin = -30. if self.levels_min=='min' else self.levels_min
            cmax = 0 if self.levels_max=='max' else self.levels_max
            self.levels = np.linspace(cmin, cmax, 31)
            self.cbar_ticks = np.linspace(cmin, cmax, 6)

        elif self.eval_opt == 'phase_deg':
            cmin = -180. if self.levels_min=='min' else self.levels_min
            cmax = 180. if self.levels_max=='max' else self.levels_max
            self.levels = np.linspace(cmin, cmax, 12)
            self.cbar_ticks = np.linspace(cmin, cmax, 5)

        elif self.eval_opt == 'phase':
            if self.levels_min=='min' and self.levels_max=='max':
                self.levels = np.linspace(-np.pi, np.pi, 12)
                self.cbar_ticks = np.linspace(-np.pi, np.pi, 5)
                self.cbar_ticklabels = ['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$']
            else:
                cmin = -np.pi if self.levels_min=='min' else self.levels_min
                cmax = np.pi if self.levels_max=='max' else self.levels_max
                self.levels = np.linspace(cmin, cmax, 12)
                self.cbar_ticks = np.linspace(cmin, cmax, 5)
        else:
            self.levels = np.linspace(self.cached_min, self.cached_max, 31)
            self.cbar_ticks = np.linspace(self.cached_min, self.cached_max, 6)

    def compute_quantity(self, data, warn_about_nans=True):
        assert isinstance(data, np.ndarray), 'input must be ndarray'

        self.cached_db_ref = 0.

        if warn_about_nans and np.any(np.isnan(data)):
            _LOGGER.warn("input data contains NaNs")

        if self.eval_opt == '':
            self.set_min_max_dbref(data)
            self.update_levels()
            return data

        if self.eval_opt == 'mag_db' or self.eval_opt == 'mag_db_power':
            mag = np.abs(data)
            self.set_min_max_dbref(mag)
            self.update_levels()
            return self.db_scale_factor*np.log10(mag/self.cached_db_ref)                

        if self.eval_opt == 'mag':
            mag = np.abs(data)
            self.set_min_max_dbref(mag)
            self.update_levels()
            return np.abs(data)

        if self.eval_opt == 'rms':
            rms = np.abs(data)/np.sqrt(2)
            self.set_min_max_dbref(rms)
            self.update_levels()
            return rms

        if self.eval_opt == 'rms_db':
            rms = np.abs(data)/np.sqrt(2)
            self.set_min_max_dbref(rms)
            self.update_levels()
            return self.db_scale_factor*np.log10(rms/self.cached_db_ref)

        if self.eval_opt == 'phase_deg':
            self.update_levels()
            return np.angle(data)*180./np.pi

        if self.eval_opt == 'phase':
            self.update_levels()
            return np.angle(data)

        if self.eval_opt == 'real':
            re = np.real(data)
            self.set_min_max_dbref(re)
            self.update_levels()
            return re

        if self.eval_opt == 'imag':
            im = np.imag(data)
            self.set_min_max_dbref(im)
            self.update_levels()
            return im

        raise ValueError('Unsupported evaluation option: {}'.format(self.eval_opt))


    def set_min_max_dbref(self, processed_data):
        if self.levels_min == 'min':
            self.cached_min = np.nanmin(processed_data)
        else:
            self.cached_min = self.levels_min
        if self.levels_max == 'max':
            self.cached_max = np.nanmax(processed_data)
        else:
            self.cached_max = self.levels_max
        if self.db_ref == 'max':
            self.cached_db_ref = np.nanmax(processed_data)
        else:
            self.cached_db_ref = self.db_ref

    def get_db_ref(self):
        if self.db_ref == 'max':
            return self.cached_db_ref
        else:
            return self.db_ref

    def desc(self):
        '''Return a TeX-formatted string discribing the quantity'''

        if self.component > 1:
            comp_str = r'\mathrm{' + self.component + '}'
        else:
            comp_str = self.component
        return '$' + self.tex_prefix + self.symbol + '_' + comp_str + self.tex_suffix + '$ ' + self.desc_suffix


    def range_str(self):
        '''Return a TeX-formatted string suitable for the colobar'''

        range_str = ''

        number_format_str = '{{:.{}g}}'.format(self.print_nsignificant)
        if '_db' in self.eval_opt:
            range_str =  'dB/ref=' + str(float(number_format_str.format(self.get_db_ref())))

        if self.unit_str is not '':
            range_str = range_str + ' ' + self.unit_str
        
        if 'rms' in self.eval_opt:
            range_str = range_str + ' (RMS)'

        return range_str


    def plt_levels(self):
        if self.levels[-1]==self.levels[0]:
            return np.linspace(self.levels[0], self.levels[0]+1., 5)
        return self.levels

    def colorbar_ticks(self):
        if self.cbar_ticks is None:
            return np.linspace(self.levels[0], self.levels[-1], 6)
        return self.cbar_ticks

    def colorbar_ticklabels(self):
        if self.cbar_ticklabels is None:
            format_str = '{{:.{}g}}'.format(self.print_nsignificant)
            if np.any(np.iscomplex(self.colorbar_ticks())):
                return [complex(format_str.format(t)) for t in self.colorbar_ticks()]
            else:
                return [float(format_str.format(np.real(t))) for t in self.colorbar_ticks()]
        return self.cbar_ticklabels

    def plt_extend(self):
        return self.level_extend

    

class SpectralQuantity(Quantity):

    def compute_from_spatial(self, X_m, Y_m, F_complex, wavelength, warn_about_nans=True):
        Nx, Ny = X_m.shape
        dx = X_m[1,0]-X_m[0,0]
        dy = Y_m[0,1]-Y_m[0,0]

        # for spectrum, X,Y need to be equidistant and even
        need_resampling = False
        if Nx%2!=0 or Ny%2!=0:
            need_resampling = True
        if not need_resampling:
            try:
                npx.validate_meshgrid(X_m, Y_m, require_equidistant_dx=True, require_equidistant_dy=True)
            except(ValueError):
                need_resampling = True
        
        if need_resampling:
            npx.validate_meshgrid(X_m, Y_m)
            source_points = (X_m[:,0], Y_m[0,:]) # regular grid, i.e. axis ticks

            # create equidistant grid:
            #Nx = 2**int(np.ceil(np.log2(Nx))) # next power of 2 greater than Nx
            #Ny = 2**int(np.ceil(np.log2(Ny))) # next power of 2 greater than Ny
            # use reasonable values:
            if wavelength is not None:
                Nx_approx = (X_m[-1,0] - X_m[0,0] ) /(wavelength/8)
                Nx = 2**int(np.ceil(np.log2(Nx_approx))) # next power of 2 greater than Nx
                Ny_approx = (Y_m[0,-1] - Y_m[0,0] ) /(wavelength/8)
                Ny = 2**int(np.ceil(np.log2(Ny_approx))) # next power of 2 greater than Nx
            else:
                Nx = 128
                Ny = 128
            
            print("resampling to equidistant grid for spectrum computation (shape: ({}, {}))".format(Nx, Ny))

            x = np.linspace(X_m[0,0], X_m[-1,0], Nx)
            y = np.linspace(Y_m[0,0], Y_m[0,-1], Ny)
            dx = x[1]-x[0]
            dy = y[1]-y[0]
            X, Y = npx.meshgrid(x,y)
            target_points = np.c_[X.ravel(), Y.ravel()]

            interpolator = RegularGridInterpolator(source_points, F_complex)
            F_equidistant = interpolator(target_points).reshape(X.shape)
        else:
            F_equidistant = F_complex
        
        Kx, Ky = pwe.spectral_meshgrid(Nx, dx, Ny, dy)
        S = pwe.to_spectral(F_equidistant)

        if wavelength is not None:
            k0 = 2*np.pi/wavelength
            Kx = Kx/k0
            Ky = Ky/k0

        # put center of spectrum is at center of plot:
        Kx = fftshift(Kx)
        Ky = fftshift(Ky)
        Dq = fftshift(S)

        Do = self.compute_quantity( Dq, warn_about_nans=warn_about_nans )
        return Kx, Ky, Do


    def desc(self):
        ''' Return a TeX-formatted string discribing the quantity
        '''
        if self.component > 1:
            comp_str = r'\mathrm{' + self.component + '}'
        else:
            comp_str = self.component
        return '$' + self.tex_prefix +  '\mathcal{F} (' + self.symbol + '_' + comp_str + ')' + self.tex_suffix + '$ ' + self.desc_suffix



def set_equal_xyz_scaling(ax):
    assert hasattr(ax, 'get_zlim'), "Supplied (sub)plot axes need to be 3D"
    scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    centers = (scaling[:,0] + scaling[:,1])*0.5
    scale_xyz = np.array([[np.min(scaling), np.max(scaling)]]*3)
    centers_current = (scale_xyz[:,0] + scale_xyz[:,1])*0.5
    shift = centers-centers_current
    scale_centered = np.c_[scale_xyz[:,0] + shift, scale_xyz[:,1] + shift]
    ax.auto_scale_xyz(*[c for c in scale_centered])


def plot_ff_3d(ax, theta_meshgrid, phi_meshgrid, ff_values, N_theta=64, N_phi=64, **kwargs):
    assert hasattr(ax, 'get_zlim'), "Supplied (sub)plot axes need to be 3D"
    assert npx.validate_meshgrid(theta_meshgrid, phi_meshgrid), "Theta and phi not in expected meshgrids of : theta, phi = npx.meshgrid(np.linspace(0,np.pi), np.linspace(0,2*np.pi,Np))"
    assert np.all(theta_meshgrid.shape==ff_values.shape), "Dimensions of theta_meshgrid and ff_values must match"
    default_plotargs = dict(alpha=0.9, cmap=Z43_CMAP, linewidth=0.1, edgecolors='gray', antialiased=True)
    default_plotargs.update(kwargs)
    
    r = ff_values.ravel()/np.nanmax(ff_values)

    #
    # resample because 3D plots have problem with too many datapoints:
    #
    Nt, Np = theta_meshgrid.shape    
    th_p = np.linspace(0., np.pi, np.max([N_theta, Nt]))
    ph_p = np.linspace(0., 2*np.pi, np.max([N_phi, Np]))
    th_mg, ph_mg = npx.meshgrid(th_p, ph_p)
    nodes_2d = np.array([[xc, yc] for xc,yc in zip(th_mg.ravel(), ph_mg.ravel())])

    if N_theta > Nt or N_phi > Np:
        rgi = RegularGridInterpolator((theta_meshgrid[:,0], phi_meshgrid[0,:]), r.reshape(Nt, Np))
        r_p = rgi(nodes_2d)
    else:
        nodes_2d = np.array([[xc, yc] for xc,yc in zip(theta_meshgrid.ravel(), phi_meshgrid.ravel())])
        r_p = r

    # nodes of radiation pattern
    nodes = np.c_[r_p*np.sin(nodes_2d[:,0])*np.cos(nodes_2d[:,1]), 
                 r_p*np.sin(nodes_2d[:,0])*np.sin(nodes_2d[:,1]), 
                 r_p*np.cos(nodes_2d[:,0])]
    
    # triangulation for trisurf
    tri_del = Delaunay(nodes_2d)
    triangles = tri_del.simplices

    # color the triangles with their r (same as S4L)
    collec = ax.plot_trisurf(nodes[:,0], nodes[:,1], nodes[:,2], triangles=triangles, **default_plotargs)
    # compute mean of the 3 triangle nodes
    tri_colors = np.zeros( triangles.shape[0] )
    for i, tri in enumerate(triangles):
        tri_colors[i] = np.mean( r_p[tri] )

    collec.set_clim(vmin=0.0, vmax=1.0)
    collec.set_array(tri_colors)
    set_equal_xyz_scaling(ax)


def plot_ff_2d(ax, theta_meshgrid, phi_meshgrid, ff_values, phi=None, theta=None, plot_polar=True, use_db=None, **kwargs):
    ''' Plot a cut of the far-field
        assumes theta in range [0, np.pi] and phi in range [0, 2*np.pi]
    '''
    assert (phi is None and theta is not None) or (theta is None and phi is not None), "Specify plot cut by phi or theta"
    assert npx.validate_meshgrid(theta_meshgrid, phi_meshgrid), "Theta and phi not in expected meshgrids of : theta, phi = npx.meshgrid(np.linspace(0,np.pi), np.linspace(0,2*np.pi,Np))"
    assert np.all(theta_meshgrid.shape==ff_values.shape), "Dimensions of theta_meshgrid and ff_values must match"
    assert np.all(theta_meshgrid <= np.pi), "Theta must be smaller than pi (rad)"
    assert np.all(phi_meshgrid <= 2*np.pi), "Phi must be smaller than 2 pi (rad)"
    if plot_polar:
        assert isinstance(ax, PolarAxes), "For polar plot use plt.subplots(1,1, subplot_kw=dict(projection='polar'))"
    else:
        if use_db is None:
            use_db = True

    # for constant phi, plot for theta in [-90, 90]
    if phi is not None:
        assert phi <= 2*np.pi, "Phi must be smaller than 2 pi (rad)"

        idx_p = np.argmin(np.abs(phi_meshgrid[0,:]-phi))
        idx_m = np.argmin(np.abs(phi_meshgrid[0,:]-np.pi-phi))

        theta_m = np.flipud(-theta_meshgrid[:,idx_m])
        theta_p = theta_meshgrid[:,idx_p]

        ff_m = np.flipud(ff_values[:,idx_m])
        ff_p = ff_values[:,idx_p]

        # now glue together:
        theta = np.concatenate( (theta_m[:-1], theta_p) )
        ff = np.concatenate( (ff_m[:-1], ff_p) )
        
        if use_db:
            ff = 20*np.log10(ff/ff.max())
        
        if plot_polar:
            ax.plot(theta, ff, **kwargs)
        else:
            # only plot -90 to 90 degrees
            mask = (theta >= -np.pi/2) & (theta <= np.pi/2)
            ax.plot(theta[mask]*180/np.pi, ff[mask], **kwargs)
            ax.set_xlabel('theta (deg)')


    elif theta is not None:
        assert theta <= np.pi, "Theta must be smaller than pi (rad)"

        idx = np.argmin(np.abs(theta_meshgrid[:,0]-theta))
        phi = phi_meshgrid[idx,:]
        ff = ff_values[idx,:]
        if use_db:
            ff = 20*np.log10(ff/ff.max())

        if plot_polar:
            ax.plot(phi, ff, **kwargs)
        else:
            ax.plot(phi*180/np.pi, ff, **kwargs)
            ax.set_xlabel('phi (deg)')



def compute_quantity(X_m, Y_m, F_complex, quantity, wavelength=None):
    ''' Compute different quantities for the given complex field

        Supported quantities:
        - 'mag_db'
        - 'mag_db_field'
        - 'mag_db_power'
        - 'mag'
        - 'phase'
        - 'phase_deg'
        - 'real'
        - 'imag'
        - 'mag_spectrum'
        - 'mag_db_spectrum'
        - 'phase_spectrum'
        - 'phase_deg_spectrum'

    '''
    assert isinstance(F_complex, np.ndarray)
    assert np.ndim(F_complex)==2

    #
    # spectral quantities
    #
    if 'spectrum' in quantity:
        # for spectrum, X,Y need to be equidistant
        npx.validate_meshgrid(X_m, Y_m, require_equidistant_dx=True, require_equidistant_dy=True)
        Nx, Ny = X_m.shape
        Kx, Ky = pwe.spectral_meshgrid(Nx, X_m[1,0]-X_m[0,0], Ny, Y_m[0,1]-Y_m[0,0])

        S = pwe.to_spectral(F_complex)

        if wavelength is not None:
            k0 = 2*np.pi/wavelength
            Kx = Kx/k0
            Ky = Ky/k0

        if 'mag_db' in quantity:
            mag = np.abs(S)
            Dq = 20*np.log10(mag/np.nanmax(mag))
            # hack to be able to plot 0.0, put to -120dB
            mask = mag==0
            Dq[mask] = -120.0
        elif 'mag' in quantity:
            Dq = np.abs(S)
        elif 'phase_deg' in quantity:
            Dq = np.angle(S)*180./np.pi
        elif 'phase' in quantity:
            Dq = np.angle(S)

        # put center of spectrum is at center of plot:
        Kx = fftshift(Kx)
        Ky = fftshift(Ky)
        Dq = fftshift(Dq)

        return Kx, Ky, Dq

    #
    # spatial quantities
    #
    if wavelength is not None:
        X = X_m/wavelength
        Y = Y_m/wavelength
    else:
        X = X_m
        Y = Y_m

    if 'mag_db_power' in quantity:
        mag = np.abs(F_complex)
        Dq = 10*np.log10(mag/np.nanmax(mag))
        # hack to be able to plot 0.0, put to -120dB
        mask = mag==0
        Dq[mask] = -120.0
    elif 'mag_db_field' in quantity:
        mag = np.abs(F_complex)
        Dq = 20*np.log10(mag/np.nanmax(mag))
        # hack to be able to plot 0.0, put to -120dB
        mask = mag==0
        Dq[mask] = -120.0
    elif 'mag_db' in quantity:
        mag = np.abs(F_complex)
        Dq = 20*np.log10(mag/np.nanmax(mag))
        # hack to be able to plot 0.0, put to -120dB
        mask = mag==0
        Dq[mask] = -120.0
    elif 'real' in quantity:
        Dq = np.real(F_complex)
    elif 'imag' in quantity:
        Dq = np.imag(F_complex)
    elif 'mag' in quantity:
        Dq = np.abs(F_complex)
    elif 'phase_deg' in quantity:
        Dq = np.angle(F_complex)*180./np.pi
    elif 'phase' in quantity:
        Dq = np.angle(F_complex)
    else:
        raise ValueError('Undefined quantity : {}'.format(quantity))

    return X, Y, Dq

def spatial_labels(normalized=True, ax=None):
    if ax is None:
        ax = plt.gca()
    labels = ComplexVectorPlotter.SPATIAL_LABELS[int(normalized)]
    _ = ax.set_xlabel(labels[0])
    _ = ax.set_ylabel(labels[1])

def spectral_labels(normalized=True, ax=None):
    if ax is None:
        ax = plt.gca()
    labels = ComplexVectorPlotter.SPECTRAL_LABELS[int(normalized)]
    _ = ax.set_xlabel(labels[0])
    _ = ax.set_ylabel(labels[1])

def add_ring(r=1.0, ax=None, **kargs):
    if ax is None:
        ax = plt.gca()
    ckargs = dict(color='grey', fill=False, linestyle='--')
    ckargs.update(kargs)
    _ = ax.add_artist(plt.Circle((0, 0), r, **ckargs))

def add_visible_region(beta=1, ax=None, **kargs):
    add_ring(beta, ax, **kargs)

def all_figures():
    """ Returns all figures in canvas """
    # http://stackoverflow.com/questions/3783217/get-the-list-of-figures-in-matplotlib
    import matplotlib._pylab_helpers
    figures=[manager.canvas.figure for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    return figures

# REPORTS ----------------------------------------
def format_figure(fig):
    #mpl.rcParams['axes.titlesize'] = 'x-large'
    #mpl.rcParams['axes.labelsize'] = 13#'x-large' # fontsize of the x any y labels
    #mpl.rcParams['axes.grid'] = True # display grid or not
    #mpl.rcParams['font.size'] = 11
    pass

def set_report_mode(fontsize=16):
    mpl.rcParams['axes.titlesize'] = fontsize# 'x-large'
    mpl.rcParams['axes.labelsize'] = fontsize#'x-large' # fontsize of the x any y labels
    mpl.rcParams['axes.grid'] = True # display grid or not
    mpl.rcParams['font.size'] = fontsize

def save_figure(filename, fig=None, destdir="."):
    """
    Example:
        from pyrec.mpl_helpers import save_figure

        figsdir = r'../../../trunk/reports/1612_apple/pics'
        for i, fig in enumerate(figs):
            print save_figure('foo{}'.format(i), fig, destdir=figsdir)
    """
    #import pyrec.mpl_helpers
    import matplotlib.pyplot as plt

    # TODO: format figure size

    kargs = {'format':'png', 'bbox_inches':'tight', 'transparent': False, 'dpi':300}
    filepath = os.path.join(destdir, filename+'.png')
    if fig:
        # http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.savefig
        fig.savefig(filepath, **kargs)
    else:
        # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.savefig
        plt.savefig(filepath, **kargs)
    
    latex_str = r'\includegraphics[keepaspectratio,width=\textwidth]{%s.png}'%filename

    _LOGGER.info(latex_str)

    return latex_str, filepath

