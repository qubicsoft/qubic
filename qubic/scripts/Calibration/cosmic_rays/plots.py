import os
import shutil

import numpy as np
from model import Model as md
from mixedDistribution import MixedDistribution
from scipy.stats import rv_continuous, linregress, landau, expon, norm, cauchy

import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import qubicpack.pixel_translation as pt

# Set the backend for matplotlib to 'Agg', which is suitable for generating images without interactive display.
matplotlib.use('Agg')  # Use 'Agg' backend to render static visualizations for file output.

# If LaTeX is installed on the system, configure matplotlib to use it for text formatting.
if shutil.which('latex'):
    plt.rcParams['text.usetex'] = True



def latex_percent(x, pos):
    """
    Format a tick value as a percentage string in LaTeX math mode.

    This function is intended to be used as a formatter for y-axis ticks,
    formatting it to two decimal places, followed by a percent sign.

    Parameters
    ----------
    x : float
        The numerical tick value.
    pos : int or None
        Tick position (not used in this formatter).

    Returns
    -------
    str
        The formatted tick label in the form "XX.XX%"
    """
    return rf"${x: .3g} \% $"


class Plots:

    """
      The Plots class serves as a utility container that groups together static methods
      for generating various visualizations related to cosmic ray event analysis.
      It is designed as a static-only class and does not require instantiation.

    Methods:
      - custom_subplots: Creates a custom figure with a grid of subplots.
      - plot_tau_candidates: Plots candidate events and their residuals for a given TES.
      - plot_fp: Visualizes the focal plane, highlighting TES with tau candidates.
      - plot_hist: Generates histogram plots with an optional statistical fit overlay.
      - plot_taus: Orchestrates the generation of tau candidate plots or a summary plot for multiple TESs.
    """

    dpi = 500

    @staticmethod
    def custom_subplots(rows: int, cols: list[int] | int, **fg_kw) -> plt.subplots:
        """
        Create a figure with a mosaic of subplots arranged in a grid defined by the number of rows and columns.

        Parameters
        ----------
        rows : int
            The number of rows in the subplot grid.
        cols : list[int] or int
            Either a list specifying the number of columns for each row, or a single integer that will be applied to every row.
        **fg_kw : dict
            Additional keyword arguments to pass to plt.figure(), such as 'figsize' or 'tight_layout'.

        Returns
        -------
        tuple[plt.Figure, list[plt.Axes]]
            A tuple containing the created figure and a list of axes objects.

        Raises
        ------
        ValueError
            If the length of the provided list for 'cols' does not match the number of rows.
        """

        # If 'cols' is provided as an integer, convert it to a list by repeating its value.
        if isinstance(cols, int):
            cols = [cols] * cols

        if len(cols) != rows:
            raise ValueError("Number of columns does not match number of rows")

        # Initialize an empty list to store the axes objects.
        axes = []
        # Create a new figure with the given keyword arguments.
        fig = plt.figure(**fg_kw)

        # Define a GridSpec with the specified number of rows and a single column for the overall layout.
        gs = gridspec.GridSpec(rows, 1, figure=fig)

        # Iterate over each row defined in the GridSpec.
        for r, gs_row in enumerate(gs):
            # For each row, create a subgrid with 1 row and a number of columns specified by cols[r].
            line = gs_row.subgridspec(1, cols[r])

            # For each cell in the subgrid, add a subplot to the figure and append it to the axes list.
            for plot in line:
                axes.append(fig.add_subplot(plot))

        return fig, axes

    @staticmethod
    def plot_tau_candidates(taus_fig: plt.Figure,
                            paths: tuple[str, ...],
                            n_taus: int,
                            start_taus_idx: int,
                            n_tes: str,
                            taus: dict):
        """
        Plot TES candidate signals, their residuals, and a summary of tau values on a single figure.

        This function creates multiple subplots within the given figure. Each candidate tau signal is plotted
        along with its residuals, where:
          - Data points preceding and following the candidate are shown.
          - The candidate signal is highlighted.
          - Residuals between the model and data are calculated and displayed.
        An additional subplot is used to present all tau values over the entire time domain.

        Parameters
        ----------
        taus_fig : plt.Figure
            The matplotlib figure in which to draw the plots.
        paths : tuple[str, ...]
            A tuple containing the file paths for time data and the cleaned signals.
        n_taus : int
            The number of tau candidates to plot.
        start_taus_idx : int
            The starting index for the tau candidates to be plotted.
        n_tes : str
            The TES index (in the range [0, 255]) indicating which TES data to analyze.
        taus : dict
            A dictionary with tau candidate information, including:
              - 'indexes': start and stop indices for each candidate,
              - 'exp fit params': parameters from the exponential fit,
              - 'chi square': chi-square and p-value information,
              - 'taus': tau values,
              - 'sigma': uncertainties for tau values.
        """

        # Unpack file paths: first for time data, second for cleaned signals
        time_fname, signals_clean_fname = paths

        # # Load the full time series from the specified time data file
        time_raw = np.load(time_fname)

        # Load the clean signal data for the specified TES
        clean_signal = np.load(signals_clean_fname)[n_tes]

        # Determine the starting index for plotting based on the number of taus.
        # If only one tau is to be plotted, center it in the first row; otherwise, follow row-major order.
        ax_start = 1 * (n_taus == 1)  # if n_taus equals 1, start plotting from axis index 1, else index 0
        ax_stop = ax_start + n_taus  # end index for tau candidate axes

        # Loop over the axes for tau candidates and their corresponding residual axes
        for index, (taus_ax, res_ax) in enumerate(
                zip(taus_fig.axes[ax_start: ax_stop], taus_fig.axes[3 + ax_start: -1])):

            ## Adjust index by adding the starting tau index; if it equals the number of candidates, exit the loop
            if (index := index + start_taus_idx) == len(taus['indexes']):
                break  # exit if all available tau candidates have been processed

            # Configure the candidate subplot's appearance
            taus_ax.set(facecolor='whitesmoke',
                        title=rf'$Candidate \ \#{index + 1}$',
                        xlabel=r'$time \ [s]$',
                        ylabel=r'$clean \ signal \ [ADU]$' if not (index % 3) else '')

            # Configure the residual subplot's appearance
            res_ax.set(facecolor='whitesmoke',
                       xlabel=r'$time \ [s]$',
                       ylabel=r'$\frac{Model \ - \ Data}{Data}$' if index == start_taus_idx else '')

            # Format the y-axis ticks to display percentages
            res_ax.yaxis.set_major_formatter(ticker.FuncFormatter(latex_percent))

            # Retrieve the start and stop indices for the current candidate from the dictionary.
            start, stop = taus['indexes'][index]  # extract candidate start and stop indices
            popt = taus['exp fit params'][index]  # extract exponential fit parameters for candidate
            chi_square = taus['chi square'][index][1]  # get chi-square value from candidate's data
            pvalue = taus['chi square'][index][2]  # get p-value from candidate's chi-square info
            pcov = np.array(taus['covariance matrix'][index])  # get the estimated approximate covariance of popt

            # Collect overview points: a few points before and after the candidate for context
            ov_t = [*time_raw[max(0, start - 5): start], *time_raw[stop + 1: min(len(time_raw), stop + 5)]]
            ov_s = [*clean_signal[max(0, start - 5): start], *clean_signal[stop + 1: min(len(clean_signal), stop + 5)]]

            # Extract the exact time and signal segments for the candidate.
            t = time_raw[start: stop]  # candidate's time segment
            s = clean_signal[start: stop]  # candidate's signal segment

            # Perform linear regression on the rising edge of the candidate (up to the maximum signal point)
            lin_offset = t[0]  # define the initial time as the offset
            lin_raise_x = t[:s.argmax() + 1]  # time values from start until the maximum signal
            lin_raise_y = s[:s.argmax() + 1]  # corresponding signal values

            # Calculate linear regression on adjusted time values
            lin_reg = linregress(lin_raise_x - lin_offset, lin_raise_y)
            lin_y_est = lin_reg.intercept + lin_reg.slope * (lin_raise_x - lin_offset)

            # Define the offset for the exponential decay
            exp_offset = t[s.argmax()]  # time at maximum signal value
            exp_dec_x = t[s.argmax():]  # time segment from maximum to end (decay portion)
            exp_dec_y = s[s.argmax():]  # corresponding signal values in the decay portion
            # Generate a finely spaced time array for plotting the exponential decay fit curve
            exp_dec_fit_x = np.linspace(exp_dec_x[0], exp_dec_x[-1], 1000)
            # Compute the exponential decay model values using the fitted parameters
            exp_dec_y_est = md.exp_decay(exp_dec_fit_x - exp_offset, *popt)

            # Set an offset to define a range for calculating the signal standard deviation around the candidate
            std_offset = 20
            # Calculate the standard deviation of signal values before and after the candidate window
            std_candidate = np.array([*clean_signal[max(0, start - std_offset): start],
                                      *clean_signal[stop: min(len(clean_signal), stop + std_offset)]]).std()

            # Evaluate the exponential decay model on the candidate's decay segment
            mod_val_data = md.exp_decay(exp_dec_x - exp_offset, *popt)

            # Calculate residuals as the relative difference between the model and the observed decay data
            y_err = (mod_val_data - exp_dec_y) / exp_dec_y
            # An alternative calculation using standard deviation is provided
            # y_err = (mod_val_data - exp_dec_y) / std_candidate

            # Plot overview points, the rising segment, and the decay segment on the candidate subplot.
            taus_ax.scatter(ov_t, ov_s)  # scatter plot for contextual overview points
            taus_ax.scatter(lin_raise_x, lin_raise_y)  # scatter plot for the rising edge points
            taus_ax.scatter(exp_dec_x, exp_dec_y)  # scatter plot for the decay segment points

            taus_ax.plot(exp_dec_fit_x, exp_dec_y_est, 'c--', label=rf'$\tau = {round(-1 / popt[2], 5)} ~ s$')

            # Fill the area between the model's upper and lower bounds based on residual error, with transparency.
            sigma_y = md.exp_model_uncertainty_jacobian(exp_dec_fit_x, popt, pcov)

            taus_ax.fill_between(exp_dec_fit_x, exp_dec_y_est - sigma_y, exp_dec_y_est + sigma_y,
                                 color='grey', alpha=0.2,
                                 label=rf'$\chi_{{\nu}}^2 = {chi_square:.2f} ~ | ~ p-value = {pvalue:.2f}$')

            taus_ax.plot(lin_raise_x, lin_y_est, 'k--', label=rf'$slope = {round(lin_reg.slope, 2)} ~ s^{{-1}}$')

            taus_ax.legend()
            taus_ax.grid(True)

            # residuals plots
            res_ax.plot(exp_dec_x, y_err * 100, marker='o', linestyle='dashed',
                        label=rf"$ \sigma: {std_candidate:.4f}$")
            res_ax.legend()
            res_ax.grid(True)

        # Retrieve the last axis from the figure to plot the summary of all tau values
        all_taus_ax = taus_fig.axes[-1]

        #  Loop over all tau values along with their uncertainties and index ranges
        for index, (tau, sigma, (start, stop)) in enumerate(zip(taus['taus'], taus['sigma'], taus['indexes'])):
            # Extract the time segment corresponding to the current tau candidate.
            t = time_raw[start:stop]

            # Set default transparency (alpha) and label for plotting.
            alpha = 0.1  # default low visibility
            label = ''  # default no label

            # For tau candidates within a selected range, increase visibility and add a label.
            if start_taus_idx <= index < 3 + start_taus_idx:
                alpha = 1  # full opacity for highlighted candidates
                label = rf"${tau} \pm {round(sigma, 5)}$"  # label showing tau value and its sigma

            all_taus_ax.errorbar(t[0], tau, yerr=sigma,
                                 marker='o', linestyle='dashed',
                                 linewidth=1, markersize=5,
                                 alpha=alpha,
                                 label=label)

        all_taus_ax.set(
            facecolor='whitesmoke',
            title=r'$\tau \ along \ time \ domain \ $',
            xlabel=r'$time \ [s]$',
            ylabel=r'$\tau \ [s]$')
        all_taus_ax.legend(title=r'$\tau \pm \sigma$', loc='best', ncol=len(taus['taus']) // 10 + 1)
        all_taus_ax.grid(True)

    @staticmethod
    def plot_fp(fp_ax: plt.Axes, taus: dict, thermometers: tuple):
        """
        Plot the focal plane representation with two ASICs colored differently.
        Highlight TES that have at least one tau candidate.

        Parameters
        ----------
        fp_ax : plt.Axes
            The matplotlib Axes object representing the focal plane.
        taus : dict
            Dictionary containing the tau candidate data for each TES under analysis.
        thermometers : tuple
            Tuple containing the TES identifiers that correspond to thermometers.
        """

        fp_ax.set_axis_off()

        fp_ax.set(facecolor='whitesmoke', title=r'$Focal \ Plane$')
        fp_ax.tick_params(labelbottom=False, labelleft=False)

        # Retrieve the physical layout (positions) of the TES on the focal plane.
        fp_identity = pt.make_id_focalplane()
        # Define colors to differentiate the two ASICs.
        asic_colors = ['cyan', 'orange', 'magenta', 'red', 'black']

        # TODO: generalizzare su tutti gli asic disponibili
        for asic in [1, 2]:

            # Initialize counters for thermometer positions and TES with tau candidates
            therm_x, therm_y = 0, 0
            tes_with_taus = 0

            for tes in range(1, 129):

                if tes not in thermometers:
                    # For TES that are not thermometers, retrieve their physical position on the focal plane.
                    coords = (fp_identity.TES == tes) & (fp_identity.ASIC == asic)
                    x, y = fp_identity[coords].col, fp_identity[coords].row

                else:
                    # For thermometers, increment the counter based on the ASIC being processed.
                    therm_y += asic == 1
                    therm_x += asic == 2
                    x, y = therm_x, therm_y

                # Set full opacity if the TES has a candidate tau, otherwise use lower opacity
                alpha = 1 if [tes, asic] in [taus[tes]['tes'] for tes in taus] else 0.4

                # Count the number of TES with tau candidates (alpha > 0.5 implies candidate presence)
                tes_with_taus += alpha > 0.5

                # Define the bounding box for the TES label with rounded edges.
                bbox = dict(boxstyle=mpatches.BoxStyle("Round", pad=0.4),
                            edgecolor=asic_colors[asic - 1], facecolor='white', alpha=alpha)

                fp_ax.text(**dict(x=x / 17,
                                  y=y / 17,
                                  s=rf"${tes:03}$", size=5.5, rotation='horizontal', bbox=bbox), alpha=alpha)

            bbox = dict(boxstyle=mpatches.BoxStyle("Round", pad=0.4),
                        edgecolor=asic_colors[asic - 1],
                        facecolor='white')

            fp_ax.text(**dict(x=2 / 17, y=(4 - asic) / 17, s=rf'$ASIC~{asic}:{tes_with_taus}$',
                              size=6, rotation='horizontal', bbox=bbox))

            fp_ax.set_aspect('equal')

    @staticmethod
    def plot_hist(hist_ax: plt.Figure.add_subplot,
                  title: str,
                  xlabel: str,
                  taus: dict,
                  data_path: str,
                  datatype: str = 'signal',
                  to_plot: np.ndarray | list = None,
                  fit_func: rv_continuous | MixedDistribution = None,
                  fit_name_args: tuple[str, ...] = None,
                  fit_kwargs: dict = None):

        """
        Plot histograms for various data types derived from TES tau candidates.

        This method generates a histogram on the specified axes. It extracts data to be histogrammed
        either directly from the provided 'to_plot' parameter or by computing values based on the 'taus'
        dictionary and a data file. The 'datatype' parameter controls the computation of values:
          - 'signal': Computes the difference between the maximum and starting value in a signal segment.
          - 'elevation': Computes the mean elevation (or its complementary value if data is descending).
          - 'time': Computes the duration of the event.
          - 'residuals': Flattens and aggregates residuals from the candidate data.
        Optionally, if a fit function is provided, the method fits a probability density function (PDF)
        to the histogram data and overlays the fitted curve.

        Parameters
        ----------
        hist_ax : plt.Figure.add_subplot
            The matplotlib axes object on which the histogram will be drawn.
        title : str
            Title of the histogram plot.
        xlabel : str
            Label for the x-axis.
        taus : dict
            Dictionary containing tau candidate data for each TES under analysis.
        data_path : str
            File path to the data file to load data if 'to_plot' is not provided.
        datatype : str, optional
            Type of data to plot. Options include:
                'signal'    - Histogram of the tau amplitudes.
                'elevation' - Histogram of instrument elevation for cosmic ray events.
                'time'      - Histogram of the duration of cosmic ray events.
                'residuals' - Histogram of residual values.
            Default is 'signal'.
        to_plot : np.ndarray or list, optional
            Data to plot directly. If provided, this data will be used instead of loading from file.
        fit_func : callable, optional
            A function that fits the data and returns fit parameters. It must implement 'fit' and 'pdf' methods.
        fit_name_args : tuple[str, ...], optional
            Tuple of strings used to format the fit parameters into a label for the fitted function.
        fit_kwargs : dict, optional
            Additional keyword arguments to pass to the fit function.
        """

        tot_taus = 0

        # Initialize the 'data' variable as an empty dictionary or array.
        data: dict[str, ...] | np.ndarray = {}
        # Initialize an empty list to collect values for the histogram.
        to_hist: list[float] | np.ndarray = []

        # If no external data is provided to plot...
        if not to_plot:
            # For non-residual data, attempt to load data from the specified file path
            if datatype != "residuals":
                # If the file does not exist, exit the function
                if not os.path.isfile(data_path):
                    return
                else:
                    # Load the data from the file
                    data = np.load(data_path)

            # Loop over each TES in the 'taus' dictionary
            for tes in taus:
                sig = data[str(tes)] if datatype == 'signal' else data

                for start, stop in taus[tes]['indexes']:
                    # Depending on the datatype, calculate and collect histogram values.
                    if datatype == 'signal':
                        # Compute the amplitude (difference between the maximum signal value in the candidate and its
                        # starting value)
                        to_hist.append(sig[start:stop].max() - sig[start])
                    elif datatype == 'elevation':
                        # Compute the mean of the elevation segment; adjust based on whether data is descending.
                        to_hist.append(data[start:stop].mean()) #  if data[-1] > data[0] else 90 - data[start:stop].mean())
                    elif datatype == 'time':
                        tot_taus += 1
                        # Compute the duration of the event.
                        to_hist.append(data[stop - 1] - data[start])
                    elif datatype == 'residuals':
                        # For residuals, flatten the nested list of residuals and extend the histogram data list.
                        to_hist.extend([res for row in taus[tes][datatype] for res in row])
        else:
            to_hist = to_plot

        fit_kwargs = fit_kwargs or {}

        hist_ax.grid(True)
        hist_ax.set(facecolor='whitesmoke', title=title, xlabel=xlabel)

        hist_label = ''
        to_hist = np.array(to_hist)
        p_range = np.arange(len(to_hist))

        # If a fit function is provided, perform a fit to the histogram data and overlay the PDF curve
        if fit_func:

            if isinstance(fit_func, MixedDistribution):
                res_std = np.std(to_hist)
                fit_kwargs = {"p0": [0.8, 0, res_std, 0, res_std],
                              "bounds": [(0, 1), (0, 0), (1e-6, None), (0, None), (1e-6, None)],
                              # "method": "TNC"
                              }  # "L-BFGS-B"

            if datatype == 'residuals':
                min_hist, max_hist = (np.percentile(to_hist, 5), np.percentile(to_hist, 95))
                p_range = (to_hist >= min_hist) & (to_hist <= max_hist)

            elif datatype == 'time':
                taus_freq, to_hist = (to_hist[0], to_hist[1:]) if to_plot else (
                tot_taus / (data[1] * len(data)), to_hist)

                min_hist, max_hist = (np.min(to_hist), 0.2)
                p_range = (to_hist >= min_hist) & (to_hist <= max_hist)
                fit_kwargs = {"floc": min_hist}
                hist_label = rf"$ \nu_{{\tau}}: \; {taus_freq:.3g} Hz$"

            elif datatype == 'signal':
                min_hist, max_hist = (0, np.percentile(to_hist, 95))
                p_range = (to_hist >= min_hist) & (to_hist <= max_hist)

            else:
                # Determine the x-axis limits for the fit based on the datatype
                min_hist, max_hist = (min(to_hist), max(to_hist))

            if params := fit_func.fit(to_hist, **fit_kwargs):
                # Generate a linearly spaced array of x-values for the fitted PDF curve.
                x = np.linspace(min_hist, max_hist, 1000)
                # Compute the fitted PDF values using the fit parameters.
                pdf_fit = fit_func.pdf(x, *params)

                if isinstance(fit_func, MixedDistribution):
                    params = [(1 - list(params)[0]) * 100]

                label_fit = "~".join(fit_name_args).format(*params)
                hist_ax.plot(x, pdf_fit, '--', color='orange', label=rf'${fit_func.name} : ~ {label_fit}$')
                hist_ax.legend(loc='best')

        # Attempt to plot the histogram with automatically determined bins
        try:
            hist_ax.hist(to_hist[p_range],
                         bins='auto',
                         color='black',
                         density=fit_func is not None,
                         label=hist_label,
                         histtype='step')

        except Exception as e:
            # If an error occurs (e.g., due to binning issues), fall back to a fixed number of bins
            hist_ax.hist(to_hist[p_range],
                         bins=1296,
                         color='black',
                         density=fit_func is not None,
                         label=hist_label,
                         histtype='step')

        if hist_label:
            hist_ax.legend(loc='best')

    @staticmethod
    def plot_taus(n_tes: int, tes_taus: dict, paths: tuple[str, ...],
                  start_taus_idx: int = 0,
                  thermometers: tuple[int, ...] = None,
                  summary_plot: bool = False):

        """
        Generate plots for tau candidates from TES analysis.

        Depending on the 'summary_plot' flag, this method either creates a final plot for the analyzed dataset,
        a specific scanning strategy or a summary plot that aggregates results from all scanning strategies.
        The plots include details such as tau candidates, focal plane representation, and histograms for residuals,
        amplitudes, elevations, and time frames.

        Parameters
        ----------
        n_tes : int
            The total number of TES.
        tes_taus : dict
            Dictionary containing tau candidate data for the TES under analysis.
        paths : tuple[str, ...]
            Tuple of file paths: observation directory, plots directory, signals file, interpolated elevation file, and time file.
        start_taus_idx : int, optional
            Index of the first tau candidate to plot (default is 0).
        thermometers : tuple[int, ...], optional
            Tuple of TES identifiers corresponding to thermometers to be highlighted in the plot.
        summary_plot : bool, optional
            Flag indicating whether to generate a summary plot aggregating all scanning strategies.
        """

        # Exit the function if there is no tau data or if the number of TES is negative
        if not tes_taus or n_tes < 0:
            return None

        # Unpack the provided paths into specific variables
        observation_dir, plots_dir, signals_clean_fname, interp_elev_fname, time_fname = paths
        # Extract the dataset identifier from the observation directory path.
        dataset = observation_dir.split(os.sep)[-1][3:].replace('_', '~')

        fig_kw = dict(figsize=(16, 10), tight_layout=True)

        if not summary_plot:

            # Final plot for the results of a single scanning strategy.
            # Extract the TES location and ASIC information from the tau candidate data.
            loc_tes, asic = tes_taus['tes']

            # Retrieve the list of tau candidates.
            taus = tes_taus['taus']

            fig, ax = Plots.custom_subplots(3, [3, 3, 1], **fig_kw)
            fig.suptitle(rf'${dataset}~| ~TES,~ASIC ~:~{loc_tes, asic}~ | ~ \# \tau: ~ {len(taus)} $')
            Plots.plot_tau_candidates(fig, (time_fname, signals_clean_fname), len(taus[start_taus_idx:]),
                                      start_taus_idx, str(n_tes), tes_taus)

            # Adjust the subplot axes if the number of tau candidates is less than expected.
            if len(taus[start_taus_idx:]) == 1:
                # If there is only one tau candidate, remove the extra axes.
                fig.axes[0].remove()  # Remove the first tau plot.
                fig.axes[1].remove()  # Remove the last tau plot.
                fig.axes[1].remove()  # Remove the first residual plot.
                fig.axes[2].remove()  # Remove the last residual plot.

            if len(taus[start_taus_idx:]) == 2:
                # If there are only two tau candidates, remove the remaining extra axis.
                fig.axes[2].remove()  # Remove the last tau plot.
                fig.axes[4].remove()  # Remove the last residual plot.

            # Build the file path for saving the plot
            plot_path = os.path.join(plots_dir, f"{n_tes}_{start_taus_idx // 3 + 1:02}_taus_plot.png")

            # Check if a file already exists at the target path to avoid conflicts.
            if os.path.exists(plot_path):
                # Modify the file name by appending 'error' if a conflict is detected.
                plot_path = os.path.join(plots_dir, f"{n_tes}_error_{start_taus_idx // 3 + 1:02}_taus_plot.png")

            fig.savefig(plot_path, dpi=Plots.dpi)

        else:

            average_tau = [tau for tes in tes_taus for tau in tes_taus[tes]['taus']]
            average_sigma = [tau for tes in tes_taus for tau in tes_taus[tes]['sigma']]

            average_tau = np.mean(average_tau)
            average_sigma = np.mean(average_sigma) / len(average_sigma) ** 0.5

            # Summary plot that aggregates the results across analysed dataset or all scanning strategies.

            fig, (ax_fp, ax_res, ax_amp, ax_elev, ax_tm_frame) = Plots.custom_subplots(2, [2, 3], **fig_kw)
            fig.suptitle(rf"${dataset}$")

            normCauchy = MixedDistribution(norm, cauchy)

            # Plot the focal plane, highlighting TES and thermometers
            Plots.plot_fp(ax_fp, tes_taus, thermometers=thermometers)

            # Plot the histogram for cosmic ray residual fits
            Plots.plot_hist(hist_ax=ax_res,
                            title=r'$Cosmic ~ Rays ~ Residual ~ Fit$',
                            xlabel=r'$ Residuals $',
                            taus=tes_taus,
                            data_path="",
                            fit_func=normCauchy,
                            fit_name_args=(r'{:.2f} \% ~ outliers',),
                            datatype="residuals")

            # Plot the histogram for tau versus signal amplitude.
            Plots.plot_hist(hist_ax=ax_amp,
                            title=r'$ \tau \ vs \ Amplitudes$',
                            xlabel=r'$ Amplitude ~ [ADU]$',
                            taus=tes_taus,
                            data_path=signals_clean_fname,
                            fit_func=landau,
                            fit_name_args=("loc = {:.4f}", "scale = {:.4f}"))

            # Plot the histogram for tau versus elevation
            Plots.plot_hist(hist_ax=ax_elev,
                            title=r'$ \tau \ vs \ Elevation$',
                            xlabel=r'$ Elevation ~ [degrees]$',
                            taus=tes_taus,
                            data_path=interp_elev_fname,
                            datatype='elevation')

            # Plot the histogram for tau versus the time frame (duration of cosmic ray events)
            Plots.plot_hist(hist_ax=ax_tm_frame,
                            title=rf'$ \tau \ vs \ Time \ Frame: {average_tau:.3g} \pm {average_sigma:.3g}$',
                            xlabel=r'$ Time ~ [s]$',
                            taus=tes_taus,
                            data_path=time_fname,
                            datatype='time',
                            fit_func=expon,
                            fit_name_args=("loc = {:.2f}", "scale = {:.2f}"))

            fig.savefig(os.path.join(plots_dir, f"taus_for_all_tes.png"), dpi=Plots.dpi)

        plt.close(fig)
