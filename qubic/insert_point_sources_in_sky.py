#!/usr/bin/env python
## Module to insert point sources into a PySM sky map
## Version 1.0 - Dec 3rd 2021
## Aniello Mennella
##
## CHANGELOG
## Version 1.0 - first working version of the module


def gaussian2D(theta, theta0, phi, phi0, fwhm):
    """
    This the 2D gaussian function centered in theta0, phi0
        
    Input
    theta        - FLOAT - angle theta in radians
    theta0       - FLOAT - center angle theta in radians
    phi          - FLOAT - angle phi in radians
    phi0         - FLOAT - center angle phi in radians
    fwhm         - FLOAT - fwhm of the gaussian in radians
        
    Output
    The gaussian function defined as: 
    1/(np.pi * sigma**2) * np.exp(-((theta-theta0)**2 + (phi-phi0)**2) / sigma**2) (Normalized to unit 
    integral)
    """
    import numpy as np

    # Here we center the angles so that theta is in [-180, 180]  or  [-pi,pi]
    # and phi is in [-90, 90] or in [-pi/2, pi/2]

    sigma = 1.0 / (2 * np.sqrt(2.0 * np.log(2.0))) * fwhm

    new_angles = center_ang((theta, phi), degree=False)
    new_center = center_ang((theta0, phi0), degree=False)

    # This is the Gaussian normalized so that the integral is 1

    gaussval = np.exp(
        -((new_angles[0] - new_center[0]) ** 2 + (new_angles[1] - new_center[1]) ** 2)
        / (2 * sigma ** 2)
    )

    return gaussval


# ### Generate a 2D grid for the Gaussian values

# In[6]:


def gaussian2D_grid(source_center_deg, fwhm_deg):
    """
    This generates the grid for the gaussian smoothing around the source
    center. The grid is generated in an interval of 4 times the fwhm
        
    Input
    source_center_deg      - TUPLE - the LonLat coordinates of the source in degrees
    fwhm_deg               - FLOAT - fwhm of the gaussian in degrees
        
    Output
    theta_range, phi_range - NDARRAYS - Two arrays containing the theta and phi ranges in degrees
    """

    import numpy as np

    # The number of points for each axis in the grid is 30 points times the fwhm in degrees
    if fwhm_deg >= 1:
        npoints = 30 * fwhm_deg
    else:
        npoints = 50

    factor_fwhm = 2

    # Convert angles in radians
    #     source_center_rad = np.pi * np.array(source_center_deg) / 180.
    #     fwhm_rad = np.pi * fwhm_deg / 180.

    # Generate ranges
    theta_range_deg = np.arange(
        -factor_fwhm * fwhm_deg, factor_fwhm * fwhm_deg, fwhm_deg / npoints
    )
    phi_range_deg = theta_range_deg.copy()

    #     integral = -1/(2*np.sqrt(np.log(4))) * \
    #                 np.exp(-(fwhm_rad**2/np.log(256))) * \
    #                 fwhm_rad * np.pi ** (3/2) *\
    #                 (\
    #                  -2 * special.erfi(fwhm_rad/(2 * np.sqrt(np.log(4)))) + \
    #                       special.erfi((fwhm_rad**2 - 2j * np.pi * np.log(4))/(2 * fwhm_rad * np.sqrt(np.log(4)))) + \
    #                       special.erfi((fwhm_rad**2 + 2j * np.pi * np.log(4))/(2 * fwhm_rad * np.sqrt(np.log(4)))))

    return theta_range_deg + source_center_deg[0], phi_range_deg + source_center_deg[1]


# ### Insert the source to a map that may be provided in input smoothing it with a Gaussian

# In[7]:


def insert_source(
    source_center_deg,
    fwhm_deg,
    flux_Jy,
    nside,
    units="uK_CMB",
    input_map=None,
    frequency=None,
):
    """
    This function smooth a source with a given flux with a gaussian and returns a map 
    with the source in its position
        
    Input
    source_center_deg      - TUPLE   - the LonLat coordinates of the source in degrees
    fwhm_deg               - FLOAT   - fwhm of the gaussian in degrees
    flux_Jy                - FLOAT   - source flux in Jy
    nside                  - INT     - output map NSIDE
    units                  - STRING  - the units we want the map to be produced. Can be 'Flux'
                                       [Jy/sr], 'K_CMB' or 'uK_CMB'
    input_map              - NDARRAY - An input map that may be used to add the point source to. Default
                                       is input_map = None, which means that the map will be initialized
                                       at the desired NSIDE
    frequency              - FLOAT   - the frequency at which we want the conversion to take place
        
    Output
    return_map             - NDARRAY - The HEALPix map
    """

    import healpy as h
    import numpy as np

    # Check if the nside provided by the input map is consistent with the required nside
    if input_map is not None:
        map_nside = h.get_nside(input_map)
        if map_nside != nside:
            print(
                "The required Nside (%s) is not consistent with the input map (%s)"
                % (nside, map_nside)
            )
            return -1

    if nside < 1024:
        internal_nside = 1024  # The Nside at which the smoothing is done
    else:
        internal_nside = 2 * nside

    internal_npixels = 12 * internal_nside ** 2  # The corresponding number of pixels
    degrade_map = True  # If to degrade the map at the end

    # Here we check if we require a higher Nside than the internal. In general this should
    # not happen. In case the required Nside is larger than the internal, then all the computation
    # is run at the required Nside and there is no degradation at the end
    if internal_nside <= nside:
        internal_nside = nside
        degrade_map = False

    # Generate 2D gaussian array
    source_center_rad = np.pi * np.array(source_center_deg) / 180.0
    fwhm_rad = np.pi * fwhm_deg / 180.0

    theta_deg, phi_deg = gaussian2D_grid(source_center_deg, fwhm_deg)

    # Find correct pixels removing duplicates
    pixels = [
        h.ang2pix(internal_nside, th, ph, lonlat=True)
        for th in theta_deg
        for ph in phi_deg
    ]
    pixels = list(set(pixels))

    # Find now the angles corresponding to pixels
    angles_deg = [h.pix2ang(internal_nside, px, lonlat=True) for px in pixels]

    # Shift the angles in the [-180,180] and [-90,90] ranges for gaussian computation
    angles_deg = [center_ang(a) for a in angles_deg]

    # Convert to radians
    newtheta_rad = np.pi / 180 * np.array([ang[0] for ang in angles_deg])
    newphi_rad = np.pi / 180 * np.array([ang[1] for ang in angles_deg])

    # Now calculate gaussian values
    gaussian_values = np.array(
        [
            gaussian2D(th, source_center_rad[0], ph, source_center_rad[1], fwhm_rad)
            for th, ph in zip(newtheta_rad, newphi_rad)
        ]
    )

    # Multiply beam values by flux normalized to beam solid angle
    sigma = 1.0 / (2 * np.sqrt(2.0 * np.log(2.0))) * fwhm_rad
    beam_solid_angle = 2 * np.pi * sigma ** 2
    factor = flux_Jy / beam_solid_angle

    # Pixel solid angle
    output_map = np.zeros(internal_npixels)
    output_map[pixels] = gaussian_values * factor

    # Perform the conversion to K_CMB if required
    if units == "K_CMB" or units == "uK_CMB":
        if frequency == None:
            print(
                "Need to specify frequency to convert to thermodynamic temperature. Proceeding in Jansky"
            )
        else:
            # If we want in uK_CMB then multiply by 1e6
            output_factor = {}
            output_factor["K_CMB"] = 1
            output_factor["uK_CMB"] = 1e6

            Jysr2K_CMB = Jansky_invsr_to_K_CMB(frequency)
            output_map = output_map * Jysr2K_CMB * output_factor[units]

    # Final degradation
    #     degrade_map = False
    if degrade_map:
        return_map = h.ud_grade(output_map, nside)
    else:
        return_map = output_map

    if input_map is not None:
        return_map = return_map + input_map

    return return_map


# ### Center angles in the [-180,180] and [-90,90] ranges

# In[8]:


def center_ang(angle, degree=True):
    """
    This function centers a pair of (theta,phi) angles so that it lies in the [-180,180]
    and [-90,90] degrees ranges
        
    Input
    angle                  - TUPLE   - the angle pair to be centered
    degree                 - BOOL    - whether the angles are in degree or radians

    Output
    new_angle              - TUPLE   - the centered angle pair
    """
    import numpy as np

    th = angle[0]
    ph = angle[1]

    if degree:
        shift = 180
    else:
        shift = np.pi

    if th > shift:
        th = th - 2.0 * shift
    if th < -shift:
        th = 2.0 * shift + th

    # Phi in the -Pi/2, Pi/2 range

    if ph > shift / 2.0:
        ph = -shift + ph
    if ph < -shift / 2.0:
        ph = shift + ph

    return (th, ph)


# ### Convert Jansky to K_CMB

# In[9]:


def Jansky_invsr_to_K_CMB(frequency):
    """
    This function converts flux density from Jansy/sr to K_CMB using the astropy units.equivalencies
    class
        
    Input
    frequency        - FLOAT - the frequency in hertz at which we the conversion is needed

    Output
    Jysr2K_CMB.value - TUPLE - the conversion factor from 1 Jy to K_CMB
    """
    from astropy import units
    from astropy.cosmology import Planck15

    freq = frequency * units.Hz
    equiv = units.equivalencies.thermodynamic_temperature(freq, Planck15.Tcmb0)
    Jysr2K_CMB = (1.0 * units.Jy / units.sr).to(units.K, equivalencies=equiv)

    return Jysr2K_CMB.value


# ## Add sources to a PYSM map array (defined in the QUBIC band)

# <p style="font-size:120%;line-height:1.5">
# This is the main function in the module. It takes a PySM map array and adds point sources with a frequency behaviour consistent with their SED as derived from the PCCS catalog
# </p>
#
# <p style="font-size:120%;line-height:1.5">
# It adds the point source both in temperature and polarization ($Q$ and $U$). For polarization the $Q$ and $U$ components are calculated according to the following equation:
# </p>
#
# <p style="font-size:120%;line-height:1.5">
# $$
# Q = P\cos{2\Psi},\,U = P\sin{2\Psi}
# $$
# </p>
#
# <p style="font-size:120%;line-height:1.5">
# \noindent where $\Psi$ is the polarization angle and $P$ the point source polarized flux. Both are derived from the PCCS. The total intensity flux in mJy is extracted from the catalog field specified by the 'DETFLUX' keyword, the polarized flux in mJy is extracted from the catalog field specified by the 'PFLUX' keyword, the polarization angle in degrees is extracted from the catalog field specified by the 'ANGLE_P' keyword.
# </p>

# In[54]:


def add_sources_to_sky_map(
    input_map,
    frequencies,
    sources,
    fwhm_deg=("Auto", 1),
    catalog_file="Auto",
    reference_frequency="143",
):
    """
    This function takes a PySM map array and adds point sources with a frequency behaviour
    consistent with their SED as derived from the PCCS catalog. The point source is added in total intensity
    and polarization (Q and U). Stokes parameters Q and U are derived using the polarization angle present in
    the catalog
        
    Input
    input_map        - NDARRAY         - An array shaped (nfreq, npix, 3) containing sky maps in the 
                                         frequencies defined in the QUBIC dictionary (n_sub is the number 
                                         of frequencies, filter_nu is the center frequency, 
                                         filter_relative_bandwidth is the relative bandwidth
    frequencies      - LIST or NDARRAY - list of frequencies in Hz
    sources          - LIST or NDARRAY - list of sources ad defined in the QUBIC pccs
    fwhm_deg         - TUPLE           - The fwhm in degrees of the gaussian used to smooth the source. 
                                         fwhm_deg is specified as a tuple. If fwhm_deg[0] == 'Auto' then
                                         the fwhm is derived automatically by the pixel size multiplying it
                                         by the factor specified in fwhm_deg[1]. If fwhm_deg[0] == 'Man' then
                                         the fwhm is specified directly, in degrees, in fwhm_deg[1]. Default
                                         is fwhm_deg = ('Auto', 1)
    catalog_file        - STRING       - the catalog filename (in pickle format) If catalog_file = 'Auto'
                                         (Default) then catalog_file = qubic.data.PATH + 'qubic_pccs2.pickle'
    reference_frequency - STRING       - the reference frequency in GHz in the catalog to derive the source
                                         coordinates. It defaults to 143 GHz. Can be left to this value also
                                         for 220 GHz, as the source location is weakly dependent on the 
                                         frequency
    
    Output
    output_map       - NDARRAY         - An array shaped (nfreq, npix, 3) containing sky + point 
                                         sources maps 
    """

    import qubic
    import pickle
    import numpy as np
    import healpy as h

    catalog_frequencies = ['030', '044', '070', '100', '143', '217', '353']
    complement_frequencies = [f for f in catalog_frequencies if f != reference_frequency]
    
    output_map = input_map.copy()

    # Check that the number of frequencies is consistent with the input map
    if len(input_map) != len(frequencies):
        print("The number of maps and frequencies are inconsistent.")
        print(
            "There are %i maps and %i frequencies" % (len(input_map), len(frequencies))
        )
        return -1

    nside = h.get_nside(input_map[0, :, 0])

    # Define catalog file if not provided manually
    if catalog_file == "Auto":
        catalog_file = qubic.data.PATH + "qubic_pccs2.pickle"

    # Define fwhm in degrees
    if type(fwhm_deg) is not tuple:
        print(
            "The variable fwhm_deg must be a tuple. Call help(add_sources_to_sky_map) for more info"
        )
        return -1

    if fwhm_deg[0] != "Auto" and fwhm_deg[0] != "Man":
        print("fwhm_deg[0] must be either 'Auto' or 'Man'")
        return -1

    if fwhm_deg[0] == "Auto":  # Then fwhm is the map pixel size
        npix = h.get_map_size(input_map[0, :, 0])
        pix_size_deg = (
            2 * np.sqrt(np.pi / npix) * 180 / np.pi
        )  # It's sqrt(4pi/npix) converted to deg
        fwhm = fwhm_deg[1] * pix_size_deg
    else:
        fwhm = fwhm_deg[1]

    # Open point source catalog
    with open(catalog_file, "rb") as handle:
        catalog = pickle.load(handle)

    for source in sources:
        
        # Check if source is in catalog with the reference frequency otherwise shift to the previous
        # frequency
        
        if source not in catalog[reference_frequency].keys():
            isincatalog = [source in catalog[f].keys() for f in complement_frequencies]
            if True not in isincatalog:
                print('Source %s is not in catalog' % source)
                return -1
            print('Source %s is not in catalog at frequency %s GHz' % (source, reference_frequency))
            goodfreq = [i for i, x in enumerate(isincatalog) if x] 
            diff_freq = np.abs(np.array(list(map(float, [complement_frequencies[i] for i in goodfreq]))) - float(reference_frequency))
            index = np.where(diff_freq == np.min(diff_freq))[0]
            reference_frequency = complement_frequencies[index[0]]
            print('Switched to new reference frequency %s GHz' % reference_frequency)
            	            
        print(
            "Processing source %s (%i/%i)"
            % (source, list(sources).index(source) + 1, len(sources))
        )
        source_center_deg = (
            catalog[reference_frequency][source]["GLON"],
            catalog[reference_frequency][source]["GLAT"],
        )

        # Calculate SED of source in T and P and fitting polynomial
        sed = qubic.compact_sources_sed.build_sed(source, catalog, plot=False)
        fi = np.poly1d(sed[source]["i_fit"])
        fp = np.poly1d(sed[source]["p_fit"])

        # Loop over frequencies, for each frequency get the corresponding flux in I and P
        for fq in frequencies:
            fq_index = list(frequencies).index(fq)
            print("Processing frequency # %i of %i" % (fq_index + 1, len(frequencies)))
            i_flux = fi(fq / 1e9) / 1e3  # Conversion from mJy -> Jy
            p_flux = fp(fq / 1e9) / 1e3  # Conversion from mJy -> Jy

            polarization_angle = (
                catalog[reference_frequency][source]["ANGLE_P"] * np.pi / 180.0
            )
            if p_flux > 0.0:
                q_flux = p_flux * np.cos(2.0 * polarization_angle)
                u_flux = p_flux * np.sin(2.0 * polarization_angle)
            else:
                q_flux = 0
                u_flux = 0

            flux = [i_flux, q_flux, u_flux]

            for index in [0, 1, 2]:
                outmap = insert_source(
                    source_center_deg,
                    fwhm,
                    flux[index],
                    nside,
                    units="uK_CMB",
                    input_map=input_map[fq_index, :, index],
                    frequency=fq,
                )

                output_map[fq_index, :, index] = outmap

        input_map = output_map.copy()

        print("")

    return output_map


# ## Get source from catalog

# In[11]:


def getsource(source, frequency, catalog):
    """
    This function gets a source from the compact source database and returnts its inputs
        
    Input
    source     - STRING - the source name in the convention of the database or with the common name
    frequency  - FLOAT  - the frequency in GHz
    catalog    - DICT   - the QUBIC PCCS catalog
    
    Output
    out_source - DICT   - the source parameters
    """
    import qubic.compact_sources_sed as pccs
    import numpy as np

    # Check if source is in catalog
    exists, sourcename = pccs.isincatalog(source, catalog)

    if not exists:
        print("Source %s does not exist in catalog" % source)
        return -1

    # Check if frequency is present
    sourcefreqs = pccs.source2freqs(source, catalog, altnames=pccs.altnames)
    freq_string = "%03i" % frequency
    if freq_string not in sourcefreqs:
        print("Frequency %s GHz is not in the catalog" % frequency)
        return -1

    out_source = catalog[freq_string][sourcename]

    return out_source
