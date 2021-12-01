#!/usr/bin/env python
# coding: utf-8

# <p style="font-size:260%;line-height:1.5">Generate SEDs of compact sources from the Planck Catalog of Compact Sources </p>

# <p style="font-size:140%;line-height:1.2">
#     Here I develop a set of functions to read the Planck PCCS and derive the SEDs of compact sources that are seen at all frequencies
#     </p>

# # Common data


import qubic

freqs    = ['030','044','070','100','143','217','353']
freqs_ex = ['100','143','217','353']
altnames = {           'Crab'       : '184.5-05.8',           'RCW38'      : '267.9-01.1',           'Orion'      : '209.0-19.4'}

catalog_dir = qubic.data.PATH


# # Functions

# ## Check if source is in catalog

# In[1]:


def isincatalog(source, catalog):
    
    freqs    = ['030','044','070','100','143','217','353']
    
    # Make list of all sources
    allsources = []
    for f in freqs:
        allsources = allsources + list(catalog[f].keys())
    allsources = list(set(allsources))

    # Check if source exists in catalog
    if source in list(altnames.keys()):
        return True, altnames[source]
    elif source in allsources:
        return True, source
    else:
        return False, ''       


# ## Build catalog from PCCS

# In[2]:


def build_catalog(freqs = 'All', freqs_ex = 'All', excluded = True):
    '''
    This function builds a dictionary containing the main parameters of the compact sources
    contained in the PCCS
        
    Input
    freqs       - LIST  - List of frequencies in the catalog (Default 
                          ['030','044','070','100','143','217','353'])
    freqs_ex     - LIST - List of frequencies of excluded catalogs (Default
                          ['100','143','217','353'])
    excluded     - BOOL - Whether to include the excluded catalogs (Default: True)
        
    Output
    catalog      - DICT - Dictionary containing the data
    '''
    from astropy.io.fits import open as fitsOpen # For FITS files
    import numpy as np
    
    if freqs == 'All':
        freqs    = ['030','044','070','100','143','217','353']

    if freqs_ex == 'All':
        freqs_ex = ['100','143','217','353']
    
    catalog = {}

    # Read normal catalogs
    global_namelist = []
    for f in freqs:
        
        print('Building catalog at %s GHz from PCCS2' % f)
        catalog[f] = {}
        fname = '%sCOM_PCCS_%s_R2.01.fits' % (catalog_dir, f)
        fd = fitsOpen(fname, "readonly")
        names    = fd[1].data['NAME    ']
        
        # Position
        ras      = fd[1].data['RA      ']
        decs     = fd[1].data['DEC     ']
        gLons    = fd[1].data['GLON    ']
        gLats    = fd[1].data['GLAT    ']
        
        # Intensity flux
        detFluxs       = fd[1].data['DETFLUX      ']
        detFluxs_err   = fd[1].data['DETFLUX_ERR  ']
        aperFluxs      = fd[1].data['APERFLUX     ']
        aperFluxs_err  = fd[1].data['APERFLUX_ERR ']
        psfFluxs       = fd[1].data['PSFFLUX      ']
        psfFluxs_err   = fd[1].data['PSFFLUX_ERR  ']
        gauFluxs       = fd[1].data['GAUFLUX      ']
        gauFluxs_err   = fd[1].data['GAUFLUX_ERR  ']
        
        # Polarized flux
        ps                = fd[1].data['P                  ']
        ps_err            = fd[1].data['P_ERR              ']
        angle_ps          = fd[1].data['ANGLE_P            ']
        angle_ps_err      = fd[1].data['ANGLE_P_ERR        ']
        aper_ps           = fd[1].data['APER_P             ']
        aper_ps_err       = fd[1].data['APER_P_ERR         ']
        aper_angle_ps     = fd[1].data['APER_ANGLE_P       ']
        aper_angle_ps_err = fd[1].data['APER_ANGLE_P_ERR   ']
        
        fd.close()
        for name, ra, dec, gLon, gLat, detFlux, detFlux_err, aperFlux, aperFlux_err,              psfFlux, psfFlux_err, gauFlux, gauFlux_err, p, p_err, angle_p, angle_p_err, aper_p,            aper_p_err, aper_angle_p, aper_angle_p_err in         zip (names, ras, decs, gLons, gLats, detFluxs, detFluxs_err, aperFluxs, aperFluxs_err,              psfFluxs, psfFluxs_err, gauFluxs, gauFluxs_err, ps, ps_err, angle_ps, angle_ps_err, aper_ps,            aper_ps_err, aper_angle_ps, aper_angle_ps_err):
            
            if f == freqs[0]:
                # If we are scanning the first frequency then define names based on GLON and GLAT
                # Rounded to 1 decimal place
                
                new_name = build_name(name)
                
                global_namelist.append(new_name)

            else:
                # For other frequencies see if each source is close enough to be one of the first frequency
                # set. In this case use the name already used in the first set, otherwise define new name 
                # based on rounded GLON GLAT
                
                new_name = build_name(name)
                
                source_exist, new_name = duplicate_source(new_name, global_namelist)
                
                if source_exist == False:
                    global_namelist.append(new_name)
                
            catalog[f][new_name]    = {}
            
            # Name
            catalog[f][new_name]['NAME']    = new_name
            
            # Position
            catalog[f][new_name]['RA'  ]    = np.float(ra)
            catalog[f][new_name]['DEC' ]    = np.float(dec)
            catalog[f][new_name]['GLON']    = np.float(gLon)
            catalog[f][new_name]['GLAT']    = np.float(gLat)
            
            # Intensity flux
            catalog[f][new_name]['DETFLUX'      ] = np.float(detFlux)
            catalog[f][new_name]['DETFLUX_ERR'  ] = np.float(detFlux_err)            
            catalog[f][new_name]['APERFLUX'     ] = np.float(aperFlux)
            catalog[f][new_name]['APERFLUX_ERR' ] = np.float(aperFlux_err)            
            catalog[f][new_name]['PSFFLUX'      ] = np.float(psfFlux)
            catalog[f][new_name]['PSFFLUX_ERR'  ] = np.float(psfFlux_err)            
            catalog[f][new_name]['GAUFLUX'      ] = np.float(gauFlux)
            catalog[f][new_name]['GAUFLUX_ERR'  ] = np.float(gauFlux_err) 
            
            # Polarized flux
            catalog[f][new_name]['PFLUX'           ] = np.float(p)
            catalog[f][new_name]['PFLUX_ERR'       ] = np.float(p_err)    
            catalog[f][new_name]['ANGLE_P'         ] = np.float(angle_p)
            catalog[f][new_name]['ANGLE_P_ERR'     ] = np.float(angle_p_err)    
            catalog[f][new_name]['APER_P'          ] = np.float(aper_p)
            catalog[f][new_name]['APER_P_ERR'      ] = np.float(aper_p_err)    
            catalog[f][new_name]['APER_ANGLE_P'    ] = np.float(aper_angle_p)
            catalog[f][new_name]['APER_ANGLE_P_ERR'] = np.float(aper_angle_p_err)    
            
            catalog[f][new_name]['ALTNAME'] = ''
        
    if excluded:

        # Read excluded catalogs
        for f in freqs_ex:
            print('Building catalog at %s GHz from PCCS2E' % f)
            fname = '%sCOM_PCCS_%s-excluded_R2.01.fits' % (catalog_dir, f)
            fd = fitsOpen(fname, "readonly")
            names    = fd[1].data['NAME    ']

            # Position
            ras      = fd[1].data['RA      ']
            decs     = fd[1].data['DEC     ']
            gLons    = fd[1].data['GLON    ']
            gLats    = fd[1].data['GLAT    ']

            # Intensity flux
            detFluxs       = fd[1].data['DETFLUX      ']
            detFluxs_err   = fd[1].data['DETFLUX_ERR  ']
            aperFluxs      = fd[1].data['APERFLUX     ']
            aperFluxs_err  = fd[1].data['APERFLUX_ERR ']
            psfFluxs       = fd[1].data['PSFFLUX      ']
            psfFluxs_err   = fd[1].data['PSFFLUX_ERR  ']
            gauFluxs       = fd[1].data['GAUFLUX      ']
            gauFluxs_err   = fd[1].data['GAUFLUX_ERR  ']

            # Polarized flux
            ps                = fd[1].data['P                  ']
            ps_err            = fd[1].data['P_ERR              ']
            angle_ps          = fd[1].data['ANGLE_P            ']
            angle_ps_err      = fd[1].data['ANGLE_P_ERR        ']
            aper_ps           = fd[1].data['APER_P             ']
            aper_ps_err       = fd[1].data['APER_P_ERR         ']
            aper_angle_ps     = fd[1].data['APER_ANGLE_P       ']
            aper_angle_ps_err = fd[1].data['APER_ANGLE_P_ERR   ']
            fd.close()
            for name, ra, dec, gLon, gLat, detFlux, detFlux_err, aperFlux, aperFlux_err,                  psfFlux, psfFlux_err, gauFlux, gauFlux_err, p, p_err, angle_p, angle_p_err, aper_p,                aper_p_err, aper_angle_p, aper_angle_p_err in             zip (names, ras, decs, gLons, gLats, detFluxs, detFluxs_err, aperFluxs, aperFluxs_err,                  psfFluxs, psfFluxs_err, gauFluxs, gauFluxs_err, ps, ps_err, angle_ps, angle_ps_err, aper_ps,                aper_ps_err, aper_angle_ps, aper_angle_ps_err):

                new_name = build_name(name)

                source_exist, new_name = duplicate_source(new_name, global_namelist)
                
                if source_exist == False:
                    global_namelist.append(new_name)
 
                catalog[f][new_name]    = {}
            
                # Name
                catalog[f][new_name]['NAME']    = new_name

                # Position
                catalog[f][new_name]['RA'  ]    = np.float(ra)
                catalog[f][new_name]['DEC' ]    = np.float(dec)
                catalog[f][new_name]['GLON']    = np.float(gLon)
                catalog[f][new_name]['GLAT']    = np.float(gLat)

                # Intensity flux
                catalog[f][new_name]['DETFLUX'      ] = np.float(detFlux)
                catalog[f][new_name]['DETFLUX_ERR'  ] = np.float(detFlux_err)            
                catalog[f][new_name]['APERFLUX'     ] = np.float(aperFlux)
                catalog[f][new_name]['APERFLUX_ERR' ] = np.float(aperFlux_err)            
                catalog[f][new_name]['PSFFLUX'      ] = np.float(psfFlux)
                catalog[f][new_name]['PSFFLUX_ERR'  ] = np.float(psfFlux_err)            
                catalog[f][new_name]['GAUFLUX'      ] = np.float(gauFlux)
                catalog[f][new_name]['GAUFLUX_ERR'  ] = np.float(gauFlux_err) 

                # Polarized flux
                catalog[f][new_name]['PFLUX'           ] = np.float(p)
                catalog[f][new_name]['PFLUX_ERR'       ] = np.float(p_err)    
                catalog[f][new_name]['ANGLE_P'         ] = np.float(angle_p)
                catalog[f][new_name]['ANGLE_P_ERR'     ] = np.float(angle_p_err)    
                catalog[f][new_name]['APER_P'          ] = np.float(aper_p)
                catalog[f][new_name]['APER_P_ERR'      ] = np.float(aper_p_err)    
                catalog[f][new_name]['APER_ANGLE_P'    ] = np.float(aper_angle_p)
                catalog[f][new_name]['APER_ANGLE_P_ERR'] = np.float(aper_angle_p_err)    

                catalog[f][new_name]['ALTNAME'] = ''

    return catalog


# In[3]:


def build_name(name):
    '''
    This function builds a source name from the PCCS name by rounding l and b to the first decimal place
        
    Input
    name         - STRING - source name as defined in the PCCS (Glll.ll±bb.bb)
        
    Output
    new_name     - STRING - source new name defined as lll.l±bb.b
    '''
    import numpy as np
    
    name_l = np.round(np.float(name[-12:-6]),1)
    str_l  = '%05.1f' % name_l
    sign_b = name[-6]
    name_b = np.round(np.float(name[-5:]),1)
    str_b = '%04.1f' % name_b
    new_name = str_l.rjust(5,'0') + sign_b + str_b.rjust(4,'0')
    
    return new_name


# In[4]:


def duplicate_source(name, global_namelist, threshold = 0.1):
    '''
    This function finds if a given source is a duplicate of others already found in catalogs
    relative to other frequencies by checking the distance in GLON and GLAT
        
    Input
    name                   - STRING       - source name modified by the catalog building routine
                                            (Glll.l±bb.b)
    global_namelist        - LIST         - list of modified names of sources already loaded from other 
                                            frequency catalogs
    threshold              - FLOAT        - maximum distance in degrees to decide whether two sources  
                                            coincide (Default threshold = 0.1)
    Output
    isduplicate, new_name  - BOOL, STRING - whether a duplicate has been found, new name
    '''
    import numpy as np
    
    name_l = np.float(name[0:5])
    name_b = np.float(name[-5:])
    
    for item in global_namelist:
        ex_l = np.float(item[0:5])
        ex_b = np.float(item[-5:])
        
        if (np.abs(name_l - ex_l) <= threshold) and (np.abs(name_b - ex_b) <= threshold):
            # In this case we have the same source, return True and the name
            return True, item
    
    return False, name    


# ## Build SEDs

# ### SEDs of common sources

# In[6]:


def build_sed_allfreqs(catalog, freqs = ['030','044','070','100','143','217','353']):
    '''
    This function builds the SED of the sources in the catalog using data across frequencies specified
    in freqs
        
    Input
    catalog      - DICT - The dictionary with the source catalog

    freqs        - LIST - List of frequencies (Default ['030', '044', '070', 100','143','217','353'])
        
    Output
    SED          - DICT - Dictionary containing the SED (frequencies, measured I_flux, measured P_flux
                          4th order polinomial fits to measured I_flux and P_flux
    '''   
    import numpy as np
    
    # Build common set of sources
    inters = ''
    for f in freqs:
        inters = inters + ('set(catalog["%s"].keys()) & ' % f)
    inters = 'list(' + inters[0:-2] + ')'
    common_sources = eval(inters)
    
    flist = np.array(list(map(float,freqs)))
    
    SED = {}
    
    for source in common_sources:
        SED[source] = {}
        i_flux    = np.array([catalog[f][source]['DETFLUX'] for f in freqs])
        p_flux    = np.array([catalog[f][source]['PFLUX'] for f in freqs])
        sed_i_fit = np.polyfit(flist, i_flux,4)
        sed_p_fit = np.polyfit(flist, p_flux,4)
        SED[source]['freq']   = flist
        SED[source]['i_flux'] = i_flux
        SED[source]['p_flux'] = p_flux
        SED[source]['i_fit']  = sed_i_fit
        SED[source]['p_fit']  = sed_p_fit

    return SED


# ### SED of a given source

# In[7]:


def build_sed(source, catalog, plot = False, polyfit = 3):
    '''
    This function builds the SED of a given source
        
    Input
    source       - STRING - The source name, either in the stanard lll.ll±bb.bb format or in the common 
                            name format if a translation is available (e.g. Crab)

    catalog      - DICT - The dictionary with the source catalog

    plot         - BOOL - Whether to plot intensity and polarized fluxes. Default: False

    polyfit      - INT  - Order of the polynomial fit. Default: 3

    Output
    SED          - DICT - Dictionary containing the SED (frequencies, measured I_flux, measured P_flux
                          4th order polinomial fits to measured I_flux and P_flux
    '''    
    import numpy as np
    import pylab as pl
    
    # Check if source is in catalog
    exists, sourcename = isincatalog(source, catalog)
    
    if not exists:
        print('Source %s is not in catalog' % source)
        return -1
    
    # Get the number of frequencies at which we have data
    s_freqs = source2freqs(source, catalog)
    if len(s_freqs) <= 1:
        print('Not enough frequencies to build a SED')
        return -1

    flist = np.array(list(map(float,s_freqs)))

    SED = {}
    
    SED[sourcename] = {}
    
    i_flux    = np.array([catalog[f][sourcename]['DETFLUX'] for f in s_freqs])
    p_flux    = np.array([catalog[f][sourcename]['PFLUX']   for f in s_freqs])
    sed_i_fit = np.polyfit(flist, i_flux, polyfit)
    sed_p_fit = np.polyfit(flist, p_flux, polyfit)
    SED[sourcename]['freq']   = flist
    SED[sourcename]['i_flux'] = i_flux
    SED[sourcename]['p_flux'] = p_flux
    SED[sourcename]['i_fit']  = sed_i_fit
    SED[sourcename]['p_fit']  = sed_p_fit
    
    if plot:
        newfreq = np.arange(flist[0], flist[-1] + 1, 1)
        fi = np.poly1d(SED[sourcename]['i_fit'])
        fp = np.poly1d(SED[sourcename]['p_fit'])
        
        pl.figure(figsize = (13,7))
        
        # Intensity plot
        pl.subplot(121)
        pl.plot(SED[sourcename]['freq'],SED[sourcename]['i_flux'],'.')
        pl.plot(newfreq, fi(newfreq))
        pl.xlabel('Frequency [GHz]')
        pl.ylabel('Flux [mJy]')
        pl.title('%s - Intensity flux' % source)
        
        # Polarization plot
        pl.subplot(122)
        pl.plot(SED[sourcename]['freq'],SED[sourcename]['p_flux'],'.')
        pl.plot(newfreq, fp(newfreq))
        pl.xlabel('Frequency [GHz]')
        pl.ylabel('Flux [mJy]')
        pl.title('%s - Polarized flux' % source)
        

    return SED


# ## Translate from common source name to catalog name

# In[8]:


def name2cat(name, altnames):

    if name not in list(altnames.keys()):
        print('Name %s not known' % name)
        return -1
    
    return altnames[name]


# ## Return the frequencies of a given source name 

# In[12]:


def source2freqs(source, catalog, altnames = altnames):
    '''
    This function return the list of frequencies in the catalog given a certain source
        
    Input
    source       - STRING - The source name, either in the stanard lll.ll±bb.bb format or in the common 
                            name format if a translation is available (e.g. Crab)

    catalog      - DICT   - The PCCS in Qubic format
        
    altnames     - DICT   - The correspondence between common name and catalog standard name (Defaults to 
                            altnames defined at the top of the notebook)
 
    Output
    freqlist     - LIST - List of frequencies where a source is found
    '''        
    import numpy as np
    
    exists, sourcename = isincatalog(source, catalog)
    
    if not exists:
        print('Source %s is not in catalog' % source)
        return -1        
    
    isinfreq = [sourcename in list(catalog[f].keys()) for f in freqs]
    
    return [freqs[i] for i in list(np.where(isinfreq)[0])]


