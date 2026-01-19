# Python source files
py.install_sources(
  ['__init__.py'], 
  subdir: 'qubic',
)

py.install_sources(['calfiles/__init__.py'], subdir: 'qubic/calfiles')

py.install_sources(['dicts/__init__.py'], subdir: 'qubic/dicts')

py.install_sources(['data/__init__.py'], subdir: 'qubic/data')

# Install lib package
py.install_sources([
    'lib/__init__.py',
    'lib/Qbeams.py',
    'lib/Qbilin_interp.py',
    'lib/Qcoaddition.py',
    'lib/QdataHandling.py',
    'lib/Qdictionary.py',
    'lib/Qfit.py',
    'lib/Qfoldertools.py',
    'lib/Qforecast.py',
    'lib/Qgps.py',
    'lib/Qhdf5.py',
    'lib/Qhorns.py',
    'lib/Qmpi_tools.py',
    'lib/Qplotters.py',
    'lib/Qripples.py',
    'lib/Qsamplings.py',
    'lib/Qscene.py',
    'lib/QskySim.py',
    'lib/Qspectra.py',
    'lib/Qutilities.py'
],
  subdir: 'qubic/lib',
)

# Install calibration subpackage
py.install_sources(
  [
    'lib/Calibration/__init__.py',
    'lib/Calibration/Qcalibration.py',
    'lib/Calibration/Qdemodulation.py',
    'lib/Calibration/Qfiber.py',
    'lib/Calibration/Qlinearity.py',
    'lib/Calibration/Qselfcal.py',
    'lib/Calibration/QtesEvaluation.py'
  ],
  subdir: 'qubic/lib/Calibration',
)

# Install fitting subpackage
py.install_sources(
  [
    'lib/Fitting/__init__.py',
    'lib/Fitting/QanalysisMC.py',
    'lib/Fitting/Qcamb.py',
    'lib/Fitting/Qmcmc.py',
    'lib/Fitting/Qnamaster.py',
    'lib/Fitting/QreadMC.py',
    'lib/Fitting/QsynthesizedBeam.py'
  ],
  subdir: 'qubic/lib/Fitting',
)

# Install instrument subpackage
py.install_sources(
  [
    'lib/Instrument/__init__.py',
    'lib/Instrument/Qinstrument.py',
    'lib/Instrument/Qacquisition.py',
    'lib/Instrument/Qnoise.py'
  ],
  subdir: 'qubic/lib/Instrument',
)

# Install map-making subpackage
py.install_sources(
  [
    'lib/MapMaking/__init__.py',
    'lib/MapMaking/Qatmosphere.py',
    'lib/MapMaking/Qcg.py',
    'lib/MapMaking/Qmap_plotter.py',
    'lib/MapMaking/Qmaps.py',
  ],
  subdir: 'qubic/lib/MapMaking',
)
py.install_sources([    
    'lib/MapMaking/ComponentMapMaking/__init__.py',
    'lib/MapMaking/ComponentMapMaking/Qchi2MM.py',
    'lib/MapMaking/ComponentMapMaking/Qcomponent_model.py',
    'lib/MapMaking/ComponentMapMaking/Qcostfunc.py',
    'lib/MapMaking/ComponentMapMaking/preset/__init__.py',
    'lib/MapMaking/ComponentMapMaking/preset/preset_acquisition.py',
    'lib/MapMaking/ComponentMapMaking/preset/preset_components.py',
    'lib/MapMaking/ComponentMapMaking/preset/preset_external_data.py',
    'lib/MapMaking/ComponentMapMaking/preset/preset_gain.py',
    'lib/MapMaking/ComponentMapMaking/preset/preset_mixingmatrix.py',
    'lib/MapMaking/ComponentMapMaking/preset/preset_qubic.py',
    'lib/MapMaking/ComponentMapMaking/preset/preset_sky.py',
    'lib/MapMaking/ComponentMapMaking/preset/preset_tools.py',
    'lib/MapMaking/ComponentMapMaking/preset/preset.py',],
  subdir: 'qubic/lib/MapMaking/ComponentMapMaking',)

py.install_sources([    
    'lib/MapMaking/FrequencyMapMaking/__init__.py',
    'lib/MapMaking/FrequencyMapMaking/FMM_errors_checking.py',
    'lib/MapMaking/FrequencyMapMaking/Qfmm.py',
    'lib/MapMaking/FrequencyMapMaking/Qspectra_component.py'],
    subdir: 'qubic/lib/MapMaking/FrequencyMapMaking',)

# Compile fortran
subdir('flib')
