This directory contains: 

To use synth_beam_many_frquencies.Rmd and SynthesizedBeam.Rmd or scanSource.py you have to define two environment variables: QUBIC_DATADIR and QUBIC_DICT

QUBIC_DATADIR points to the directory where qubicsoft script's (instrument.py, acquisition.py, etc) are.
QUBIC_DICT points to directory where dictionaries are save: $QUBIC_DATADIR/dicts/

to define those variables you have to edit your ~/.bashrc file and add the following lines:

export QUBIC_DATADIR=/home/user/path/to/qubic/directory
export QUBIC_DICT=/home/user/path/to/qubic/directory/dicts



# Documentation on how QUBIC works and some improvements in end-to-end simulation pipeline.
## scan.pdf: Notes with detailed explanations of the methods and configurations used in the pipeline (by Thibaut Louise)



# Notebooks for beginners qubicsoft users.
## beamtest.Rmd: 
		This code plots the beam profiles for the 3 possible values of the dictionary entry 'beam_shape' at 150 and 220 GHz

## monofreq_pipeline_end2end.Rmd:
		This notebook is the typical pipeline for data simulation and analysis. There are 2 parts :

		    From a given sky map, simulate Time Order Data (TOD)
    		From those TOD, perform Map Making to reconstruct the input sky

		Here we work with only one frequency band.

## SynthesizedBeam.Rmd: 
		This notebook aims at showing how to obtain the QUBIC Synthesized beam for a given set of horns accounting for various effects (detector size, bandwidth...).