import os
import sys

folder = 'E2E_nrec2_withoutconv_MM/cmbdust_convolved_DB/maps/'

files = os.listdir(folder)

for i in files:
    #print(folder + i)
	os.system(f'sbatch main.sh {folder + i}')
