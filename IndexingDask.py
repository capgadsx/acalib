import acalib
import glob
import os.path

indexing = acalib.IndexingDask()
indexing.config['PRECISION'] = 0.01
indexing.config['SAMPLES'] = 100
folder = '/mnt/nfs/cycle0fits_copia/cycle0fits/'
files = glob.glob(folder+'*.fits')
print files
'''checkResults = indexing.runChecks(files)
for indx, result in enumerate(checkResults):
    if result[0]:
        print(os.path.basename(files[indx])+' failed because of '+result[1])
'''
