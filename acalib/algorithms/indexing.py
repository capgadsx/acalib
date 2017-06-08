import acalib
from .algorithm import Algorithm
from .gms import GMS
from astropy.nddata import support_nddata, NDDataRef, NDData
from collections import namedtuple


class Indexing(Algorithm):
    """
    Perform an unsupervised region of interest detection and extract shape features.

    Parameters
    ----------
    params : dict (default = None)
        Algorithm parameters, allowed keys:

        P : float (default = 0.05)
            Thresholding quantile for multiscale segmentation.
        PRECISION : float (default = 0.02)
            Smallest scale percentage for the multiscale segmentation.
        SAMPLES : int (default = 1000)
            Number of pixels used to generate the spectra sketch.
        RANDOM_STATE : int (default = None)
            Seed for random smpling.


    References
    ----------

    .. [1] Araya, M., Candia, G., Gregorio, R., Mendoza, M., & Solar, M. (2016). Indexing data cubes for content-based searches in radio astronomy. Astronomy and Computing, 14, 23-34.

    """
    def default_params(self):
        if 'P' not in self.config:
            self.config['P'] = 0.05
        if 'PRECISION' not in self.config:
            self.config['PRECISION'] = 0.02
        if 'RANDOM_STATE' not in self.config:
            self.config['RANDOM_STATE'] = None
        if 'SAMPLES' not in self.config:
            self.config["SAMPLES"] = 1000


    def run(self, cube):
        """
            Run the indexing algorithm on a given data cube.

            Parameters
            ----------
            data : (M,N,Z) numpy.ndarray or astropy.nddata.NDData or astropy.nddata.NDDataRef
                Astronomical data cube.

            Returns
            -------
            List of ROI with the cube slice, segmented images for each resolution and ROI table.
        """

        if type(cube) is NDData or type(cube) is NDDataRef:
            if cube.wcs:
                wcs = cube.wcs
            else:
                wcs = None
            data = cube.data
        else:
            data = cube
            wcs = None


        c = []
        ROI = namedtuple('RegionsOfInterest', ['cube_slice','segmented_images','table'])
        params = {"P":self.config["P"], "PRECISION":self.config["PRECISION"]}
        gms = GMS(params)


        spectra, slices = acalib.core.spectra_sketch(data, self.config["SAMPLES"], self.config["RANDOM_STATE"])

        pp_slices = []
        for slice in slices:
            pp_slice = acalib.core.vel_stacking(cube, slice)
            labeled_images = gms.run(pp_slice)

            if wcs is not None:
                freq_min = float(wcs.all_pix2world(0, 0, slice.start, 1)[2])
                freq_max = float(wcs.all_pix2world(0, 0, slice.stop, 1)[2])
            else:
                freq_min = None
                freq_max = None

            table = acalib.core.measure_shape(pp_slice, labeled_images, freq_min, freq_max)
            if len(table) > 0:
                c.append(ROI(cube_slice=pp_slice, segmented_images=labeled_images,table=table))

        if wcs:
            wcs = wcs.dropaxis(2)
            for i,roi in enumerate(c):
                for j, im in enumerate(roi.segmented_images):
                    c[i].segmented_images[j] = NDData(data=im, wcs = wcs)
        return c

class IndexingDask(object):
    valid_fields = ['gms_percentile', 'precision', 'random_state', 'samples', 'partitions', 'partition_size', 'scheduler']

    def __init__(self):
        self.gms_percentile = 0.05
        self.precision = 0.02
        self.random_state = None
        self.samples = 1000
        self.partitions = None
        self.partition_size = None
        self.scheduler = '127.0.0.1:8786'

    def __getattr__(self, name):
        if name not in self.valid_fields:
            raise ValueError(name+' is not a valid field')

    def __setattr__(self, name, value):
        if name not in self.valid_fields:
            raise ValueError(name+' is not a valid field')
        super(IndexingDask, self).__setattr__(name, value)

    def runSafe(self, files):
        log.info('Connecting to dask-scheduler at ['+self.scheduler+']')
        client = distributed.Client(self.scheduler)
        check = lambda x: self.check(x)
        check.__name__ = 'check'
        denoise = lambda x: self.denoiseCube(x)
        denoise.__name__ = 'denoise'
        indexing = lambda x: self.runIndexing(x)
        indexing.__name__ = 'indexing'

    def check(self, fits):
        if not os.path.isabs(fits):
            return (False, 'FITS file path is not absolute')
        try:
            cube = acalib.io.loadFITS_PrimaryOnly(fits)
        except IOError:
            return (False, 'IOError')
        except MemoryError:
            return (False, 'MemoryError')
        if np.isnan(cube.data).any():
            return (False, 'NaN')
        return (True, cube)

    def denoiseCube(self, cube_data):
        if cube_data[0]:
            return (True, acalib.denoise(cube_data[1], threshold=acalib.noise_level(cube_data[1])))
        return cube_data

    def runIndexing(self, cube_data):
        if cube_data[0]:
            pass
        return cube_data

    def computeIndexing(self, data):
        gmsParams = {'P': self.config['P'], 'PRECISION': self.config['PRECISION']}
        gms = GMS(gmsParams)
        spectra, slices = acalib.core.spectra_sketch(data.data, self.config["SAMPLES"], self.config["RANDOM_STATE"])
        result = []
        print(slices)
        for slice in slices:
            slice_stacked = acalib.core.vel_stacking(data, slice)
            labeled_images = gms.run(slice_stacked)
            freq_min = None
            freq_max = None
            if data.wcs:
                freq_min = float(data.wcs.all_pix2world(0, 0, slice.start, 1)[2])
                freq_max = float(data.wcs.all_pix2world(0, 0, slice.stop, 1)[2])
            table = acalib.core.measure_shape(slice_stacked, labeled_images, freq_min, freq_max)
            print(table)
            print(len(table))
            if len(table) > 0:
                result.append(table)
        return result

    def checkAbsoluteLocalFilePaths(self, files):
        for f in files:
            if not os.path.isabs(f):
                log.error('FITS file path should be absolute when running in local-filesystem mode')
                raise ValueError('FITS file path should be absolute when running in local-filesystem mode')

    def checkCube(self, x):
        try:
            cube = acalib.io.loadFITS_PrimaryOnly(x)
        except IOError:
            log.error('Failed to load: '+os.path.basename(x))
            return (True, 'IOError') #Dafuq (?)
        except MemoryError:
            log.error('Failed to load: '+os.path.basename(x))
            return (True, 'MemoryError')
        if np.isnan(cube.data).any():
            log.error(os.path.basename(x)+' contains NaN values!!')
            return (True, 'NaN') #What we have to do when data is NaN ???
        return (False, None)

    def runChecks(self, files):
        self.checkAbsoluteLocalFilePaths(files)
        log.info('Connecting to dask-scheduler at ['+self.config['SCHEDULER_ADDR']+']')
        client = distributed.Client(self.config['SCHEDULER_ADDR'])
        check = lambda x: self.checkCube(x)
        check.__name__ = 'check'
        cores = sum(client.ncores().values())
        log.info('Running Checks on '+str(len(files))+' elements with '+str(cores)+' cores')
        data = db.from_sequence(files, self.config['PARTITION_SIZE'], self.config['N_PARTITIONS'])
        results = data.map(check).compute()
        log.info('Gathering results')
        results = client.gather(results)
        log.info('Removing dask-client')
        client.shutdown()
        return results


    def run(self, files):
        self.checkAbsoluteLocalFilePaths(files)
        log.info('Connecting to dask-scheduler at ['+self.config['SCHEDULER_ADDR']+']')
        client = distributed.Client(self.config['SCHEDULER_ADDR'])
        indexing = lambda x: self.computeIndexing(x)
        indexing.__name__ = 'computeIndexing'
        load = lambda x: acalib.io.loadFITS_PrimmaryOnly(x)
        load.__name__ = 'loadData'
        denoise = lambda x: acalib.denoise(x, threshold=acalib.noise_level(x))
        denoise.__name__ = 'denoise'
        cores = sum(client.ncores().values())
        log.info('Computing "Indexing" on '+str(len(files))+' elements with '+str(cores)+' cores')
        data = db.from_sequence(files, self.config['PARTITION_SIZE'], self.config['N_PARTITIONS'])
        results = data.map(load).map(denoise).map(indexing).compute()
        log.info('Gathering results')
        results = client.gather(results)
        log.info('Removing dask-client')
        client.shutdown()
        return results
