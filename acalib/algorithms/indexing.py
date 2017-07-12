import os
import dask
import numpy
import acalib
import distributed
from .gms import GMS
from .algorithm import Algorithm
from astropy.nddata import NDDataRef, NDData
from skimage.filters import threshold_local
from skimage.measure import label,regionprops
from skimage.morphology import binary_opening, disk
from skimage.segmentation import clear_border
from acalib.core.analysis import _kernelsmooth, _kernel_shift

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
    __valid_fields = ['gms_percentile', 'precision', 'random_state', 'samples', 'scheduler']

    def __init__(self):
        self.gms_percentile = 0.05
        self.precision = 0.02 
        self.random_state = None
        self.samples = 1000
        self.scheduler = '127.0.0.1:8786'

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            return super(IndexingDask, self).__getattr__(name)
        if name not in self.__valid_fields:
            raise ValueError(name+' is not a valid field')

    def __setattr__(self, name, value):
        if name not in self.__valid_fields:
            raise ValueError(name+' is not a valid field')
        super(IndexingDask, self).__setattr__(name, value)
    
    def run(self, files):
        client = distributed.Client(self.scheduler)
        indexing_pipeline = self.__create_pipeline(files)
        results = client.compute(indexing_pipeline)
        for future in distributed.as_completed(results):
            print('Compute finished for '+future.result()[1])
        pass

    def __create_pipeline(self, files):
        load = lambda x : self.__indexing_load(x)
        load.__name__ = 'load-fits'
        denoise = lambda x: self.__indexing_denoise(x)
        denoise.__name__ = 'denoise-cube'
        get_slices = lambda x: self.__indexing_spectra(x)
        get_slices.__name__ = 'slice-cube'
        vel_stacking = lambda x: self.__gms_vel_stacking(x[0], x[1])
        vel_stacking.__name__ = 'vel-stacking'
        get_w_gms = lambda x: self.__gms_optimal_w(x)
        get_w_gms.__name__ = 'gms-optimal-w'
        gms = lambda x: self.__gms(x[0], x[1])
        gms.__name__ = 'gms'
        measure_shape = lambda x: self.__indexing_measure_shape(x[0], x[1], x[2], x[3])
        measure_shape.__name__ = 'measure-shape'
        items_denoised_cubes = []
        for i in files:
            item = dask.delayed(load)(i)
            item = dask.delayed(denoise)(item)
            items_denoised_cubes.append(item)
        items_cube_slices = []
        for i in items_denoised_cubes:
            item = dask.delayed(get_slices)(i)
            items_cube_slices.append(item)
        items_velocity_stacked_cubes = []
        for index, item_cube_sliced in enumerate(items_cube_slices):
            item = dask.delayed(vel_stacking)((items_denoised_cubes[index], item_cube_sliced))
            items_velocity_stacked_cubes.append(item)
        items_w_values_for_gms = []
        for i in items_velocity_stacked_cubes:
            item = dask.delayed(get_w_gms)(i)
            items_w_values_for_gms.append(item)
        items_gms_results = []
        for index, item_stacked_cube in enumerate(items_velocity_stacked_cubes):
            item = dask.delayed(gms)((item_stacked_cube, items_w_values_for_gms[index]))
            items_gms_results.append(item)            
        items_indexing_results = []
        for index, item_gms_result in enumerate(items_gms_results):
            item = dask.delayed(measure_shape)((items_denoised_cubes[index],
                                                items_velocity_stacked_cubes[index],
                                                items_cube_slices[index],
                                                item_gms_result))
            items_indexing_results.append(item)
        return items_indexing_results
    
    def __indexing_load(self, x):
        if not os.path.isabs(x):
            return [False, x, 'FITS file path is not absolute']
        try:
            cube = acalib.io.loadFITS_PrimaryOnly(x)
        except IOError:
            return [False, x, 'IOError']
        except MemoryError:
            return [False, x, 'MemoryError']
        return [True, x, cube]

    def __indexing_denoise(self, item):
        if item[0]:
            noise_level = acalib.noise_level(item[2])
            return [True, item[1], acalib.denoise(item[2], threshold=noise_level)]
        return item

    def __indexing_spectra(self, item):
        if item[0]:
            slices = acalib.core.spectra_sketch(item[2].data, self.samples, self.random_state)[1]
            if len(slices) > 0:
                return [True, item[1], slices]
            else:
                return [False, item[1], 'spectra_sketch returned empty slices']
        return item

    def __gms_vel_stacking(self, item_cube, item_slice):
        if item_slice[0]:
            vel_stacking = lambda x: self.__gms_vel_stacking_delayed(x[0], x[1])
            vel_stacking.__name__ = 'vel-stacking-acalib'
            velocity_stacked_cubes = []
            for slice in item_slice[2]:
                velocity_stacked_cube = dask.delayed(vel_stacking)((item_cube[2], slice))
                velocity_stacked_cubes.append(velocity_stacked_cube)
            with distributed.worker_client() as client:
                velocity_stacked_cubes = client.compute(velocity_stacked_cubes)
                velocity_stacked_cubes = client.gather(velocity_stacked_cubes)
            return [True, item_cube[1], velocity_stacked_cubes]
        return item_slice

    def __gms_vel_stacking_delayed(self, cube, slice):
        cube = acalib.core.vel_stacking(cube.data, slice)
        cube[numpy.isnan(cube)] = 0
        return cube

    def __gms_optimal_w(self, item_with_stacked_images):
        if item_with_stacked_images[0]:
            w_delayed = lambda x : self.__gms_optimal_w_compute(x[0], x[1])
            w_delayed.__name__ = 'compute-optimal-w'
            optimal_w_results = []
            for stacked_image in item_with_stacked_images[2]:
                x = dask.delayed(w_delayed)((stacked_image, self.gms_percentile)) #TODO: Change precision to gms_percentile
                optimal_w_results.append(x)
            with distributed.worker_client() as client:
                optimal_w_results = client.compute(optimal_w_results)
                optimal_w_results = client.gather(optimal_w_results)
            return [True, item_with_stacked_images[1], optimal_w_results]
        return item_with_stacked_images

    def __gms_optimal_w_compute(self, image, p):
        radiusMin = 5
        radiusMax = 40
        inc = 1
        image = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
        dims = image.shape
        rows = dims[0]
        cols = dims[1]
        maxsize = numpy.max([rows, cols])
        imagesize = cols * rows
        radius_thresh = numpy.round(numpy.min([rows, cols]) / 4.)
        unit = numpy.round(maxsize / 100.)
        radiusMin = radiusMin * unit
        radiusMax = radiusMax * unit
        radiusMax = int(numpy.min([radiusMax, radius_thresh]))
        radius = radiusMin
        inc = inc * unit
        bg = numpy.percentile(image, p * 100)
        fg = numpy.percentile(image, (1 - p) * 100)
        min_ov = imagesize
        overalls = []
        threshold = lambda x: self.__gms_optimal_w_threshold(x[0], x[1], x[2], x[3])
        threshold.__name__ = 'max-w-threshold'
        get_radius = lambda x: self.__gms_optimal_w_get_min_overall(x[0], x[1], x[2])
        get_radius.__name__ = 'find-minimal-w'
        while radius <= radiusMax:
            x = dask.delayed(threshold)((image, radius, bg, fg))
            overalls.append(x)
            radius += inc
        radius_with_min_overall = dask.delayed(get_radius)((overalls, min_ov, radius))
        with distributed.worker_client() as client:
            radius_with_min_overall = client.compute(radius_with_min_overall)
            radius_with_min_overall = client.gather(radius_with_min_overall)
        return radius_with_min_overall

    def __gms_optimal_w_threshold(self, image, radius, bg, fg):
        tt = int(radius ** 2)
        if tt % 2 == 0:
            tt += 1
        threshold_val = threshold_local(image, tt, method='mean', offset=0)
        g = image > threshold_val
        overall = self.__gms_optimal_w_bg_fg(image, g, bg, fg)
        return (overall, radius)

    def __gms_optimal_w_bg_fg(self, f, g, bg, fg):
        dims = f.shape
        rows = dims[0]
        cols = dims[1]
        fp_result = 0
        fn_result = 0
        for row in range(rows):
            for col in range(cols):
                if g[row][col] == True:
                    if (numpy.abs(f[row][col] - bg) < numpy.abs(f[row][col] - fg)):
                        fp_result += 1
                if g[row][col] == False:
                    if (numpy.abs(f[row][col] - bg) > numpy.abs(f[row][col] - fg)):
                        fn_result += 1
        overall = fp_result + fn_result
        return overall

    def __gms_optimal_w_get_min_overall(self, overalls, minimum, radius):
        for overall in overalls:
            if overall[0] < minimum:
                minimum = overall[0]
                radius = overall[1]
        return radius

    def __gms(self, item_stacked_images, item_w):
        if item_stacked_images[0]:
            w_max = item_w[2][0]
            gms_results = []
            compute_gms = lambda x: self.__gms_compute(x[0], x[1])
            compute_gms.__name__ = 'compute-gms'
            for image in item_stacked_images[2]:
                x = dask.delayed(compute_gms)((image, w_max))
                gms_results.append(x)
            with distributed.worker_client() as client:
                gms_results = client.compute(gms_results)
                gms_results = client.gather(gms_results)
            return [True, item_stacked_images[1], gms_results]
        return item_stacked_images

    #TODO: Test with numba jit(?)
    def __gms_compute(self, image, w_max):
        if len(image.shape) > 2:
            raise ValueError('Only 2D data cubes supported')
        dims = image.shape
        rows = dims[0]
        cols = dims[1]
        size = numpy.min([rows, cols])
        precision = size * self.precision
        image = image.astype('float64')
        diff = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
        tt = w_max ** 2
        if tt % 2 == 0:
            tt += 1
        threshold_val = threshold_local(diff, int(tt), method='mean', offset=0)
        g = diff > threshold_val
        r = w_max / 2
        rMin = 2 * numpy.round(self.precision)
        image_list = []
        while r > rMin:
            background = numpy.zeros((rows, cols))
            selem = disk(r)
            sub = binary_opening(g, selem)
            sub = clear_border(sub)
            sub = label(sub)
            fts = regionprops(sub)
            image_list.append(sub)
            if len(fts) > 0:
                for props in fts:
                    C_x, C_y = props.centroid
                    radius = int(props.equivalent_diameter / 2.)
                    kern = 0.01 * numpy.ones((2 * radius, 2 * radius))
                    krn = _kernelsmooth(x=numpy.ones((2 * radius, 2 * radius)), kern=kern)
                    krn = numpy.exp(numpy.exp(krn))
                    if numpy.max(krn) > 0:
                        krn = (krn - numpy.min(krn)) / (numpy.max(krn) - numpy.min(krn))
                        background = _kernel_shift(background, krn, C_x, C_y)
            if numpy.max(background) > 0:
                background = (background - numpy.min(background)) / (numpy.max(background) - numpy.min(background))
                diff = diff - background
            diff = (diff - numpy.min(diff)) / (numpy.max(diff) - numpy.min(diff))
            tt = int(r * r)
            if tt % 2 == 0:
                tt += 1
            adaptive_threshold = threshold_local(diff, tt, offset=0, method='mean')
            g = diff > adaptive_threshold
            r = numpy.round(r / 2.)
        return image_list

    def __indexing_measure_shape(self, item_cube, item_stacked, item_slices, item_labeled_images):
        if item_labeled_images[0]:
            cube = item_cube[2]
            vel_stacked_images = item_stacked[2]
            slices = item_slices[2]
            labeled_images = item_labeled_images[2]
            assert len(vel_stacked_images) == len(slices) == len(labeled_images)
            tables_results = []
            get_table = lambda x: acalib.core.measure_shape(x[0], x[1], x[2], x[3])
            get_table.__name__ = 'compute-measure-shape'
            for i, vel_stacked_image in enumerate(vel_stacked_images):
                freq_min = None
                freq_max = None
                if cube.wcs:
                    freq_min = float(cube.wcs.all_pix2world(0, 0, slices[i].start, 1)[2])
                    freq_max = float(cube.wcs.all_pix2world(0, 0, slices[i].stop, 1)[2])
                table = dask.delayed(get_table)((vel_stacked_image, labeled_images[i], freq_min, freq_max))
                tables_results.append(table)
            with distributed.worker_client() as client:
                tables_results = client.compute(tables_results)
                tables_results = client.gather(tables_results)
            return [True, item_cube[1], tables_results]
        return item_labeled_images