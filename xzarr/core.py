# -*- coding: utf-8 -*-
"""
"""

import collections
import dask.array as da
from dask.utils import infer_storage_options
import numpy as np
import pickle
import xarray as xr
import zarr

__all__ = ['XZarr', 'dask_to_zarr', 'dask_from_zarr', 'dataset_from_zarr',
           'dataset_to_zarr', 'xarray_from_zarr', 'xarray_to_zarr']


def _get_chunks(darr):
    for i, c in enumerate(darr.chunks):
        if len(set(c[:-1])) != 1:
            # I believe arbitrary chunking is not possible
            # but last chunk may be different
            raise ValueError("Must use regular chunking for zarr, but"
                             "dimension %s has %s"
                             % (i, darr.chunks))
    return [c[0] for c in darr.chunks]


def dask_to_zarr(darr, url, compressor='default', ret=False,
                 storage_options=None):
    """
    Save dask array to a zarr

    Parameters
    ----------
    darr: dask array
    url: location
        May include protocol, e.g., ``s3://mybucket/mykey.zarr``
    compressor: string ['default']
        Compression to use, see [zarr compressors](http://zarr.readthedocs.io/en/latest/api/codecs.html)
    """
    url = make_mapper(url, storage_options)
    chunks = _get_chunks(darr)
    out = zarr.open_array(url, mode='w', shape=darr.shape,
                          chunks=chunks, dtype=darr.dtype,
                          compressor=compressor)
    da.store(darr, out, lock=False)
    if ret:
        return out


def dask_from_zarr(url, ret=False, storage_options=None):
    """
    Load zarr data into a dask array

    Parameters
    ----------
    url: location
        May include protocol, e.g., ``s3://mybucket/mykey.zarr``
    ret: bool (False)
        To also return the raw zarr.
    """
    url = make_mapper(url, storage_options)
    d = zarr.open_array(url)
    out = da.from_array(d, chunks=d.chunks)
    if ret:
        return out, d
    return out


def xarray_to_zarr(arr, url, storage_options=None, **kwargs):
    """
    Save xarray.DataArray to a zarr

    This is a simplified method, where all metadata, including coordinates,
    is stored into a special key within the zarr.

    Parameters
    ----------
    arr: data to store
    url: location to store into
    kwargs: passed on to zarr
    """
    coorddict = [(name, arr.coords[name].values) for name in arr.dims]
    z = dask_to_zarr(arr.data, url,
                     compressor=kwargs.get('compressor', 'default'), ret=True)
    z.store['.xarray'] = pickle.dumps({'coords': coorddict, 'attrs': arr.attrs,
                                      'name': arr.name, 'dims': arr.dims}, -1)


def xarray_from_zarr(url, storage_options=None):
    """
    Load xarray.DataArray from a zarr, stored using ``xarray_to_zarr``

    Parameters
    ----------
    url: location of zarr

    Returns
    -------
    xarray.DataArray instance
    """
    z, d = dask_from_zarr(url, True)
    meta = pickle.loads(d.store['.xarray'])
    out = xr.DataArray(z, **meta)
    return out


def _safe_attrs(attrs, order=True):
    """
    Rationalize numpy contents of attributes for serialization

    Since number in xarray attributes are often numpy numbers or arrays,
    which will not JSON serialize, replace them with simple values or lists.

    Parameters
    ----------
    attrs: attributes set, dict-like
    order: bool, whether to produce an ordered or simple dict.

    Returns
    -------
    Altered attribute set
    """
    out = collections.OrderedDict() if order else {}
    for k, v in attrs.items():
        if isinstance(v, (np.number, np.ndarray)):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def dataset_to_zarr(ds, url, path_in_dataset='/', storage_options=None,
                    **kwargs):
    """
    Save xarray.Dataset in to zarr

    All coordinates, variables and their attributes will be saved. If the
    variables are based on dask.Arrays, chunking will be preserved; otherwise,
    zarr will guess a suitable chunking scheme. The user may wish to define
    the chunking manually by calling ``ds.chunk`` first.

    Parameters
    ----------
    ds: data
    url: location to save to
    path_in_dataset: string ('/')
        If only writing to some sub-set of a larger dataset, in the sense
        used by netCDF
    kwargs: passed on to zarr
    """
    url = make_mapper(url, storage_options)
    comp = kwargs.get('compressor', 'default')
    if path_in_dataset == '/':
        root = zarr.open_group(url, 'w')
    else:
        root = zarr.open_group(url, 'a')
        root = root.create_group(path_in_dataset, overwrite=True)
    attrs = {'coords': {}, 'variables': {}, 'dims': {}}
    attrs['attrs'] = _safe_attrs(ds.attrs)
    # for standard, should coords be children of the real root?
    coords = root.create_group('coords')
    for coord in ds.coords:
        # if coords are global, maybe this should be overwrite or skip-if-extant
        coords.create_dataset(name=coord, data=np.asarray(ds.coords[coord]))
        attrs['coords'][coord] = _safe_attrs(ds.coords[coord].attrs)
    variables = root.create_group('variables')
    for variable in set(ds.variables) - set(ds.coords):
        v = ds.variables[variable]
        if isinstance(v.data, da.Array):
            chunks = _get_chunks(v)
            out = variables.create_dataset(name=variable, shape=v.shape,
                                           chunks=chunks, dtype=v.dtype,
                                           compressor=comp)
            da.store(v.data, out)
        else:
            variables.create_dataset(name=variable, data=v, compressor=comp)
        attrs['dims'][variable] = v.dims
        attrs['variables'][variable] = _safe_attrs(v.attrs)
    root.attrs.update(attrs)


class XZarr(object):
    def __init__(self, url, path_in_dataset='/', storage_options=None,
                 **kwargs):
        """
        Initiate zarr storage for Xarrays

        Parameters
        ----------
        url: str
            Location to write to, maybe including protocol prefix
        path_in_dataset: str ['/']
            Location within the dataset; if '/' (default), any existing data
            will be clobbered; if anything else, that part of the data
            collection will be clobbered
        storage_options: dict [None]
            Parameters to be passed to the file-system backend
        kwargs: key-value
            extra parameters for zarr. 'compressor' will define the default
            compression applied to all data valiables.
        """
        self.url = url
        self.storage_options = storage_options
        self.path = path_in_dataset
        url = make_mapper(url, storage_options)
        self.comp = kwargs.get('compressor', 'default')
        if path_in_dataset == '/':
            self.root = zarr.open_group(url, 'w')
        else:
            root = zarr.open_group(url, 'a')
            self.root = root.create_group(path_in_dataset, overwrite=True)
        self.attrs = {'coords': {}, 'variables': {}, 'dims': {}}
        self.coords = self.root.create_group('coords')
        self.variables = self.root.create_group('variables')

    def add_coordinate(self, data, name, attrs, compressor=None):
        """
        Named 1D coordinate array for labeling data

        Parameters
        ----------
        data: 1d array-like
        name: str
            coordinate's name
        attrs: dict
            arbitrary extra information, e.g., units
        compressor: None, str or zarr compressor instance
            Override default compressor for this dataset
        """
        comp = compressor if compressor is not None else self.comp
        self.coords.create_dataset(name=name, data=data, compressor=comp)
        self.attrs['coords'][name] = _safe_attrs(attrs)

    def add_variable(self, data, name, dims, attrs, compressor=None):
        """
        Labeled data

        Parameters
        ----------
        data: numpy.ndarray or dask.array
        name: str
            dataset's name
        dims: list of str
            named coordinated corresponding to the dimensions of the data
        attrs: dict
            Arbitrary extra information, e.g., scaling, units
        compressor: None or str or zarr compressor
            Override default compressor for this dataset
        """
        comp = compressor or self.comp
        if isinstance(data, da.Array):
            chunks = _get_chunks(data)
            out = self.variables.create_dataset(
                    name=name, shape=data.shape, chunks=chunks,
                    dtype=data.dtype, compressor=comp)
            da.store(data, out)
        else:
            self.variables.create_dataset(name=name, data=data,
                                          compressor=comp)
        self.attrs['dims'][name] = dims
        self.attrs['variables'][name] = _safe_attrs(attrs)

    def create_empty(self, name, dims, dtype, chunks,attrs=None,
                     compressor=None, fill=0, as_zarr=True):
        """
        New empty dataset, to be filled in

        Parameters
        ----------
        name: str
            Dataset's name
        dims: list of str
            named coordinated corresponding to the dimensions of the data
        dtype: DType or str
            data type
        chunks: tuple of lists
            chunking sizes
        attrs: dict
            Arbitrary extra information, e.g., scaling, units
        compressor: None or str or zarr compressor
        fill: value
        as_zarr: bool (True)
            Whether to pass back the raw zarr created or the same data as
            viewer by Xarray.

        Returns
        -------
        zarr dataset or Xarray variable
        """
        comp = compressor or self.comp
        shape = [self.coords[c].size for c in dims]
        z = self.variables.create_dataset(name=name, shape=shape,
                                          dtype=dtype, compressor=comp,
                                          chunks=chunks, fill_value=fill)
        self.attrs['dims'][name] = dims
        self.attrs['variables'][name] = _safe_attrs(attrs or {})
        if as_zarr:
            return z
        else:
            return self.as_xarray()[name]

    def as_xarray(self):
        """Open this dataset as an Xarray"""
        self.flush()
        return dataset_from_zarr(self.url, self.path, self.storage_options)

    def as_zarr(self):
        return self.root

    def flush(self):
        """Write attributes to file store"""
        self.root.attrs.update(self.attrs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.flush()


def dataset_from_zarr(url, path_in_dataset='/', storage_options=None):
    """
    Load a zarr into a xarray.Dataset.

    Variables and coordinates will be loaded, with applicable attributes.
    Variables will load lazily and respect the on-disc chunking scheme;
    coordinates will be loaded eagerly into memory.

    Parameters
    ----------
    url: location to load from
    path_in_dataset: string ('/')
        For some sub-set of a larger dataset, in the sense used by netCDF

    Returns
    -------
    xarray.Dataset instance
    """
    url = make_mapper(url, storage_options)
    root = zarr.open_group(url, 'r')
    if path_in_dataset != '/':
        root = root[path_in_dataset]
    attrs = dict(root.attrs)
    coords = {}
    for coord in root['coords']:
        coords[coord] = (coord, np.array(root['coords'][coord]),
                         attrs['coords'].get(coord, None))
    out = {}
    for variable in root['variables']:
        d = root['variables'][variable]
        out[variable] = (root.attrs['dims'][variable],
                         da.from_array(d, chunks=d.chunks),
                         attrs['variables'].get(variable, None))
    ds = xr.Dataset(out, coords, attrs=attrs.get('attrs', None))
    return ds


def s3mapper(url, key=None, username=None, secret=None, password=None,
             path=None, host=None, s3=None, **kwargs):
    import s3fs
    if not s3:
        if username is not None:
            key = username
        if key is not None:
            kwargs['key'] = key
        if password is not None:
            secret = password
        if secret is not None:
            kwargs['secret'] = secret
        s3 = s3fs.S3FileSystem(**kwargs)
    return s3fs.S3Map('/'.join([host, path]), s3)


def gcsmapper(url, gcs=None, **kwargs):
    if not gcs:
        project = kwargs.pop('project', None)
        access = kwargs.pop('access', 'full_control')
        token = kwargs.pop('token', None)
        gcs = gcsfs.GCSFileSystem(project, access, token)
    return gcsfs.GCSMap(url, gcs)


mappers = {'file': lambda x, **kw: x,
           's3': s3mapper}

try:
    import gcsfs
    mappers['gs'] = gcsmapper
except ImportError:
    pass


def make_mapper(url, storage_options=None):
    options = infer_storage_options(url, storage_options)
    protocol = options.pop('protocol')
    return mappers[protocol](url, **options)


def fits_to_coords(h):
    """
    Given cartesian WCS in a FITS header, generate coordinates for the data

    This is only for CAR coordinate systems, where the pixel axes are
    along the WCS axes and there are no distorsions.

    Parameters
    ----------
    h: HDU header

    Returns
    -------
    list of (coord name, 1D coordinate values)
    """
    from astropy import wcs
    import re
    w = wcs.WCS(h)
    out = []
    for i in range(h['NAXIS']):
        l = h['NAXIS%i' % (i+1)]
        name = h['CTYPE%i' % (i+1)].split('-', 1)[0].rstrip()
        s = re.compile(r'(\(.+?\))')
        units = s.findall(h.comments['CRVAL%i' % (i+1)])
        units = units[0] if units else ''
        c = np.arange(l)
        inputs = [0, 0, 0, 0, 0]
        inputs[i] = c
        c2 = w.wcs_pix2world(*inputs)
        out.append((c2[i], name, {'units': units}))
    return out

# Test pieces
testfile = '/Users/mdurant/data/smith_sandwell_topo_v8_2.nc'


def test_dask_roundtrip():
    arr = xr.open_dataset(testfile, chunks={'latitude': 6336//11,
                                            'longitude': 10800//15}).ROSE
    darr = arr.data
    dask_to_zarr(darr, 'out.zarr', compressor=zarr.Blosc())
    assert dask_from_zarr('out.zarr').mean().compute() == darr.mean().compute()


def test_xarray_roundtrip():
    arr = xr.open_dataset(testfile, chunks={'latitude': 6336//11,
                                            'longitude': 10800//15}).ROSE
    darr = arr.data
    xarray_to_zarr(arr, 'out.xarr')
    out = xarray_from_zarr('out.xarr')
    assert out.mean().values == darr.mean().compute()


def test_dataset_roundtrip():
    ds = xr.open_dataset(testfile, chunks={'latitude': 6336//11,
                                           'longitude': 10800//15})
    dataset_to_zarr(ds, 'out.xzarr')
    out = dataset_from_zarr('out.xzarr')
    assert isinstance(out.ROSE.data, da.Array)
    for c in out.coords:
        assert (ds.coords[c].data == out.coords[c].data).all()
        assert ds.coords[c].attrs == out.coords[c].attrs
    for c in out.ROSE.coords:
        assert (ds.coords[c].data == out.coords[c].data).all()
        assert ds.coords[c].attrs == out.coords[c].attrs
    assert _safe_attrs(ds.ROSE.attrs, False) == _safe_attrs(
            out.ROSE.attrs, False)
    assert _safe_attrs(ds.attrs, False) == _safe_attrs(out.attrs, False)
    assert (ds.ROSE == out.ROSE).all().values
