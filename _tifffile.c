#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tifffile.py

# Copyright (c) 2008-2016, Christoph Gohlke
# Copyright (c) 2008-2016, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read image and meta data from (bio)TIFF files. Save numpy arrays as TIFF.

Image and metadata can be read from TIFF, BigTIFF, OME-TIFF, STK, LSM, NIH,
SGI, ImageJ, MicroManager, FluoView, SEQ and GEL files.
Only a subset of the TIFF specification is supported, mainly uncompressed
and losslessly compressed 2**(0 to 6) bit integer, 16, 32 and 64-bit float,
grayscale and RGB(A) images, which are commonly used in bio-scientific imaging.
Specifically, reading JPEG and CCITT compressed image data, chroma subsampling,
or EXIF, IPTC, GPS, and XMP metadata is not implemented. Only primary info
records are read for STK, FluoView, MicroManager, and NIH Image formats.

TIFF, the Tagged Image File Format aka Thousands of Incompatible File Formats,
is under the control of Adobe Systems. BigTIFF allows for files greater than
4 GB. STK, LSM, FluoView, SGI, SEQ, GEL, and OME-TIFF, are custom extensions
defined by Molecular Devices (Universal Imaging Corporation), Carl Zeiss
MicroImaging, Olympus, Silicon Graphics International, Media Cybernetics,
Molecular Dynamics, and the Open Microscopy Environment consortium
respectively.

For command line usage run `python tifffile.py --help`

:Author:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2016.06.21

Requirements
------------
* `CPython 2.7 or 3.5 <http://www.python.org>`_ (64 bit recommended)
* `Numpy 1.10 <http://www.numpy.org>`_
* `Matplotlib 1.5 <http://www.matplotlib.org>`_ (optional for plotting)
* `Tifffile.c 2016.04.13 <http://www.lfd.uci.edu/~gohlke/>`_
  (recommended for faster decoding of PackBits and LZW encoded strings)

Revisions
---------
2016.06.21
    Do not always memmap contiguous data in page series.
2016.05.13
    Add option to specify resolution unit.
    Write grayscale images with extra samples when planarconfig is specified.
    Do not write RGB color images with 2 samples.
    Reorder TiffWriter.save keyword arguments (backwards incompatible).
2016.04.18
    Pass 1932 tests.
    TiffWriter, imread, and imsave accept open binary file streams.
2016.04.13
    Correctly handle reversed fill order in 2 and 4 bps images (bug fix).
    Implement reverse_bitorder in C.
2016.03.18
    Fixed saving additional ImageJ metadata.
2016.02.22
    Pass 1920 tests.
    Write 8 bytes double tag values using offset if necessary (bug fix).
    Add option to disable writing second image description tag.
    Detect tags with incorrect counts.
    Disable color mapping for LSM.
2015.11.13
    Read LSM 6 mosaics.
    Add option to specify directory of memory-mapped files.
    Add command line options to specify vmin and vmax values for colormapping.
2015.10.06
    New helper function to apply colormaps.
    Renamed is_palette attributes to is_indexed (backwards incompatible).
    Color-mapped samples are now contiguous (backwards incompatible).
    Do not color-map ImageJ hyperstacks (backwards incompatible).
    Towards supporting Leica SCN.
2015.09.25
    Read images with reversed bit order (fill_order is lsb2msb).
2015.09.21
    Read RGB OME-TIFF.
    Warn about malformed OME-XML.
2015.09.16
    Detect some corrupted ImageJ metadata.
    Better axes labels for 'shaped' files.
    Do not create TiffTags for default values.
    Chroma subsampling is not supported.
    Memory-map data in TiffPageSeries if possible (optional).
2015.08.17
    Pass 1906 tests.
    Write ImageJ hyperstacks (optional).
    Read and write LZMA compressed data.
    Specify datetime when saving (optional).
    Save tiled and color-mapped images (optional).
    Ignore void byte_counts and offsets if possible.
    Ignore bogus image_depth tag created by ISS Vista software.
    Decode floating point horizontal differencing (not tiled).
    Save image data contiguously if possible.
    Only read first IFD from ImageJ files if possible.
    Read ImageJ 'raw' format (files larger than 4 GB).
    TiffPageSeries class for pages with compatible shape and data type.
    Try to read incomplete tiles.
    Open file dialog if no filename is passed on command line.
    Ignore errors when decoding OME-XML.
    Rename decoder functions (backwards incompatible)
2014.08.24
    TiffWriter class for incremental writing images.
    Simplified examples.
2014.08.19
    Add memmap function to FileHandle.
    Add function to determine if image data in TiffPage is memory-mappable.
    Do not close files if multifile_close parameter is False.
2014.08.10
    Pass 1730 tests.
    Return all extrasamples by default (backwards incompatible).
    Read data from series of pages into memory-mapped array (optional).
    Squeeze OME dimensions (backwards incompatible).
    Workaround missing EOI code in strips.
    Support image and tile depth tags (SGI extension).
    Better handling of STK/UIC tags (backwards incompatible).
    Disable color mapping for STK.
    Julian to datetime converter.
    TIFF ASCII type may be NULL separated.
    Unwrap strip offsets for LSM files greater than 4 GB.
    Correct strip byte counts in compressed LSM files.
    Skip missing files in OME series.
    Read embedded TIFF files.
2014.02.05
    Save rational numbers as type 5 (bug fix).
2013.12.20
    Keep other files in OME multi-file series closed.
    FileHandle class to abstract binary file handle.
    Disable color mapping for bad OME-TIFF produced by bio-formats.
    Read bad OME-XML produced by ImageJ when cropping.
2013.11.03
    Allow zlib compress data in imsave function (optional).
    Memory-map contiguous image data (optional).
2013.10.28
    Read MicroManager metadata and little endian ImageJ tag.
    Save extra tags in imsave function.
    Save tags in ascending order by code (bug fix).
2012.10.18
    Accept file like objects (read from OIB files).
2012.08.21
    Rename TIFFfile to TiffFile and TIFFpage to TiffPage.
    TiffSequence class for reading sequence of TIFF files.
    Read UltraQuant tags.
    Allow float numbers as resolution in imsave function.
2012.08.03
    Read MD GEL tags and NIH Image header.
2012.07.25
    Read ImageJ tags.
    ...

Notes
-----
The API is not stable yet and might change between revisions.

Tested on little-endian platforms only.

Other Python packages and modules for reading bio-scientific TIFF files:

*  `Imread <https://github.com/luispedro/imread>`_
*  `PyLibTiff <https://github.com/pearu/pylibtiff>`_
*  `SimpleITK <http://www.simpleitk.org>`_
*  `PyLSM <https://launchpad.net/pylsm>`_
*  `PyMca.TiffIO.py <https://github.com/vasole/pymca>`_ (same as fabio.TiffIO)
*  `BioImageXD.Readers <http://www.bioimagexd.net/>`_
*  `Cellcognition.io <http://cellcognition.org/>`_
*  `CellProfiler.bioformats
   <https://github.com/CellProfiler/python-bioformats>`_

Acknowledgements
----------------
*   Egor Zindy, University of Manchester, for cz_lsm_scan_info specifics.
*   Wim Lewis for a bug fix and some read_cz_lsm functions.
*   Hadrien Mary for help on reading MicroManager files.
*   Christian Kliche for help writing tiled and color-mapped files.

References
----------
(1) TIFF 6.0 Specification and Supplements. Adobe Systems Incorporated.
    http://partners.adobe.com/public/developer/tiff/
(2) TIFF File Format FAQ. http://www.awaresystems.be/imaging/tiff/faq.html
(3) MetaMorph Stack (STK) Image File Format.
    http://support.meta.moleculardevices.com/docs/t10243.pdf
(4) Image File Format Description LSM 5/7 Release 6.0 (ZEN 2010).
    Carl Zeiss MicroImaging GmbH. BioSciences. May 10, 2011
(5) File Format Description - LSM 5xx Release 2.0.
    http://ibb.gsf.de/homepage/karsten.rodenacker/IDL/Lsmfile.doc
(6) The OME-TIFF format.
    http://www.openmicroscopy.org/site/support/file-formats/ome-tiff
(7) UltraQuant(r) Version 6.0 for Windows Start-Up Guide.
    http://www.ultralum.com/images%20ultralum/pdf/UQStart%20Up%20Guide.pdf
(8) Micro-Manager File Formats.
    http://www.micro-manager.org/wiki/Micro-Manager_File_Formats
(9) Tags for TIFF and Related Specifications. Digital Preservation.
    http://www.digitalpreservation.gov/formats/content/tiff_tags.shtml

Examples
--------
>>> data = numpy.random.rand(5, 301, 219)
>>> imsave('temp.tif', data)

>>> image = imread('temp.tif')
>>> numpy.testing.assert_array_equal(image, data)

>>> with TiffFile('temp.tif') as tif:
...     images = tif.asarray()
...     for page in tif:
...         for tag in page.tags.values():
...             t = tag.name, tag.value
...         image = page.asarray()

"""

from __future__ import division, print_function

import sys
import os
import re
import glob
import math
import zlib
import time
import json
import struct
import warnings
import tempfile
import datetime
import collections
from fractions import Fraction
from xml.etree import cElementTree as etree

import numpy

try:
    import lzma
except ImportError:
    try:
        import backports.lzma as lzma
    except ImportError:
        lzma = None

try:
    if __package__:
        from . import _tifffile
    else:
        import _tifffile
except ImportError:
    warnings.warn(
        "ImportError: No module named '_tifffile'. "
        "Loading of some compressed images will be very slow. "
        "Tifffile.c can be obtained at http://www.lfd.uci.edu/~gohlke/")


__version__ = '2016.06.21'
__docformat__ = 'restructuredtext en'
__all__ = (
    'imsave', 'imread', 'imshow', 'TiffFile', 'TiffWriter', 'TiffSequence',
    # utility functions used in oiffile and czifile
    'FileHandle', 'lazyattr', 'natural_sorted', 'decode_lzw', 'stripnull')


def imsave(file, data, **kwargs):
    """Write image data to TIFF file.

    Refer to the TiffWriter class and member functions for documentation.

    Parameters
    ----------
    file : str or binary stream
        File name or writable binary stream, such as a open file or BytesIO.
    data : array_like
        Input image. The last dimensions are assumed to be image depth,
        height, width, and samples.
    kwargs : dict
        Parameters 'byteorder', 'bigtiff', 'software', and 'imagej', are passed
        to the TiffWriter class.
        Parameters 'photometric', 'planarconfig', 'resolution', 'compress',
        'colormap', 'tile', 'description', 'datetime', 'metadata', 'contiguous'
        and 'extratags' are passed to the TiffWriter.save function.

    Examples
    --------
    >>> data = numpy.random.rand(2, 5, 3, 301, 219)
    >>> imsave('temp.tif', data, compress=6, metadata={'axes': 'TZCYX'})

    """
    tifargs = parse_kwargs(kwargs, 'bigtiff', 'byteorder', 'software',
                           'imagej')

    if 'bigtiff' not in tifargs and 'imagej' not in tifargs and (
            data.size*data.dtype.itemsize > 2000*2**20):
        tifargs['bigtiff'] = True

    with TiffWriter(file, **tifargs) as tif:
        tif.save(data, **kwargs)


class TiffWriter(object):
    """Write image data to TIFF file.

    TiffWriter instances must be closed using the 'close' method, which is
    automatically called when using the 'with' context manager.

    Examples
    --------
    >>> data = numpy.random.rand(2, 5, 3, 301, 219)
    >>> with TiffWriter('temp.tif', bigtiff=True) as tif:
    ...     for i in range(data.shape[0]):
    ...         tif.save(data[i], compress=6)

    """
    TYPES = {'B': 1, 's': 2, 'H': 3, 'I': 4, '2I': 5, 'b': 6,
             'h': 8, 'i': 9, 'f': 11, 'd': 12, 'Q': 16, 'q': 17}
    TAGS = {
        'new_subfile_type': 254, 'subfile_type': 255,
        'image_width': 256, 'image_length': 257, 'bits_per_sample': 258,
        'compression': 259, 'photometric': 262, 'document_name': 269,
        'image_description': 270, 'strip_offsets': 273, 'orientation': 274,
        'samples_per_pixel': 277, 'rows_per_strip': 278,
        'strip_byte_counts': 279, 'x_resolution': 282, 'y_resolution': 283,
        'planar_configuration': 284, 'page_name': 285, 'resolution_unit': 296,
        'software': 305, 'datetime': 306, 'predictor': 317, 'color_map': 320,
        'tile_width': 322, 'tile_length': 323, 'tile_offsets': 324,
        'tile_byte_counts': 325, 'extra_samples': 338, 'sample_format': 339,
        'smin_sample_value': 340, 'smax_sample_value': 341,
        'image_depth': 32997, 'tile_depth': 32998}

    def __init__(self, file, bigtiff=False, byteorder=None,
                 software='tifffile.py', imagej=False):
        """Open a TIFF file for writing.

        Use bigtiff=True when creating files larger than 2 GB.

        Parameters
        ----------
        file : str, binary stream, or FileHandle
            File name or writable binary stream, such as a open file
            or BytesIO.
        bigtiff : bool
            If True, the BigTIFF format is used.
        byteorder : {'<', '>'}
            The endianness of the data in the file.
            By default this is the system's native byte order.
        software : str
            Name of the software used to create the file.
            Saved with the first page in the file only.
        imagej : bool
            If True, write an ImageJ hyperstack compatible file.
            This format can handle data types uint8, uint16, or float32 and
            data shapes up to 6 dimensions in TZCYXS order.
            RGB images (S=3 or S=4) must be uint8.
            ImageJ's default byte order is big endian but this implementation
            uses the system's native byte order by default.
            ImageJ does not support BigTIFF format or LZMA compression.
            The ImageJ file format is undocumented.

        """
        if byteorder not in (None, '<', '>'):
            raise ValueError("invalid byteorder %s" % byteorder)
        if byteorder is None:
            byteorder = '<' if sys.byteorder == 'little' else '>'
        if imagej and bigtiff:
            warnings.warn("writing incompatible bigtiff ImageJ")

        self._byteorder = byteorder
        self._software = software
        self._imagej = bool(imagej)
        self._metadata = None
        self._colormap = None

        self._description_offset = 0
        self._description_len_offset = 0
        self._description_len = 0

        self._tags = None
        self._shape = None  # normalized shape of data in consecutive pages
        self._data_shape = None  # shape of data in consecutive pages
        self._data_dtype = None  # data type
        self._data_offset = None  # offset to data
        self._data_byte_counts = None  # byte counts per plane
        self._tag_offsets = None  # strip or tile offset tag code

        self._fh = FileHandle(file, mode='wb', size=0)
        self._fh.write({'<': b'II', '>': b'MM'}[byteorder])

        if bigtiff:
            self._bigtiff = True
            self._offset_size = 8
            self._tag_size = 20
            self._numtag_format = 'Q'
            self._offset_format = 'Q'
            self._value_format = '8s'
            self._fh.write(struct.pack(byteorder+'HHH', 43, 8, 0))
        else:
            self._bigtiff = False
            self._offset_size = 4
            self._tag_size = 12
            self._numtag_format = 'H'
            self._offset_format = 'I'
            self._value_format = '4s'
            self._fh.write(struct.pack(byteorder+'H', 42))

        # first IFD
        self._ifd_offset = self._fh.tell()
        self._fh.write(struct.pack(byteorder+self._offset_format, 0))

    def save(self, data, photometric=None, planarconfig=None, tile=None,
             contiguous=True, compress=0, colormap=None,
             description=None, datetime=None, resolution=None,
             metadata={}, extratags=()):
        """Write image data and tags to TIFF file.

        Image data are written in one stripe per plane by default.
        Dimensions larger than 2 to 4 (depending on photometric mode, planar
        configuration, and SGI mode) are flattened and saved as separate pages.
        The 'sample_format' and 'bits_per_sample' tags are derived from
        the data type.

        Parameters
        ----------
        data : numpy.ndarray
            Input image. The last dimensions are assumed to be image depth,
            height (length), width, and samples.
            If a colormap is provided, the dtype must be uint8 or uint16 and
            the data values are indices into the last dimension of the
            colormap.
        photometric : {'minisblack', 'miniswhite', 'rgb', 'palette'}
            The color space of the image data.
            By default this setting is inferred from the data shape and the
            value of colormap.
        planarconfig : {'contig', 'planar'}
            Specifies if samples are stored contiguous or in separate planes.
            By default this setting is inferred from the data shape.
            If this parameter is set, extra samples are used to store grayscale
            images.
            'contig': last dimension contains samples.
            'planar': third last dimension contains samples.
        tile : tuple of int
            The shape (depth, length, width) of image tiles to write.
            If None (default), image data are written in one stripe per plane.
            The tile length and width must be a multiple of 16.
            If the tile depth is provided, the SGI image_depth and tile_depth
            tags are used to save volume data. Few software can read the
            SGI format, e.g. MeVisLab.
        contiguous : bool
            If True (default) and the data and parameters are compatible with
            previous ones, if any, the data are stored contiguously after
            the previous one. Parameters 'photometric' and 'planarconfig' are
            ignored.
        compress : int or 'lzma'
            Values from 0 to 9 controlling the level of zlib compression.
            If 0, data are written uncompressed (default).
            Compression cannot be used to write contiguous files.
            If 'lzma', LZMA compression is used, which is not available on
            all platforms.
        colormap : numpy.ndarray
            RGB color values for the corresponding data value.
            Must be of shape (3, 2**(data.itemsize*8)) and dtype uint16.
        description : str
            The subject of the image. Saved with the first page only.
            Cannot be used with the ImageJ format.
        datetime : datetime
            Date and time of image creation. Saved with the first page only.
            If None (default), the current date and time is used.
        resolution : (float, float[, str]) or ((int, int), (int, int)[, str])
            X and Y resolutions in pixels per resolution unit as float or
            rational numbers.
            A third, optional parameter specifies the resolution unit,
            which must be None (default for ImageJ), 'inch' (default), or 'cm'.
        metadata : dict
            Additional meta data to be saved along with shape information
            in JSON or ImageJ formats in an image_description tag.
            If None, do not write a second image_description tag.
        extratags : sequence of tuples
            Additional tags as [(code, dtype, count, value, writeonce)].

            code : int
                The TIFF tag Id.
            dtype : str
                Data type of items in 'value' in Python struct format.
                One of B, s, H, I, 2I, b, h, i, f, d, Q, or q.
            count : int
                Number of data values. Not used for string values.
            value : sequence
                'Count' values compatible with 'dtype'.
            writeonce : bool
                If True, the tag is written to the first page only.

        """
        # TODO: refactor this function
        fh = self._fh
        byteorder = self._byteorder
        numtag_format = self._numtag_format
        value_format = self._value_format
        offset_format = self._offset_format
        offset_size = self._offset_size
        tag_size = self._tag_size

        data = numpy.asarray(data, dtype=byteorder+data.dtype.char, order='C')
        if data.size == 0:
            raise ValueError("can not save empty array")

        # just append contiguous data if possible
        if self._data_shape:
            if (not contiguous or
                    self._data_shape[1:] != data.shape or
                    self._data_dtype != data.dtype or
                    (compress and self._tags) or
                    tile or
                    not numpy.array_equal(colormap, self._colormap)):
                # incompatible shape, dtype, compression mode, or colormap
                self._write_remaining_pages()
                self._write_image_description()
                self._description_offset = 0
                self._description_len_offset = 0
                self._data_shape = None
                self._colormap = None
                if self._imagej:
                    raise ValueError(
                        "ImageJ does not support non-contiguous data")
            else:
                # consecutive mode
                self._data_shape = (self._data_shape[0] + 1,) + data.shape
                if not compress:
                    # write contiguous data, write ifds/tags later
                    fh.write_array(data)
                    return

        if photometric not in (None, 'minisblack', 'miniswhite',
                               'rgb', 'palette'):
            raise ValueError("invalid photometric %s" % photometric)
        if planarconfig not in (None, 'contig', 'planar'):
            raise ValueError("invalid planarconfig %s" % planarconfig)

        # prepare compression
        if not compress:
            compress = False
            compress_tag = 1
        elif compress == 'lzma':
            compress = lzma.compress
            compress_tag = 34925
            if self._imagej:
                raise ValueError("ImageJ can not handle LZMA compression")
        elif not 0 <= compress <= 9:
            raise ValueError("invalid compression level %s" % compress)
        elif compress:
            def compress(data, level=compress):
                return zlib.compress(data, level)
            compress_tag = 32946

        # prepare ImageJ format
        if self._imagej:
            if description:
                warnings.warn("not writing description to ImageJ file")
                description = None
            volume = False
            if data.dtype.char not in 'BHhf':
                raise ValueError("ImageJ does not support data type '%s'"
                                 % data.dtype.char)
            ijrgb = photometric == 'rgb' if photometric else None
            if data.dtype.char not in 'B':
                ijrgb = False
            ijshape = imagej_shape(data.shape, ijrgb)
            if ijshape[-1] in (3, 4):
                photometric = 'rgb'
                if data.dtype.char not in 'B':
                    raise ValueError("ImageJ does not support data type '%s' "
                                     "for RGB" % data.dtype.char)
            elif photometric is None:
                photometric = 'minisblack'
                planarconfig = None
            if planarconfig == 'planar':
                raise ValueError("ImageJ does not support planar images")
            else:
                planarconfig = 'contig' if ijrgb else None

        # verify colormap and indices
        if colormap is not None:
            if data.dtype.char not in 'BH':
                raise ValueError("invalid data dtype for palette mode")
            colormap = numpy.asarray(colormap, dtype=byteorder+'H')
            if colormap.shape != (3, 2**(data.itemsize * 8)):
                raise ValueError("invalid color map shape")
            self._colormap = colormap

        # verify tile shape
        if tile:
            tile = tuple(int(i) for i in tile[:3])
            volume = len(tile) == 3
            if (len(tile) < 2 or tile[-1] % 16 or tile[-2] % 16 or
                    any(i < 1 for i in tile)):
                raise ValueError("invalid tile shape")
        else:
            tile = ()
            volume = False

        # normalize data shape to 5D or 6D, depending on volume:
        #   (pages, planar_samples, [depth,] height, width, contig_samples)
        data_shape = data.shape

        if photometric == 'rgb':
            data = reshape_nd(data, 3)
        else:
            data = reshape_nd(data, 2)

        shape = data.shape

        samplesperpixel = 1
        extrasamples = 0
        if volume and data.ndim < 3:
            volume = False
        if colormap is not None:
            photometric = 'palette'
            planarconfig = None
        if photometric is None:
            photometric = 'minisblack'
            if planarconfig == 'contig':
                if data.ndim > 2 and shape[-1] in (3, 4):
                    photometric = 'rgb'
            elif planarconfig == 'planar':
                if volume and data.ndim > 3 and shape[-4] in (3, 4):
                    photometric = 'rgb'
                elif data.ndim > 2 and shape[-3] in (3, 4):
                    photometric = 'rgb'
            elif data.ndim > 2 and shape[-1] in (3, 4):
                photometric = 'rgb'
            elif self._imagej:
                photometric = 'minisblack'
            elif volume and data.ndim > 3 and shape[-4] in (3, 4):
                photometric = 'rgb'
            elif data.ndim > 2 and shape[-3] in (3, 4):
                photometric = 'rgb'
        if planarconfig and len(shape) <= (3 if volume else 2):
            planarconfig = None
            photometric = 'minisblack'
        if photometric == 'rgb':
            if len(shape) < 3:
                raise ValueError("not a RGB(A) image")
            if len(shape) < 4:
                volume = False
            if planarconfig is None:
                if shape[-1] in (3, 4):
                    planarconfig = 'contig'
                elif shape[-4 if volume else -3] in (3, 4):
                    planarconfig = 'planar'
                elif shape[-1] > shape[-4 if volume else -3]:
                    planarconfig = 'planar'
                else:
                    planarconfig = 'contig'
            if planarconfig == 'contig':
                data = data.reshape((-1, 1) + shape[(-4 if volume else -3):])
                samplesperpixel = data.shape[-1]
            else:
                data = data.reshape(
                    (-1,) + shape[(-4 if volume else -3):] + (1,))
                samplesperpixel = data.shape[1]
            if samplesperpixel > 3:
                extrasamples = samplesperpixel - 3
        elif planarconfig and len(shape) > (3 if volume else 2):
            if planarconfig == 'contig':
                data = data.reshape((-1, 1) + shape[(-4 if volume else -3):])
                samplesperpixel = data.shape[-1]
            else:
                data = data.reshape(
                    (-1,) + shape[(-4 if volume else -3):] + (1,))
                samplesperpixel = data.shape[1]
            extrasamples = samplesperpixel - 1
        else:
            planarconfig = None
            # remove trailing 1s
            while len(shape) > 2 and shape[-1] == 1:
                shape = shape[:-1]
            if len(shape) < 3:
                volume = False
            data = data.reshape(
                (-1, 1) + shape[(-3 if volume else -2):] + (1,))

        # normalize shape to 6D
        assert len(data.shape) in (5, 6)
        if len(data.shape) == 5:
            data = data.reshape(data.shape[:2] + (1,) + data.shape[2:])
        shape = data.shape

        if tile and not volume:
            tile = (1, tile[-2], tile[-1])

        if photometric == 'palette':
            if (samplesperpixel != 1 or extrasamples or
                    shape[1] != 1 or shape[-1] != 1):
                raise ValueError("invalid data shape for palette mode")

        if photometric == 'rgb' and samplesperpixel == 2:
            raise ValueError("not a RGB image (samplesperpixel=2)")

        bytestr = bytes if sys.version[0] == '2' else (
            lambda x: bytes(x, 'utf-8') if isinstance(x, str) else x)
        tags = []  # list of (code, ifdentry, ifdvalue, writeonce)

        strip_or_tile = 'tile' if tile else 'strip'
        tag_byte_counts = TiffWriter.TAGS[strip_or_tile + '_byte_counts']
        tag_offsets = TiffWriter.TAGS[strip_or_tile + '_offsets']
        self._tag_offsets = tag_offsets

        def pack(fmt, *val):
            return struct.pack(byteorder+fmt, *val)

        def addtag(code, dtype, count, value, writeonce=False):
            # Compute ifdentry & ifdvalue bytes from code, dtype, count, value
            # Append (code, ifdentry, ifdvalue, writeonce) to tags list
            code = int(TiffWriter.TAGS.get(code, code))
            try:
                tifftype = TiffWriter.TYPES[dtype]
            except KeyError:
                raise ValueError("unknown dtype %s" % dtype)
            rawcount = count
            if dtype == 's':
                value = bytestr(value) + b'\0'
                count = rawcount = len(value)
                rawcount = value.find(b'\0\0')
                if rawcount < 0:
                    rawcount = count
                else:
                    rawcount += 1  # length of string without buffer
                value = (value,)
            if len(dtype) > 1:
                count *= int(dtype[:-1])
                dtype = dtype[-1]
            ifdentry = [pack('HH', code, tifftype),
                        pack(offset_format, rawcount)]
            ifdvalue = None
            if struct.calcsize(dtype) * count <= offset_size:
                # value(s) can be written directly
                if count == 1:
                    if isinstance(value, (tuple, list, numpy.ndarray)):
                        value = value[0]
                    ifdentry.append(pack(value_format, pack(dtype, value)))
                else:
                    ifdentry.append(pack(value_format,
                                         pack(str(count)+dtype, *value)))
            else:
                # use offset to value(s)
                ifdentry.append(pack(offset_format, 0))
                if isinstance(value, numpy.ndarray):
                    assert value.size == count
                    assert value.dtype.char == dtype
                    ifdvalue = value.tobytes()
                elif isinstance(value, (tuple, list)):
                    ifdvalue = pack(str(count)+dtype, *value)
                else:
                    ifdvalue = pack(dtype, value)
            tags.append((code, b''.join(ifdentry), ifdvalue, writeonce))

        def rational(arg, max_denominator=1000000):
            # return nominator and denominator from float or two integers
            try:
                f = Fraction.from_float(arg)
            except TypeError:
                f = Fraction(arg[0], arg[1])
            f = f.limit_denominator(max_denominator)
            return f.numerator, f.denominator

        if description:
            # user provided description
            addtag('image_description', 's', 0, description, writeonce=True)

        # write shape and metadata to image_description
        self._metadata = {} if not metadata else metadata
        if self._imagej:
            description = imagej_description(
                data_shape, shape[-1] in (3, 4), self._colormap is not None,
                **self._metadata)
        elif metadata or metadata == {}:
            description = image_description(
                data_shape, self._colormap is not None, **self._metadata)
        else:
            description = None
        if description:
            # add 32 bytes buffer
            # the image description might be updated later with the final shape
            description += b'\0'*32
            self._description_len = len(description)
            addtag('image_description', 's', 0, description, writeonce=True)

        if self._software:
            addtag('software', 's', 0, self._software, writeonce=True)
            self._software = None  # only save to first page in file
        if datetime is None:
            datetime = self._now()
        addtag('datetime', 's', 0, datetime.strftime("%Y:%m:%d %H:%M:%S"),
               writeonce=True)
        addtag('compression', 'H', 1, compress_tag)
        addtag('image_width', 'I', 1, shape[-2])
        addtag('image_length', 'I', 1, shape[-3])
        if tile:
            addtag('tile_width', 'I', 1, tile[-1])
            addtag('tile_length', 'I', 1, tile[-2])
            if tile[0] > 1:
                addtag('image_depth', 'I', 1, shape[-4])
                addtag('tile_depth', 'I', 1, tile[0])
        addtag('new_subfile_type', 'I', 1, 0)
        addtag('sample_format', 'H', 1,
               {'u': 1, 'i': 2, 'f': 3, 'c': 6}[data.dtype.kind])
        addtag('photometric', 'H', 1, {'miniswhite': 0, 'minisblack': 1,
                                       'rgb': 2, 'palette': 3}[photometric])
        if colormap is not None:
            addtag('color_map', 'H', colormap.size, colormap)
        addtag('samples_per_pixel', 'H', 1, samplesperpixel)
        if planarconfig and samplesperpixel > 1:
            addtag('planar_configuration', 'H', 1, 1
                   if planarconfig == 'contig' else 2)
            addtag('bits_per_sample', 'H', samplesperpixel,
                   (data.dtype.itemsize * 8,) * samplesperpixel)
        else:
            addtag('bits_per_sample', 'H', 1, data.dtype.itemsize * 8)
        if extrasamples:
            if photometric == 'rgb' and extrasamples == 1:
                addtag('extra_samples', 'H', 1, 1)  # associated alpha channel
            else:
                addtag('extra_samples', 'H', extrasamples, (0,) * extrasamples)
        if resolution:
            addtag('x_resolution', '2I', 1, rational(resolution[0]))
            addtag('y_resolution', '2I', 1, rational(resolution[1]))
            if len(resolution) > 2:
                resolution_unit = {None: 1, 'inch': 2, 'cm': 3}[resolution[2]]
            elif self._imagej:
                resolution_unit = 1
            else:
                resolution_unit = 2
            addtag('resolution_unit', 'H', 1, resolution_unit)
        if not tile:
            addtag('rows_per_strip', 'I', 1, shape[-3])  # * shape[-4]

        if tile:
            # use one chunk per tile per plane
            tiles = ((shape[2] + tile[0] - 1) // tile[0],
                     (shape[3] + tile[1] - 1) // tile[1],
                     (shape[4] + tile[2] - 1) // tile[2])
            numtiles = product(tiles) * shape[1]
            strip_byte_counts = [
                product(tile) * shape[-1] * data.dtype.itemsize] * numtiles
            addtag(tag_byte_counts, offset_format, numtiles, strip_byte_counts)
            addtag(tag_offsets, offset_format, numtiles, [0] * numtiles)
            # allocate tile buffer
            chunk = numpy.empty(tile + (shape[-1],), dtype=data.dtype)
        else:
            # use one strip per plane
            strip_byte_counts = [
                data[0, 0].size * data.dtype.itemsize] * shape[1]
            addtag(tag_byte_counts, offset_format, shape[1], strip_byte_counts)
            addtag(tag_offsets, offset_format, shape[1], [0] * shape[1])

        # add extra tags from user
        for t in extratags:
            addtag(*t)

        # TODO: check TIFFReadDirectoryCheckOrder warning in files containing
        #   multiple tags of same code
        # the entries in an IFD must be sorted in ascending order by tag code
        tags = sorted(tags, key=lambda x: x[0])

        if not (self._bigtiff or self._imagej) and (
                fh.tell() + data.size*data.dtype.itemsize > 2**31-1):
            raise ValueError("data too large for standard TIFF file")

        # if not compressed or tiled, write the first ifd and then all data
        # contiguously; else, write all ifds and data interleaved
        for pageindex in range(shape[0] if (compress or tile) else 1):
            # update pointer at ifd_offset
            pos = fh.tell()
            fh.seek(self._ifd_offset)
            fh.write(pack(offset_format, pos))
            fh.seek(pos)

            # write ifdentries
            fh.write(pack(numtag_format, len(tags)))
            tag_offset = fh.tell()
            fh.write(b''.join(t[1] for t in tags))
            self._ifd_offset = fh.tell()
            fh.write(pack(offset_format, 0))  # offset to next IFD

            # write tag values and patch offsets in ifdentries, if necessary
            for tagindex, tag in enumerate(tags):
                if tag[2]:
                    pos = fh.tell()
                    fh.seek(tag_offset + tagindex*tag_size + offset_size + 4)
                    fh.write(pack(offset_format, pos))
                    fh.seek(pos)
                    if tag[0] == tag_offsets:
                        strip_offsets_offset = pos
                    elif tag[0] == tag_byte_counts:
                        strip_byte_counts_offset = pos
                    elif tag[0] == 270 and tag[2].endswith(b'\0\0\0\0'):
                        # image description buffer
                        self._description_offset = pos
                        self._description_len_offset = (
                            tag_offset + tagindex * tag_size + 4)
                    fh.write(tag[2])

            # write image data
            data_offset = fh.tell()
            if compress:
                strip_byte_counts = []
            if tile:
                for plane in data[pageindex]:
                    for tz in range(tiles[0]):
                        for ty in range(tiles[1]):
                            for tx in range(tiles[2]):
                                c0 = min(tile[0], shape[2] - tz*tile[0])
                                c1 = min(tile[1], shape[3] - ty*tile[1])
                                c2 = min(tile[2], shape[4] - tx*tile[2])
                                chunk[c0:, c1:, c2:] = 0
                                chunk[:c0, :c1, :c2] = plane[
                                    tz*tile[0]:tz*tile[0]+c0,
                                    ty*tile[1]:ty*tile[1]+c1,
                                    tx*tile[2]:tx*tile[2]+c2]
                                if compress:
                                    t = compress(chunk)
                                    strip_byte_counts.append(len(t))
                                    fh.write(t)
                                else:
                                    fh.write_array(chunk)
                                    fh.flush()
            elif compress:
                for plane in data[pageindex]:
                    plane = compress(plane)
                    strip_byte_counts.append(len(plane))
                    fh.write(plane)
            else:
                fh.write_array(data)

            # update strip/tile offsets and byte_counts if necessary
            pos = fh.tell()
            for tagindex, tag in enumerate(tags):
                if tag[0] == tag_offsets:  # strip/tile offsets
                    if tag[2]:
                        fh.seek(strip_offsets_offset)
                        strip_offset = data_offset
                        for size in strip_byte_counts:
                            fh.write(pack(offset_format, strip_offset))
                            strip_offset += size
                    else:
                        fh.seek(tag_offset + tagindex*tag_size +
                                offset_size + 4)
                        fh.write(pack(offset_format, data_offset))
                elif tag[0] == tag_byte_counts:  # strip/tile byte_counts
                    if compress:
                        if tag[2]:
                            fh.seek(strip_byte_counts_offset)
                            for size in strip_byte_counts:
                                fh.write(pack(offset_format, size))
                        else:
                            fh.seek(tag_offset + tagindex*tag_size +
                                    offset_size + 4)
                            fh.write(pack(offset_format, strip_byte_counts[0]))
                    break
            fh.seek(pos)
            fh.flush()

            # remove tags that should be written only once
            if pageindex == 0:
                tags = [tag for tag in tags if not tag[-1]]

        # if uncompressed, write remaining ifds/tags later
        if not (compress or tile):
            self._tags = tags

        self._shape = shape
        self._data_shape = (1,) + data_shape
        self._data_dtype = data.dtype
        self._data_offset = data_offset
        self._data_byte_counts = strip_byte_counts

    def _write_remaining_pages(self):
        """Write outstanding IFDs and tags to file."""
        if not self._tags:
            return

        fh = self._fh
        byteorder = self._byteorder
        numtag_format = self._numtag_format
        offset_format = self._offset_format
        offset_size = self._offset_size
        tag_size = self._tag_size
        data_offset = self._data_offset
        page_data_size = sum(self._data_byte_counts)
        tag_bytes = b''.join(t[1] for t in self._tags)
        numpages = self._shape[0] * self._data_shape[0] - 1

        pos = fh.tell()
        if not self._bigtiff and pos + len(tag_bytes) * numpages > 2**32 - 256:
            if self._imagej:
                warnings.warn("truncating ImageJ file")
                return
            raise ValueError("data too large for non-bigtiff file")

        def pack(fmt, *val):
            return struct.pack(byteorder+fmt, *val)

        for _ in range(numpages):
            # update pointer at ifd_offset
            pos = fh.tell()
            fh.seek(self._ifd_offset)
            fh.write(pack(offset_format, pos))
            fh.seek(pos)

            # write ifd entries
            fh.write(pack(numtag_format, len(self._tags)))
            tag_offset = fh.tell()
            fh.write(tag_bytes)
            self._ifd_offset = fh.tell()
            fh.write(pack(offset_format, 0))  # offset to next IFD

            # offset to image data
            data_offset += page_data_size

            # write tag values and patch offsets in ifdentries, if necessary
            for tagindex, tag in enumerate(self._tags):
                if tag[2]:
                    pos = fh.tell()
                    fh.seek(tag_offset + tagindex*tag_size + offset_size + 4)
                    fh.write(pack(offset_format, pos))
                    fh.seek(pos)
                    if tag[0] == self._tag_offsets:
                        strip_offsets_offset = pos
                    fh.write(tag[2])

            # update strip/tile offsets if necessary
            pos = fh.tell()
            for tagindex, tag in enumerate(self._tags):
                if tag[0] == self._tag_offsets:  # strip/tile offsets
                    if tag[2]:
                        fh.seek(strip_offsets_offset)
                        strip_offset = data_offset
                        for size in self._data_byte_counts:
                            fh.write(pack(offset_format, strip_offset))
                            strip_offset += size
                    else:
                        fh.seek(tag_offset + tagindex*tag_size +
                                offset_size + 4)
                        fh.write(pack(offset_format, data_offset))
                    break
            fh.seek(pos)

        self._tags = None
        self._data_dtype = None
        self._data_offset = None
        self._data_byte_counts = None
        # do not reset _shape or _data_shape

    def _write_image_description(self):
        """Write meta data to image_description tag."""
        if (not self._data_shape or self._data_shape[0] == 1 or
                self._description_offset <= 0):
            return

        colormapped = self._colormap is not None
        if self._imagej:
            isrgb = self._shape[-1] in (3, 4)
            description = imagej_description(
                self._data_shape, isrgb, colormapped, **self._metadata)
        else:
            description = image_description(
                self._data_shape, colormapped, **self._metadata)

        # rewrite description and its length to file
        description = description[:self._description_len-1]
        pos = self._fh.tell()
        self._fh.seek(self._description_offset)
        self._fh.write(description)
        self._fh.seek(self._description_len_offset)
        self._fh.write(struct.pack(self._byteorder+self._offset_format,
                                   len(description)+1))
        self._fh.seek(pos)

        self._description_offset = 0
        self._description_len_offset = 0
        self._description_len = 0

    def _now(self):
        """Return current date and time."""
        return datetime.datetime.now()

    def close(self, truncate=False):
        """Write remaining pages (if not truncate) and close file handle."""
        if not truncate:
            self._write_remaining_pages()
        self._write_image_description()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def imread(files, **kwargs):
    """Return image data from TIFF file(s) as numpy array.

    The first image series is returned if no arguments are provided.

    Parameters
    ----------
    files : str, binary stream, or sequence
        File name, seekable binary stream, glob pattern, or sequence of
        file names.
    key : int, slice, or sequence of page indices
        Defines which pages to return as array.
    series : int
        Defines which series of pages in file to return as array.
    multifile : bool
        If True (default), OME-TIFF data may include pages from multiple files.
    pattern : str
        Regular expression pattern that matches axes names and indices in
        file names.
    kwargs : dict
        Additional parameters passed to the TiffFile or TiffSequence asarray
        function.

    Examples
    --------
    >>> imsave('temp.tif', numpy.random.rand(3, 4, 301, 219))
    >>> im = imread('temp.tif', key=0)
    >>> im.shape
    (4, 301, 219)
    >>> ims = imread(['temp.tif', 'temp.tif'])
    >>> ims.shape
    (2, 3, 4, 301, 219)

    """
    kwargs_file = parse_kwargs(kwargs, multifile=True)
    kwargs_seq = parse_kwargs(kwargs, 'pattern')

    if isinstance(files, basestring) and any(i in files for i in '?*'):
        files = glob.glob(files)
    if not files:
        raise ValueError('no files found')
    if not hasattr(files, 'seek') and len(files) == 1:
        files = files[0]

    if isinstance(files, basestring) or hasattr(files, 'seek'):
        with TiffFile(files, **kwargs_file) as tif:
            return tif.asarray(**kwargs)
    else:
        with TiffSequence(files, **kwargs_seq) as imseq:
            return imseq.asarray(**kwargs)


class lazyattr(object):
    """Lazy object attribute whose value is computed on first access."""
    __slots__ = ('func',)

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return self
        value = self.func(instance)
        if value is NotImplemented:
            return getattr(super(owner, instance), self.func.__name__)
        setattr(instance, self.func.__name__, value)
        return value


class TiffFile(object):
    """Read image and metadata from TIFF, STK, LSM, and FluoView files.

    TiffFile instances must be closed using the 'close' method, which is
    automatically called when using the 'with' context manager.

    Attributes
    ----------
    pages : list of TiffPage
        All TIFF pages in file.
    series : list of TiffPageSeries
        TIFF pages with compatible shapes and types.
    micromanager_metadata: dict
        Extra MicroManager non-TIFF metadata in the file, if exists.

    All attributes are read-only.

    Examples
    --------
    >>> with TiffFile('temp.tif') as tif:
    ...     data = tif.asarray()
    ...     data.shape
    (5, 301, 219)

    """
    def __init__(self, arg, name=None, offset=None, size=None,
                 multifile=True, multifile_close=True, maxpages=None,
                 fastij=True):
        """Initialize instance from file.

        Parameters
        ----------
        arg : str or open file
            Name of file or open file object.
            The file objects are closed in TiffFile.close().
        name : str
            Optional name of file in case 'arg' is a file handle.
        offset : int
            Optional start position of embedded file. By default this is
            the current file position.
        size : int
            Optional size of embedded file. By default this is the number
            of bytes from the 'offset' to the end of the file.
        multifile : bool
            If True (default), series may include pages from multiple files.
            Currently applies to OME-TIFF only.
        multifile_close : bool
            If True (default), keep the handles of other files in multifile
            series closed. This is inefficient when few files refer to
            many pages. If False, the C runtime may run out of resources.
        maxpages : int
            Number of pages to read (default: no limit).
        fastij : bool
            If True (default), try to use only the metadata from the first page
            of ImageJ files. Significantly speeds up loading movies with
            thousands of pages.

        """
        self._fh = FileHandle(arg, mode='rb',
                              name=name, offset=offset, size=size)
        self.offset_size = None
        self.pages = []
        self._multifile = bool(multifile)
        self._multifile_close = bool(multifile_close)
        self._files = {self._fh.name: self}  # cache of TiffFiles
        try:
            self._fromfile(maxpages, fastij)
        except Exception:
            self._fh.close()
            raise

    @property
    def filehandle(self):
        """Return file handle."""
        return self._fh

    @property
    def filename(self):
        """Return name of file handle."""
        return self._fh.name

    def close(self):
        """Close open file handle(s)."""
        for tif in self._files.values():
            tif._fh.close()
        self._files = {}

    def _fromfile(self, maxpages=None, fastij=True):
        """Read TIFF header and all page records from file."""
        self._fh.seek(0)
        try:
            self.byteorder = {b'II': '<', b'MM': '>'}[self._fh.read(2)]
        except KeyError:
            raise ValueError("not a valid TIFF file")
        self._is_native = self.byteorder == {'big': '>',
                                             'little': '<'}[sys.byteorder]
        version = struct.unpack(self.byteorder+'H', self._fh.read(2))[0]
        if version == 43:
            # BigTiff
            self.offset_size, zero = struct.unpack(self.byteorder+'HH',
                                                   self._fh.read(4))
            if zero or self.offset_size != 8:
                raise ValueError("not a valid BigTIFF file")
        elif version == 42:
            self.offset_size = 4
        else:
            raise ValueError("not a TIFF file")
        self.pages = []
        while True:
            try:
                page = TiffPage(self)
                self.pages.append(page)
            except StopIteration:
                break
            if maxpages and len(self.pages) > maxpages:
                break
            if fastij and page.is_imagej:
                if page._patch_imagej():
                    break  # only read the first page of ImageJ files
                fastij = False

        if not self.pages:
            raise ValueError("empty TIFF file")

        if self.is_micromanager:
            # MicroManager files contain metadata not stored in TIFF tags.
            self.micromanager_metadata = read_micromanager_metadata(self._fh)

        if self.is_lsm:
            self._fix_lsm_strip_offsets()
            self._fix_lsm_strip_byte_counts()

    def _fix_lsm_strip_offsets(self):
        """Unwrap strip offsets for LSM files greater than 4 GB."""
        # each series and position require separate unwrappig (undocumented)
        for series in self.series:
            positions = 1
            for i in 0, 1:
                if series.axes[i] in 'PM':
                    positions *= series.shape[i]
            positions = len(series.pages) // positions
            for i, page in enumerate(series.pages):
                if not i % positions:
                    wrap = 0
                    previous_offset = 0
                strip_offsets = []
                for current_offset in page.strip_offsets:
                    if current_offset < previous_offset:
                        wrap += 2**32
                    strip_offsets.append(current_offset + wrap)
                    previous_offset = current_offset
                page.strip_offsets = tuple(strip_offsets)

    def _fix_lsm_strip_byte_counts(self):
        """Set strip_byte_counts to size of compressed data.

        The strip_byte_counts tag in LSM files contains the number of bytes
        for the uncompressed data.

        """
        if not self.pages:
            return
        strips = {}
        for page in self.pages:
            assert len(page.strip_offsets) == len(page.strip_byte_counts)
            for offset, bytecount in zip(page.strip_offsets,
                                         page.strip_byte_counts):
                strips[offset] = bytecount
        offsets = sorted(strips.keys())
        offsets.append(min(offsets[-1] + strips[offsets[-1]], self._fh.size))
        for i, offset in enumerate(offsets[:-1]):
            strips[offset] = min(strips[offset], offsets[i+1] - offset)
        for page in self.pages:
            if page.compression:
                page.strip_byte_counts = tuple(
                    strips[offset] for offset in page.strip_offsets)

    def asarray(self, key=None, series=None, memmap=False, tempdir=None):
        """Return image data from multiple TIFF pages as numpy array.

        By default the first image series is returned.

        Parameters
        ----------
        key : int, slice, or sequence of page indices
            Defines which pages to return as array.
        series : int or TiffPageSeries
            Defines which series of pages to return as array.
        memmap : bool
            If True, return an read-only array stored in a binary file on disk
            if possible. The TIFF file is used if possible, else a temporary
            file is created.
        tempdir : str
            The directory where the memory-mapped file will be created.

        """
        if key is None and series is None:
            series = 0
        if series is not None:
            try:
                series = self.series[series]
            except (KeyError, TypeError):
                pass
            pages = series.pages
        else:
            pages = self.pages

        if key is None:
            pass
        elif isinstance(key, int):
            pages = [pages[key]]
        elif isinstance(key, slice):
            pages = pages[key]
        elif isinstance(key, collections.Iterable):
            pages = [pages[k] for k in key]
        else:
            raise TypeError("key must be an int, slice, or sequence")

        if not len(pages):
            raise ValueError("no pages selected")

        if self.is_nih:
            if pages[0].is_indexed:
                result = stack_pages(pages, colormapped=False, squeeze=False)
                result = apply_colormap(result, pages[0].color_map)
            else:
                result = stack_pages(pages, memmap=memmap, tempdir=tempdir,
                                     colormapped=False, squeeze=False)
        elif len(pages) == 1:
            result = pages[0].asarray(memmap=memmap)
        elif self.is_ome:
            assert not self.is_indexed, "color mapping disabled for ome-tiff"
            if any(p is None for p in pages):
                # zero out missing pages
                firstpage = next(p for p in pages if p)
                nopage = numpy.zeros_like(
                    firstpage.asarray(memmap=False))
            if memmap:
                with tempfile.NamedTemporaryFile() as fh:
                    result = numpy.memmap(fh, series.dtype, shape=series.shape)
                    result = result.reshape(-1)
            else:
                result = numpy.empty(series.shape, series.dtype).reshape(-1)
            index = 0

            class KeepOpen:
                # keep Tiff files open between consecutive pages
                def __init__(self, parent, close):
                    self.master = parent
                    self.parent = parent
                    self._close = close

                def open(self, page):
                    if self._close and page and page.parent != self.parent:
                        if self.parent != self.master:
                            self.parent.filehandle.close()
                        self.parent = page.parent
                        self.parent.filehandle.open()

                def close(self):
                    if self._close and self.parent != self.master:
                        self.parent.filehandle.close()

            keep = KeepOpen(self, self._multifile_close)
            for page in pages:
                keep.open(page)
                if page:
                    a = page.asarray(memmap=False, colormapped=False,
                                     reopen=False)
                else:
                    a = nopage
                try:
                    result[index:index + a.size] = a.reshape(-1)
                except ValueError as e:
                    warnings.warn("ome-tiff: %s" % e)
                    break
                index += a.size
            keep.close()
        elif key is None and series and series.offset:
            if memmap:
                result = self.filehandle.memmap_array(
                    series.dtype, series.shape, series.offset)
            else:
                self.filehandle.seek(series.offset)
                result = self.filehandle.read_array(
                    series.dtype, product(series.shape))
        else:
            result = stack_pages(pages, memmap=memmap, tempdir=tempdir)

        if key is None:
            try:
                result.shape = series.shape
            except ValueError:
                try:
                    warnings.warn("failed to reshape %s to %s" % (
                        result.shape, series.shape))
                    # try series of expected shapes
                    result.shape = (-1,) + series.shape
                except ValueError:
                    # revert to generic shape
                    result.shape = (-1,) + pages[0].shape
        elif len(pages) == 1:
            result.shape = pages[0].shape
        else:
            result.shape = (-1,) + pages[0].shape
        return result

    @lazyattr
    def series(self):
        """Return pages with compatible properties as TiffPageSeries."""
        if not self.pages:
            return []

        series = []
        if self.is_ome:
            series = self._ome_series()
        elif self.is_fluoview:
            series = self._fluoview_series()
        elif self.is_lsm:
            series = self._lsm_series()
        elif self.is_imagej:
            series = self._imagej_series()
        elif self.is_nih:
            series = self._nih_series()

        if not series:
            # generic detection of series
            shapes = []
            pages = {}
            index = 0
            for page in self.pages:
                if not page.shape:
                    continue
                if page.is_shaped:
                    index += 1  # shape starts a new series
                shape = page.shape + (index, page.axes,
                                      page.compression in TIFF_DECOMPESSORS)
                if shape in pages:
                    pages[shape].append(page)
                else:
                    shapes.append(shape)
                    pages[shape] = [page]
            series = []
            for s in shapes:
                shape = ((len(pages[s]),) + s[:-3] if len(pages[s]) > 1
                         else s[:-3])
                axes = (('I' + s[-2]) if len(pages[s]) > 1 else s[-2])
                page0 = pages[s][0]
                if page0.is_shaped:
                    metadata = image_description_dict(page0.is_shaped)
                    reshape = metadata['shape']
                    if 'axes' in metadata:
                        reaxes = metadata['axes']
                        if len(reaxes) == len(reshape):
                            axes = reaxes
                            shape = reshape
                        else:
                            warnings.warn("axes do not match shape")
                    try:
                        axes = reshape_axes(axes, shape, reshape)
                        shape = reshape
                    except ValueError as e:
                        warnings.warn(e.message)
                series.append(
                    TiffPageSeries(pages[s], shape, page0.dtype, axes))

        # remove empty series, e.g. in MD Gel files
        series = [s for s in series if sum(s.shape) > 0]
        return series

    def _fluoview_series(self):
        """Return image series in FluoView file."""
        page0 = self.pages[0]
        dims = {
            b'X': 'X', b'Y': 'Y', b'Z': 'Z', b'T': 'T',
            b'WAVELENGTH': 'C', b'TIME': 'T', b'XY': 'R',
            b'EVENT': 'V', b'EXPOSURE': 'L'}
        mmhd = list(reversed(page0.mm_header.dimensions))
        axes = ''.join(dims.get(i[0].strip().upper(), 'Q')
                       for i in mmhd if i[1] > 1)
        shape = tuple(int(i[1]) for i in mmhd if i[1] > 1)
        return [TiffPageSeries(self.pages, shape, page0.dtype, axes)]

    def _lsm_series(self):
        """Return image series in LSM file."""
        page0 = self.pages[0]
        lsmi = page0.cz_lsm_info
        axes = CZ_SCAN_TYPES[lsmi.scan_type]
        if page0.is_rgb:
            axes = axes.replace('C', '').replace('XY', 'XYC')
        if hasattr(lsmi, 'dimension_p') and lsmi.dimension_p > 1:
            axes += 'P'
        if hasattr(lsmi, 'dimension_m') and lsmi.dimension_m > 1:
            axes += 'M'
        axes = axes[::-1]
        shape = tuple(getattr(lsmi, CZ_DIMENSIONS[i]) for i in axes)
        pages = [p for p in self.pages if not p.is_reduced]
        dtype = pages[0].dtype
        series = [TiffPageSeries(pages, shape, dtype, axes)]
        if len(pages) != len(self.pages):  # reduced RGB pages
            pages = [p for p in self.pages if p.is_reduced]
            cp = 1
            i = 0
            while cp < len(pages) and i < len(shape)-2:
                cp *= shape[i]
                i += 1
            shape = shape[:i] + pages[0].shape
            axes = axes[:i] + 'CYX'
            dtype = pages[0].dtype
            series.append(TiffPageSeries(pages, shape, dtype, axes))
        return series

    def _imagej_series(self):
        """Return image series in ImageJ file."""
        # ImageJ's dimension order is always TZCYXS
        # TODO: fix loading of color, composite or palette images
        shape = []
        axes = []
        page0 = self.pages[0]
        ij = page0.imagej_tags
        if 'frames' in ij:
            shape.append(ij['frames'])
            axes.append('T')
        if 'slices' in ij:
            shape.append(ij['slices'])
            axes.append('Z')
        if 'channels' in ij and not (self.is_rgb and not
                                     ij.get('hyperstack', False)):
            shape.append(ij['channels'])
            axes.append('C')
        remain = ij.get('images', len(self.pages)) // (product(shape)
                                                       if shape else 1)
        if remain > 1:
            shape.append(remain)
            axes.append('I')
        if page0.axes[0] == 'I':
            # contiguous multiple images
            shape.extend(page0.shape[1:])
            axes.extend(page0.axes[1:])
        elif page0.axes[:2] == 'SI':
            # color-mapped contiguous multiple images
            shape = page0.shape[0:1] + tuple(shape) + page0.shape[2:]
            axes = list(page0.axes[0]) + axes + list(page0.axes[2:])
        else:
            shape.extend(page0.shape)
            axes.extend(page0.axes)
        return [TiffPageSeries(self.pages, shape, page0.dtype, axes)]

    def _nih_series(self):
        """Return image series in NIH file."""
        page0 = self.pages[0]
        if len(self.pages) == 1:
            shape = page0.shape
            axes = page0.axes
        else:
            shape = (len(self.pages),) + page0.shape
            axes = 'I' + page0.axes
        return [TiffPageSeries(self.pages, shape, page0.dtype, axes)]

    def _ome_series(self):
        """Return image series in OME-TIFF file(s)."""
        omexml = self.pages[0].tags['image_description'].value
        try:
            root = etree.fromstring(omexml)
        except etree.ParseError as e:
            # TODO: test this
            warnings.warn("ome-xml: %s" % e)
            omexml = omexml.decode('utf-8', 'ignore').encode('utf-8')
            root = etree.fromstring(omexml)
        uuid = root.attrib.get('UUID', None)
        self._files = {uuid: self}
        dirname = self._fh.dirname
        modulo = {}
        series = []
        for element in root:
            if element.tag.endswith('BinaryOnly'):
                warnings.warn("ome-xml: not an ome-tiff master file")
                break
            if element.tag.endswith('StructuredAnnotations'):
                for annot in element:
                    if not annot.attrib.get('Namespace',
                                            '').endswith('modulo'):
                        continue
                    for value in annot:
                        for modul in value:
                            for along in modul:
                                if not along.tag[:-1].endswith('Along'):
                                    continue
                                axis = along.tag[-1]
                                newaxis = along.attrib.get('Type', 'other')
                                newaxis = AXES_LABELS[newaxis]
                                if 'Start' in along.attrib:
                                    labels = range(
                                        int(along.attrib['Start']),
                                        int(along.attrib['End']) + 1,
                                        int(along.attrib.get('Step', 1)))
                                else:
                                    labels = [label.text for label in along
                                              if label.tag.endswith('Label')]
                                modulo[axis] = (newaxis, labels)
            if not element.tag.endswith('Image'):
                continue
            for pixels in element:
                if not pixels.tag.endswith('Pixels'):
                    continue
                atr = pixels.attrib
                dtype = atr.get('Type', None)
                axes = ''.join(reversed(atr['DimensionOrder']))
                shape = list(int(atr['Size'+ax]) for ax in axes)
                size = product(shape[:-2])
                ifds = [None] * (size // self.pages[0].samples_per_pixel)
                for data in pixels:
                    if not data.tag.endswith('TiffData'):
                        continue
                    atr = data.attrib
                    ifd = int(atr.get('IFD', 0))
                    num = int(atr.get('NumPlanes', 1 if 'IFD' in atr else 0))
                    num = int(atr.get('PlaneCount', num))
                    idx = [int(atr.get('First'+ax, 0)) for ax in axes[:-2]]
                    try:
                        idx = numpy.ravel_multi_index(idx, shape[:-2])
                    except ValueError:
                        # ImageJ produces invalid ome-xml when cropping
                        warnings.warn("ome-xml: invalid TiffData index")
                        continue
                    for uuid in data:
                        if not uuid.tag.endswith('UUID'):
                            continue
                        if uuid.text not in self._files:
                            if not self._multifile:
                                # abort reading multifile OME series
                                # and fall back to generic series
                                return []
                            fname = uuid.attrib['FileName']
                            try:
                                tif = TiffFile(os.path.join(dirname, fname))
                            except (IOError, ValueError):
                                tif.close()
                                warnings.warn(
                                    "ome-xml: failed to read '%s'" % fname)
                                break
                            self._files[uuid.text] = tif
                            if self._multifile_close:
                                tif.close()
                        pages = self._files[uuid.text].pages
                        try:
                            for i in range(num if num else len(pages)):
                                ifds[idx + i] = pages[ifd + i]
                        except IndexError:
                            warnings.warn("ome-xml: index out of range")
                        # only process first uuid
                        break
                    else:
                        pages = self.pages
                        try:
                            for i in range(num if num else len(pages)):
                                ifds[idx + i] = pages[ifd + i]
                        except IndexError:
                            warnings.warn("ome-xml: index out of range")
                if all(i is None for i in ifds):
                    # skip images without data
                    continue
                dtype = next(i for i in ifds if i).dtype
                series.append(TiffPageSeries(ifds, shape, dtype, axes, self))
        for serie in series:
            shape = list(serie.shape)
            for axis, (newaxis, labels) in modulo.items():
                i = serie.axes.index(axis)
                size = len(labels)
                if shape[i] == size:
                    serie.axes = serie.axes.replace(axis, newaxis, 1)
                else:
                    shape[i] //= size
                    shape.insert(i+1, size)
                    serie.axes = serie.axes.replace(axis, axis+newaxis, 1)
            serie.shape = tuple(shape)
        # squeeze dimensions
        for serie in series:
            serie.shape, serie.axes = squeeze_axes(serie.shape, serie.axes)
        return series

    def __len__(self):
        """Return number of image pages in file."""
        return len(self.pages)

    def __getitem__(self, key):
        """Return specified page."""
        return self.pages[key]

    def __iter__(self):
        """Return iterator over pages."""
        return iter(self.pages)

    def __str__(self):
        """Return string containing information about file."""
        result = [
            self._fh.name.capitalize(),
            format_size(self._fh.size),
            {'<': 'little endian', '>': 'big endian'}[self.byteorder]]
        if self.is_bigtiff:
            result.append("bigtiff")
        if len(self.pages) > 1:
            result.append("%i pages" % len(self.pages))
        if len(self.series) > 1:
            result.append("%i series" % len(self.series))
        if len(self._files) > 1:
            result.append("%i files" % (len(self._files)))
        return ", ".join(result)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @lazyattr
    def fstat(self):
        try:
            return os.fstat(self._fh.fileno())
        except Exception:  # io.UnsupportedOperation
            return None

    @lazyattr
    def is_bigtiff(self):
        """File has BigTIFF format."""
        return self.offset_size != 4

    @lazyattr
    def is_rgb(self):
        """File contains only RGB images."""
        return all(p.is_rgb for p in self.pages)

    @lazyattr
    def is_indexed(self):
        """File contains only indexed images."""
        return all(p.is_indexed for p in self.pages)

    @lazyattr
    def is_mdgel(self):
        """File has MD Gel format."""
        return any(p.is_mdgel for p in self.pages)

    @lazyattr
    def is_mediacy(self):
        """File was created by Media Cybernetics software."""
        return any(p.is_mediacy for p in self.pages)

    @lazyattr
    def is_stk(self):
        """File has MetaMorph STK format."""
        return all(p.is_stk for p in self.pages)

    @lazyattr
    def is_lsm(self):
        """File was created by Carl Zeiss software."""
        return len(self.pages) and self.pages[0].is_lsm

    @lazyattr
    def is_vista(self):
        """File was created by ISS Vista."""
        return len(self.pages) and self.pages[0].is_vista

    @lazyattr
    def is_imagej(self):
        """File has ImageJ format."""
        return len(self.pages) and self.pages[0].is_imagej

    @lazyattr
    def is_micromanager(self):
        """File was created by MicroManager."""
        return len(self.pages) and self.pages[0].is_micromanager

    @lazyattr
    def is_nih(self):
        """File has NIH Image format."""
        return len(self.pages) and self.pages[0].is_nih

    @lazyattr
    def is_fluoview(self):
        """File was created by Olympus FluoView."""
        return len(self.pages) and self.pages[0].is_fluoview

    @lazyattr
    def is_ome(self):
        """File has OME-TIFF format."""
        return len(self.pages) and self.pages[0].is_ome

    @lazyattr
    def is_scn(self):
        """File has Leica SCN format."""
        return len(self.pages) and self.pages[0].is_scn


class TiffPage(object):
    """A TIFF image file directory (IFD).

    Attributes
    ----------
    index : int
        Index of page in file.
    dtype : str {TIFF_SAMPLE_DTYPES}
        Data type of image, color-mapped if applicable.
    shape : tuple
        Dimensions of the image array in TIFF page,
        color-mapped and with extra samples if applicable.
    axes : str
        Axes label codes:
        'X' width, 'Y' height, 'S' sample, 'I' image series|page|plane,
        'Z' depth, 'C' color|em-wavelength|channel, 'E' ex-wavelength|lambda,
        'T' time, 'R' region|tile, 'A' angle, 'P' phase, 'H' lifetime,
        'L' exposure, 'V' event, 'Q' unknown, '_' missing
    tags : TiffTags
        Dictionary of tags in page.
        Tag values are also directly accessible as attributes.
    color_map : numpy.ndarray
        Color look up table, if exists.
    cz_lsm_scan_info: Record(dict)
        LSM scan info attributes, if exists.
    imagej_tags: Record(dict)
        Consolidated ImageJ description and metadata tags, if exists.
    uic_tags: Record(dict)
        Consolidated MetaMorph STK/UIC tags, if exists.

    All attributes are read-only.

    Notes
    -----
    The internal, normalized '_shape' attribute is 6 dimensional:

    0. number planes/images  (stk, ij).
    1. planar samples_per_pixel.
    2. image_depth Z  (sgi).
    3. image_length Y.
    4. image_width X.
    5. contig samples_per_pixel.

    """
    def __init__(self, parent):
        """Initialize instance from file."""
        self.parent = parent
        self.index = len(parent.pages)
        self.shape = self._shape = ()
        self.dtype = self._dtype = None
        self.axes = ""
        self.tags = TiffTags()
        self._offset = 0

        self._fromfile()
        self._process_tags()

    def _fromfile(self):
        """Read TIFF IFD structure and its tags from file.

        File cursor must be at storage position of IFD offset and is left at
        offset to next IFD.

        Raises StopIteration if offset (first bytes read) is 0
        or a corrupted page list is encountered.

        """
        fh = self.parent.filehandle
        byteorder = self.parent.byteorder
        offset_size = self.parent.offset_size

        # read offset to this IFD
        fmt = {4: 'I', 8: 'Q'}[offset_size]
        offset = struct.unpack(byteorder + fmt, fh.read(offset_size))[0]
        if not offset:
            raise StopIteration()
        if offset >= fh.size:
            warnings.warn("invalid page offset > file size")
            raise StopIteration()
        self._offset = offset

        # read standard tags
        tags = self.tags
        fh.seek(offset)
        fmt, size = {4: ('H', 2), 8: ('Q', 8)}[offset_size]
        try:
            numtags = struct.unpack(byteorder + fmt, fh.read(size))[0]
            if numtags > 4096:
                raise ValueError("suspicious number of tags")
        except Exception:
            warnings.warn("corrupted page list at offset %i" % offset)
            raise StopIteration()

        tagcode = 0
        for _ in range(numtags):
            try:
                tag = TiffTag(self.parent)
            except TiffTag.Error as e:
                warnings.warn(str(e))
                continue
            if tagcode > tag.code:
                # expected for early LSM and tifffile versions
                warnings.warn("tags are not ordered by code")
            tagcode = tag.code
            if tag.name not in tags:
                tags[tag.name] = tag
            else:
                # some files contain multiple IFD with same code
                # e.g. MicroManager files contain two image_description
                i = 1
                while True:
                    name = "%s_%i" % (tag.name, i)
                    if name not in tags:
                        tags[name] = tag
                        break

        pos = fh.tell()  # where offset to next IFD can be found

        if self.is_lsm or (self.index and self.parent.is_lsm):
            # correct non standard LSM bitspersample tags
            self.tags['bits_per_sample']._fix_lsm_bitspersample(self)

        if self.is_lsm:
            # read LSM info subrecords
            for name, reader in CZ_LSM_INFO_READERS.items():
                try:
                    offset = self.cz_lsm_info['offset_'+name]
                except KeyError:
                    continue
                if offset < 8:
                    # older LSM revision
                    continue
                fh.seek(offset)
                try:
                    setattr(self, 'cz_lsm_'+name, reader(fh))
                except ValueError:
                    pass
        elif self.is_stk and 'uic1tag' in tags and not tags['uic1tag'].value:
            # read uic1tag now that plane count is known
            uic1tag = tags['uic1tag']
            fh.seek(uic1tag.value_offset)
            tags['uic1tag'].value = Record(
                read_uic1tag(fh, byteorder, uic1tag.dtype, uic1tag.count,
                             tags['uic2tag'].count))
        fh.seek(pos)

    def _process_tags(self):
        """Validate standard tags and initialize attributes.

        Raise ValueError if tag values are not supported.

        """
        tags = self.tags
        for code, (name, default, dtype, count, validate) in TIFF_TAGS.items():
            if name in tags:
                #tags[name] = TiffTag(code, dtype=dtype, count=count,
                #                     value=default, name=name)
                if validate:
                    try:
                        if tags[name].count == 1:
                            setattr(self, name, validate[tags[name].value])
                        else:
                            setattr(self, name, tuple(
                                validate[value] for value in tags[name].value))
                    except KeyError:
                        raise ValueError("%s.value (%s) not supported" %
                                         (name, tags[name].value))
            elif default is not None:
                setattr(self, name, validate[default] if validate else default)

        if 'bits_per_sample' in tags:
            tag = tags['bits_per_sample']
            if tag.count == 1:
                self.bits_per_sample = tag.value
            else:
                # LSM might list more items than samples_per_pixel
                value = tag.value[:self.samples_per_pixel]
                if any((v-value[0] for v in value)):
                    self.bits_per_sample = value
                else:
                    self.bits_per_sample = value[0]

        if 'sample_format' in tags:
            tag = tags['sample_format']
            if tag.count == 1:
                self.sample_format = TIFF_SAMPLE_FORMATS[tag.value]
            else:
                value = tag.value[:self.samples_per_pixel]
                if any((v-value[0] for v in value)):
                    self.sample_format = [TIFF_SAMPLE_FORMATS[v]
                                          for v in value]
                else:
                    self.sample_format = TIFF_SAMPLE_FORMATS[value[0]]

        if 'photometric' not in tags:
            self.photometric = None

        if 'image_length' in tags:
            self.strips_per_image = int(math.floor(
                float(self.image_length + self.rows_per_strip - 1) /
                self.rows_per_strip))
        else:
            self.strips_per_image = 0

        key = (self.sample_format, self.bits_per_sample)
        self.dtype = self._dtype = TIFF_SAMPLE_DTYPES.get(key, None)

        if 'image_length' not in self.tags or 'image_width' not in self.tags:
            # some GEL file pages are missing image data
            self.image_length = 0
            self.image_width = 0
            self.image_depth = 0
            self.strip_offsets = 0
            self._shape = ()
            self.shape = ()
            self.axes = ''

        if self.is_vista or self.parent.is_vista:
            # ISS Vista writes wrong image_depth tag
            self.image_depth = 1

        if self.is_indexed:
            self.dtype = self.tags['color_map'].dtype[1]
            self.color_map = numpy.array(self.color_map, self.dtype)
            dmax = self.color_map.max()
            if dmax < 256:
                self.dtype = numpy.uint8
                self.color_map = self.color_map.astype(self.dtype)
            #else:
            #    self.dtype = numpy.uint8
            #    self.color_map >>= 8
            #    self.color_map = self.color_map.astype(self.dtype)
            # TODO: support other photometric modes than RGB
            self.color_map.shape = (3, -1)

        # determine shape of data
        image_length = self.image_length
        image_width = self.image_width
        image_depth = self.image_depth
        samples_per_pixel = self.samples_per_pixel

        if self.is_stk:
            assert self.image_depth == 1
            planes = self.tags['uic2tag'].count
            if self.is_contig:
                self._shape = (planes, 1, 1, image_length, image_width,
                               samples_per_pixel)
                if samples_per_pixel == 1:
                    self.shape = (planes, image_length, image_width)
                    self.axes = 'YX'
                else:
                    self.shape = (planes, image_length, image_width,
                                  samples_per_pixel)
                    self.axes = 'YXS'
            else:
                self._shape = (planes, samples_per_pixel, 1, image_length,
                               image_width, 1)
                if samples_per_pixel == 1:
                    self.shape = (planes, image_length, image_width)
                    self.axes = 'YX'
                else:
                    self.shape = (planes, samples_per_pixel, image_length,
                                  image_width)
                    self.axes = 'SYX'
            # detect type of series
            if planes == 1:
                self.shape = self.shape[1:]
            elif numpy.all(self.uic2tag.z_distance != 0):
                self.axes = 'Z' + self.axes
            elif numpy.all(numpy.diff(self.uic2tag.time_created) != 0):
                self.axes = 'T' + self.axes
            else:
                self.axes = 'I' + self.axes
            # DISABLED
            if self.is_indexed:
                assert False, "color mapping disabled for stk"
                if self.color_map.shape[1] >= 2**self.bits_per_sample:
                    if image_depth == 1:
                        self.shape = (planes, image_length, image_width,
                                      self.color_map.shape[0])
                    else:
                        self.shape = (planes, image_depth, image_length,
                                      image_width, self.color_map.shape[0])
                    self.axes = self.axes + 'S'
                else:
                    warnings.warn("palette cannot be applied")
                    self.is_indexed = False
        elif self.is_indexed:
            samples = 1
            if 'extra_samples' in self.tags:
                samples += self.tags['extra_samples'].count
            if self.is_contig:
                self._shape = (1, 1, image_depth, image_length, image_width,
                               samples)
            else:
                self._shape = (1, samples, image_depth, image_length,
                               image_width, 1)
            if self.color_map.shape[1] >= 2**self.bits_per_sample:
                if image_depth == 1:
                    self.shape = (image_length, image_width,
                                  self.color_map.shape[0])
                    self.axes = 'YXS'
                else:
                    self.shape = (image_depth, image_length, image_width,
                                  self.color_map.shape[0])
                    self.axes = 'ZYXS'
            else:
                warnings.warn("palette cannot be applied")
                self.is_indexed = False
                if image_depth == 1:
                    self.shape = (image_length, image_width)
                    self.axes = 'YX'
                else:
                    self.shape = (image_depth, image_length, image_width)
                    self.axes = 'ZYX'
        elif self.is_rgb or samples_per_pixel > 1:
            if self.is_contig:
                self._shape = (1, 1, image_depth, image_length, image_width,
                               samples_per_pixel)
                if image_depth == 1:
                    self.shape = (image_length, image_width, samples_per_pixel)
                    self.axes = 'YXS'
                else:
                    self.shape = (image_depth, image_length, image_width,
                                  samples_per_pixel)
                    self.axes = 'ZYXS'
            else:
                self._shape = (1, samples_per_pixel, image_depth,
                               image_length, image_width, 1)
                if image_depth == 1:
                    self.shape = (samples_per_pixel, image_length, image_width)
                    self.axes = 'SYX'
                else:
                    self.shape = (samples_per_pixel, image_depth,
                                  image_length, image_width)
                    self.axes = 'SZYX'
            if False and self.is_rgb and 'extra_samples' in self.tags:
                # DISABLED: only use RGB and first alpha channel if exists
                extra_samples = self.extra_samples
                if self.tags['extra_samples'].count == 1:
                    extra_samples = (extra_samples,)
                for exs in extra_samples:
                    if exs in ('unassalpha', 'assocalpha', 'unspecified'):
                        if self.is_contig:
                            self.shape = self.shape[:-1] + (4,)
                        else:
                            self.shape = (4,) + self.shape[1:]
                        break
        else:
            self._shape = (1, 1, image_depth, image_length, image_width, 1)
            if image_depth == 1:
                self.shape = (image_length, image_width)
                self.axes = 'YX'
            else:
                self.shape = (image_depth, image_length, image_width)
                self.axes = 'ZYX'
        if not self.compression and 'strip_byte_counts' not in tags:
            self.strip_byte_counts = (
                product(self.shape) * (self.bits_per_sample // 8),)

        assert len(self.shape) == len(self.axes)

    def _patch_imagej(self):
        """Return if ImageJ data are contiguous and adjust page attributes.

        Patch 'strip_offsets' and 'strip_byte_counts' tags to span the
        complete contiguous data.

        ImageJ stores all image metadata in the first page and image data is
        stored contiguously before the second page, if any. No need to
        read other pages.

        """
        if not self.is_imagej or not self.is_contiguous or self.parent.is_ome:
            return
        images = self.imagej_tags.get('images', 0)
        if images <= 1:
            return
        offset, count = self.is_contiguous
        shape = self.shape
        if self.is_indexed:
            shape = shape[:-1]
        if (count != product(shape) * self.bits_per_sample // 8 or
                offset + count*images > self.parent.filehandle.size):
            self.is_imagej = False
            warnings.warn("corrupted ImageJ metadata or file")
            return

        pre = 'tile' if self.is_tiled else 'strip'
        self.tags[pre+'_offsets'].value = (offset,)
        self.tags[pre+'_byte_counts'].value = (count * images,)
        self.shape = (images,) + self.shape
        self._shape = (images,) + self._shape[1:]
        self.axes = 'I' + self.axes
        return True

    def asarray(self, squeeze=True, colormapped=True, rgbonly=False,
                scale_mdgel=False, memmap=False, reopen=True,
                maxsize=64*1024*1024*1024):
        """Read image data from file and return as numpy array.

        Raise ValueError if format is unsupported.
        If any of 'squeeze', 'colormapped', or 'rgbonly' are not the default,
        the shape of the returned array might be different from the page shape.

        Parameters
        ----------
        squeeze : bool
            If True, all length-1 dimensions (except X and Y) are
            squeezed out from result.
        colormapped : bool
            If True, color mapping is applied for palette-indexed images.
        rgbonly : bool
            If True, return RGB(A) image without additional extra samples.
        memmap : bool
            If True, use numpy.memmap to read arrays from file if possible.
            For use on 64 bit systems and files with few huge contiguous data.
        reopen : bool
            If True and the parent file handle is closed, the file is
            temporarily re-opened (and closed if no exception occurs).
        scale_mdgel : bool
            If True, MD Gel data will be scaled according to the private
            metadata in the second TIFF page. The dtype will be float32.
        maxsize: int or None
            Maximum size of data before a ValueError is raised.
            Can be used to catch DOS. Default: 64 GB.

        """
        if not self._shape:
            return
        if maxsize and product(self._shape) > maxsize:
            raise ValueError("data is too large %s" % str(self._shape))

        if self.dtype is None:
            raise ValueError("data type not supported: %s%i" % (
                self.sample_format, self.bits_per_sample))
        if self.compression not in TIFF_DECOMPESSORS:
            raise ValueError("cannot decompress %s" % self.compression)
        if 'sample_format' in self.tags:
            tag = self.tags['sample_format']
            if tag.count != 1 and any((i-tag.value[0] for i in tag.value)):
                raise ValueError("sample formats do not match %s" % tag.value)

        if self.is_chroma_subsampled:
            # TODO: implement chroma subsampling
            raise NotImplementedError("chroma subsampling not supported")

        fh = self.parent.filehandle
        closed = fh.closed
        if closed:
            if reopen:
                fh.open()
            else:
                raise IOError("file handle is closed")

        dtype = self._dtype
        shape = self._shape
        image_width = self.image_width
        image_length = self.image_length
        image_depth = self.image_depth
        typecode = self.parent.byteorder + dtype
        bits_per_sample = self.bits_per_sample
        lsb2msb = self.fill_order == 'lsb2msb'

        byte_counts, offsets = self._byte_counts_offsets

        if self.is_tiled:
            tile_width = self.tile_width
            tile_length = self.tile_length
            tile_depth = self.tile_depth if 'tile_depth' in self.tags else 1
            tw = (image_width + tile_width - 1) // tile_width
            tl = (image_length + tile_length - 1) // tile_length
            td = (image_depth + tile_depth - 1) // tile_depth
            shape = (shape[0], shape[1],
                     td*tile_depth, tl*tile_length, tw*tile_width, shape[-1])
            tile_shape = (tile_depth, tile_length, tile_width, shape[-1])
            runlen = tile_width
        else:
            runlen = image_width

        if memmap and self._is_memmappable(rgbonly, colormapped):
            result = fh.memmap_array(typecode, shape, offset=offsets[0])
        elif self.is_contiguous:
            fh.seek(offsets[0])
            result = fh.read_array(typecode, product(shape))
            result = result.astype('=' + dtype)
            if lsb2msb:
                reverse_bitorder(result)
        else:
            if self.is_contig:
                runlen *= self.samples_per_pixel
            if bits_per_sample in (8, 16, 32, 64, 128):
                if (bits_per_sample * runlen) % 8:
                    raise ValueError("data and sample size mismatch")

                def unpack(x, typecode=typecode):
                    if self.predictor == 'float':
                        # the floating point horizontal differencing decoder
                        # needs the raw byte order
                        typecode = dtype
                    try:
                        return numpy.fromstring(x, typecode)
                    except ValueError as e:
                        # strips may be missing EOI
                        warnings.warn("unpack: %s" % e)
                        xlen = ((len(x) // (bits_per_sample // 8)) *
                                (bits_per_sample // 8))
                        return numpy.fromstring(x[:xlen], typecode)

            elif isinstance(bits_per_sample, tuple):
                def unpack(x):
                    return unpack_rgb(x, typecode, bits_per_sample)
            else:
                def unpack(x):
                    return unpack_ints(x, typecode, bits_per_sample, runlen)

            decompress = TIFF_DECOMPESSORS[self.compression]
            if self.compression == 'jpeg':
                table = self.jpeg_tables if 'jpeg_tables' in self.tags else b''

                def decompress(x):
                    return decode_jpeg(x, table, self.photometric)

            if self.is_tiled:
                result = numpy.empty(shape, dtype)
                tw, tl, td, pl = 0, 0, 0, 0
                for offset, bytecount in zip(offsets, byte_counts):
                    fh.seek(offset)
                    tile = fh.read(bytecount)
                    if lsb2msb:
                        tile = reverse_bitorder(tile)
                    tile = decompress(tile)
                    tile = unpack(tile)
                    try:
                        tile.shape = tile_shape
                    except ValueError:
                        # incomplete tiles; see gdal issue #1179
                        warnings.warn("invalid tile data")
                        t = numpy.zeros(tile_shape, dtype).reshape(-1)
                        s = min(tile.size, t.size)
                        t[:s] = tile[:s]
                        tile = t.reshape(tile_shape)
                    if self.predictor == 'horizontal':
                        numpy.cumsum(tile, axis=-2, dtype=dtype, out=tile)
                    elif self.predictor == 'float':
                        raise NotImplementedError()
                    result[0, pl, td:td+tile_depth,
                           tl:tl+tile_length, tw:tw+tile_width, :] = tile
                    del tile
                    tw += tile_width
                    if tw >= shape[4]:
                        tw, tl = 0, tl + tile_length
                        if tl >= shape[3]:
                            tl, td = 0, td + tile_depth
                            if td >= shape[2]:
                                td, pl = 0, pl + 1
                result = result[...,
                                :image_depth, :image_length, :image_width, :]
            else:
                strip_size = (self.rows_per_strip * self.image_width *
                              self.samples_per_pixel)
                result = numpy.empty(shape, dtype).reshape(-1)
                index = 0
                for offset, bytecount in zip(offsets, byte_counts):
                    fh.seek(offset)
                    strip = fh.read(bytecount)
                    if lsb2msb:
                        strip = reverse_bitorder(strip)
                    strip = decompress(strip)
                    strip = unpack(strip)
                    size = min(result.size, strip.size, strip_size,
                               result.size - index)
                    result[index:index+size] = strip[:size]
                    del strip
                    index += size

        result.shape = self._shape

        if self.predictor and not (self.is_tiled and not self.is_contiguous):
            if self.parent.is_lsm and not self.compression:
                pass  # work around bug in LSM510 software
            elif self.predictor == 'horizontal':
                numpy.cumsum(result, axis=-2, dtype=dtype, out=result)
            elif self.predictor == 'float':
                result = decode_floats(result)
        if colormapped and self.is_indexed:
            if self.color_map.shape[1] >= 2**bits_per_sample:
                # FluoView and LSM might fail here
                result = apply_colormap(result[:, 0:1, :, :, :, 0:1],
                                        self.color_map)
        elif rgbonly and self.is_rgb and 'extra_samples' in self.tags:
            # return only RGB and first alpha channel if exists
            extra_samples = self.extra_samples
            if self.tags['extra_samples'].count == 1:
                extra_samples = (extra_samples,)
            for i, exs in enumerate(extra_samples):
                if exs in ('unassalpha', 'assocalpha', 'unspecified'):
                    if self.is_contig:
                        result = result[..., [0, 1, 2, 3+i]]
                    else:
                        result = result[:, [0, 1, 2, 3+i]]
                    break
            else:
                if self.is_contig:
                    result = result[..., :3]
                else:
                    result = result[:, :3]

        if squeeze:
            try:
                result.shape = self.shape
            except ValueError:
                warnings.warn("failed to reshape from %s to %s" % (
                    str(result.shape), str(self.shape)))

        if scale_mdgel and self.parent.is_mdgel:
            # MD Gel stores private metadata in the second page
            tags = self.parent.pages[1]
            if tags.md_file_tag in (2, 128):
                scale = tags.md_scale_pixel
                scale = scale[0] / scale[1]  # rational
                result = result.astype('float32')
                if tags.md_file_tag == 2:
                    result **= 2  # squary root data format
                result *= scale

        if closed:
            # TODO: file should remain open if an exception occurred above
            fh.close()
        return result

    @lazyattr
    def _byte_counts_offsets(self):
        """Return simplified byte_counts and offsets."""
        if 'tile_offsets' in self.tags:
            byte_counts = self.tile_byte_counts
            offsets = self.tile_offsets
        else:
            byte_counts = self.strip_byte_counts
            offsets = self.strip_offsets

        j = 0
        for i, (b, o) in enumerate(zip(byte_counts, offsets)):
            if b > 0 and o > 0:
                if i > j:
                    byte_counts[j] = b
                    offsets[j] = o
                j += 1
            elif b > 0 and o <= 0:
                raise ValueError("invalid offset")
            else:
                warnings.warn("empty byte count")
        if j == 0:
            j = 1

        return byte_counts[:j], offsets[:j]

    def _is_memmappable(self, rgbonly, colormapped):
        """Return if page's image data in file can be memory-mapped."""
        return (self.parent.filehandle.is_file and
                self.is_contiguous and
                (self.bits_per_sample == 8 or self.parent._is_native) and
                self.fill_order == 'msb2lsb' and
                not self.predictor and
                not self.is_chroma_subsampled and
                not (rgbonly and 'extra_samples' in self.tags) and
                not (colormapped and self.is_indexed))

    @lazyattr
    def is_contiguous(self):
        """Return offset and size of contiguous data, else None.

        Excludes prediction, fill_order, and colormapping.

        """
        if self.compression or self.bits_per_sample not in (8, 16, 32, 64):
            return
        if self.is_tiled:
            if (self.image_width != self.tile_width or
                    self.image_length % self.tile_length or
                    self.tile_width % 16 or self.tile_length % 16):
                return
            if ('image_depth' in self.tags and 'tile_depth' in self.tags and
                    (self.image_length != self.tile_length or
                     self.image_depth % self.tile_depth)):
                return
            offsets = self.tile_offsets
            byte_counts = self.tile_byte_counts
        else:
            offsets = self.strip_offsets
            byte_counts = self.strip_byte_counts
        if len(offsets) == 1:
            return offsets[0], byte_counts[0]
        if self.is_stk or all(offsets[i] + byte_counts[i] == offsets[i+1] or
                              byte_counts[i+1] == 0  # no data/ignore offset
                              for i in range(len(offsets)-1)):
            return offsets[0], sum(byte_counts)

    def __str__(self):
        """Return string containing information about page."""
        s = ', '.join(s for s in (
            ' x '.join(str(i) for i in self.shape),
            str(numpy.dtype(self.dtype)),
            '%s bit' % str(self.bits_per_sample),
            self.photometric if 'photometric' in self.tags else '',
            self.compression if self.compression else 'raw',
            '|'.join(t[3:] for t in (
                'is_stk', 'is_lsm', 'is_nih', 'is_ome', 'is_imagej',
                'is_micromanager', 'is_fluoview', 'is_mdgel', 'is_mediacy',
                'is_scn', 'is_sgi', 'is_reduced', 'is_tiled',
                'is_contiguous') if getattr(self, t))) if s)
        return "Page %i: %s" % (self.index, s)

    def __getattr__(self, name):
        """Return tag value."""
        if name in self.tags:
            value = self.tags[name].value
            setattr(self, name, value)
            return value
        raise AttributeError(name)

    @lazyattr
    def uic_tags(self):
        """Consolidate UIC tags."""
        if not self.is_stk:
            raise AttributeError("uic_tags")
        tags = self.tags
        result = Record()
        result.number_planes = tags['uic2tag'].count
        if 'image_description' in tags:
            result.plane_descriptions = self.image_description.split(b'\x00')
        if 'uic1tag' in tags:
            result.update(tags['uic1tag'].value)
        if 'uic3tag' in tags:
            result.update(tags['uic3tag'].value)  # wavelengths
        if 'uic4tag' in tags:
            result.update(tags['uic4tag'].value)  # override uic1 tags
        uic2tag = tags['uic2tag'].value
        result.z_distance = uic2tag.z_distance
        result.time_created = uic2tag.time_created
        result.time_modified = uic2tag.time_modified
        try:
            result.datetime_created = [
                julian_datetime(*dt) for dt in
                zip(uic2tag.date_created, uic2tag.time_created)]
            result.datetime_modified = [
                julian_datetime(*dt) for dt in
                zip(uic2tag.date_modified, uic2tag.time_modified)]
        except ValueError as e:
            warnings.warn("uic_tags: %s" % e)
        return result

    @lazyattr
    def imagej_tags(self):
        """Consolidate ImageJ metadata."""
        if not self.is_imagej:
            raise AttributeError("imagej_tags")
        result = imagej_description_dict(self.is_imagej)
        if 'imagej_metadata' in self.tags:
            try:
                result.update(imagej_metadata(
                    self.tags['imagej_metadata'].value,
                    self.tags['imagej_byte_counts'].value,
                    self.parent.byteorder))
            except Exception as e:
                warnings.warn(str(e))
        return Record(result)

    @lazyattr
    def is_rgb(self):
        """Page contains a RGB image."""
        return ('photometric' in self.tags and
                self.tags['photometric'].value == 2)

    @lazyattr
    def is_contig(self):
        """Page contains contiguous image."""
        if 'planar_configuration' in self.tags:
            return self.tags['planar_configuration'].value == 1
        return True

    @lazyattr
    def is_indexed(self):
        """Page contains indexed, palette-colored image.

        Disable color-mapping for OME, LSM, STK, and ImageJ hyperstacks.

        """
        if (self.is_stk or self.is_lsm or self.parent.is_lsm or
                self.is_ome or self.parent.is_ome):
            return False
        if self.is_imagej:
            if b'mode' in self.is_imagej:
                return False
        elif self.parent.is_imagej:
            return self.parent.is_indexed
        return ('photometric' in self.tags and
                self.tags['photometric'].value == 3)

    @lazyattr
    def is_tiled(self):
        """Page contains tiled image."""
        return 'tile_width' in self.tags

    @lazyattr
    def is_reduced(self):
        """Page is reduced image of another image."""
        return ('new_subfile_type' in self.tags and
                self.tags['new_subfile_type'].value & 1)

    @lazyattr
    def is_chroma_subsampled(self):
        """Page contains chroma subsampled image."""
        return ('ycbcr_subsampling' in self.tags and
                self.tags['ycbcr_subsampling'].value != (1, 1))

    @lazyattr
    def is_mdgel(self):
        """Page contains md_file_tag tag."""
        return 'md_file_tag' in self.tags

    @lazyattr
    def is_mediacy(self):
        """Page contains Media Cybernetics Id tag."""
        return ('mc_id' in self.tags and
                self.tags['mc_id'].value.startswith(b'MC TIFF'))

    @lazyattr
    def is_stk(self):
        """Page contains UIC2Tag tag."""
        return 'uic2tag' in self.tags

    @lazyattr
    def is_lsm(self):
        """Page contains LSM CZ_LSM_INFO tag."""
        return 'cz_lsm_info' in self.tags

    @lazyattr
    def is_fluoview(self):
        """Page contains FluoView MM_STAMP tag."""
        return 'mm_stamp' in self.tags

    @lazyattr
    def is_nih(self):
        """Page contains NIH image header."""
        return 'nih_image_header' in self.tags

    @lazyattr
    def is_sgi(self):
        """Page contains SGI image and tile depth tags."""
        return 'image_depth' in self.tags and 'tile_depth' in self.tags

    @lazyattr
    def is_vista(self):
        """Software tag is 'ISS Vista'."""
        return ('software' in self.tags and
                self.tags['software'].value == b'ISS Vista')

    @lazyattr
    def is_ome(self):
        """Page contains OME-XML in image_description tag."""
        if 'image_description' not in self.tags:
            return False
        d = self.tags['image_description'].value.strip()
        return d.startswith(b'<?xml version=') and d.endswith(b'</OME>')

    @lazyattr
    def is_scn(self):
        """Page contains Leica SCN XML in image_description tag."""
        if 'image_description' not in self.tags:
            return False
        d = self.tags['image_description'].value.strip()
        return d.startswith(b'<?xml version=') and d.endswith(b'</scn>')

    @lazyattr
    def is_shaped(self):
        """Return description containing shape if exists, else None."""
        if 'image_description' in self.tags:
            description = self.tags['image_description'].value
            if b'"shape":' in description or b'shape=(' in description:
                return description
        if 'image_description_1' in self.tags:
            description = self.tags['image_description_1'].value
            if b'"shape":' in description or b'shape=(' in description:
                return description

    @lazyattr
    def is_imagej(self):
        """Return ImageJ description if exists, else None."""
        if 'image_description' in self.tags:
            description = self.tags['image_description'].value
            if description.startswith(b'ImageJ='):
                return description
        if 'image_description_1' in self.tags:
            # Micromanager
            description = self.tags['image_description_1'].value
            if description.startswith(b'ImageJ='):
                return description

    @lazyattr
    def is_micromanager(self):
        """Page contains Micro-Manager metadata."""
        return 'micromanager_metadata' in self.tags


class TiffTag(object):
    """A TIFF tag structure.