import os
import shutil
import numpy as np
import zipfile
import natsort
from natsort import os_sorted
import imagecodecs
import tifffile
from concurrent.futures import ThreadPoolExecutor
import pydicom
from pydicom.dataelem import DataElement
from pydicom.tag import Tag
from pydicom.encaps import decode_data_sequence, generate_pixel_data_frame
from pydicom.pixel_data_handlers.util import convert_color_space, _expand_segmented_lut
# from pylibjpeg import decode
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
import argparse
import sys
from struct import unpack, unpack_from
from sys import byteorder
from typing import (
    Dict, Optional, Union, List, Tuple, TYPE_CHECKING, cast, Iterable,
    ByteString
)
import warnings
from timeit import default_timer as timer
import configparser


def find_icc_profile(dataset):
    icc_profile_tag = Tag(0x0028, 0x2000)

    for elem in dataset.iterall():
        if elem.tag == icc_profile_tag:
            return elem.value
    return None


def extract_icc_profile(ds):
    # Find ICC Profile in the dataset
    icc_profile = find_icc_profile(ds)

    if icc_profile is not None:
        return icc_profile
    else:
        raise ValueError("No ICC Profile found in the DICOM file.")


def extract_metadata_filemeta(ds):
    metadata = {}
    for item in ds.file_meta:
        metadata[str(item.tag)] = item.value
    return metadata


def extract_metadata(ds):
    metadata = {}
    for elem in ds:
        if elem.VR in ['OB', 'OW', 'OF', 'OD', 'UN']:  # Skip byte data elements
            continue
        if elem.VR == 'SQ':  # Check if the element is a sequence
            metadata[str(elem.tag)] = [extract_metadata(item) for item in elem.value]
        else:
            metadata[str(elem.tag)] = elem.value
    return metadata


def decipher_dcm_folder(path_to_dcm):
    # Define the four possible image types
    possible_types = ['THUMBNAIL', 'VOLUME', 'LABEL', 'OVERVIEW', 'REGIONLOCALIZER', 'LOCALIZER']

    # list of files .dcm
    dcm_list = [f for f in os.listdir(path_to_dcm) if f.endswith('.dcm') and 'graphics' not in f]

    thumbnail_dcm = None
    overview_dcm = None
    label_dcm = None

    # volume
    dcm_levels_list = []
    dcm_levels_width_list = []
    dcm_levels_height_list = []
    for dcm in dcm_list:
        ds = pydicom.dcmread(path_to_dcm + '/' + dcm, force=True)
        img_type_str = ds[0x0008, 0x0008].value
        # img_type_bytes = ds.get_item(Tag(0x0008, 0x0008)).value
        # img_type_str = img_type_bytes.decode('utf-8')
        # Extract the nature of the image type
        extracted_type = [img_type for img_type in possible_types if img_type in img_type_str][0]
        if extracted_type == 'VOLUME':
            dcm_levels_list.append(ds)
            dcm_levels_width_list.append(ds.TotalPixelMatrixColumns)  # width
        elif extracted_type == 'THUMBNAIL':
            thumbnail_dcm = ds
        elif extracted_type == 'OVERVIEW':
            overview_dcm = ds
        elif extracted_type == 'LABEL':
            label_dcm = ds

    # find the base level and the pyramidal levels (based on width)
    pyramidal_levels = [round(max(dcm_levels_width_list) / f) for f in dcm_levels_width_list]

    # reorder lists
    sorted_indices = sorted(range(len(dcm_levels_width_list)), key=lambda i: dcm_levels_width_list[i], reverse=True)
    pyramidal_levels = [pyramidal_levels[i] for i in sorted_indices]
    dcm_levels_list = [dcm_levels_list[i] for i in sorted_indices]

    dcm_levels_dict = dict(zip(pyramidal_levels, dcm_levels_list))

    return thumbnail_dcm, overview_dcm, label_dcm, dcm_levels_dict


def get_main_metada(ds, pixel_size):
    # List of other tags
    tags = [
        (0x0008, 0x0070),  # Manufacturer
        (0x0008, 0x1090),  # Manufacturer's Model Name
        (0x0018, 0x1000),  # Device Serial Number
        (0x0018, 0x1020),  # Software Versions
        (0x0925, 0x0010),  # Private Creator
        (0x0028, 0x0010),  # Rows
        (0x0028, 0x0011),  # Columns
        (0x0028, 0x0004),  # Photometric Interpretation
        (0x0048, 0x0001),  # Imaged Volume Width
        (0x0048, 0x0002),  # Imaged Volume Height
        (0x0048, 0x0006),  # Total Pixel Matrix Columns
        (0x0048, 0x0007),  # Total Pixel Matrix Rows
    ]
    tag_name = ['Manufacturer',
                "Manufacturer Model Name",
                'Device Serial Number',
                'Software Versions',
                'Private Creator',
                'Rows',
                'Columns',
                'Photometric Interpretation',
                'Imaged Volume Width',
                'Imaged Volume Height',
                'Total Pixel Matrix Columns',
                'Total Pixel Matrix Rows',
                'Compression',
                'Objective Lens Power',
                'Session Mode'
                ]

    # Map the TransferSyntaxUID to the corresponding compression type
    compression_types = {
        '1.2.840.10008.1.2.4.50': 'JPEG Baseline (Process 1)',
        '1.2.840.10008.1.2.4.51': 'JPEG Extended (Process 2 & 4)',
        '1.2.840.10008.1.2.4.57': 'JPEG Lossless (Process 14)',
        '1.2.840.10008.1.2.4.70': 'JPEG Lossless, Non-Hierarchical (Process 14)',
        '1.2.840.10008.1.2.4.80': 'JPEG-LS Lossless Image Compression',
        '1.2.840.10008.1.2.4.81': 'JPEG-LS Lossy (Near-Lossless) Image Compression',
        '1.2.840.10008.1.2.4.90': 'JPEG 2000 Image Compression (Lossless Only)',
        '1.2.840.10008.1.2.4.91': 'JPEG 2000 Image Compression',
        '1.2.840.10008.1.2.4.92': 'JPEG 2000 Part 2 Multicomponent Image Compression (Lossless Only)',
        '1.2.840.10008.1.2.4.93': 'JPEG 2000 Part 2 Multicomponent Image Compression'
    }

    values = []
    for tag in tags:
        try:
            values.append(ds[tag].value)
        except:
            values.append('Unknown')
    # compression
    transfer_syntax_uid = ds.file_meta.TransferSyntaxUID
    compression_type = compression_types.get(str(transfer_syntax_uid), 'Unknown')
    values.append(compression_type)
    # tags within sequence
    try:
        obj_power = int(ds.get_item(Tag(0x0048, 0x0105))[0][0x0048, 0x0112].value)  # Objective Lens Power
    except Exception as e:
        pixelsize_x = round(float(ds.ImagedVolumeWidth) / float(ds.TotalPixelMatrixColumns) * 1000, 6)  # mm=>µm
        if pixelsize_x < 0.30:  # some web-based platforms like TeleSlide seem to require a value
            obj_power = 40
        else:
            if pixelsize_x < 0.60:
                obj_power = 20
            else:
                obj_power = 10

    try:
        session_mode = ds.get_item(Tag(0x0040, 0x0555))[1][0x0040, 0xa160].value  # Session Mode
    except Exception as e:
        session_mode = 'Unknown'
    # obj_power = int(ds.get_item(Tag(0x0048, 0x0105))[0][0x0048, 0x0112].value) #Objective Lens Power
    # session_mode = ds.get_item(Tag(0x0040, 0x0555))[1][0x0040, 0xa160].value #Session Mode
    values.append(obj_power)
    values.append(session_mode)
    tag_dict = dict(zip(tag_name, values))
    return tag_dict


def create_img_from_tiles(ds, thumbnail_tiles):
    # tile_size_x = ds.Columns
    # tile_size_y = ds.Rows
    if ds.TotalPixelMatrixColumns % ds.Columns != 0:  # pas un multiple de la tile size
        nb_tile_x = ds.TotalPixelMatrixColumns // ds.Columns + 1  # integer division
    else:
        nb_tile_x = ds.TotalPixelMatrixColumns // ds.Columns  # integer division
    if ds.TotalPixelMatrixRows % ds.Rows != 0:  # pas un multiple de la tile size
        nb_tile_y = int(ds.TotalPixelMatrixRows // ds.Rows) + 1
    else:
        nb_tile_y = ds.TotalPixelMatrixRows // ds.Rows
    img_width = ds.TotalPixelMatrixColumns  # width
    img_height = ds.TotalPixelMatrixRows  # height
    photometric_interpretation = ds.PhotometricInterpretation
    expected_nb_tiles = nb_tile_x * nb_tile_y

    # Iterate through the mask and extract non-overlapping tiles with positive pixels
    # i = 0
    coords = []
    tile_size = []
    for col in range(0, nb_tile_y):
        for row in range(0, nb_tile_x):
            if row == (nb_tile_x - 1):
                tile_size_x = ds.TotalPixelMatrixColumns % ds.Columns
            else:
                tile_size_x = ds.Columns
            if col == (nb_tile_y - 1):
                tile_size_y = ds.TotalPixelMatrixRows % ds.Rows
            else:
                tile_size_y = ds.Rows
            coords.append((col, row))
            tile_size.append((tile_size_y, tile_size_x))

    assert len(coords) == len(tile_size) == ds.NumberOfFrames

    if photometric_interpretation == 'MONOCHROME2':  # grayscale
        if ds.NumberOfFrames == 1:  # only one grayscale frame/tile of only 2 dimensions
            thumbnail_array = thumbnail_tiles[0:img_height, 0:img_width]
        else:
            idx = 0
            thumbnail_array = np.zeros((ds.TotalPixelMatrixRows, ds.TotalPixelMatrixColumns), dtype=np.uint8)
            for tile_array in thumbnail_tiles:
                thumbnail_array[int(coords[idx][0] * ds.Rows):int(coords[idx][0] * ds.Rows) + tile_size[idx][0],
                int(coords[idx][1] * ds.Columns):int(coords[idx][1] * ds.Columns) + tile_size[idx][1]] = \
                    tile_array[0:tile_size[idx][0], 0:tile_size[idx][1]]
                idx += 1
    else:  # brightfield
        if ds.NumberOfFrames == 1:  # only one color frame/tile of 3 dimensions
            thumbnail_array = thumbnail_tiles[0:img_height, 0:img_width, :]
            if photometric_interpretation != 'RGB':  # sometimes YBR_FULL_422
                thumbnail_array = convert_color_space(thumbnail_array, photometric_interpretation, 'RGB')
        else:
            idx = 0
            thumbnail_array = np.zeros((ds.TotalPixelMatrixRows, ds.TotalPixelMatrixColumns, 3), dtype=np.uint8)
            for tile_array in thumbnail_tiles:
                if photometric_interpretation != 'RGB':  # sometimes YBR_FULL_422
                    tile_array = convert_color_space(tile_array, photometric_interpretation, 'RGB')
                thumbnail_array[int(coords[idx][0] * ds.Rows):int(coords[idx][0] * ds.Rows) + tile_size[idx][0],
                int(coords[idx][1] * ds.Columns):int(coords[idx][1] * ds.Columns) + tile_size[idx][1]] = \
                    tile_array[0:tile_size[idx][0], 0:tile_size[idx][1]]
                idx += 1
    return thumbnail_array


def get_lut(ds):
    # Apply a color palette lookup table to `arr`. From pydicom

    # Note: input value (IV) is the stored pixel value in `arr`
    # LUTs[IV] -> [R, G, B] values at the IV pixel location in `arr`

    ds = cast("Dataset", ds)

    # if 'RedPaletteColorLookupTableDescriptor' not in ds:
    #    raise ValueError("No suitable Palette Color Lookup Table Module found")

    # RedPaletteColorLookupTableDescriptor = ds.RedPaletteColorLookupTableDescriptor
    RedPaletteColorLookupTableDescriptor = ds[(0x0048, 0x0105)][0][(0x0048, 0x0120)][0][(0x0028, 0x1101)].value

    # All channels are supposed to be identical
    lut_desc = cast(List[int], RedPaletteColorLookupTableDescriptor)
    # A value of 0 = 2^16 entries
    nr_entries = lut_desc[0] or 2**16

    # May be negative if Pixel Representation is 1
    first_map = lut_desc[1]
    # Actual bit depth may be larger (8 bit entries in 16 bits allocated)
    nominal_depth = lut_desc[2]
    dtype = np.dtype('uint{:.0f}'.format(nominal_depth))

    luts = []
    if 'RedPaletteColorLookupTableData' in ds:
        # LUT Data is described by PS3.3, C.7.6.3.1.6
        r_lut = cast(bytes, ds.RedPaletteColorLookupTableData)
        g_lut = cast(bytes, ds.GreenPaletteColorLookupTableData)
        b_lut = cast(bytes, ds.BluePaletteColorLookupTableData)
        a_lut = cast(
            Optional[bytes],
            getattr(ds, 'AlphaPaletteColorLookupTableData', None)
        )

        actual_depth = len(r_lut) / nr_entries * 8
        dtype = np.dtype('uint{:.0f}'.format(actual_depth))

        for lut_bytes in [ii for ii in [r_lut, g_lut, b_lut, a_lut] if ii]:
            luts.append(np.frombuffer(lut_bytes, dtype=dtype))
    elif 'SegmentedRedPaletteColorLookupTableData' in ds:
        # Segmented LUT Data is described by PS3.3, C.7.9.2
        r_lut = cast(bytes, ds.SegmentedRedPaletteColorLookupTableData)
        g_lut = cast(bytes, ds.SegmentedGreenPaletteColorLookupTableData)
        b_lut = cast(bytes, ds.SegmentedBluePaletteColorLookupTableData)
        a_lut = cast(
            Optional[bytes],
            getattr(ds, 'SegmentedAlphaPaletteColorLookupTableData', None)
        )

        endianness = '<' if ds.is_little_endian else '>'
        byte_depth = nominal_depth // 8
        fmt = 'B' if byte_depth == 1 else 'H'
        actual_depth = nominal_depth

        for seg in [ii for ii in [r_lut, g_lut, b_lut, a_lut] if ii]:
            len_seg = len(seg) // byte_depth
            s_fmt = endianness + str(len_seg) + fmt
            lut_ints = _expand_segmented_lut(unpack(s_fmt, seg), s_fmt)
            luts.append(np.asarray(lut_ints, dtype=dtype))
    elif 'RedPaletteColorLookupTableData' not in ds and 'SegmentedRedPaletteColorLookupTableData' not in ds:
        # faire un try except à l'usage
        # Segmented LUT Data is described by PS3.3, C.7.9.2
        r_lut = cast(bytes, ds[(0x0048, 0x0105)][0][(0x0048, 0x0120)][0][(0x0028, 0x1221)].value)
        g_lut = cast(bytes, ds[(0x0048, 0x0105)][0][(0x0048, 0x0120)][0][(0x0028, 0x1222)].value)
        b_lut = cast(bytes, ds[(0x0048, 0x0105)][0][(0x0048, 0x0120)][0][(0x0028, 0x1223)].value)
        a_lut = cast(
            Optional[bytes],
            getattr(ds, 'SegmentedAlphaPaletteColorLookupTableData', None)
        )
        endianness = '<' if ds.is_little_endian else '>'
        byte_depth = nominal_depth // 8
        fmt = 'B' if byte_depth == 1 else 'H'
        actual_depth = nominal_depth

        for seg in [ii for ii in [r_lut, g_lut, b_lut, a_lut] if ii]:
            len_seg = len(seg) // byte_depth
            s_fmt = endianness + str(len_seg) + fmt
            lut_ints = _expand_segmented_lut(unpack(s_fmt, seg), s_fmt)
            luts.append(np.asarray(lut_ints, dtype=dtype))

    else:
        raise ValueError("No suitable Palette Color Lookup Table Module found")

    if actual_depth not in [8, 16]:
        raise ValueError(
            f"The bit depth of the LUT data '{actual_depth:.1f}' "
            "is invalid (only 8 or 16 bits per entry allowed)"
        )

    lut_lengths = [len(ii) for ii in luts]
    if not all(ii == lut_lengths[0] for ii in lut_lengths[1:]):
        raise ValueError("LUT data must be the same length")

    return luts


def create_frame_list_tiled_sparse(ds):
    # expected number of tiles and coordinates if tiled_full
    tile_size_x = ds.Columns
    tile_size_y = ds.Rows
    photometric_interpretation = ds.PhotometricInterpretation
    if ds.TotalPixelMatrixColumns % ds.Columns != 0:  # pas un multiple de la tile size
        nb_tile_x = ds.TotalPixelMatrixColumns // ds.Columns + 1  # integer division
    else:
        nb_tile_x = ds.TotalPixelMatrixColumns // ds.Columns  # integer division
    if ds.TotalPixelMatrixRows % ds.Rows != 0:  # pas un multiple de la tile size
        nb_tile_y = int(ds.TotalPixelMatrixRows // ds.Rows) + 1
    else:
        nb_tile_y = ds.TotalPixelMatrixRows // ds.Rows
    # wsi_width = ds.TotalPixelMatrixColumns  # width
    # wsi_height = ds.TotalPixelMatrixRows  # height
    # expected_nb_tiles = nb_tile_x * nb_tile_y

    # theoretical frame position if tiled_full
    expected_tile_pos = []
    for col in range(0, nb_tile_y):
        for row in range(0, nb_tile_x):
            expected_tile_pos.append((col * tile_size_y + 1, row * tile_size_x + 1))

    # list of frame position
    col_row_list = []
    for pffgs_item in ds.PerFrameFunctionalGroupsSequence:
        row = pffgs_item[(0x0048, 0x021a)][0][(0x0048, 0x021e)].value  # column  #width
        column = pffgs_item[(0x0048, 0x021a)][0][(0x0048, 0x021f)].value  # row  #height
        col_row_list.append((column, row))

    # list of encoded frames
    frame_list = []
    for frame in generate_pixel_data_frame(ds.PixelData, ds.NumberOfFrames):
        frame_list.append(frame)

    assert len(frame_list) == len(col_row_list)

    # Step 1: Combine the lists using zip
    combined = list(zip(col_row_list, frame_list))
    # Step 2: Sort the combined list based on the first list (list1)
    sorted_combined = os_sorted(combined, key=lambda x: x[0])
    # Step 3: Separate the sorted pairs back into two lists
    col_row_list, frame_list = zip(*sorted_combined)
    # Convert back to lists if needed
    col_row_list = list(col_row_list)
    frame_list = list(frame_list)

    # create a blank tile
    photometric_interpretation = ds.PhotometricInterpretation
    if photometric_interpretation == 'MONOCHROME2':  # singleplex fluorescence
        blank_tile = np.zeros((tile_size_y, tile_size_x), dtype=np.uint8)  # grayscale, black background
    else:  # assume brightfield otherwise
        blank_tile = np.ones((tile_size_y, tile_size_x, 3), dtype=np.uint8) * 255  # RGB, white background
        if photometric_interpretation != 'RGB':  # sometimes YBR_FULL_422
            blank_tile = convert_color_space(blank_tile, 'RGB', photometric_interpretation)
    # encode blank tile
    if ds.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.50' or ds.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.51':  # JPEG
        blank_tile = imagecodecs.jpeg8_encode(blank_tile, colorspace='JCS_YCbCr', bitspersample=int(ds.BitsAllocated))
    elif ds.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.91':  # JPEG2000
        if photometric_interpretation != 'RGB':
            blank_tile = imagecodecs.jpeg2k_encode(blank_tile, codecformat='J2K', bitspersample=int(ds.BitsAllocated),
                                                   colorspace='SYCC')
        else:  # RGB
            blank_tile = imagecodecs.jpeg2k_encode(blank_tile, codecformat='J2K', bitspersample=int(ds.BitsAllocated),
                                                   colorspace='SRGB')

    # loop through each expected tile and create them if not present
    frame_list_tiled_full = []
    for tile_pos in expected_tile_pos:
        if tile_pos in col_row_list:
            index = col_row_list.index(tile_pos)
            frame_list_tiled_full.append(frame_list[index])
        else:
            frame_list_tiled_full.append(blank_tile)
    return frame_list_tiled_full


def from_DICOM_to_SVS(path_to_folder, is_zipped: bool, label: bool, macro: bool):
    '''
    Parameters:
        path_to_folder (str): the path to the folder containing the DICOM WSI files
        is_zipped (bool): True if the files need to be unzipped, False otherwise. It is assumed that all files will be either zipped or not. Default to True.
        label (bool): whether to add the label image if it exists. Default to True.
        macro (bool): whether to add the macro image if it exists. Default to True.
    '''

    # list of all files +/- unzip them
    if is_zipped == True:
        # list of zipped files
        WSI_list = [f for f in os.listdir(path_to_folder) if f.endswith(".zip")]
        print('Unzipping files...')
        # unzip
        path_unzip = path_to_folder + '_unzip'
        if not os.path.exists(path_unzip):
            os.mkdir(path_unzip)
        for WSI in WSI_list:
            path = os.path.join(path_to_folder + '/' + WSI)
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(os.path.splitext(path)[0])  # same name, just without the .zip extension
        # list of folder of DICOM images. One folder per image
        WSI_dir = [f for f in os.listdir(path_unzip)]
        WSI_dir = os_sorted(WSI_dir)
    else:  # no zipped files, list all image folder
        WSI_dir = [f for f in os.listdir(path_to_folder)]
        WSI_dir = os_sorted(WSI_dir)
        path_unzip = path_to_folder
    print(f'Number of identified WSI is: {len(WSI_dir)}')
    # define output_path
    path_output = path_to_folder + '_output'
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    # loop through each WSI
    y = 0
    for WSI_name in WSI_dir:
        print(f'Starting the conversion of {WSI_name}, slide number {y + 1} out of {len(WSI_dir)}')
        # list all files of the DICOM folder
        WSI_files = [f for f in os.listdir(path_unzip + '/' + WSI_name) if f.endswith(".dcm") and 'graphics' not in f]  # we ignore .dcm.import and graphics.dcm files
        thumbnail_dcm, overview_dcm, label_dcm, dcm_levels_dict = \
            decipher_dcm_folder(path_to_dcm=os.path.join(path_unzip, WSI_name))
        biggest_file_path = dcm_levels_dict[1].filename

        # open the image with WSIDicom and get some image properties
        ds = dcm_levels_dict[1]
        metadata_filemeta = extract_metadata_filemeta(ds)
        metadata = extract_metadata(ds)
        metadata = {**metadata_filemeta, **metadata}

        WSI_shape = (ds.TotalPixelMatrixColumns, ds.TotalPixelMatrixRows)  # width, height (yes)

        pixelsize_x = round(float(ds.ImagedVolumeWidth) / float(ds.TotalPixelMatrixColumns) * 1000, 6)  # mm=>µm
        pixelsize_y = round(float(ds.ImagedVolumeHeight) / float(ds.TotalPixelMatrixRows) * 1000, 6)  # mm=>µm
        pixel_size = round((pixelsize_x + pixelsize_y) / 2, 6)

        # tag_dict = get_main_metada(biggest_file_path, pixel_size)
        tag_dict = get_main_metada(ds, pixel_size)

        # quality_jpeg
        try:
            # Aperio GT450DX. Not found in 3DHistech slides from Pannoramic scan II
            quality_jpeg = int(metadata['(0040, 0555)'][6]['(0040, a30a)'])
        except Exception as e:
            quality_jpeg = None

        # Define pyramid levels
        # already defined by decipher_dcm_folder()

        # compression type : JPEG or JPEG2000
        compression_arg = None
        if tag_dict['Compression'] == 'JPEG Baseline (Process 1)':
            compression_name = 'JPEG/YCC'
            compression_arg = 'jpeg'
        elif tag_dict['Compression'] == 'JPEG 2000 Image Compression':
            if tag_dict['Photometric Interpretation'] == 'RGB':
                compression_name = 'J2K/KDU'
                compression_arg = 33005
            elif tag_dict['Photometric Interpretation'] == 'YBR_ICT':
                compression_name = 'J2K/YCC'
                compression_arg = 33003
            else:
                compression_name = 'J2K'
                compression_arg = 34712

        if compression_arg is None:
            print(f'Unsupported compression type for image: {WSI_name}. Try a lossy conversion.')

        else:
            # JPEG/YCC #if JPEG compression
            # JPEG2000: J2K/YUV16  #YCC #33003
            # JPEG2000: J2K/KDU   #RGB #33005

            # define the image description tag, which contains important information such as resolution and compression arguments
            # it is necessary that this image description starts with Aperio, as some software use this to recognize the file as svs
            image_description_base = f'Aperio Leica Biosystems (fake): {tag_dict["Private Creator"]} {tag_dict["Manufacturer"]} {tag_dict["Manufacturer Model Name"]} v{tag_dict["Software Versions"]} \n{WSI_shape[0]}x{WSI_shape[1]} [0,0,{WSI_shape[0]}x{WSI_shape[1]}] ({tag_dict["Columns"]}x{tag_dict["Rows"]}) {compression_name} Q={quality_jpeg}|AppMag = {tag_dict["Objective Lens Power"]}|MPP = {pixel_size}|ScanScope ID = {tag_dict["Device Serial Number"]}|ScannerType = {tag_dict["Manufacturer Model Name"]}|SessionMode = {tag_dict["Session Mode"]}|'

            # number of tiles and coordinates
            # tile_size_x = ds.Columns
            # tile_size_y = ds.Rows
            if ds.TotalPixelMatrixColumns % ds.Columns != 0:  # pas un multiple de la tile size
                nb_tile_x = ds.TotalPixelMatrixColumns // ds.Columns + 1  # integer division
            else:
                nb_tile_x = ds.TotalPixelMatrixColumns // ds.Columns  # integer division
            if ds.TotalPixelMatrixRows % ds.Rows != 0:  # pas un multiple de la tile size
                nb_tile_y = int(ds.TotalPixelMatrixRows // ds.Rows) + 1
            else:
                nb_tile_y = ds.TotalPixelMatrixRows // ds.Rows
            # wsi_width = ds.TotalPixelMatrixColumns  # width
            # wsi_height = ds.TotalPixelMatrixRows  # height
            photometric_interpretation = ds.PhotometricInterpretation
            expected_nb_tiles = nb_tile_x * nb_tile_y

            assert expected_nb_tiles == ds.NumberOfFrames

            # tile_size = (tag_dict['Columns'], tag_dict['Rows'])
            # tiles_indexes = list(range(0, ds.NumberOfFrames))

            # create the TIFF file
            if ds.DimensionOrganizationType == 'TILED_FULL':  # tiled_full
                def generate_tiles(ds, frame_list_tiled_full):
                    for frame in generate_pixel_data_frame(ds.PixelData, ds.NumberOfFrames):
                        yield frame
            else:  # tiled_sparse
                def generate_tiles(ds, frame_list_tiled_full):
                    for frame in frame_list_tiled_full:
                        yield frame

            # Brightfield VS Fluorescence
            if photometric_interpretation == 'MONOCHROME2':  # grayscale, deemed to encode a fluorescence image
                try:
                    color_map = get_lut(ds)
                    photometric_arg = 'palette'
                    shape_arg = (WSI_shape[1], WSI_shape[0])
                except:
                    print('Failed to extract the LookUp Table/LUT, default to FITC palette')

                    # create the FITC colormap
                    def generate_lists(total_length, range_length):
                        # Create the second list with the specified range
                        lut = [i // (total_length // range_length) for i in range(total_length)]
                        return lut

                    r_lut = np.array(generate_lists(256, 128), dtype=np.uint16)
                    g_lut = np.array(generate_lists(256, 256), dtype=np.uint16)
                    b_lut = np.zeros((256), dtype=np.uint16)
                    colormap_FITC = np.array((r_lut, g_lut, b_lut))
                    # define args for grayscale tiff writing with colormap
                    photometric_arg = 'palette'
                    color_map = colormap_FITC
                    shape_arg = (WSI_shape[1], WSI_shape[0])

            else:  # brightfield
                photometric_arg = 'rgb'
                color_map = None
                shape_arg = (WSI_shape[1], WSI_shape[0], 3)

            # ICC profile
            icc_profile_bytes = None
            try:
                icc_profile_bytes = extract_icc_profile(ds)
                print("ICC Profile extracted successfully.")
            except Exception as e:
                print(f"An error occurred: {e}")

            if icc_profile_bytes is not None:
                extratag = [(34675, 7, None, icc_profile_bytes, True)]  # ICC extratag
            else:
                extratag = None  # no extratag

            # Write tiles to TIFF file
            with tifffile.TiffWriter(os.path.join(path_output, WSI_name + '.svs'), shaped=False,
                                     bigtiff=True) as tif:  # all WSI from the Aperio GT450 DX seem to be BigTIFF, whatever the file size
                if ds.DimensionOrganizationType == 'TILED_FULL':  # tiled_full
                    frame_list_tiled_full = None  # decoy
                else:  # tiled_sparse
                    frame_list_tiled_full = create_frame_list_tiled_sparse(ds)

                # write the full resolution image
                tif.write(data=generate_tiles(ds, frame_list_tiled_full),
                          dtype='uint8',
                          shape=shape_arg,
                          subfiletype=0,
                          resolutionunit='CENTIMETER',
                          resolution=(1e4 / pixel_size, 1e4 / pixel_size),
                          # 1e-4 because resolution is in centimeter #Number of pixels per `resolutionunit` in X and Y directions
                          photometric=photometric_arg,  # will be automatically converted to YCbCr if RGB
                          compression=compression_arg,
                          compressionargs={'level': 91},  # the quality parameter found in WSI from the Aperio GT450 DX
                          tile=(tag_dict['Rows'], tag_dict['Columns']),
                          colormap=color_map,
                          description=image_description_base,
                          metadata=metadata,  # add other DICOM tags as a dictionary
                          extratags=extratag)

                # add the thumbnail image as a separate series, in second position
                if thumbnail_dcm is not None:
                    ds = thumbnail_dcm
                    photometric_interpretation = ds.PhotometricInterpretation
                    thumbnail_array = ds.pixel_array
                    if photometric_interpretation != 'MONOCHROME2':  # if not a grayscale image/fluorescence image => brightfield image
                        thumbnail_array = convert_color_space(thumbnail_array, photometric_interpretation, 'RGB')
                    thumbnail_shape = thumbnail_array.shape
                    # mpp_thumbnail = round(WSI_shape[0] / thumbnail_shape[1], 6)
                else:  # create the thumbnail from level 16
                    print('No thumbnail detected, creating one')
                    # load the pyramidal level 16 to create the thumbnail
                    if 32 in dcm_levels_dict:  # if this level exists
                        ds = dcm_levels_dict[32]
                    else:
                        if 16 in dcm_levels_dict:  # if this level exists
                            ds = dcm_levels_dict[16]
                        else:
                            max_level = max(dcm_levels_dict.keys())  # create thumbnail from maximum level/lowest resolution
                            ds = dcm_levels_dict[max_level]

                    # photometric_interpretation = ds.PhotometricInterpretation
                    thumbnail_tiles = ds.pixel_array
                    thumbnail_array = create_img_from_tiles(ds, thumbnail_tiles)
                    thumbnail_pil = Image.fromarray(thumbnail_array)
                    thumbnail_width, thumbnail_height = thumbnail_pil.size
                    # Calculate the aspect ratio of the image
                    aspect_ratio = thumbnail_width / thumbnail_height
                    # resize to a target width
                    target_width = 1920  # the default width with the Aperio GT450 DX
                    new_height = round(1920 / aspect_ratio)
                    thumbnail_pil = thumbnail_pil.resize((target_width, new_height), Image.BICUBIC)
                    # pillow to numpy
                    thumbnail_array = np.asarray(thumbnail_pil)
                    # thumbnail_array = thumbnail_array.astype('uint8')
                    thumbnail_shape = thumbnail_array.shape
                    # mpp_thumbnail = round(WSI_shape[0] / thumbnail_shape[1], 6)

                image_description_thumbnail = f'Aperio Leica Biosystems (fake): {tag_dict["Private Creator"]} {tag_dict["Manufacturer"]} {tag_dict["Manufacturer Model Name"]} v{tag_dict["Software Versions"]} \n{thumbnail_shape[1]}x{thumbnail_shape[0]} [0,0,{thumbnail_shape[1]}x{thumbnail_shape[0]}] ({tag_dict["Columns"]}x{tag_dict["Rows"]}) JPEG Q=100|AppMag = {tag_dict["Objective Lens Power"]}|MPP = {pixel_size}|ScanScope ID = {tag_dict["Device Serial Number"]}|ScannerType = {tag_dict["Manufacturer Model Name"]}|SessionMode = {tag_dict["Session Mode"]}|'

                tif.write(thumbnail_array,
                          subfiletype=0,
                          photometric=photometric_arg,
                          compression='jpeg',
                          compressionargs={'level': 100},  # atypical but is what was found in the image description of SVS files
                          colormap=color_map,
                          description=image_description_thumbnail,
                          extratags=extratag)  # no tiling, the image must be stripped

                # subresolutions images
                # a function to define pyramidal levels
                def write_pyramidal_level(level, tile_size):  # absolute level ID. 4 means that the width and height are divided by 4 as compared to the full resolution image.
                    ds = dcm_levels_dict[level]
                    level_shape = (ds.TotalPixelMatrixColumns, ds.TotalPixelMatrixRows)
                    image_description_level = f'Aperio Leica Biosystems (fake): {tag_dict["Private Creator"]} {tag_dict["Manufacturer"]} {tag_dict["Manufacturer Model Name"]} v{tag_dict["Software Versions"]} \n{level_shape[0]} [0,0,{level_shape[0]}] ({tag_dict["Columns"]}x{tag_dict["Rows"]}) {compression_name} Q={quality_jpeg}|AppMag = {tag_dict["Objective Lens Power"]}|MPP = {round(pixel_size * level, 6)}|ScanScope ID = {tag_dict["Device Serial Number"]}|ScannerType = {tag_dict["Manufacturer Model Name"]}|SessionMode = {tag_dict["Session Mode"]}|'
                    photometric_interpretation = ds.PhotometricInterpretation
                    if photometric_interpretation == 'MONOCHROME2':  # grayscale,
                        shape_arg = (level_shape[1], level_shape[0])
                    else:  # brightfield
                        shape_arg = (level_shape[1], level_shape[0], 3)

                    if ds.DimensionOrganizationType == 'TILED_FULL':  # tiled_full
                        frame_list_tiled_full = None  # decoy
                    else:  # tiled_sparse
                        frame_list_tiled_full = create_frame_list_tiled_sparse(ds)

                    # write level
                    tif.write(generate_tiles(ds, frame_list_tiled_full),
                              dtype='uint8',
                              shape=shape_arg,
                              subfiletype=0,
                              resolutionunit='CENTIMETER',
                              resolution=(1e4 / level / pixel_size, 1e4 / level / pixel_size),
                              photometric=photometric_arg,
                              compression=compression_arg,
                              compressionargs={'level': 91},
                              tile=(tag_dict['Rows'], tag_dict['Columns']),
                              colormap=color_map,
                              description=image_description_level,
                              extratags=extratag)

                for level in list(dcm_levels_dict.keys())[1:]:  # all levels except the first=full resolution
                    write_pyramidal_level(level, tile_size=(tag_dict['Rows'], tag_dict['Columns']))

                # label image?
                if label and label_dcm is not None:
                    ds = label_dcm
                    try:
                        label_array = ds.pixel_array
                    except Exception as e:
                        print("Could not load pixel data (pip install pydicom pylibjpeg pylibjpeg-libjpeg):", e)
                    else:
                        label_shape = label_array.shape
                        photometric_interpretation = ds.PhotometricInterpretation
                        if photometric_interpretation != 'RGB':  # sometimes YBR_FULL_422
                            label_array = convert_color_space(ds.pixel_array, photometric_interpretation, 'RGB')
                        image_description_label = f'{tag_dict["Private Creator"]} {tag_dict["Manufacturer"]} {tag_dict["Manufacturer Model Name"]} v{tag_dict["Software Versions"]} \nlabel {label_shape[1]}x{label_shape[0]}'
                        tif.write(label_array,
                                  subfiletype=1,  # reduced type of image
                                  photometric='rgb',
                                  compression='lzw',  # compression is not jpeg for the label image
                                  description=image_description_label,
                                  predictor=2,  # horizontal
                                  extratags=extratag)  # no tiling, the image must be stripped
                else:
                    print("No label image found.")

                # macro image?
                if macro and overview_dcm is not None:
                    ds = overview_dcm
                    macro_array = ds.pixel_array
                    macro_shape = macro_array.shape
                    photometric_interpretation = ds.PhotometricInterpretation
                    if photometric_interpretation != 'RGB':  # sometimes YBR_FULL_422
                        macro_array = convert_color_space(ds.pixel_array, photometric_interpretation, 'RGB')
                    image_description_macro = f'{tag_dict["Private Creator"]} {tag_dict["Manufacturer"]} {tag_dict["Manufacturer Model Name"]} v{tag_dict["Software Versions"]} \nmacro {macro_shape[1]}x{macro_shape[0]}'
                    tif.write(macro_array,
                              subfiletype=9,  # macro/reduced
                              photometric='rgb',
                              compression='jpeg',
                              compressionargs={'level': 95, 'outcolorspace': 'rgb'},
                              # needed to specifically ask for RGB and not YCbCr
                              description=image_description_macro,
                              extratags=extratag)  # no tiling, the image must be stripped
                else:
                    # Handle the case where no associated images are present
                    print("No macro/overview image found.")

            # # rename file extension from .tiff to .svs #not required to read the image in Aperio, but for most other software
            # path = os.path.join(path_output, WSI_name + '.svs')
            # if os.path.exists(path):  # if a file has the same name as what we want, remove it
            #     os.remove(path)
            # os.rename(os.path.join(path_output, WSI_name) + '.tiff', path)
            print(f'Pyramidal TIFF image saved as: {WSI_name}.svs')
            y += 1
    # delete the intermediate directory (unzipped folder)
    if is_zipped == True:
        try:
            shutil.rmtree(path_to_folder + '_unzip')
        except Exception as e:
            print(f"Could not delete folder {path_to_folder}_unzip: {e}")

    print('All done')


def main():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Create a ConfigParser instance
    config = configparser.ConfigParser()

    # Read settings from an INI file
    config.read('settings.ini')

    # Ensure the section 'DICOM2svs' exists
    if not config.has_section('DICOM2svs'):
        config.add_section('DICOM2svs')

    # read settings
    path_to_folder = config.get('DICOM2svs', 'path_to_folder', fallback="")
    is_zipped = config.get('DICOM2svs', 'is_zipped', fallback=False)
    is_label = config.get('DICOM2svs', 'is_label', fallback=False)
    is_macro = config.get('DICOM2svs', 'is_macro', fallback=False)

    def select_folder():
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            path_entry.delete(0, tk.END)
            path_entry.insert(0, folder_selected)

    def run_conversion():
        path_to_folder = path_entry.get()
        zip_option = is_zipped_var.get()
        label_option = label_var.get()
        macro_option = macro_var.get()

        if not os.path.isdir(path_to_folder):
            messagebox.showerror("Error", "Invalid folder path")
            return

        config.set('DICOM2svs', 'path_to_folder', path_to_folder)
        config.set('DICOM2svs', 'is_zipped', str(zip_option))
        config.set('DICOM2svs', 'is_label', str(label_option))
        config.set('DICOM2svs', 'is_macro', str(macro_option))
        # Write changes back to the configuration file
        with open('settings.ini', 'w') as configfile:
            config.write(configfile)

        # # Close the window before running the conversion
        # window.destroy()

        start_time = timer()
        from_DICOM_to_SVS(path_to_folder, zip_option, label_option, macro_option)
        elapsed_time = timer() - start_time

        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        messagebox.showinfo("Success", f"Conversion completed in {elapsed_time:.2f} seconds")
        # sys.exit()  # exit python once done

    # Create the main window
    # window = tk.Tk()
    window = tk.Toplevel()
    window.title("DICOM to SVS Converter")

    tk.Label(window, text="Path to folder:").grid(row=0, column=0, padx=10, pady=10)
    path_to_folder_var = tk.StringVar(value=path_to_folder)
    path_entry = tk.Entry(window, textvariable=path_to_folder_var, width=50)
    path_entry.grid(row=0, column=1, padx=10, pady=10)
    tk.Button(window, text="Browse...", command=select_folder).grid(row=0, column=2, padx=10, pady=10)

    is_zipped_var = tk.BooleanVar(value=is_zipped.lower() == 'true')
    label_var = tk.BooleanVar(value=is_label.lower() == 'true')
    macro_var = tk.BooleanVar(value=is_macro.lower() == 'true')

    tk.Checkbutton(window, text="To check if files are zipped", variable=is_zipped_var, onvalue=True,
                   offvalue=False).grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=5)
    tk.Checkbutton(window, text="Add label image if exists", variable=label_var, onvalue=True, offvalue=False).grid(
        row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5)
    tk.Checkbutton(window, text="Add overview/macro image if exists", variable=macro_var, onvalue=True,
                   offvalue=False).grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    # Adding the warning label
    warning_text = (
        "Warning: Whole slide images are large files. Ensure your disk has enough space: at least as much as the original files for uncompressed DICOMs, or twice as much for .zip files.")
    tk.Label(window, text=warning_text, wraplength=400).grid(row=5, column=0, columnspan=3, padx=10, pady=10)

    tk.Button(window, text="Convert", command=run_conversion).grid(row=6, column=0, columnspan=3, pady=20)

    window.mainloop()


if __name__ == '__main__':
    main()

# Bertrand Chauveau
# August 2024
# University of Bordeaux
# Revised by Alexandre KOROVIN, 14 Nov 2024