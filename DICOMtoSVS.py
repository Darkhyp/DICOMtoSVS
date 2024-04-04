from openslide import open_slide
import openslide
import os
import shutil
from PIL import Image
import numpy as np
from PIL import Image
from PIL import ImageCms
import zipfile
from natsort import os_sorted
import imagecodecs
import tifffile
from concurrent.futures import ThreadPoolExecutor
from skimage.transform import resize
import argparse

def get_biggest_file(directory_path):
  biggest_file = (None, 0)  # Initialize with None path and 0 size
  WSI_files = [f for f in os.listdir(directory_path) if f.endswith(".dcm")]
  for file_path in WSI_files:
    if os.path.isfile(directory_path+ '/' + file_path):  # Check if it's a valid file
      size = os.path.getsize(directory_path+ '/' + file_path)
      size_in_mb = round(size / 1048576,2)  # Convert bytes to Megabytes
      if size_in_mb > biggest_file[1]:
        biggest_file = (directory_path+ '/' + file_path, size_in_mb)
  return biggest_file


def from_DICOM_to_SVS(path_to_folder, zip=True, ICC=False, multithreading=True, max_workers=12, label=True, macro=True):
    '''
    Parameters:
        path_to_folder (str): the path to the folder containing the DICOM WSI files
        zip (bool): True if the files need to be unzipped, False otherwise. It is assumed that all files will be either zipped or not. Default to True.
        ICC (bool): whether to apply the ICC profile when writing the file (True, to obtain 'ready to use' pixels), or to simply embed it if it exists (False). Default to False?
        multithreading (bool): whether to use multithreading or not. Default to True.
        max_workers (int): number of workers for multithreading. Default to 12.
        label (bool): whether to add the label image if it exists. Default to True.
        macro (bool): whether to add the macro image if it exists. Default to True.
    '''

    #list of all files +/- unzip them
    if zip==True:
        #list of zipped files
        WSI_list = [f for f in os.listdir(path_to_folder) if f.endswith(".zip")]
        print('Unzipping files...')
        #unzip
        path_unzip = path_to_folder+'_unzip'
        if not os.path.exists(path_unzip):
            os.mkdir(path_unzip)
        for WSI in WSI_list:
            with zipfile.ZipFile(path_to_folder + '/' + WSI, 'r') as zip_ref:
                zip_ref.extractall(path_unzip + '/' + WSI[:-4])  #same name, just without the .zip extension
        #list of folder of DICOM images. One folder per image
        WSI_dir = [f for f in os.listdir(path_unzip) ]
        WSI_dir = os_sorted(WSI_dir)
    else: #no zipped files, list all image folder
        WSI_dir = [f for f in os.listdir(path_to_folder)]
        WSI_dir = os_sorted(WSI_dir)
        path_unzip = path_to_folder
    print(f'Number of identified WSI is: {len(WSI_dir)}')
    #define output_path
    path_output = path_to_folder+'_output'
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    #loop through each WSI
    y=0
    for WSI_name in WSI_dir:
        print(f'Starting the conversion of {WSI_name}, slide number {y+1} out of {len(WSI_dir)}')
        #list all files of the DICOM folder
        WSI_files = [f for f in os.listdir(path_unzip + '/' + WSI_name) if f.endswith(".dcm")]  #we ignore .dcm.import
        biggest_file_path, biggest_file_size = get_biggest_file(path_unzip +'/' + WSI_name)  #possibly not required but to ensure to use the baseline/full resolution image

        #open the image with OpenSlide and get some image properties
        wsi = openslide.OpenSlide(biggest_file_path)
        WSI_shape = wsi.dimensions
        #level_count = wsi.level_count
        #level_downsamples= wsi.level_downsamples
        properties = wsi.properties
        #scanner = properties['dicom.ManufacturerModelName']
        pixelsize_x = float(properties['openslide.mpp-x'])  #µm/pixel, str => float()
        pixelsize_y = float(properties['openslide.mpp-y'])  #µm/pixel, str => float()
        pixel_size = round((pixelsize_x + pixelsize_y)/2, 6)
        if 'openslide.objective-power' in properties:
            obj_power = int(properties['openslide.objective-power'])
        else: #if objective power is not specified, get a rough estimate from the pixel_size between objective 20 and 40
            obj_power = 40 if pixelsize_x < 0.3 else 20
        icc_profile = wsi.color_profile  #the ICC profile if it exists
        if icc_profile is not None:  #if an ICC profile exists, capture it in bytes
            tile = wsi.read_region((0,0),0,(100,100)).convert("RGB")
            icc_content_bytes = tile.info['icc_profile']
        if pixelsize_x==pixelsize_y:
            pass
        else:
            print(f'image resolution is DIFFERENT BETWEEN x and y axes for WSI {WSI_name}')
        if ICC==True:
            #create a profile for RGB
            rgbp=ImageCms.createProfile("sRGB")
            icc2rgb = ImageCms.buildTransformFromOpenProfiles(icc_profile,rgbp, "RGB", "RGB")

        # Define pyramid levels
        pyramid_levels = [1, 4, 16, 64]  # Adjust levels as needed. WSI from Aperio scanners usually use these levels.
        if WSI_shape[0] <= 65536 or WSI_shape[1] <= 65536:  #empirically based from some Aperio WSI examples
            pyramid_levels = [1, 4, 16]

        WSI_shape_level1 = (int(WSI_shape[1] / 4), int(WSI_shape[0] / 4), 3)  #height, width, channel  #integral multiple of initial resolution
        WSI_shape_level2 = (int(WSI_shape[1] / 16), int(WSI_shape[0] / 16), 3)  #height, width, channel
        WSI_shape_level3 = (int(WSI_shape[1] / 64), int(WSI_shape[0] / 64), 3)  #height, width, channel

        #define the image description tag, which contains important information such as resolution and compression arguments
        image_description_base = f'Aperio Leica Biosystems GT450 DX v1.2.0 (fake) \n{WSI_shape[0]}x{WSI_shape[1]} [0,0,{WSI_shape[0]}x{WSI_shape[1]}] (256x256) JPEG/YCC Q=91|AppMag = {obj_power}|MPP = {pixel_size}|ScanScope ID = SS45371|ScannerType = GT450 DX|SessionMode = PDX|'
        image_description_level1 = f'Aperio Leica Biosystems GT450 DX v1.2.0 (fake) \n{WSI_shape_level1[1]}x{WSI_shape_level1[0]} [0,0,{WSI_shape_level1[1]}x{WSI_shape_level1[0]}] (256x256) JPEG/YCC Q=91|AppMag = {obj_power}|MPP = {round(pixel_size*4, 6)}|ScanScope ID = SS45371|ScannerType = GT450 DX|SessionMode = PDX|'
        image_description_level2 = f'Aperio Leica Biosystems GT450 DX v1.2.0 (fake) \n{WSI_shape_level3[1]}x{WSI_shape_level3[0]} [0,0,{WSI_shape_level3[1]}x{WSI_shape_level3[0]}] (256x256) JPEG/YCC Q=91|AppMag = {obj_power}|MPP = {round(pixel_size*16, 6)}|ScanScope ID = SS45371|ScannerType = GT450 DX|SessionMode = PDX|'
        image_description_level3 = f'Aperio Leica Biosystems GT450 DX v1.2.0 (fake) \n{WSI_shape_level3[1]}x{WSI_shape_level3[0]} [0,0,{WSI_shape_level3[1]}x{WSI_shape_level3[0]}] (256x256) JPEG/YCC Q=91|AppMag = {obj_power}|MPP = {round(pixel_size*64, 6)}|ScanScope ID = SS45371|ScannerType = GT450 DX|SessionMode = PDX|'
        image_description_level = [None, image_description_level1, image_description_level2, image_description_level3]


        #create the TIFF file
        #due to the size of most WSI, the full resolution image will possibly not fit into RAM.
        #Hence, the full resolution image will be written through tiling, while the subresolution level 1 (downsample factor 4) will be created as a numpy array that will fit in most RAM
        #with multithreading
        def create_tiles_coord(WSI_dimension, tile_size):
            result_tiles = []
            # Iterate through the mask and extract non-overlapping tiles with positive pixels
            for col in range(0, WSI_dimension[1], tile_size[1]):
                for row in range(0, WSI_dimension[0], tile_size[0]):
                    result_tiles.append((row, col))
            return result_tiles

        if ICC==True and icc_profile is not None:
            def process_tile(tile_coord):
                tile = wsi.read_region(tile_coord, 0, tile_size).convert("RGB")
                #apply ICC profile
                tile = ImageCms.applyTransform(tile, icc2rgb)
                # convert to numpy array and in the meantime create the WSI_array at level 1 (as the full resolution array/base level requires lots of RAM depending on WSI shape)
                tile_array = np.array(tile)
                WSI_array_level1[int(tile_coord[1]/4):int((tile_coord[1]+tile_size[1])/4), int(tile_coord[0]/4):int((tile_coord[0]+tile_size[0])/4), :] = tile_array[::4,::4,:]
                return tile_array

        else: #no ICC to be applied, just to be embedded in the file if it exists
            def process_tile(tile_coord):
                tile = wsi.read_region(tile_coord, 0, tile_size).convert("RGB")
                # convert to numpy array and in the meantime create the WSI_array at level 1 (as the full resolution array/base level requires lots of RAM depending on WSI shape)
                tile_array = np.array(tile)
                WSI_array_level1[int(tile_coord[1]/4):int((tile_coord[1]+tile_size[1])/4), int(tile_coord[0]/4):int((tile_coord[0]+tile_size[0])/4), :] = tile_array[::4,::4,:]
                return tile_array

        tile_size = (256,256)
        tiles_coord = create_tiles_coord(WSI_shape, tile_size=tile_size)
        num_tiles = len(tiles_coord)
        #initialize the level 1 numpy array of the WSI
        WSI_array_level1 = np.zeros((round(WSI_shape[1]/4)+tile_size[1], round(WSI_shape[0]/4)+tile_size[0],3), dtype=np.uint8)

        # Function to process tiles concurrently
        if multithreading==True:
            def generate_tiles():
                with ThreadPoolExecutor(max_workers=max_workers) as executor:  #adjust here the number depending on your configuration
                    for tile_array in executor.map(process_tile, tiles_coord):
                        yield tile_array
        else: #no multithreading
            if ICC==True and icc_profile is not None:
                def generate_tiles():
                    for tile_coord in tiles_coord:
                        tile = wsi.read_region(tile_coord, 0, tile_size).convert("RGB")
                        tile = ImageCms.applyTransform(tile, icc2rgb)
                        tile_array = np.array(tile)
                        WSI_array_level1[int(tile_coord[1]/4):int((tile_coord[1]+tile_size[1])/4), int(tile_coord[0]/4):int((tile_coord[0]+tile_size[0])/4), :] = tile_array[::4,::4,:]
                        yield tile_array
            else: #no ICC to be applied, but will be embedded if it exists
                def generate_tiles():
                    for tile_coord in tiles_coord:
                        tile = wsi.read_region(tile_coord, 0, tile_size).convert("RGB")
                        #tile = ImageCms.applyTransform(tile, icc2rgb)
                        tile_array = np.array(tile)
                        WSI_array_level1[int(tile_coord[1]/4):int((tile_coord[1]+tile_size[1])/4), int(tile_coord[0]/4):int((tile_coord[0]+tile_size[0])/4), :] = tile_array[::4,::4,:]
                        yield tile_array

        # Write tiles to TIFF file
        if icc_profile is not None and ICC!=True: #if an ICC profile exists for this slide and it is not intended to apply it, then embed it
            extratag = [(34675, 7, None, icc_content_bytes, True)]  #ICC extratag
        else:
            extratag = None  #no extratag
        if ICC==True and icc_profile is None:
            print(f'No detected ICC profile for slide named {WSI_name}')
        with tifffile.TiffWriter(path_output + '/' + WSI_name + '.tiff', shaped=False, bigtiff=True) as tif:   #all WSI from the Aperio GT450 DX seem to be BigTIFF, whatever the file size
            #write the full resolution image
            tif.write(data=generate_tiles(),
                      dtype='uint8',
                      shape=(WSI_shape[1], WSI_shape[0], 3),
                      subfiletype=0,
                      resolutionunit='CENTIMETER',
                      resolution=(1e4 / pixelsize_x, 1e4 / pixelsize_y),  #1e-4 because resolution is in centimeter #Number of pixels per `resolutionunit` in X and Y directions
                      photometric='rgb',  #will be automatically converted to YCbCr
                      compression='jpeg',
                      compressionargs={'level':91},  #the quality parameter found in WSI from the Aperio GT450 DX
                      tile=(256,256),
                      description=image_description_base,
                      extratags=extratag)

            #resize WSI_array_level1 to target_size WSI_shape_level1  #because if WSI_shape is not an integral multiple of tile_size /4, additional pixels are present and will create image shifts when viewing
            WSI_array_level1 = WSI_array_level1[0:WSI_shape_level1[0], 0:WSI_shape_level1[1], :]  #a[::level, ::level, :] returns one additional value if shape is not an integral multiple of level

            # add a thumbnail image as a separate series, in second position
            thumbnail = (WSI_array_level1[::4, ::4]).astype('uint8')  #to obtain a reduced image to rezise it after
            # Calculate the aspect ratio of the image
            aspect_ratio = thumbnail.shape[1] / thumbnail.shape[0]
            # resize to a target width
            target_width=1920  #the default width with the Aperio GT450 DX
            new_height = round(1920 / aspect_ratio)
            # Resize the thumbnail while preserving the aspect ratio
            mpp_thumbnail = round(WSI_shape[0]/target_width,6)
            thumbnail = resize(thumbnail, (new_height, 1920), anti_aliasing=True)*255
            thumbnail= thumbnail.astype('uint8')
            image_description_thumbnail = f'Aperio Leica Biosystems GT450 DX v1.2.0 \n{target_width}x{new_height} [0,0,{target_width}x{new_height}] (256x256) JPEG/YCC Q=100|AppMag = 40|MPP = {mpp_thumbnail}|ScanScope ID = SS45371|ScannerType = GT450 DX|SessionMode = PDX|'

            tif.write(thumbnail,
                      subfiletype=0,
                      photometric='rgb',
                      compression='jpeg',
                      compressionargs={'level':100},  #atypical but is what was found in the image description of SVS files
                      description=image_description_thumbnail,
                      extratags=extratag)  #no tiling, the image must be stripped

            # subresolutions images
            #WSI_array_level1  #already defined when writing the full resolution image
            #level2
            WSI_array_level2 = WSI_array_level1[::4, ::4,:]  #4*4 = 16
            WSI_array_level2 = WSI_array_level2[0:WSI_shape_level2[0], 0:WSI_shape_level2[1], :]
            #level3
            WSI_array_level3 = WSI_array_level1[::16, ::16,:]  #16*4 = 64
            WSI_array_level3 = WSI_array_level3[0:WSI_shape_level3[0], 0:WSI_shape_level3[1], :]

            #assert WSI_array_level1.shape==WSI_shape_level1 and WSI_array_level2.shape==WSI_shape_level2 and WSI_array_level3.shape==WSI_shape_level3

            WSI_array_level = [None, WSI_array_level1, WSI_array_level2, WSI_array_level3]
            for i in range(1, len(pyramid_levels[1:])+1):
                tif.write(WSI_array_level[i],
                        subfiletype=0,
                        resolutionunit='CENTIMETER',
                        resolution=(1e4 / pyramid_levels[i] / pixelsize_x, 1e4 / pyramid_levels[i] /pixelsize_y),
                        photometric='rgb',
                        compression='jpeg',
                        compressionargs={'level':91},
                        tile=(256,256),
                        description=image_description_level[i],
                        extratags=extratag)

            #label and macro images, if these images exist
            try:
                associated_images = wsi.associated_images

                # Access specific images using dictionary keys (if present)
                label_pil = associated_images.get('label')
                macro_pil = associated_images.get('macro')
            except openslide.OpenSlideError:
                # Handle the case where no associated images are present
                print("No associated images found.")
            except KeyError as e:
                # Handle cases where specific keys ('label' or 'macro') are missing
                missing_key = e.args[0]
                print(f"Image '{missing_key}' not found in associated images.")
            # If images are found, access and process them
            if label==True and 'label_pil' in locals():
                #convert to numpy array
                label_array = np.array(label_pil)[:,:,0:3] #RGBA=> RGB
                label_shape = label_array.shape
                image_description_label = f'Aperio Leica Biosystems GT450 DX v1.2.0 \nlabel {label_shape[1]}x{label_shape[0]}'
                tif.write(label_array,
                          subfiletype=1,  #reduced type of image
                          photometric='rgb',
                          compression='lzw',  #compression is not jpeg for the label image
                          description=image_description_label,
                          predictor=2,  #horizontal
                          extratags=extratag)  #no tiling, the image must be stripped

            if macro==True and 'macro_pil' in locals():
                macro_array = np.array(macro_pil)[:,:,0:3]  #RGBA=> RGB
                macro_shape = macro_array.shape
                image_description_macro = f'Aperio Leica Biosystems GT450 DX v1.2.0 \nmacro {macro_shape[1]}x{macro_shape[0]}'
                tif.write(macro_array,
                          subfiletype=9,  #macro/reduced
                          photometric='rgb',
                          compression='jpeg',
                          compressionargs={'level':95, 'outcolorspace': 'rgb'},  #needed to specifically ask for RGB and not YCbCr
                          description=image_description_macro,
                          extratags=extratag)  #no tiling, the image must be stripped

        #rename file extension from .tiff to .svs #not required to read the image in Aperio, but for most other softwares
        os.rename(path_output + '/' + WSI_name + '.tiff', path_output + '/' + WSI_name + '.svs')
        print(f'Pyramidal TIFF image saved as: {WSI_name}.svs')
        y+=1
    #delete the intermediate directory (unzipped folder)
    if zip==True:
        shutil.rmtree(path_to_folder+'_unzip')
    print('All done')

def main():
    parser = argparse.ArgumentParser(description='Convert DICOM WSI files to SVS format.')
    parser.add_argument('path_to_folder', type=str, help='Path to the folder containing DICOM files')
    parser.add_argument('--zip', default=True, action=argparse.BooleanOptionalAction, help='Unzip files if needed (default: True)')
    parser.add_argument('--ICC', default=False, action=argparse.BooleanOptionalAction, help='Apply ICC profile when writing (default: False)')
    parser.add_argument('--multithreading', default=True, action=argparse.BooleanOptionalAction, help='Use multithreading (default: True)')
    parser.add_argument('--max_workers', type=int, default=12, help='Maximum number of concurrent threads (default: 12)')   
    parser.add_argument('--label', default=True, action=argparse.BooleanOptionalAction, help='Add label image if exists (default: True)')
    parser.add_argument('--macro', default=True, action=argparse.BooleanOptionalAction, help='Add macro image if exists (default: True)')

    args = parser.parse_args()

    # Call your function with parsed arguments
    from_DICOM_to_SVS(path_to_folder = args.path_to_folder,
                      zip = args.zip,
                      ICC = args.ICC,
                      multithreading = args.multithreading,
                      max_workers = args.max_workers,
                      label = args.label,
                      macro = args.macro)

if __name__ == '__main__':
    main()
