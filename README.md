# DICOMtoSVS
To convert DICOM whole slide images into SVS-like TIFF pyramidal images

This work is currently under review at a scientific journal. A link to the published article will be displayed in due time.

Pathology Departments are encouraged to use Digital Imaging and Communication in Medicine (DICOM), a standardized nonproprietary format, for their workflow of whole slide images (WSI), like radiologists before them. While this allows for a universal workflow in routine with WSI from scanners of various manufacturers, as of April 2024, the DICOM format for WSI remains recent and not supported by many collaborative softwares for diagnostic imaging (e.g. TeleSlide) and many free commercial softwares (e.g. Aperio ImageScope, NDP.view2). This limits the actual usability of these DICOM WSI for collaborative diagnosis, research and teaching purposes. Open source tools recently added support for DICOM WSI (Bio-Formats in 2021 and OpenSlide in 2024), but implemented solutions (like in the QuPath software) do not support the International Color Consortium (ICC) profile, which can result in dull color rendering.

If the DICOM format is to be the future reference WSI format, as in Radiology, commercial software will add support for this format. In the meantime, one of the solution is to convert the DICOM WSI into a more common WSI file format. Here, we use a pyramidal TIFF organization in the same way of SVS files. SVS files are actual TIFF files, with no proprietary extensions, and is the WSI file format of Aperio (Leica Biosystems). Proper solutions with thorough explanations about the conversion of DICOM to SVS-like TIFF is sparse, and dealing with the ICC profile is, to our knowledge, lacking.

Here, we propose a python solution to convert DICOM WSI into SVS-like TIFF pyramidal images. These slides can then be opened with common software supporting the SVS format (Aperio ImageScope, QuPath, TeleSlide, ...). This conversion mainly relies on OpenSlide for reading DICOM slides and tifffile for writing the SVS-like file.

The code is provided in 2 ways: 
- a colab-compatible jupyter notebook
- a python script to be run through Command Line Interface

The code was tested using a Windows operating system with WSL2.
The dependencies are listed in the requirements.txt file used for creating the virtual environment.

Using a 13th Gen Intel(R) Core(TM) i7-13700K with 32Gb of RAM, the required time to convert a 1.5 Gb DICOM WSI into a SVS-like is around 1 minute.

The ICC profile, when embedded in the original DICOM file, can be either embedded in the SVS-like file (can be read with viewer supporting ICC profiles such as Aperio ImageScope) or directly applied during conversion and thus writing ready-to-use pixels (allowing an optimized color rendering whatever the software used for reading. The ICC profile will not be embedded in this case). It should be expected an increase of 50% of required time for conversion when applying the ICC profile. 
Label and macro images, when present in the original DICOM file, can either be removed or retained during conversion.

Using CLI, the arguments are:
- path_to_folder : string, path to the folder containing DICOM files
- --zip : boolean, if the original files are zipped, default=True
- --ICC : boolean, apply the ICC profile when writing, default=False (will then be embedded if exists)
- --multithreading : boolean, whether multithreading must be used, default=True
- --label : boolean, add label image if exists, default=True
- --macro : boolean, add macro image if exists, default=True

Example usage:
- unzipping files, embedding the ICC profile, using multithreading and adding label and macro images:
```python /path_to/DICOMtoSVS.py /path_to_folder/DCM_zip```

- unzipping files, applying the ICC profile, using multithreading and adding label and macro images:
```python /path_to/DICOMtoSVS.py /path_to_folder/DCM_zip --ICC --no-label --no-macro```

![Figure 1_github](https://github.com/bertrandchauveau/DICOMtoSVS/assets/110421330/366ac44e-8521-4338-a475-c829192bf125)
