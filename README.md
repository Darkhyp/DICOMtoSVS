# DICOMtoSVS
To convert brightfield DICOM whole slide images into SVS-like TIFF pyramidal images

This work is currently under review at a scientific journal. A link to the published article will be displayed in due time.

![Figure 1_github](https://github.com/bertrandchauveau/DICOMtoSVS/assets/110421330/366ac44e-8521-4338-a475-c829192bf125)

Pathology Departments are encouraged to use Digital Imaging and Communication in Medicine (DICOM), a standardized format, for their workflow of whole slide images (WSI), like radiologists before them. While this allows for a secure and universal workflow in routine with WSI from scanners of various manufacturers, as of April 2024, DICOM adoption for WSI remains emerging and DICOM is not supported by some collaborative softwares for diagnostic imaging (e.g. TeleSlide) and many free commercial softwares (e.g. Aperio ImageScope, NDP.view2). This limits the actual usability of these DICOM WSI for collaborative diagnosis, research and teaching purposes. Open source tools recently added support for DICOM WSI (Bio-Formats in 2021 and OpenSlide in October 2023), but implemented solutions (like in the QuPath software) do not support the International Color Consortium (ICC) profile, which can result in dull color rendering.

If the DICOM format is to be the future reference WSI format, as in Radiology, commercial software will add support for this format. In the meantime, one of the solution is to convert the DICOM WSI into a more common WSI file format. Here, we use an SVS-like pyramidal TIFF organization. SVS files are actual TIFF files, with no proprietary extensions, and is the WSI file format of Aperio (Leica Biosystems). Proper solutions with thorough explanations about the conversion of DICOM to SVS-like TIFF is scarce, and dealing with the ICC profile is, to our knowledge, lacking.

Here, we propose a python solution to convert DICOM WSI into SVS-like TIFF pyramidal images. These slides can then be opened by common software supporting the SVS format (Aperio ImageScope, QuPath, TeleSlide, ...). This conversion mainly relies on OpenSlide for reading DICOM slides and tifffile for writing the SVS-like file.

The code is provided in 2 ways: 
- a colab-compatible jupyter notebook
- a python script to be run through Command Line Interface
<a target="_blank" href="https://colab.research.google.com/github/bertrandchauveau/DICOMtoSVS/blob/main/DICOM_to_SVS.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The code was tested using a Windows operating system with WSL2.
The main dependencies are:
- openslide (> 4.0.0) and openslide-python
- imagecodecs
- tifffile
- scikit-image
- natsort
- numpy
- pillow

Using a 13th Gen Intel(R) Core(TM) i7-13700K with 32Gb of RAM, the required time to convert a 800 Mb DICOM WSI into a SVS-like is around 42 seconds. Please note that the Colab implementation is much slower, especially without a high CPU RAM environment. Using a local machine must be favored with an optimized max_workers argument.

The ICC profile, when embedded in the original DICOM file, can be either embedded in the SVS-like file (can be read with viewer supporting ICC profiles such as Aperio ImageScope) or directly applied during conversion and thus writing ready-to-use pixels (allowing an optimized color rendering whatever the software used for reading; the ICC profile will not be embedded in this case). It should be expected an mean increase of 15% of required time for conversion when applying the ICC profile (up to 50%). 
Label and macro images, when present in the original DICOM file, can either be removed or retained during conversion.

Using CLI, the arguments are:
- path_to_folder : string, path to the folder containing DICOM files
- --zip : boolean, if the original files are zipped, default=True
- --ICC : boolean, apply the ICC profile when writing, default=False (will then be embedded if exists)
- --multithreading : boolean, whether multithreading must be used, default=True
- --max_workers : integer, the number of max_workers for multithreading, default=12
- --label : boolean, add label image if exists, default=True
- --macro : boolean, add macro image if exists, default=True

Example usage:
- unzipping files, embedding the ICC profile, using multithreading and adding label and macro images:
  
```python /path_to/DICOMtoSVS.py /path_to_folder/DCM_zip```

- unzipping files, applying the ICC profile, using multithreading and adding label and macro images:

```python /path_to/DICOMtoSVS.py /path_to_folder/DCM_zip --ICC --no-label --no-macro```
