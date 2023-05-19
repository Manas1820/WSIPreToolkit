# WSIPreToolkit Documentation

The `WSIPreToolkit` is a Python package that provides various tools for working with tissue images, particularly whole slide images (WSIs). It offers functionality for detecting tissue regions, cropping images, generating patches, and more. This document provides an overview of the package and its usage.

## Installation

To install the `WSIPreToolkit` package, you can use `pip`:

```bash
pip install WSIPreToolkit
```

## Usage

### Importing the Package

To use the `WSIPreToolkit` package, you need to import it in your Python script or notebook:

```python
from WSIPreToolkit import WSIPreToolkit
```

### Tissue Detection

The `WSIPreToolkit` class provides a method `detect_tissue` for detecting tissue regions in a whole slide image (WSI). It takes the path to the WSI file as input and returns the contours outlining the detected tissue regions, the computed tier array from the mask, the RGB image of the downsampled WSI, and the actual downsampling factor used. Here's an example:

```python
toolkit = WSIPreToolkit()
mask_contours, tier, slide, downsampling_factor = toolkit.detect_tissue(image_location)
```

### Drawing Tissue Polygons

The `WSIPreToolkit` class includes a method `draw_tissue_polygons` for drawing tissue polygons on a mask image. It takes the input mask image, a list of polygons to be drawn, the type of polygons ('line' or 'area'), and an optional line thickness for 'line' type polygons. Here's an example:

```python
mask = toolkit.draw_tissue_polygons(mask, polygons, polygon_type='line', line_thickness=2)
```

### Cropping Tissue Regions

The `WSIPreToolkit` class provides a method `tissue_cutout` for extracting the tissue region from a slide image based on tissue contours. It takes the tissue slide image, the list of tissue contours, and the original slide image as input and returns the extracted tissue region. Here's an example:

```python
tissue_only = toolkit.tissue_cutout(tissue_slide, tissue_contours, slide)
```

### Cropping and Removing Empty Space

The `WSIPreToolkit` class includes a method `detect_and_crop` for detecting tissue in an image, cropping it, and removing empty space. It takes the path or location of the image file as input, along with optional parameters such as sensitivity, downsample rate, and show plot type. It returns the cropped image with removed empty space or None if no tissue contours were found. Here's an example:

```python
cropped_image = toolkit.detect_and_crop(image_location, sensitivity=1500, downsample_rate=4, show_plots='simple')
```

### Generating Patches

The `WSIPreToolkit` class provides a method `generate_patches` for generating patches from an image with specified patch size and overlap. It takes the image location, patch size, overlap, and TIL score threshold as input and returns an array containing the generated patches or None if the image cannot be loaded. Here's an example:

```python
patches = toolkit.generate_patches(image_location, patch_size=(224, 224, 3), overlap=(112, 112, 0), til_score=0)
```

## Conclusion

The `WSI_Pre_Toolkit` package offers a range of tools for working with tissue images, including tissue detection, cropping, patch generation, and more. By using this package, you can streamline your workflow when working with tissue images and extract relevant information efficiently.
