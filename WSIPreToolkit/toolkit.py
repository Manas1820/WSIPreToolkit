import cv2
from skimage import morphology
from PIL import Image, ImageFile
import openslide
import numpy as np
import logging
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


class WSIPreToolkit:
    @staticmethod
    def get_array_disk_size(array):
        """
        Returns the size in megabytes (MB) of a NumPy array on disk.

        Args:
            array (numpy.ndarray): NumPy array.

        Returns:
            float: Size of the array in megabytes (MB).

        Raises:
            None

        """

        array_size_bytes = array.size * array.itemsize
        array_size_mb = array_size_bytes / (1024 * 1024)  # Convert bytes to megabytes

        return array_size_mb

    @staticmethod
    def apply_otsu_threshold(channel, apply_gaussian_blur=True):
        """
        Applies Otsu's thresholding to a grayscale image channel.

        Args:
            channel (numpy.ndarray): Grayscale image channel.
            apply_gaussian_blur (bool, optional): Whether to apply Gaussian blur before thresholding.
                Defaults to True.

        Returns:
            numpy.ndarray: Thresholded binary image.

        Raises:
            None

        """

        if apply_gaussian_blur:
            blurred_channel = cv2.GaussianBlur(channel, (5, 5), 0)
        else:
            blurred_channel = channel

        flattened_channel = blurred_channel.reshape(
            (blurred_channel.shape[0], blurred_channel.shape[1])
        )

        _, thresholded_image = cv2.threshold(
            flattened_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return thresholded_image

    def detect_tissue(self, image_location, sensitivity=3000, downsampling_factor=4):
        """
        Detects tissue regions in a whole slide image (WSI).

        Args:
            image_location (str): Path to the whole slide image (WSI) file.
            sensitivity (int, optional): Sensitivity parameter for removing small holes and objects in the tissue mask.
                Defaults to 3000.
            downsampling_factor (int, optional): Desired downsampling factor to obtain a smaller version of the WSI.
                Defaults to 4.

        Returns:
            tuple: A tuple containing the following elements:
                - mask_contours (list): List of contours outlining the detected tissue regions.
                - tier (numpy.ndarray): The computed tier array from the mask.
                - slide (numpy.ndarray): The RGB image of the downsampled WSI.
                - downsampling_factor (int): The actual downsampling factor used.

        Raises:
            openslide.OpenSlideUnsupportedFormatError: If the format of the WSI is not supported.

        """

        try:
            wsi = openslide.OpenSlide(image_location)

            # Get a downsample of the whole slide image (to fit in memory)
            downsampling_factor = min(
                wsi.level_downsamples, key=lambda x: abs(x - downsampling_factor)
            )
            level = wsi.level_downsamples.index(downsampling_factor)
            logging.info(f"Downsampling level: {level}")

            slide = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
            slide = np.array(slide)[:, :, :3]

            # Convert from RGB to HSV color space
            slide_hsv = cv2.cvtColor(slide, cv2.COLOR_BGR2HSV)

            # Compute optimal threshold values in each channel using Otsu algorithm
            _, saturation, _ = np.split(slide_hsv, 3, axis=2)

            mask = self.apply_otsu_threshold(saturation, apply_gaussian_blur=True)

            # Make mask boolean
            mask = mask != 0

            mask = morphology.remove_small_holes(mask, area_threshold=sensitivity)
            mask = morphology.remove_small_objects(mask, min_size=sensitivity)

            mask = mask.astype(np.uint8)
            _, mask_contours, tier = cv2.findContours(
                mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            return mask_contours, tier, slide, downsampling_factor

        except openslide.OpenSlideUnsupportedFormatError as e:
            logging.error(f"Unsupported format for image: {image_location}")
            raise e

    @staticmethod
    def draw_tissue_polygons(mask, polygons, polygon_type, line_thickness=None):
        """
        Draws tissue polygons on a mask image.

        Args:
            mask (numpy.ndarray): Input mask image.
            polygons (list): List of polygons to be drawn.
            polygon_type (str): Type of polygons. Accepted values are 'line' or 'area'.
            line_thickness (int, optional): Thickness of the lines for 'line' type polygons.
                Only used if polygon_type is 'line'. Defaults to None.

        Returns:
            numpy.ndarray: The updated mask image with tissue polygons drawn.

        Raises:
            ValueError: If the polygon_type is not 'line' or 'area'.

        """

        tissue_color = 1
        for poly in polygons:
            if polygon_type == "line":
                mask = cv2.polylines(mask, [poly], True, tissue_color, line_thickness)
            elif polygon_type == "area":
                if line_thickness is not None:
                    warnings.warn(
                        '"line_thickness" is only used if "polygon_type" is "line".'
                    )
                mask = cv2.fillPoly(mask, [poly], tissue_color)
            else:
                raise ValueError('Accepted "polygon_type" values are "line" or "area".')

        return mask

    @staticmethod
    def get_sub_image(rectangle, source_image):
        """
        Extracts a sub-image from the source image based on the specified rectangle.

        Args:
            rectangle (tuple): A tuple containing the rectangle information in the format (center, size, angle),
                where 'center' is the (x, y) coordinates of the rectangle center, 'size' is the (width, height) of the rectangle,
                and 'angle' is the rotation angle in degrees.
            source_image (numpy.ndarray): The source image from which to extract the sub-image.

        Returns:
            numpy.ndarray: The sub-image extracted from the source image based on the specified rectangle.

        """

        width = int(rectangle[1][0])
        height = int(rectangle[1][1])
        box = cv2.boxPoints(rectangle)

        source_points = box.astype("float32")
        destination_points = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )
        transformation_matrix = cv2.getPerspectiveTransform(
            source_points, destination_points
        )
        warped = cv2.warpPerspective(
            source_image, transformation_matrix, (width, height)
        )
        return warped

    @staticmethod
    def tissue_cutout(tissue_slide, tissue_contours, slide):
        """
        Extracts the tissue region from a slide image based on tissue contours.

        Args:
            tissue_slide (numpy.ndarray): The slide image containing the tissue region.
            tissue_contours (list): List of tissue contours.
            slide (numpy.ndarray): The original slide image.

        Returns:
            numpy.ndarray: The extracted tissue region.

        """

        # Create a mask where white is what we want to keep
        crop_mask = np.zeros_like(tissue_slide)

        # Draw filled contours in the mask
        cv2.drawContours(crop_mask, tissue_contours, -1, 255, -1)

        # Extract the tissue region from the slide image
        tissue_only = np.zeros_like(slide)
        tissue_only[crop_mask == 255] = slide[crop_mask == 255]

        return tissue_only

    def detect_and_crop(
        self,
        image_location: str,
        sensitivity: int = 1500,
        downsample_rate: int = 4,
        show_plots: str = "simple",
    ):
        """
        Detects tissue in an image, crops it, and removes empty space.

        Args:
            image_location (str): The path or location of the image file.
            sensitivity (int): The sensitivity threshold for removing small holes and objects (default: 1500).
            downsample_rate (int): The downsample rate for the initial processing (default: 4).
            show_plots (str): Determines the type of plots to display. Options are "simple", "verbose", "none" (default: "simple").

        Returns:
            numpy.ndarray or None: The cropped image with removed empty space, or None if no tissue contours were found.

        """

        # Set up dictionary for verbose plotting
        verbose_plots = {}

        # Open Slide
        try:
            wsi = openslide.OpenSlide(str(image_location))

            # Detect tissue
            (
                tissue_contours,
                tier,
                downsampled_slide,
                downsampling_factor,
            ) = self.detect_tissue(image_location, sensitivity, downsample_rate)
            verbose_plots[
                f"Base Slide\n{self.get_array_disk_size(downsampled_slide):.2f}MB"
            ] = downsampled_slide

            # Get tissue-only slide
            base_slide_mask = np.zeros(downsampled_slide.shape[:2])
            tissue_slide = self.draw_tissue_polygons(
                base_slide_mask, tissue_contours, "line", 5
            )
            tissue_only_slide = self.tissue_cutout(
                tissue_slide, tissue_contours, downsampled_slide
            )
            verbose_plots[f"Tissue Detect\nNo Change"] = tissue_slide

            # Get minimal bounding rectangle for all tissue contours
            if len(tissue_contours) == 0:
                img_id = image_location.split("/")[-1]
                print(f"No Tissue Contours - ID: {img_id}")
                return None, 1.0

            all_bounding_rect = cv2.minAreaRect(np.concatenate(tissue_contours))

            # Crop with getSubImage()
            smart_bounding_crop = self.get_sub_image(
                all_bounding_rect, tissue_only_slide
            )
            verbose_plots[
                f"Bounding Boxes\n{self.get_array_disk_size(smart_bounding_crop):.2f}MB"
            ] = smart_bounding_crop

            # Crop empty space
            # Remove by row
            row_not_blank = [
                row.all() for row in ~np.all(smart_bounding_crop == [255, 0, 0], axis=1)
            ]
            space_cut = smart_bounding_crop[row_not_blank, :]

            # Remove by column
            col_not_blank = [
                col.all() for col in ~np.all(smart_bounding_crop == [255, 0, 0], axis=0)
            ]
            space_cut = space_cut[:, col_not_blank]
            verbose_plots[
                f"Space Cut\n{self.get_array_disk_size(space_cut):.2f}MB"
            ] = space_cut

            wsi.close()
            return space_cut

        except openslide.OpenSlideUnsupportedFormatError:
            print("Error: OpenSlideUnsupportedFormatError")

    def generate_patches(
        self,
        image_location,
        patch_size=(224, 224, 3),
        overlap=(112, 112, 0),
        til_score=0,
    ):
        """
        Generates patches from an image with specified patch size and overlap.

        Args:
            image_location (str): The path or location of the image file.
            patch_size (tuple): The size of the patches in (height, width, channels) format (default: (224, 224, 3)).
            overlap (tuple): The amount of overlap between patches in (vertical, horizontal, depth) format (default: (112, 112, 0)).
            til_score (int): TIL score threshold for patch selection (default: 0).

        Returns:
            numpy.ndarray or None: An array containing the generated patches, or None if the image cannot be loaded.

        """

        try:
            # Load image and perform tissue detection and cropping
            cropped_image = self.detect_and_crop(image_location)

            if cropped_image is None:
                print("Skipped image:", image_location)
                return None

            # Calculate the number of rows and columns of patches
            n_rows = (
                int(np.ceil((cropped_image.shape[0] - patch_size[0]) / overlap[0])) + 1
            )
            n_cols = (
                int(np.ceil((cropped_image.shape[1] - patch_size[1]) / overlap[1])) + 1
            )

            # Initialize an empty array to store patches
            patches = np.zeros((n_rows * n_cols, *patch_size))

            # Extract patches with overlap
            idx = 0
            for i in range(n_rows):
                for j in range(n_cols):
                    # Calculate the coordinates of the patch
                    row_start = i * overlap[0]
                    col_start = j * overlap[1]
                    row_end = min(row_start + patch_size[0], cropped_image.shape[0])
                    col_end = min(col_start + patch_size[1], cropped_image.shape[1])

                    # Extract the patch
                    patch = cropped_image[row_start:row_end, col_start:col_end, :]
                    patch_shape = patch.shape

                    if np.sum(patch) < 30:
                        continue

                    # Pad the patch if necessary
                    if patch_shape[0] < patch_size[0]:
                        pad_top = (patch_size[0] - patch_shape[0]) // 2
                        pad_bottom = patch_size[0] - patch_shape[0] - pad_top
                        patch = np.pad(
                            patch,
                            ((pad_top, pad_bottom), (0, 0), (0, 0)),
                            mode="constant",
                        )

                    if patch_shape[1] < patch_size[1]:
                        pad_left = (patch_size[1] - patch_shape[1]) // 2
                        pad_right = patch_size[1] - patch_shape[1] - pad_left
                        patch = np.pad(
                            patch,
                            ((0, 0), (pad_left, pad_right), (0, 0)),
                            mode="constant",
                        )

                    patches[idx] = patch
                    idx += 1

            return patches

        except FileNotFoundError:
            print("Skipped image:", image_location)
            return None
