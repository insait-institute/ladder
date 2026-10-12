import numpy as np
import PIL.Image
import roma
import torch
import torchvision
from backports.strenum import StrEnum


def rotmat_as_3x3(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert any rotmat input to [..., 3, 3] shape"""
    if rotmat.shape[-1] == 9:
        return rotmat.view(*rotmat.shape[:-1], 3, 3)
    if rotmat.shape[-2:] == torch.Size([3, 3]):
        return rotmat
    raise ValueError(f"Can't convert tensor of shape {rotmat.shape} to a 3x3 rotation matrix")


def rotmat_as_9(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert any rotmat input to [..., 9] shape"""
    if is_rotmat_9(rotmat):
        return rotmat
    if is_rotmat_3x3(rotmat):
        return rotmat.view(*rotmat.shape[:-2], 9)
    raise ValueError(f"Can't convert tensor of shape {rotmat.shape} to a 3x3 rotation matrix")


def is_rotmat_3x3(rotmat: torch.Tensor) -> bool:
    return rotmat.shape[-2:] == torch.Size([3, 3])


def is_rotmat_9(rotmat: torch.Tensor) -> bool:
    return rotmat.shape[-1] == 9


def rotmat_to_euler(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Args:
        rotmat: Batch of rotation matrices, shape [..., 3, 3]
    Returns:
        Batch of Euler angles in radiant, shape [..., 3]
    """
    rotmat = rotmat_as_3x3(rotmat)
    return roma.rotmat_to_euler(convention='xyz', rotmat=rotmat, as_tuple=False, degrees=False)


def euler_to_rotmat(angles: torch.Tensor) -> torch.Tensor:
    """
    Args:
        angles: Euler angles in radians in the format 'xyz', shape [..., 3]
    Returns:
        torch.Tensor of shape [..., 3, 3] containing rotation matrices
    """
    # NOTE:
    #   - We use extrinsic rotations, i.e. the coordinate frame remains fixed, rather than intrinsic
    #       rotation, where each rotation axis changes direction with the rotation of the previous axes.
    #       Confirmed, this is the meaning used by roma by cross check with scipy documentation and results
    #       https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html
    #   - We assume RPY (roll, pitch, yaw) order of angles. This corresponds to 'xyz' convention
    return roma.euler_to_rotmat(convention='xyz', angles=angles, degrees=False)


class ResizeMode(StrEnum):
    """
    Different modes for resizing images.
    """

    MATCH_WIDTH = 'match_width'  # Make width match the target width, keep aspect ratio
    MATCH_HEIGHT = 'match_height'  # Make height match the target height, keep aspect ratio
    MATCH_MAX = 'match_max'  # Make max(height, width) match the target size, keep aspect ratio
    NAIVE = 'naive'  # Resize naively, don't keep aspect ratio
    SMART = 'smart'  # Resize 'smart', using cropping and padding
    PAD = 'pad'  # Resize image by padding
    CROP = 'crop'  # Resize image by cropping


def assert_np_hwc_or_hw_image(image: np.ndarray | PIL.Image.Image) -> np.ndarray:
    """Make sure image is of type np.ndarray and HWC format"""
    if isinstance(image, PIL.Image.Image):
        image = np.asarray(image)

    assert isinstance(image, np.ndarray), type(image)
    assert image.ndim in [2, 3], image.shape
    # Check that shape is HWC, this check isn't foolproof
    if image.ndim == 3:
        assert image.shape[-1] <= 4, image.shape

    return image


def crop_image(
    image: np.ndarray | PIL.Image.Image,
    start_height: int,
    start_width: int,
    target_height: int,
    target_width: int,
) -> np.ndarray | PIL.Image.Image:
    np_image = assert_np_hwc_or_hw_image(image)

    height, width = np_image.shape[:2]
    assert target_width <= width, f"Can't crop image of width {width} to {target_width}"
    assert target_height <= height, f"Can't crop image of width {height} to {target_height}"

    start_height, start_width = round(start_height), round(start_width)
    target_height, target_width = round(target_height), round(target_width)

    np_image = np_image[
        start_height : start_height + target_height, start_width : start_width + target_width, ...
    ]

    image = PIL.Image.fromarray(np_image) if isinstance(image, PIL.Image.Image) else np_image

    return image


def crop_image_center(
    image: np.ndarray | PIL.Image.Image,
    target_height: int,
    target_width: int,
) -> np.ndarray | PIL.Image.Image:
    np_image = assert_np_hwc_or_hw_image(image)

    height, width = np_image.shape[:2]
    assert target_width <= width, f"Can't crop image of width {width} to {target_width}"
    assert target_height <= height, f"Can't crop image of width {height} to {target_height}"

    top = (height - target_height) // 2
    left = (width - target_width) // 2

    np_image = crop_image(np_image, top, left, target_height, target_width)

    image = PIL.Image.fromarray(np_image) if isinstance(image, PIL.Image.Image) else np_image

    return image


def pad_image_to_square(
    image: PIL.Image.Image | np.ndarray,
    pad_value: tuple[int, int, int] | tuple[float, float, float] | int | float = 0,
) -> PIL.Image.Image | np.ndarray:
    """Pad image to square by adding a symmetric border around the height/width."""

    assert isinstance(image, (PIL.Image.Image, np.ndarray)), type(image)

    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
    else:
        width, height = image.size

    if width == height:
        return image

    max_wh = max(width, height)

    horizontal_pad, vertical_pad = int((max_wh - width) / 2), int((max_wh - height) / 2)

    if isinstance(image, np.ndarray):
        padding = ((vertical_pad, vertical_pad), (horizontal_pad, horizontal_pad)) + ((0, 0),) * (
            image.ndim - 2
        )
        image = np.pad(image, padding, mode="constant", constant_values=pad_value)
    else:
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        image = torchvision.transforms.v2.functional.pad(
            image, padding=padding, fill=pad_value, padding_mode="constant"
        )

    return image


def make_image_square(
    image: PIL.Image.Image | np.ndarray,
    mode: ResizeMode | str,
    resample: PIL.Image.Resampling = PIL.Image.Resampling.LANCZOS,
    pad_value: tuple[int, int, int] | tuple[float, float, float] | int | float = 0,
) -> PIL.Image.Image | np.ndarray:
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
    else:
        width, height = image.size

    if width == height:
        return image

    if mode == ResizeMode.SMART:
        # If aspect ratio is ~16:9, crop the image to 4:3
        if max(width, height) / min(width, height) > 1.4:  # 4:3 ratio is 1.33; 16:9 ratio is 1.77
            if width > height:
                image = crop_image_center(image, target_height=height, target_width=round(height * (4 / 3)))
            else:
                image = crop_image_center(image, target_height=round(width * (4 / 3)), target_width=width)

        # Pad the image to square
        image = pad_image_to_square(image, pad_value=pad_value)

    elif mode == ResizeMode.PAD:
        image = pad_image_to_square(image, pad_value=pad_value)

    elif mode == ResizeMode.CROP:
        image = crop_image_center(image, target_height=min(height, width), target_width=min(height, width))

    else:
        if isinstance(image, np.ndarray):
            if is_depth_image(image):
                assert resample == PIL.Image.Resampling.BILINEAR
            image = PIL.Image.fromarray(image)
            is_np = True
        else:
            is_np = False

        if mode == ResizeMode.MATCH_WIDTH:
            image = image.resize((width, width), resample=resample)
        elif mode == ResizeMode.MATCH_HEIGHT:
            image = image.resize((height, height), resample=resample)
        elif mode == ResizeMode.MATCH_MAX:
            image = image.resize((max(width, height), max(width, height)), resample=resample)
        elif mode == ResizeMode.NAIVE:
            image = image.resize((min(width, height), min(width, height)), resample=resample)
        else:
            raise NotImplementedError(f"Mode {mode} not yet implemented")

        if is_np:
            image = np.asarray(image)

    return image


def assert_ratios_match(image1: PIL.Image.Image | np.ndarray, image2: PIL.Image.Image | np.ndarray) -> bool:
    width1, height1 = image1.shape[:2][::-1] if isinstance(image1, np.ndarray) else image1.size
    width2, height2 = image2.shape[:2][::-1] if isinstance(image2, np.ndarray) else image2.size

    assert np.isclose(
        width1 / height1, width2 / height2, rtol=1e-2, atol=1e-4
    ), f'{(width1, height1)} -> {width1 / height1} != {(width2, height2)} -> {width2 / height2}'

    return True


def is_depth_image(image: np.ndarray | PIL.Image.Image) -> bool:
    # https://pillow.readthedocs.io/en/latest/handbook/concepts.html#modes
    if isinstance(image, PIL.Image.Image):
        return image.mode == 'F' or image.mode.startswith('I')
    if isinstance(image, np.ndarray):
        # return np.issubdtype(image.dtype, np.floating)
        return np.issubdtype(image.dtype, np.floating) or image.itemsize > 1
    raise ValueError(f"Unsupported image type: {type(image)}")
