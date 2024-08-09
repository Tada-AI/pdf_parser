from pdfminer.pdfinterp import Color

RGBA = int
HSL = tuple[float, float, float]


def rgb_to_rgba_int(rgb: tuple[float, float, float]) -> RGBA:
    # Scale RGB values to 0-255 range and convert to integers
    r_int = int(round(rgb[0] * 255))
    g_int = int(round(rgb[1] * 255))
    b_int = int(round(rgb[2] * 255))
    a_int = 255  # Fully opaque

    # Combine into an RGBA integer
    rgba_int = (r_int << 24) | (g_int << 16) | (b_int << 8) | a_int
    return rgba_int


def grayscale_to_rgba_int(gray: float) -> RGBA:
    """
    Convert a grayscale value (0.0 to 1.0) to an RGBA integer.

    :param gray: Grayscale value (float, 0.0 to 1.0)
    :return: RGBA color as an integer
    """
    # Ensure the grayscale value is within the expected range
    gray = max(0.0, min(1.0, gray))

    # Convert to an 8-bit value
    int_val = int(gray * 255)

    # Construct RGBA
    rgba = (int_val << 24) | (int_val << 16) | (int_val << 8) | 255
    return rgba


def cmyk_to_rgba_int(cmyk: tuple[float, float, float, float]):
    c, m, y, k = cmyk

    r = 255 * (1 - c) * (1 - k)
    g = 255 * (1 - m) * (1 - k)
    b = 255 * (1 - y) * (1 - k)

    return rgb_to_rgba_int((r, g, b))


def color_to_rbga_int(color: Color):
    if isinstance(color, float):
        return grayscale_to_rgba_int(color)
    elif isinstance(color, int):
        """
        This is a non-standard case, and it's unclear what this means.
        We're returning the int since it appears to be used consistently
        within the PDFs where this occurrs
        """
        return color
    elif len(color) == 1:
        return (color[0], 0, 0)
    elif len(color) == 3:
        return rgb_to_rgba_int(color)
    else:
        return cmyk_to_rgba_int(color)


def grayscale_to_hsl(grayscale: float) -> HSL:
    # Grayscale value is directly used as lightness in HSL, with hue and saturation set to 0.
    return (0, 0, grayscale)


def rgb_to_hsl(rgb: tuple[float, float, float]) -> HSL:
    """Convert an RGB float tuple to HSL."""
    r, g, b = rgb
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    l = (max_val + min_val) / 2

    if max_val == min_val:
        h = s = 0  # achromatic
    else:
        diff = max_val - min_val
        s = diff / (2 - max_val - min_val) if l > 0.5 else diff / (max_val + min_val)

        if max_val == r:
            h = (g - b) / diff + (6 if g < b else 0)
        elif max_val == g:
            h = (b - r) / diff + 2
        else:
            h = (r - g) / diff + 4
        h /= 6

    return (h, s, l)


def cmyk_to_rgb(cmyk: tuple[float, float, float, float]):
    """Convert CMYK to RGB."""
    c, m, y, k = cmyk
    r = 1 - min(1, c + k)
    g = 1 - min(1, m + k)
    b = 1 - min(1, y + k)
    return (r, g, b)


def cmyk_to_hsl(cmyk: tuple[float, float, float, float]) -> HSL:
    """Convert a CMYK float tuple to HSL."""
    rgb = cmyk_to_rgb(cmyk)
    return rgb_to_hsl(rgb)


def color_to_hsl(color: Color | None) -> HSL:
    if isinstance(color, list):
        if len(color) == 0:
            return (0, 0, 0)
        elif len(color) == 1:
            color = color[0]
        else:
            color = tuple(color)

    if color is None:
        return (0, 0, 0)
    if isinstance(color, float):
        return grayscale_to_hsl(color)
    elif isinstance(color, int):
        """
        This is a non-standard case, and it's unclear what this means.
        We're returning a semi-arbitrary grayscale here and hoping for the best
        """
        l = color / 10 if color < 10 else 1
        return (0, 0, l)
    elif len(color) == 3:
        return rgb_to_hsl(color)
    else:
        return cmyk_to_hsl(color)
