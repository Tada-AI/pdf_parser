Rect = tuple[float, float, float, float]
Point2 = tuple[float, float]


X0 = 0
Y0 = 1
X1 = 2
Y1 = 3


def normalize_to_range(range: Point2, val: float, scale=1.0):
    lower, upper = range
    return (val - lower) / (upper - lower) * scale


def normalize_point2(bbox: Rect, point: Point2, *, scale=1.0) -> Point2:
    x0, y0, x1, y1 = bbox

    point_x, point_y = point
    normalized_x = normalize_to_range((x0, x1), point_x, scale=scale)
    normalized_y = normalize_to_range((y0, y1), point_y, scale=scale)

    return (normalized_x, normalized_y)


def normalize_bbox(outer_bbox: Rect, inner_bbox: Rect, *, scale=1.0) -> Rect:
    x0, y0, x1, y1 = inner_bbox
    normalized_x0, normalized_y0 = normalize_point2(outer_bbox, (x0, y0), scale=scale)
    normalized_x1, normalized_y1 = normalize_point2(outer_bbox, (x1, y1), scale=scale)

    return (normalized_x0, normalized_y0, normalized_x1, normalized_y1)


def denormalize_to_range(range: Point2, val: float, scale=1.0):
    lower, upper = range
    return (val + lower) * (upper + lower) / scale


def denormalize_point2(bbox: Rect, point: Point2, *, scale=1.0):
    x0, y0, x1, y1 = bbox

    point_x, point_y = point
    denom_x = denormalize_to_range((x0, x1), point_x, scale=scale)
    denom_y = denormalize_to_range((y0, y1), point_y, scale=scale)

    return (denom_x, denom_y)


def denormalize_bbox(
    outer_bbox: Rect, inner_bbox: Rect, *, scale=1.0, x_scale=1.0, y_scale=1.0
) -> Rect:
    x0, y0, x1, y1 = inner_bbox
    denomx0, denomy0 = denormalize_point2(outer_bbox, (x0, y0), scale=scale)
    denomx1, denomy1 = denormalize_point2(outer_bbox, (x1, y1), scale=scale)

    return (denomx0, denomy0, denomx1, denomy1)


def translate_bbox(bbox: Rect, direction: Point2):
    x0, y0, x1, y1 = bbox
    delta_x, delta_y = direction

    return (x0 + delta_x, y0 + delta_y, x1 + delta_x, y1 + delta_y)


def scale_bbox(bbox: Rect, scale: float):
    x0, y0, x1, y1 = bbox

    x_center = (x0 + x1) / 2
    y_center = (y0 + y1) / 2

    #  translate_bbox(bbox, (-1 * x_center, -1 * y_center))
    x0, y0, x1, y1 = translate_bbox(bbox, (-1 * x_center, -1 * y_center))
    new_bbox = x0, y0, x1, y1
    new_bbox = (x0 * scale, y0 * scale, x1 * scale, y1 * scale)

    return translate_bbox(new_bbox, (x_center, y_center))
