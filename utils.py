selected_object_id = None
selected_bbox = None
selected_center = None


def get_center(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    """Calculate the center point of a bounding box.

    Args:
        x1 (int): Top-left X coordinate.
        y1 (int): Top-left Y coordinate.
        x2 (int): Bottom-right X coordinate.
        y2 (int): Bottom-right Y coordinate.

    Returns:
        center_x (int): X-coordinate of the center point.
        center_y (int): Y-coordinate of the center point.
    """
    return (x1 + x2) // 2, (y1 + y2) // 2


def extend_line_from_edge(mid_x: int, mid_y: int, direction: str, img_shape: tuple[int, int, int]) -> tuple[int, int]:
    """Calculate the endpoint to extend a line from the center toward an image edge.

    Args:
        mid_x (int): X-coordinate of the midpoint.
        mid_y (int): Y-coordinate of the midpoint.
        direction (str): Direction to extend ('left', 'right', 'up', 'down').
        img_shape (tuple[int, int, int]): Image shape in (height, width, channels).

    Returns:
        end_x (int): X-coordinate of the endpoint.
        end_y (int): Y-coordinate of the endpoint.
    """
    h, w = img_shape[:2]
    if direction == "down":
        return mid_x, h - 1
    elif direction == "left":
        return 0, mid_y
    elif direction == "right":
        return w - 1, mid_y
    elif direction == "up":
        return mid_x, 0
    else:
        return mid_x, mid_y


def draw_tracking_scope(im, bbox: tuple, color: tuple) -> None:
    """Draw tracking scope lines extending from the bounding box to image edges.

    Args:
        im (np.ndarray): Image array to draw on.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        color (tuple): Color in BGR format for drawing.
    """
    x1, y1, x2, y2 = bbox
    mid_top = ((x1 + x2) // 2, y1)
    mid_bottom = ((x1 + x2) // 2, y2)
    mid_left = (x1, (y1 + y2) // 2)
    mid_right = (x2, (y1 + y2) // 2)
    cv2.line(im, mid_top, extend_line_from_edge(*mid_top, "up", im.shape), color, 2)
    cv2.line(im, mid_bottom, extend_line_from_edge(*mid_bottom, "down", im.shape), color, 2)
    cv2.line(im, mid_left, extend_line_from_edge(*mid_left, "left", im.shape), color, 2)
    cv2.line(im, mid_right, extend_line_from_edge(*mid_right, "right", im.shape), color, 2)


def click_event(event: int, x: int, y: int, flags: int, param) -> None:
    """Handle mouse click events to select an object for focused tracking.

    Args:
        event (int): OpenCV mouse event type.
        x (int): X-coordinate of the mouse event.
        y (int): Y-coordinate of the mouse event.
        flags (int): Any relevant flags passed by OpenCV.
        param (Any): Additional parameters (not used).
    """
    global selected_object_id
    if event == cv2.EVENT_LBUTTONDOWN:
        detections = results[0].boxes.data if results[0].boxes is not None else []
        if detections is not None:
            min_area = float("inf")
            best_match = None
            for track in detections:
                track = track.tolist()
                if len(track) >= 6:
                    x1, y1, x2, y2 = map(int, track[:4])
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        if area < min_area:
                            class_id = int(track[-1])
                            track_id = int(track[4]) if len(track) == 7 else -1
                            min_area = area
                            best_match = (track_id, model.names[class_id])
            if best_match:
                selected_object_id, label = best_match
                print(f"🔵 TRACKING STARTED: {label} (ID {selected_object_id})")
