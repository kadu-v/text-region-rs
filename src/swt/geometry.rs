use super::Rect;

pub(super) fn point_index_from_xy(width: usize, x: i32, y: i32) -> usize {
    y as usize * width + x as usize
}

pub(super) fn rect_from_opencv_points(
    xmin: i32,
    ymin: i32,
    xmax: i32,
    ymax: i32,
    inclusive: bool,
) -> Rect {
    let extra = if inclusive { 1 } else { 0 };
    Rect {
        x: xmin.min(xmax) as u32,
        y: ymin.min(ymax) as u32,
        width: (xmax - xmin).unsigned_abs() + extra,
        height: (ymax - ymin).unsigned_abs() + extra,
    }
}
