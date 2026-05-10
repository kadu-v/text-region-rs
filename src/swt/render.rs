use super::Rect;

pub(super) fn render_debug_bgr(
    width: usize,
    height: usize,
    rects: &[Rect],
) -> Vec<u8> {
    let mut output = vec![255_u8; width * height * 3];

    let colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]];
    for (i, rect) in rects.iter().enumerate() {
        let color = colors[i % colors.len()];
        draw_rect_bgr(&mut output, width, height, *rect, color);
    }
    output
}

fn draw_rect_bgr(
    output: &mut [u8],
    width: usize,
    height: usize,
    rect: Rect,
    color: [u8; 3],
) {
    if rect.width == 0 || rect.height == 0 || width == 0 || height == 0 {
        return;
    }
    let x0 = rect.x as usize;
    let y0 = rect.y as usize;
    let x1 = (rect.x as usize + rect.width.saturating_sub(1) as usize)
        .min(width - 1);
    let y1 = (rect.y as usize + rect.height.saturating_sub(1) as usize)
        .min(height - 1);
    if x0 >= width || y0 >= height {
        return;
    }

    for x in x0..=x1 {
        set_bgr(output, width, x, y0, color);
        set_bgr(output, width, x, y1, color);
    }
    for y in y0..=y1 {
        set_bgr(output, width, x0, y, color);
        set_bgr(output, width, x1, y, color);
    }
}

fn set_bgr(
    output: &mut [u8],
    width: usize,
    x: usize,
    y: usize,
    color: [u8; 3],
) {
    let idx = (y * width + x) * 3;
    output[idx] = color[0];
    output[idx + 1] = color[1];
    output[idx + 2] = color[2];
}
