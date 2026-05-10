use super::Rect;

pub(super) fn render_debug_rgb(
    width: usize,
    height: usize,
    rects: &[Rect],
) -> image::RgbImage {
    let mut output = vec![255_u8; width * height * 3];

    let colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0]];
    for (i, rect) in rects.iter().enumerate() {
        let color = colors[i % colors.len()];
        draw_rect_rgb(&mut output, width, height, *rect, color);
    }
    image::RgbImage::from_vec(width as u32, height as u32, output)
        .expect("valid RGB debug image")
}

fn draw_rect_rgb(
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
        set_rgb(output, width, x, y0, color);
        set_rgb(output, width, x, y1, color);
    }
    for y in y0..=y1 {
        set_rgb(output, width, x0, y, color);
        set_rgb(output, width, x1, y, color);
    }
}

fn set_rgb(
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
