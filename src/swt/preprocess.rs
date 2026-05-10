use crate::error::Result;

use super::SwtPreprocessed;

pub fn swt_preprocess_bgr(
    width: u32,
    height: u32,
    bgr: &[u8],
) -> Result<SwtPreprocessed> {
    super::validation::validate_bgr_input(width, height, bgr)?;

    let gray = bgr_to_gray(bgr);
    let edge =
        canny_3x3_l1(&gray, width as usize, height as usize, 175.0, 320.0);

    let gaussian = gray
        .iter()
        .map(|&value| value as f32 / 255.0)
        .collect::<Vec<_>>();
    let gaussian = gaussian_blur(&gaussian, width as usize, height as usize, 5);
    let (gradient_x, gradient_y) =
        scharr_gradients(&gaussian, width as usize, height as usize);
    let gradient_x =
        gaussian_blur(&gradient_x, width as usize, height as usize, 3);
    let gradient_y =
        gaussian_blur(&gradient_y, width as usize, height as usize, 3);

    Ok(SwtPreprocessed {
        width,
        height,
        gray,
        edge,
        gradient_x,
        gradient_y,
    })
}

fn bgr_to_gray(bgr: &[u8]) -> Vec<u8> {
    bgr.chunks_exact(3)
        .map(|pixel| {
            let b = pixel[0] as f32;
            let g = pixel[1] as f32;
            let r = pixel[2] as f32;
            (0.114 * b + 0.587 * g + 0.299 * r).round() as u8
        })
        .collect()
}

fn gaussian_blur(
    input: &[f32],
    width: usize,
    height: usize,
    ksize: usize,
) -> Vec<f32> {
    debug_assert!(ksize == 3 || ksize == 5);
    let kernel = gaussian_kernel(ksize);
    let radius = ksize as isize / 2;

    let mut temp = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (k, &weight) in kernel.iter().enumerate() {
                let dx = k as isize - radius;
                let sx = reflect101(x as isize + dx, width);
                sum += input[y * width + sx] * weight;
            }
            temp[y * width + x] = sum;
        }
    }

    let mut output = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (k, &weight) in kernel.iter().enumerate() {
                let dy = k as isize - radius;
                let sy = reflect101(y as isize + dy, height);
                sum += temp[sy * width + x] * weight;
            }
            output[y * width + x] = sum;
        }
    }
    output
}

fn gaussian_kernel(ksize: usize) -> Vec<f32> {
    let sigma = if ksize == 3 { 0.8 } else { 1.1 };
    let radius = ksize as i32 / 2;
    let mut kernel = Vec::with_capacity(ksize);
    let mut sum = 0.0;
    for i in -radius..=radius {
        let x = i as f32;
        let value = (-(x * x) / (2.0 * sigma * sigma)).exp();
        kernel.push(value);
        sum += value;
    }
    for value in &mut kernel {
        *value /= sum;
    }
    kernel
}

fn scharr_gradients(
    input: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0; input.len()];
    let mut gy = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let xm = reflect101(x as isize - 1, width);
            let xp = reflect101(x as isize + 1, width);
            let ym = reflect101(y as isize - 1, height);
            let yp = reflect101(y as isize + 1, height);

            let top_left = input[ym * width + xm];
            let top = input[ym * width + x];
            let top_right = input[ym * width + xp];
            let left = input[y * width + xm];
            let right = input[y * width + xp];
            let bottom_left = input[yp * width + xm];
            let bottom = input[yp * width + x];
            let bottom_right = input[yp * width + xp];

            gx[y * width + x] = 3.0 * (top_right - top_left)
                + 10.0 * (right - left)
                + 3.0 * (bottom_right - bottom_left);
            gy[y * width + x] = 3.0 * (bottom_left - top_left)
                + 10.0 * (bottom - top)
                + 3.0 * (bottom_right - top_right);
        }
    }
    (gx, gy)
}

fn canny_3x3_l1(
    gray: &[u8],
    width: usize,
    height: usize,
    low_threshold: f32,
    high_threshold: f32,
) -> Vec<u8> {
    let (gx, gy) = sobel_3x3_i16(gray, width, height);
    let mut magnitude = vec![0.0; gray.len()];
    for i in 0..gray.len() {
        magnitude[i] = gx[i].abs() as f32 + gy[i].abs() as f32;
    }

    let mut nms = vec![0.0; gray.len()];
    if width >= 3 && height >= 3 {
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let idx = y * width + x;
                let ax = gx[idx].abs();
                let ay = gy[idx].abs();
                let (a, b) = if ay * 2 <= ax {
                    (idx - 1, idx + 1)
                } else if ax * 2 <= ay {
                    (idx - width, idx + width)
                } else if gx[idx].signum() == gy[idx].signum() {
                    (idx - width - 1, idx + width + 1)
                } else {
                    (idx - width + 1, idx + width - 1)
                };
                let mag = magnitude[idx];
                if mag >= magnitude[a] && mag >= magnitude[b] {
                    nms[idx] = mag;
                }
            }
        }
    }

    let mut edge = vec![0_u8; gray.len()];
    let mut stack = Vec::new();
    for (idx, &mag) in nms.iter().enumerate() {
        if mag >= high_threshold {
            edge[idx] = 255;
            stack.push(idx);
        }
    }

    while let Some(idx) = stack.pop() {
        let x = idx % width;
        let y = idx / width;
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(height.saturating_sub(1));
        let x0 = x.saturating_sub(1);
        let x1 = (x + 1).min(width.saturating_sub(1));
        for ny in y0..=y1 {
            for nx in x0..=x1 {
                let nidx = ny * width + nx;
                if edge[nidx] == 0 && nms[nidx] >= low_threshold {
                    edge[nidx] = 255;
                    stack.push(nidx);
                }
            }
        }
    }

    edge
}

fn sobel_3x3_i16(
    gray: &[u8],
    width: usize,
    height: usize,
) -> (Vec<i16>, Vec<i16>) {
    let mut gx = vec![0_i16; gray.len()];
    let mut gy = vec![0_i16; gray.len()];
    for y in 0..height {
        for x in 0..width {
            let xm = reflect101(x as isize - 1, width);
            let xp = reflect101(x as isize + 1, width);
            let ym = reflect101(y as isize - 1, height);
            let yp = reflect101(y as isize + 1, height);

            let top_left = gray[ym * width + xm] as i16;
            let top = gray[ym * width + x] as i16;
            let top_right = gray[ym * width + xp] as i16;
            let left = gray[y * width + xm] as i16;
            let right = gray[y * width + xp] as i16;
            let bottom_left = gray[yp * width + xm] as i16;
            let bottom = gray[yp * width + x] as i16;
            let bottom_right = gray[yp * width + xp] as i16;

            gx[y * width + x] = -top_left + top_right - 2 * left + 2 * right
                - bottom_left
                + bottom_right;
            gy[y * width + x] = -top_left - 2 * top - top_right
                + bottom_left
                + 2 * bottom
                + bottom_right;
        }
    }
    (gx, gy)
}

fn reflect101(coord: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let mut value = coord;
    let max = len as isize - 1;
    while value < 0 || value > max {
        if value < 0 {
            value = -value;
        } else {
            value = 2 * max - value;
        }
    }
    value as usize
}
