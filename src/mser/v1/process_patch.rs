use crate::mser::v1::data::BOUNDARY_YES_MASK;

pub struct ProcessedPatch {
    pub masked_image: Vec<i16>,
    pub level_size: [u32; 257],
    pub width_with_boundary: i32,
    pub height_with_boundary: i32,
}

/// Build the masked image with border sentinels and compute gray level histogram.
///
/// The masked image has dimensions `(width+2) x (height+2)`:
/// - Border pixels are set to -1 (sentinel, marks "visited")
/// - Interior pixels store `(gray_value ^ gray_mask)` in the lower 9 bits
/// - For single-thread mode, no boundary flags are set
///
/// This mirrors C++ `img_fast_mser_v1::process_tree_patch` for the single-thread case.
pub fn process_tree_patch(
    image_data: &[u8],
    width: u32,
    height: u32,
    img_stride: u32,
    gray_mask: u8,
) -> ProcessedPatch {
    let w = width as i32;
    let h = height as i32;
    let wb = w + 2;
    let hb = h + 2;
    let total = (wb * hb) as usize;

    let mut masked_image = vec![-1i16; total];
    let mut level_size = [0u32; 257];

    for row in 0..h {
        for col in 0..w {
            let src_idx = (row as u32 * img_stride + col as u32) as usize;
            let gray = image_data[src_idx] ^ gray_mask;
            level_size[gray as usize] += 1;

            let dst_idx = ((row + 1) * wb + (col + 1)) as usize;
            masked_image[dst_idx] = gray as i16;
        }
    }

    ProcessedPatch {
        masked_image,
        level_size,
        width_with_boundary: wb,
        height_with_boundary: hb,
    }
}

/// Flip level_size for the second pass (gray_mask=255).
/// When reprocessing for bright-on-dark after dark-on-bright, the image values
/// are XOR'd in-place and the histogram is mirrored.
pub fn flip_for_second_pass(
    masked_image: &mut [i16],
    level_size: &mut [u32; 257],
    width: i32,
    height: i32,
    width_with_boundary: i32,
) {
    // XOR image values and flip the boundary flags
    for row in 0..height {
        for col in 0..width {
            let idx = ((row + 1) * width_with_boundary + (col + 1)) as usize;
            let boundary_flag = masked_image[idx] & BOUNDARY_YES_MASK;
            let gray = (0xff - (masked_image[idx] & 0x00ff)) as i16;
            masked_image[idx] = gray | boundary_flag;
        }
    }

    // Mirror histogram
    for i in 0..128 {
        level_size.swap(i, 255 - i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_image_1x1() {
        let img = [42u8];
        let result = process_tree_patch(&img, 1, 1, 1, 0);

        assert_eq!(result.width_with_boundary, 3);
        assert_eq!(result.height_with_boundary, 3);
        assert_eq!(result.masked_image.len(), 9);

        // 3x3 layout:
        // [-1, -1, -1]
        // [-1, 42, -1]
        // [-1, -1, -1]
        assert_eq!(result.masked_image[0], -1); // top-left border
        assert_eq!(result.masked_image[1], -1); // top border
        assert_eq!(result.masked_image[2], -1); // top-right border
        assert_eq!(result.masked_image[3], -1); // left border
        assert_eq!(result.masked_image[4], 42); // center pixel
        assert_eq!(result.masked_image[5], -1); // right border
        assert_eq!(result.masked_image[6], -1); // bottom-left border
        assert_eq!(result.masked_image[7], -1); // bottom border
        assert_eq!(result.masked_image[8], -1); // bottom-right border
    }

    #[test]
    fn test_masked_image_3x3_gray_mask_0() {
        // 3x3 image:
        // [10, 20, 30]
        // [40, 50, 60]
        // [70, 80, 90]
        let img = [10, 20, 30, 40, 50, 60, 70, 80, 90];
        let result = process_tree_patch(&img, 3, 3, 3, 0);

        assert_eq!(result.width_with_boundary, 5);
        assert_eq!(result.height_with_boundary, 5);
        assert_eq!(result.masked_image.len(), 25);

        // Check all border cells are -1
        // Row 0: all -1
        for i in 0..5 {
            assert_eq!(result.masked_image[i], -1);
        }
        // Row 4: all -1
        for i in 20..25 {
            assert_eq!(result.masked_image[i], -1);
        }
        // Left and right borders of rows 1-3
        for row in 1..4 {
            assert_eq!(result.masked_image[row * 5], -1);
            assert_eq!(result.masked_image[row * 5 + 4], -1);
        }

        // Check interior
        assert_eq!(result.masked_image[6], 10);
        assert_eq!(result.masked_image[7], 20);
        assert_eq!(result.masked_image[8], 30);
        assert_eq!(result.masked_image[11], 40);
        assert_eq!(result.masked_image[12], 50);
        assert_eq!(result.masked_image[13], 60);
        assert_eq!(result.masked_image[16], 70);
        assert_eq!(result.masked_image[17], 80);
        assert_eq!(result.masked_image[18], 90);
    }

    #[test]
    fn test_masked_image_3x3_gray_mask_255() {
        let img = [10, 20, 30, 40, 50, 60, 70, 80, 90];
        let result = process_tree_patch(&img, 3, 3, 3, 255);

        // gray values should be XOR'd: 10 ^ 255 = 245, etc.
        assert_eq!(result.masked_image[6], 245);
        assert_eq!(result.masked_image[7], 235);
        assert_eq!(result.masked_image[8], 225);
        assert_eq!(result.masked_image[12], 205); // 50 ^ 255
    }

    #[test]
    fn test_level_size_histogram() {
        // Image: [0, 0, 1, 1, 1, 2]  (2x3)
        let img = [0, 0, 1, 1, 1, 2];
        let result = process_tree_patch(&img, 3, 2, 3, 0);

        assert_eq!(result.level_size[0], 2);
        assert_eq!(result.level_size[1], 3);
        assert_eq!(result.level_size[2], 1);
        for i in 3..257 {
            assert_eq!(result.level_size[i], 0);
        }
    }
}
