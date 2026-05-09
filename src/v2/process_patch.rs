use crate::params::ConnectedType;
use crate::v2::data::boundary_pixel;

pub struct ProcessedPatchV2 {
    pub extended_image: Vec<u8>,
    pub points: Vec<u32>,
    pub level_size: [u32; 257],
    pub width_with_boundary: i32,
    pub height_with_boundary: i32,
}

/// Build the extended image and points array with border sentinels.
/// V2 stores gray values in `extended_image` and direction/er_index in `points`.
pub fn process_tree_patch_v2(
    image_data: &[u8],
    width: u32,
    height: u32,
    img_stride: u32,
    gray_mask: u8,
    connected_type: ConnectedType,
) -> ProcessedPatchV2 {
    let w = width as i32;
    let h = height as i32;
    let wb = w + 2;
    let hb = h + 2;
    let total = (wb * hb) as usize;
    let bp = boundary_pixel(connected_type);

    let mut extended_image = vec![0u8; total];
    let mut points = vec![bp; total]; // border pixels are boundary_pixel
    let mut level_size = [0u32; 257];

    for row in 0..h {
        for col in 0..w {
            let src_idx = (row as u32 * img_stride + col as u32) as usize;
            let gray = image_data[src_idx] ^ gray_mask;
            level_size[gray as usize] += 1;

            let dst_idx = ((row + 1) * wb + (col + 1)) as usize;
            extended_image[dst_idx] = gray;
            points[dst_idx] = 0; // interior pixels start at 0 (unvisited)
        }
    }

    ProcessedPatchV2 {
        extended_image,
        points,
        level_size,
        width_with_boundary: wb,
        height_with_boundary: hb,
    }
}

/// Flip for second pass (gray_mask=255).
pub fn flip_for_second_pass_v2(
    extended_image: &mut [u8],
    points: &mut [u32],
    level_size: &mut [u32; 257],
    width: i32,
    height: i32,
    width_with_boundary: i32,
    connected_type: ConnectedType,
) {
    let _bp = boundary_pixel(connected_type);

    for row in 0..height {
        for col in 0..width {
            let idx = ((row + 1) * width_with_boundary + (col + 1)) as usize;
            extended_image[idx] = 0xff - extended_image[idx];
            points[idx] = 0; // reset to unvisited
        }
    }

    for i in 0..128 {
        level_size.swap(i, 255 - i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_extended_image() {
        let img = [42u8];
        let result =
            process_tree_patch_v2(&img, 1, 1, 1, 0, ConnectedType::FourConnected);

        assert_eq!(result.width_with_boundary, 3);
        assert_eq!(result.height_with_boundary, 3);

        // Center pixel should have gray value in extended_image
        assert_eq!(result.extended_image[4], 42);
        // Center pixel in points should be 0 (unvisited)
        assert_eq!(result.points[4], 0);
        // Border pixels should be boundary_pixel
        let bp = boundary_pixel(ConnectedType::FourConnected);
        assert_eq!(result.points[0], bp);
        assert_eq!(result.points[1], bp);
    }

    #[test]
    fn test_v2_point_encoding() {
        // 4-connected: dir_shift=29, boundary_pixel = 5<<29
        let bp_4 = boundary_pixel(ConnectedType::FourConnected);
        assert_eq!(bp_4, 5u32 << 29);

        // 8-connected: dir_shift=28, boundary_pixel = 9<<28
        let bp_8 = boundary_pixel(ConnectedType::EightConnected);
        assert_eq!(bp_8, 9u32 << 28);
    }

    #[test]
    fn test_v2_level_size() {
        let img = [0, 0, 1, 1, 1, 2];
        let result =
            process_tree_patch_v2(&img, 3, 2, 3, 0, ConnectedType::FourConnected);

        assert_eq!(result.level_size[0], 2);
        assert_eq!(result.level_size[1], 3);
        assert_eq!(result.level_size[2], 1);
    }
}
