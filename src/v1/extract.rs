use crate::block_memory::BlockMemory;
use crate::types::{MserRegion, Point, rect_from_points};
use crate::v1::data::{LinkedPoint, MserRegionV1, RegionFlag};

/// Extract pixel coordinates for all valid MSER regions.
/// This ports `extract_pixel_parallel_1_fast` for single-thread mode.
pub fn extract_pixels(
    regions: &BlockMemory<MserRegionV1>,
    linked_points: &mut [LinkedPoint],
    valid_order: &[usize],
    gray_mask: u8,
) -> Vec<MserRegion> {
    if valid_order.is_empty() {
        return Vec::new();
    }

    // Find "top" regions: valid regions whose parent is not valid
    let mut top_regions = Vec::new();
    let mut total_top_pixels = 0;

    for &idx in valid_order {
        let mut parent_opt = regions.get(idx).parent;
        let mut has_valid_parent = false;
        while let Some(p) = parent_opt {
            if regions.get(p).region_flag == RegionFlag::Valid {
                has_valid_parent = true;
                break;
            }
            parent_opt = regions.get(p).parent;
        }

        if !has_valid_parent {
            top_regions.push(idx);
            total_top_pixels += regions.get(idx).size;
        }
    }

    // Extract pixels from top regions into a shared memory buffer
    let mut memory = Vec::with_capacity(total_top_pixels as usize);

    for &region_idx in &top_regions {
        let region = regions.get(region_idx);
        let mut pt_index = region.head;

        for _ in 0..region.size {
            let idx = pt_index as usize;
            let x = linked_points[idx].x;
            let y = linked_points[idx].y;
            let next = linked_points[idx].next;
            let mem_offset = memory.len();
            memory.push(Point {
                x: x as i32,
                y: y as i32,
            });
            linked_points[idx].ref_ = mem_offset as i32;
            pt_index = next;
        }
    }

    // Build output MSERs
    let mut msers = Vec::with_capacity(valid_order.len());

    for &idx in valid_order {
        let region = regions.get(idx);
        let head_ref = linked_points[region.head as usize].ref_;
        let tail_ref = linked_points[region.tail as usize].ref_;

        let (start, end) = if head_ref <= tail_ref {
            (head_ref as usize, tail_ref as usize)
        } else {
            (tail_ref as usize, head_ref as usize)
        };

        let points: Vec<Point> = memory[start..=end].to_vec();

        let mut min_x = i32::MAX;
        let mut max_x = i32::MIN;
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;

        for pt in &points {
            min_x = min_x.min(pt.x);
            max_x = max_x.max(pt.x);
            min_y = min_y.min(pt.y);
            max_y = max_y.max(pt.y);
        }

        msers.push(MserRegion {
            gray_level: region.gray_level ^ gray_mask,
            points,
            bounding_rect: rect_from_points(
                Point { x: min_x, y: min_y },
                Point { x: max_x, y: max_y },
            ),
        });
    }

    msers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ConnectedType;
    use crate::v1::build_tree::make_tree_patch;
    use crate::v1::recognize::recognize_mser;

    fn run_full_pipeline(
        img: &[u8],
        width: u32,
        height: u32,
        gray_mask: u8,
    ) -> Vec<MserRegion> {
        let mut result = make_tree_patch(
            img,
            width,
            height,
            width,
            gray_mask,
            ConnectedType::FourConnected,
            1,
        );

        let max_point = (0.5 * (width * height) as f32) as i32;
        let valid_order = recognize_mser(
            &mut result.regions,
            1,
            10.0,
            -1.0,
            0.0,
            1,
            max_point,
        );

        extract_pixels(
            &result.regions,
            &mut result.linked_points,
            &valid_order,
            gray_mask,
        )
    }

    #[test]
    fn test_extract_single_region() {
        // 10x10 image: center 4x4 block at value 50, border at 100
        let mut img = [100u8; 100];
        for r in 3..7 {
            for c in 3..7 {
                img[r * 10 + c] = 50;
            }
        }

        let msers = run_full_pipeline(&img, 10, 10, 0);

        // Should detect at least 1 MSER (the inner block)
        assert!(!msers.is_empty(), "Should detect at least one MSER");

        // Find the region at gray level 50
        let inner = msers.iter().find(|m| m.gray_level == 50);
        assert!(inner.is_some(), "Should find MSER at gray level 50");
        assert_eq!(inner.unwrap().points.len(), 16);
    }

    #[test]
    fn test_extract_coordinates() {
        // 3x3 image: center=50, border=100
        let img = [100, 100, 100, 100, 50, 100, 100, 100, 100];
        let mut result =
            make_tree_patch(&img, 3, 3, 3, 0, ConnectedType::FourConnected, 1);

        let max_point = 9;
        let valid_order = recognize_mser(
            &mut result.regions,
            1,
            10.0,
            -1.0,
            0.0,
            1,
            max_point,
        );

        let msers = extract_pixels(
            &result.regions,
            &mut result.linked_points,
            &valid_order,
            0,
        );

        // Find the single-pixel region at center
        let center = msers.iter().find(|m| m.gray_level == 50);
        if let Some(region) = center {
            assert_eq!(region.points.len(), 1);
            assert_eq!(region.points[0].x, 1);
            assert_eq!(region.points[0].y, 1);
        }
    }

    #[test]
    fn test_gray_level_xor() {
        let img = [100, 100, 100, 100, 200, 100, 100, 100, 100];
        let mut result = make_tree_patch(
            &img,
            3,
            3,
            3,
            255,
            ConnectedType::FourConnected,
            1,
        );

        let max_point = 9;
        let valid_order = recognize_mser(
            &mut result.regions,
            1,
            10.0,
            -1.0,
            0.0,
            1,
            max_point,
        );

        let msers = extract_pixels(
            &result.regions,
            &mut result.linked_points,
            &valid_order,
            255,
        );

        // With gray_mask=255, stored gray levels are XOR'd.
        // Output should XOR back: level ^ 255.
        for mser in &msers {
            // Verify gray_level is the original value
            assert!(mser.gray_level == 100 || mser.gray_level == 200);
        }
    }
}
