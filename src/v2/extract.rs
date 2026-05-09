use crate::block_memory::BlockMemory;
use crate::params::ConnectedType;
use crate::types::{MserRegion, Point, Rect};
use crate::v1::data::RegionFlag;
use crate::v2::data::{dir_mask_v2, MserRegionV2};

/// Extract pixel coordinates for all valid MSER regions (V2).
/// V2 scans all interior pixels, looks up er_index from points array,
/// and maps each pixel to its MSER region.
pub fn extract_pixels_v2(
    regions: &mut BlockMemory<MserRegionV2>,
    points: &[u32],
    valid_order: &[usize],
    width: i32,
    height: i32,
    width_with_boundary: i32,
    connected_type: ConnectedType,
    gray_mask: u8,
) -> Vec<MserRegion> {
    if valid_order.is_empty() {
        return Vec::new();
    }

    let the_dir_mask = dir_mask_v2(connected_type);

    // Build region_heap: er_index -> mser_index (index into valid_order), or -1
    let num_regions = regions.len();
    let mut region_heap = vec![-1i32; num_regions];

    // Map valid region er_index to mser_index
    for (mser_idx, &region_idx) in valid_order.iter().enumerate() {
        let er_index = regions.get(region_idx).er_index as usize;
        region_heap[er_index] = mser_idx as i32;
    }

    // For non-valid regions, find their nearest valid ancestor and record mapping
    for i in 0..num_regions {
        if regions.get(i).region_flag == RegionFlag::Valid {
            continue;
        }
        if regions.get(i).assigned_pointer {
            continue;
        }

        // Walk parent chain to find valid ancestor
        let mut real_region = None;
        let mut cur = regions.get(i).parent;
        while let Some(p) = cur {
            if regions.get(p).region_flag == RegionFlag::Valid {
                real_region = Some(p);
                break;
            }
            if regions.get(p).assigned_pointer {
                // Already resolved - use its mapping
                let er = regions.get(p).er_index as usize;
                let mapped = region_heap[er];
                // Propagate this mapping back
                let er_i = regions.get(i).er_index as usize;
                region_heap[er_i] = mapped;
                regions.get_mut(i).assigned_pointer = true;
                break;
            }
            cur = regions.get(p).parent;
        }

        if let Some(real_idx) = real_region {
            let real_er = regions.get(real_idx).er_index as usize;
            let real_mser_idx = region_heap[real_er];

            // Set mapping for all regions on the path
            let er_i = regions.get(i).er_index as usize;
            region_heap[er_i] = real_mser_idx;
            regions.get_mut(i).assigned_pointer = true;

            // Also set for intermediate regions on the path
            let mut cur = regions.get(i).parent;
            while let Some(p) = cur {
                if p == real_idx {
                    break;
                }
                if regions.get(p).assigned_pointer {
                    break;
                }
                let er_p = regions.get(p).er_index as usize;
                region_heap[er_p] = real_mser_idx;
                regions.get_mut(p).assigned_pointer = true;
                cur = regions.get(p).parent;
            }
        } else if !regions.get(i).assigned_pointer {
            let er_i = regions.get(i).er_index as usize;
            region_heap[er_i] = -1;
            regions.get_mut(i).assigned_pointer = true;
        }
    }

    // Prepare output: one Vec<Point> per valid MSER
    let mut mser_points: Vec<Vec<Point>> = vec![Vec::new(); valid_order.len()];

    // Scan all interior pixels
    for row in 0..height {
        for col in 0..width {
            let idx = ((row + 1) * width_with_boundary + (col + 1)) as usize;
            let point_val = points[idx];
            let er_index = (point_val & !the_dir_mask) as usize;

            if er_index < num_regions {
                let mser_index = region_heap[er_index];
                if mser_index >= 0 {
                    mser_points[mser_index as usize].push(Point {
                        x: col,
                        y: row,
                    });
                }
            }
        }
    }

    // Propagate child MSER pixels up to parent MSERs.
    // valid_order is in gray order (low to high), so processing forward
    // ensures children are completed before parents.
    let mut valid_parent: Vec<Option<usize>> = vec![None; valid_order.len()];
    for (mser_idx, &region_idx) in valid_order.iter().enumerate() {
        let mut cur = regions.get(region_idx).parent;
        while let Some(p) = cur {
            if regions.get(p).region_flag == RegionFlag::Valid {
                let parent_er = regions.get(p).er_index as usize;
                let parent_mser_idx = region_heap[parent_er];
                if parent_mser_idx >= 0 {
                    valid_parent[mser_idx] = Some(parent_mser_idx as usize);
                }
                break;
            }
            cur = regions.get(p).parent;
        }
    }

    for mser_idx in 0..valid_order.len() {
        if let Some(parent_mser_idx) = valid_parent[mser_idx] {
            let child_points = mser_points[mser_idx].clone();
            mser_points[parent_mser_idx].extend(child_points);
        }
    }

    // Build output MserRegion list
    let mut msers = Vec::with_capacity(valid_order.len());

    for (mser_idx, &region_idx) in valid_order.iter().enumerate() {
        let region = regions.get(region_idx);
        let points = std::mem::take(&mut mser_points[mser_idx]);

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
            bounding_rect: Rect::from_points(
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
    use crate::v2::build_tree::make_tree_patch_v2;
    use crate::v2::recognize::recognize_mser_v2;

    fn run_v2_pipeline(
        img: &[u8],
        width: u32,
        height: u32,
        gray_mask: u8,
    ) -> Vec<MserRegion> {
        let result =
            make_tree_patch_v2(img, width, height, width, gray_mask, ConnectedType::FourConnected);

        let mut regions = result.regions;
        let max_point = (0.5 * (width * height) as f32) as i32;
        let valid_order =
            recognize_mser_v2(&mut regions, 1, 10.0, -1.0, 0.0, 1, max_point);

        extract_pixels_v2(
            &mut regions,
            &result.points,
            &valid_order,
            result.width,
            result.height,
            result.width_with_boundary,
            result.connected_type,
            gray_mask,
        )
    }

    #[test]
    fn test_v2_extract_simple() {
        let mut img = [100u8; 100];
        for r in 3..7 {
            for c in 3..7 {
                img[r * 10 + c] = 50;
            }
        }

        let msers = run_v2_pipeline(&img, 10, 10, 0);

        assert!(!msers.is_empty(), "Should detect at least one MSER");

        let inner = msers.iter().find(|m| m.gray_level == 50);
        assert!(inner.is_some(), "Should find MSER at gray level 50");
        assert_eq!(inner.unwrap().points.len(), 16);
    }

    #[test]
    fn test_v2_extract_coordinates() {
        let img = [100, 100, 100, 100, 50, 100, 100, 100, 100];
        let result =
            make_tree_patch_v2(&img, 3, 3, 3, 0, ConnectedType::FourConnected);

        let mut regions = result.regions;
        let valid_order =
            recognize_mser_v2(&mut regions, 1, 10.0, -1.0, 0.0, 1, 9);

        let msers = extract_pixels_v2(
            &mut regions,
            &result.points,
            &valid_order,
            result.width,
            result.height,
            result.width_with_boundary,
            result.connected_type,
            0,
        );

        let center = msers.iter().find(|m| m.gray_level == 50);
        if let Some(region) = center {
            assert_eq!(region.points.len(), 1);
            assert_eq!(region.points[0].x, 1);
            assert_eq!(region.points[0].y, 1);
        }
    }

    #[test]
    fn test_v2_extract_gray_mask() {
        let img = [100, 100, 100, 100, 200, 100, 100, 100, 100];

        let msers = run_v2_pipeline(&img, 3, 3, 255);

        for mser in &msers {
            assert!(mser.gray_level == 100 || mser.gray_level == 200);
        }
    }
}
