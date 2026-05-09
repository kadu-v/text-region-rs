use crate::block_memory::BlockMemory;
use crate::heap::BucketHeap;
use crate::params::ConnectedType;
use crate::v1::data::RegionFlag;
use crate::v2::data::*;
use crate::v2::process_patch::process_tree_patch_v2;

pub struct TreeBuildResultV2 {
    pub regions: BlockMemory<MserRegionV2>,
    pub extended_image: Vec<u8>,
    pub points: Vec<u32>,
    pub width: i32,
    pub height: i32,
    pub width_with_boundary: i32,
    pub connected_type: ConnectedType,
}

fn init_comp_v2(
    comp: &mut ConnectedCompV2,
    regions: &mut BlockMemory<MserRegionV2>,
    region_idx: usize,
    patch_index: u8,
) {
    comp.size = 0;
    let region = regions.get_mut(region_idx);
    region.gray_level = comp.gray_level as u8;
    region.region_flag = RegionFlag::Unknown;
    region.size = 0;
    region.parent = None;
    region.assigned_pointer = false;
    region.calculated_var = false;
    region.patch_index = patch_index;
    comp.region_idx = region_idx;
}

fn new_region_v2(
    comp: &mut ConnectedCompV2,
    regions: &mut BlockMemory<MserRegionV2>,
    region_idx: usize,
    patch_index: u8,
) {
    let region = regions.get_mut(region_idx);
    region.gray_level = comp.gray_level as u8;
    region.region_flag = RegionFlag::Unknown;
    region.size = 0;
    region.parent = None;
    region.assigned_pointer = false;
    region.calculated_var = false;
    region.patch_index = patch_index;
    comp.region_idx = region_idx;
}

/// Build the component tree for V2 (single-threaded).
/// Ports `img_fast_mser_v2::make_tree_patch`.
pub fn make_tree_patch_v2(
    image_data: &[u8],
    width: u32,
    height: u32,
    img_stride: u32,
    gray_mask: u8,
    connected_type: ConnectedType,
    patch_index: u8,
) -> TreeBuildResultV2 {
    let processed = process_tree_patch_v2(
        image_data,
        width,
        height,
        img_stride,
        gray_mask,
        connected_type,
    );

    let w = width as i32;
    let h = height as i32;
    let wb = processed.width_with_boundary;

    let the_dir_shift = dir_shift(connected_type);
    let the_dir_mask = dir_mask_v2(connected_type);
    let _the_boundary_pixel = boundary_pixel(connected_type);
    let the_max_dir = max_dir(connected_type);
    let dir_offsets = compute_dir_offsets_v2(connected_type, wb);

    let extended_image = processed.extended_image;
    let mut points = processed.points;

    let mut heap = BucketHeap::new(&processed.level_size);
    let mut regions = BlockMemory::<MserRegionV2>::new(11);

    // Component stack
    let mut comp_stack = Vec::with_capacity(257);
    let mut sentinel = ConnectedCompV2::new();
    sentinel.gray_level = 257;
    comp_stack.push(sentinel);

    // Start at top-left interior pixel
    let start_pos = (1 + wb) as usize;
    let curr_gray_val = extended_image[start_pos];
    let mut curr_gray = curr_gray_val as i16;

    // Mark start pixel as visited: set direction to 1
    points[start_pos] = 1 << the_dir_shift;

    let mut cur_region = MserRegionV2::new();
    cur_region.er_index = regions.len() as i32;
    let cur_region_idx = regions.add(cur_region);

    let mut cur_comp = ConnectedCompV2::new();
    cur_comp.gray_level = curr_gray;
    init_comp_v2(&mut cur_comp, &mut regions, cur_region_idx, patch_index);

    let mut heap_cur_level = curr_gray as usize;
    let mut pos = start_pos;

    loop {
        let mut nbr_index = points[pos] >> the_dir_shift;

        // Explore all directions
        while nbr_index <= the_max_dir {
            let nbr_pos =
                (pos as i32 + dir_offsets[nbr_index as usize]) as usize;

            // If neighbor not visited (direction bits == 0)
            if (points[nbr_pos] & the_dir_mask) == 0 {
                let nbr_gray = extended_image[nbr_pos] as i16;
                points[nbr_pos] = 1 << the_dir_shift; // mark visited

                if nbr_gray < curr_gray {
                    // Neighbor has lower gray: push current, descend
                    heap.push(heap_cur_level, pos);
                    points[pos] = (nbr_index + 1) << the_dir_shift;
                    heap_cur_level = (heap_cur_level as i32
                        + (nbr_gray - curr_gray) as i32)
                        as usize;

                    pos = nbr_pos;

                    let mut new_r = MserRegionV2::new();
                    new_r.er_index = regions.len() as i32;
                    let new_idx = regions.add(new_r);

                    comp_stack.push(cur_comp);
                    cur_comp = ConnectedCompV2::new();
                    cur_comp.gray_level = nbr_gray;
                    init_comp_v2(
                        &mut cur_comp,
                        &mut regions,
                        new_idx,
                        patch_index,
                    );
                    curr_gray = nbr_gray;
                    nbr_index = 1;
                    continue;
                }

                // Push neighbor to its bucket
                let target_level = (heap_cur_level as i32
                    + (nbr_gray - curr_gray) as i32)
                    as usize;
                heap.push(target_level, nbr_pos);
            }

            nbr_index += 1;
        }

        // Store final direction + er_index in this point
        let er_index = regions.get(cur_comp.region_idx).er_index as u32;
        points[pos] = (nbr_index << the_dir_shift) | er_index;

        cur_comp.size += 1;

        // Get next pixel from heap
        if let Some(next_pos) = heap.pop(heap_cur_level) {
            pos = next_pos;
        } else {
            // Current bucket exhausted: finalize region
            {
                let region = regions.get_mut(cur_comp.region_idx);
                region.size = cur_comp.size;
                region.unmerged_size = cur_comp.size as u32;
            }

            // Find next non-empty bucket
            let mut pixel_val: i16 = 0;
            heap_cur_level += 1;
            for i in (curr_gray as usize + 1)..257 {
                if !heap.is_empty(i) {
                    pixel_val = i as i16;
                    heap_cur_level = i;
                    break;
                }
                heap_cur_level += 1;
            }

            if pixel_val > 0 {
                curr_gray = pixel_val;
                pos = heap.pop(heap_cur_level).unwrap();

                let prev_comp_gray = comp_stack.last().unwrap().gray_level;

                if pixel_val < prev_comp_gray {
                    let mut new_r = MserRegionV2::new();
                    new_r.er_index = regions.len() as i32;
                    let new_idx = regions.add(new_r);

                    regions.get_mut(cur_comp.region_idx).parent = Some(new_idx);
                    cur_comp.gray_level = pixel_val;
                    new_region_v2(
                        &mut cur_comp,
                        &mut regions,
                        new_idx,
                        patch_index,
                    );
                } else {
                    loop {
                        let mut prev_comp = comp_stack.pop().unwrap();
                        // Merge: set parent and accumulate size
                        regions.get_mut(cur_comp.region_idx).parent =
                            Some(prev_comp.region_idx);
                        prev_comp.size += cur_comp.size;
                        cur_comp = prev_comp;

                        if pixel_val <= cur_comp.gray_level {
                            break;
                        }

                        let prev_prev_gray =
                            comp_stack.last().unwrap().gray_level;

                        if pixel_val < prev_prev_gray {
                            let mut new_r = MserRegionV2::new();
                            new_r.er_index = regions.len() as i32;
                            let new_idx = regions.add(new_r);

                            regions.get_mut(cur_comp.region_idx).parent =
                                Some(new_idx);
                            cur_comp.gray_level = pixel_val;
                            new_region_v2(
                                &mut cur_comp,
                                &mut regions,
                                new_idx,
                                patch_index,
                            );
                            break;
                        }
                    }
                }
            } else {
                break;
            }
        }
    }

    TreeBuildResultV2 {
        regions,
        extended_image,
        points,
        width: w,
        height: h,
        width_with_boundary: wb,
        connected_type,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_roots(regions: &BlockMemory<MserRegionV2>) -> usize {
        regions.iter().filter(|r| r.parent.is_none()).count()
    }

    #[test]
    fn test_v2_uniform() {
        let img = [100u8; 9];
        let result = make_tree_patch_v2(
            &img,
            3,
            3,
            3,
            0,
            ConnectedType::FourConnected,
            0,
        );

        assert_eq!(count_roots(&result.regions), 1);

        let root_idx = (0..result.regions.len())
            .find(|&i| result.regions.get(i).parent.is_none())
            .unwrap();
        assert_eq!(result.regions.get(root_idx).size, 9);
        assert_eq!(result.regions.get(root_idx).gray_level, 100);
    }

    #[test]
    fn test_v2_two_levels() {
        let img = [100, 100, 100, 100, 50, 100, 100, 100, 100];
        let result = make_tree_patch_v2(
            &img,
            3,
            3,
            3,
            0,
            ConnectedType::FourConnected,
            0,
        );

        assert_eq!(count_roots(&result.regions), 1);

        let inner_idx = (0..result.regions.len())
            .find(|&i| result.regions.get(i).gray_level == 50)
            .unwrap();
        let inner = result.regions.get(inner_idx);
        assert_eq!(inner.size, 1);
        assert!(inner.parent.is_some());

        let outer_idx = inner.parent.unwrap();
        let outer = result.regions.get(outer_idx);
        assert_eq!(outer.gray_level, 100);
        assert!(outer.parent.is_none());
    }

    #[test]
    fn test_v2_gradient() {
        let img = [0u8, 1, 2, 3, 4];
        let result = make_tree_patch_v2(
            &img,
            5,
            1,
            5,
            0,
            ConnectedType::FourConnected,
            0,
        );

        assert_eq!(count_roots(&result.regions), 1);

        let root_idx = (0..result.regions.len())
            .find(|&i| result.regions.get(i).parent.is_none())
            .unwrap();
        assert_eq!(result.regions.get(root_idx).gray_level, 4);

        // Check cumulative sizes
        for level in 0..5u8 {
            let idx = (0..result.regions.len())
                .find(|&i| result.regions.get(i).gray_level == level)
                .unwrap();
            assert_eq!(
                result.regions.get(idx).size,
                (level as i32) + 1,
                "V2: Region at level {} should have size {}",
                level,
                level + 1
            );
        }
    }

    #[test]
    fn test_v2_er_index_mapping() {
        let img = [100, 100, 100, 100, 50, 100, 100, 100, 100];
        let result = make_tree_patch_v2(
            &img,
            3,
            3,
            3,
            0,
            ConnectedType::FourConnected,
            0,
        );

        let the_dir_mask = dir_mask_v2(result.connected_type);
        let wb = result.width_with_boundary;

        // Check that each interior point's er_index maps to a valid region
        for row in 0..result.height {
            for col in 0..result.width {
                let idx = ((row + 1) * wb + (col + 1)) as usize;
                let point_val = result.points[idx];
                let er_idx = (point_val & !the_dir_mask) as usize;

                assert!(
                    er_idx < result.regions.len(),
                    "er_index {} out of range at ({}, {})",
                    er_idx,
                    col,
                    row
                );
            }
        }
    }

    #[test]
    fn test_v2_single_pixel() {
        let img = [42u8];
        let result = make_tree_patch_v2(
            &img,
            1,
            1,
            1,
            0,
            ConnectedType::FourConnected,
            0,
        );

        assert_eq!(result.regions.len(), 1);
        assert_eq!(result.regions.get(0).size, 1);
        assert_eq!(result.regions.get(0).gray_level, 42);
    }

    #[test]
    fn test_v2_8connected_diagonal() {
        // 3x3 diagonal: 0s on diagonal, 100s elsewhere
        // [  0, 100,   0]
        // [100,   0, 100]
        // [  0, 100,   0]
        let img = [0, 100, 0, 100, 0, 100, 0, 100, 0];

        let result_4 = make_tree_patch_v2(
            &img,
            3,
            3,
            3,
            0,
            ConnectedType::FourConnected,
            0,
        );
        let result_8 = make_tree_patch_v2(
            &img,
            3,
            3,
            3,
            0,
            ConnectedType::EightConnected,
            0,
        );

        let count_4_at_0: Vec<_> = (0..result_4.regions.len())
            .filter(|&i| result_4.regions.get(i).gray_level == 0)
            .collect();
        let count_8_at_0: Vec<_> = (0..result_8.regions.len())
            .filter(|&i| result_8.regions.get(i).gray_level == 0)
            .collect();

        assert!(
            count_4_at_0.len() > count_8_at_0.len(),
            "4-conn should split diagonal pixels ({} regions), 8-conn should merge them ({} regions)",
            count_4_at_0.len(),
            count_8_at_0.len()
        );
    }

    #[test]
    fn test_v2_8connected_sizes() {
        // 3x3: all 100 except center=50
        let img = [100, 100, 100, 100, 50, 100, 100, 100, 100];
        let result = make_tree_patch_v2(
            &img,
            3,
            3,
            3,
            0,
            ConnectedType::EightConnected,
            0,
        );

        assert_eq!(count_roots(&result.regions), 1);

        let inner_idx = (0..result.regions.len())
            .find(|&i| result.regions.get(i).gray_level == 50)
            .unwrap();
        assert_eq!(result.regions.get(inner_idx).size, 1);
    }
}
