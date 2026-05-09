use crate::block_memory::BlockMemory;
use crate::heap::BucketHeap;
use crate::params::ConnectedType;
use crate::v1::data::*;
use crate::v1::process_patch::process_tree_patch;

pub struct TreeBuildResult {
    pub regions: BlockMemory<MserRegionV1>,
    pub linked_points: Vec<LinkedPoint>,
    pub linked_points_count: usize,
    pub masked_image: Vec<i16>,
    pub width: i32,
    pub height: i32,
    pub width_with_boundary: i32,
}

pub fn init_comp(
    comp: &mut ConnectedCompV1,
    regions: &mut BlockMemory<MserRegionV1>,
    region_idx: usize,
    patch_index: u8,
) {
    comp.size = 0;
    let region = regions.get_mut(region_idx);
    region.gray_level = comp.gray_level as u8;
    region.region_flag = RegionFlag::Unknown;
    region.size = 0;
    region.unmerged_size = 0;
    region.parent = None;
    region.calculated_var = false;
    region.boundary_region = false;
    region.patch_index = patch_index;
    comp.region_idx = region_idx;
}

pub fn new_region(
    comp: &mut ConnectedCompV1,
    regions: &mut BlockMemory<MserRegionV1>,
    region_idx: usize,
    patch_index: u8,
) {
    let region = regions.get_mut(region_idx);
    region.gray_level = comp.gray_level as u8;
    region.region_flag = RegionFlag::Unknown;
    region.size = 0;
    region.unmerged_size = 0;
    region.parent = None;
    region.calculated_var = false;
    region.patch_index = patch_index;
    comp.region_idx = region_idx;
}

pub fn merge_comp(
    comp_top: &ConnectedCompV1,
    comp_below: &mut ConnectedCompV1,
    regions: &mut BlockMemory<MserRegionV1>,
    linked_points: &mut [LinkedPoint],
) {
    let top_region_idx = comp_top.region_idx;
    let below_region_idx = comp_below.region_idx;

    regions.get_mut(top_region_idx).parent = Some(below_region_idx);

    let top_boundary = regions.get(top_region_idx).boundary_region;
    regions.get_mut(below_region_idx).boundary_region |= top_boundary;

    if comp_below.size > 0 {
        linked_points[comp_below.tail as usize].next = comp_top.head;
        linked_points[comp_top.head as usize].prev = comp_below.tail;
        comp_below.tail = comp_top.tail;

        if comp_top.left < comp_below.left {
            comp_below.left = comp_top.left;
        }
        if comp_top.right > comp_below.right {
            comp_below.right = comp_top.right;
        }
        if comp_top.top < comp_below.top {
            comp_below.top = comp_top.top;
        }
        if comp_top.bottom > comp_below.bottom {
            comp_below.bottom = comp_top.bottom;
        }
    } else {
        comp_below.head = comp_top.head;
        comp_below.tail = comp_top.tail;
        comp_below.left = comp_top.left;
        comp_below.right = comp_top.right;
        comp_below.top = comp_top.top;
        comp_below.bottom = comp_top.bottom;
    }

    comp_below.size += comp_top.size;
}

/// Build the component tree for a single-thread patch.
/// This is the core algorithm porting `img_fast_mser_v1::make_tree_patch`.
pub fn make_tree_patch(
    image_data: &[u8],
    width: u32,
    height: u32,
    img_stride: u32,
    gray_mask: u8,
    connected_type: ConnectedType,
    _min_point: i32,
) -> TreeBuildResult {
    let processed =
        process_tree_patch(image_data, width, height, img_stride, gray_mask);

    let w = width as i32;
    let h = height as i32;
    let wb = processed.width_with_boundary;

    let dir_offsets = compute_dir_offsets(connected_type, wb);
    let the_dir_mask = dir_mask(connected_type);
    let the_dir_max = dir_max(connected_type);

    let mut masked_image = processed.masked_image;
    let total_padded = (wb * processed.height_with_boundary) as usize;
    let mut linked_points = vec![LinkedPoint::default(); total_padded];
    let mut linked_points_count = 0usize;

    let mut heap = BucketHeap::new(&processed.level_size);

    let mut regions = BlockMemory::<MserRegionV1>::new(11); // ~2048 per block

    // Component stack (max 256 levels + 1 sentinel)
    let mut comp_stack = Vec::with_capacity(257);

    // Sentinel component at bottom of stack
    let mut sentinel = ConnectedCompV1::new();
    sentinel.gray_level = 257;
    comp_stack.push(sentinel);

    // Start at top-left interior pixel
    let start_pos = (1 + wb) as usize; // row 1, col 1 in padded image
    let start_gray = (masked_image[start_pos] & GRAY_MASK_BITS) as i16;

    let cur_region_idx = regions.add(MserRegionV1::new());
    let mut cur_comp = ConnectedCompV1::new();
    cur_comp.gray_level = start_gray;
    init_comp(&mut cur_comp, &mut regions, cur_region_idx, 0);

    // Mark as visited
    masked_image[start_pos] |= VISITED_FLAG;

    // Set heap_cur to the gray level of the start pixel
    let mut heap_cur_level = start_gray as usize;

    let mut img_pos = start_pos;

    loop {
        // Explore all directions from current pixel
        let mut dir_val = masked_image[img_pos] & the_dir_mask;
        while dir_val < the_dir_max {
            let dir_idx = (dir_val >> 9) as usize;
            let nbr_pos = (img_pos as i32 + dir_offsets[dir_idx]) as usize;

            if masked_image[nbr_pos] >= 0 {
                // Neighbor not visited
                masked_image[nbr_pos] |= VISITED_FLAG; // mark visited

                let nbr_gray = (masked_image[nbr_pos] & GRAY_MASK_BITS) as i16;
                let cur_gray = (masked_image[img_pos] & GRAY_MASK_BITS) as i16;
                let offset = nbr_gray - cur_gray;

                if offset < 0 {
                    // Neighbor has lower gray: push current to heap, descend
                    heap.push(heap_cur_level, img_pos);
                    masked_image[img_pos] += DIR_SHIFT; // increment direction
                    heap_cur_level =
                        (heap_cur_level as i32 + offset as i32) as usize;

                    img_pos = nbr_pos;

                    // Create new component
                    let new_region_idx = regions.add(MserRegionV1::new());
                    comp_stack.push(cur_comp);
                    cur_comp = ConnectedCompV1::new();
                    cur_comp.gray_level = nbr_gray;
                    init_comp(&mut cur_comp, &mut regions, new_region_idx, 0);
                    continue;
                } else {
                    // Neighbor same or higher: push neighbor to heap
                    let target_level =
                        (heap_cur_level as i32 + offset as i32) as usize;
                    heap.push(target_level, nbr_pos);
                }
            }

            masked_image[img_pos] += DIR_SHIFT; // increment direction
            dir_val = masked_image[img_pos] & the_dir_mask;
        }

        // All directions explored: add pixel to current component
        let _gray = (masked_image[img_pos] & GRAY_MASK_BITS) as u8;
        let flat_idx = img_pos as i32 - (1 + wb); // offset from start of interior
        let py = flat_idx / wb;
        let px = flat_idx - py * wb;

        let pt_idx = linked_points_count;
        linked_points[pt_idx].x = px as u16;
        linked_points[pt_idx].y = py as u16;

        if cur_comp.size > 0 {
            linked_points[pt_idx].next = cur_comp.head;
            linked_points[cur_comp.head as usize].prev = pt_idx as i32;
            linked_points[pt_idx].prev = -1;
            linked_points[pt_idx].ref_ = -1;

            if (px as u16) < cur_comp.left {
                cur_comp.left = px as u16;
            } else if (px as u16) > cur_comp.right {
                cur_comp.right = px as u16;
            }
            if (py as u16) < cur_comp.top {
                cur_comp.top = py as u16;
            } else if (py as u16) > cur_comp.bottom {
                cur_comp.bottom = py as u16;
            }
        } else {
            linked_points[pt_idx].prev = -1;
            linked_points[pt_idx].next = -1;
            linked_points[pt_idx].ref_ = -1;
            cur_comp.tail = pt_idx as i32;
            cur_comp.left = px as u16;
            cur_comp.right = px as u16;
            cur_comp.top = py as u16;
            cur_comp.bottom = py as u16;
        }

        cur_comp.head = pt_idx as i32;
        cur_comp.size += 1;
        linked_points_count += 1;

        // Get next pixel from heap
        if let Some(next_pos) = heap.pop(heap_cur_level) {
            img_pos = next_pos;
        } else {
            // Current bucket exhausted: finalize region
            {
                let region = regions.get_mut(cur_comp.region_idx);
                region.head = cur_comp.head;
                region.tail = cur_comp.tail;
                region.size = cur_comp.size;
                region.unmerged_size = cur_comp.size as u32;
                region.left = cur_comp.left;
                region.right = cur_comp.right;
                region.top = cur_comp.top;
                region.bottom = cur_comp.bottom;
            }

            // Find next non-empty bucket
            let cur_gray = (masked_image[img_pos] & GRAY_MASK_BITS) as usize;
            let mut pixel_val: i16 = 0;
            heap_cur_level += 1;
            for i in (cur_gray + 1)..257 {
                if !heap.is_empty(i) {
                    pixel_val = i as i16;
                    heap_cur_level = i;
                    break;
                }
                heap_cur_level += 1;
            }

            if pixel_val > 0 {
                img_pos = heap.pop(heap_cur_level).unwrap();

                let prev_comp_gray = comp_stack.last().unwrap().gray_level;

                if pixel_val < prev_comp_gray {
                    // New intermediate level
                    let new_region_idx = regions.add(MserRegionV1::new());

                    if cur_comp.region_idx != new_region_idx {
                        regions.get_mut(cur_comp.region_idx).parent =
                            Some(new_region_idx);
                        let boundary =
                            regions.get(cur_comp.region_idx).boundary_region;
                        regions.get_mut(new_region_idx).boundary_region =
                            boundary;
                    }

                    cur_comp.gray_level = pixel_val;
                    new_region(&mut cur_comp, &mut regions, new_region_idx, 0);
                } else {
                    // Merge with previous component(s)
                    loop {
                        let mut prev_comp = comp_stack.pop().unwrap();
                        merge_comp(
                            &cur_comp,
                            &mut prev_comp,
                            &mut regions,
                            &mut linked_points,
                        );
                        cur_comp = prev_comp;

                        if pixel_val <= cur_comp.gray_level {
                            break;
                        }

                        let prev_prev_gray =
                            comp_stack.last().unwrap().gray_level;

                        if pixel_val < prev_prev_gray {
                            let new_region_idx =
                                regions.add(MserRegionV1::new());

                            if cur_comp.region_idx != new_region_idx {
                                regions.get_mut(cur_comp.region_idx).parent =
                                    Some(new_region_idx);
                                let boundary = regions
                                    .get(cur_comp.region_idx)
                                    .boundary_region;
                                regions
                                    .get_mut(new_region_idx)
                                    .boundary_region = boundary;
                            }

                            cur_comp.gray_level = pixel_val;
                            new_region(
                                &mut cur_comp,
                                &mut regions,
                                new_region_idx,
                                0,
                            );
                            break;
                        }
                    }
                }
            } else {
                // No more pixels: done
                break;
            }
        }
    }

    TreeBuildResult {
        regions,
        linked_points,
        linked_points_count,
        masked_image,
        width: w,
        height: h,
        width_with_boundary: wb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_roots(regions: &BlockMemory<MserRegionV1>) -> usize {
        regions.iter().filter(|r| r.parent.is_none()).count()
    }

    #[test]
    fn test_uniform_image() {
        // 3x3 image, all value 100
        let img = [100u8; 9];
        let result =
            make_tree_patch(&img, 3, 3, 3, 0, ConnectedType::FourConnected, 0);

        // Should produce exactly 1 region (the root)
        assert_eq!(count_roots(&result.regions), 1);

        // The root region should have size 9
        let root_idx = (0..result.regions.len())
            .find(|&i| result.regions.get(i).parent.is_none())
            .unwrap();
        assert_eq!(result.regions.get(root_idx).size, 9);
        assert_eq!(result.regions.get(root_idx).gray_level, 100);
    }

    #[test]
    fn test_two_levels() {
        // 3x3 image: center=50, border=100
        // [100, 100, 100]
        // [100,  50, 100]
        // [100, 100, 100]
        let img = [100, 100, 100, 100, 50, 100, 100, 100, 100];
        let result =
            make_tree_patch(&img, 3, 3, 3, 0, ConnectedType::FourConnected, 0);

        // Should have exactly 1 root
        assert_eq!(count_roots(&result.regions), 1);

        // Find the inner region (gray_level=50)
        let inner_idx = (0..result.regions.len())
            .find(|&i| result.regions.get(i).gray_level == 50)
            .unwrap();
        let inner = result.regions.get(inner_idx);
        assert_eq!(inner.size, 1);
        assert!(inner.parent.is_some());

        // The parent should be the outer region
        let outer_idx = inner.parent.unwrap();
        let outer = result.regions.get(outer_idx);
        assert_eq!(outer.gray_level, 100);
        assert!(outer.parent.is_none()); // root
    }

    #[test]
    fn test_gradient_1d() {
        // 1x5 image: [0, 1, 2, 3, 4]
        let img = [0u8, 1, 2, 3, 4];
        let result =
            make_tree_patch(&img, 5, 1, 5, 0, ConnectedType::FourConnected, 0);

        // Should have exactly 1 root
        assert_eq!(count_roots(&result.regions), 1);

        // Should have 5 regions forming a chain
        assert!(result.regions.len() >= 5);

        // The root should be at gray level 4
        let root_idx = (0..result.regions.len())
            .find(|&i| result.regions.get(i).parent.is_none())
            .unwrap();
        assert_eq!(result.regions.get(root_idx).gray_level, 4);
    }

    #[test]
    fn test_gradient_sizes() {
        // 1x5 image: [0, 1, 2, 3, 4]
        let img = [0u8, 1, 2, 3, 4];
        let result =
            make_tree_patch(&img, 5, 1, 5, 0, ConnectedType::FourConnected, 0);

        // Region at level 0 should have size 1
        // Region at level 1 should have size 2 (cumulative)
        // etc.
        for level in 0..5u8 {
            let idx = (0..result.regions.len())
                .find(|&i| result.regions.get(i).gray_level == level)
                .unwrap();
            assert_eq!(
                result.regions.get(idx).size,
                (level as i32) + 1,
                "Region at level {} should have size {}",
                level,
                level + 1
            );
        }
    }

    #[test]
    fn test_parent_chain() {
        // 1x5 image: [0, 1, 2, 3, 4]
        let img = [0u8, 1, 2, 3, 4];
        let result =
            make_tree_patch(&img, 5, 1, 5, 0, ConnectedType::FourConnected, 0);

        // Verify parent chain: region(0) -> region(1) -> region(2) -> region(3) -> region(4)
        for level in 0..4u8 {
            let idx = (0..result.regions.len())
                .find(|&i| result.regions.get(i).gray_level == level)
                .unwrap();
            let parent_idx = result.regions.get(idx).parent.unwrap();
            let parent = result.regions.get(parent_idx);
            assert_eq!(
                parent.gray_level,
                level + 1,
                "Parent of region at level {} should be at level {}",
                level,
                level + 1
            );
        }
    }

    #[test]
    fn test_linked_list_integrity() {
        let img = [100, 100, 100, 100, 50, 100, 100, 100, 100];
        let result =
            make_tree_patch(&img, 3, 3, 3, 0, ConnectedType::FourConnected, 0);

        for i in 0..result.regions.len() {
            let region = result.regions.get(i);
            if region.size == 0 {
                continue;
            }

            // Walk from head to tail, count steps
            let mut count = 0;
            let mut cur = region.head;
            while cur != -1 {
                count += 1;
                cur = result.linked_points[cur as usize].next;
                assert!(count <= region.size, "Linked list longer than size");
            }
            assert_eq!(
                count, region.size,
                "Linked list length {} != region size {} for region {}",
                count, region.size, i
            );
        }
    }

    #[test]
    fn test_8connected_vs_4connected() {
        // 3x3 image with diagonal pattern:
        // [  0, 100,   0]
        // [100,   0, 100]
        // [  0, 100,   0]
        let img = [0, 100, 0, 100, 0, 100, 0, 100, 0];

        let result_4 =
            make_tree_patch(&img, 3, 3, 3, 0, ConnectedType::FourConnected, 0);
        let result_8 =
            make_tree_patch(&img, 3, 3, 3, 0, ConnectedType::EightConnected, 0);

        // With 4-connected, the 0-pixels are not connected (diagonal only)
        // With 8-connected, they form one connected component
        // Both should have different region counts at gray level 0

        let count_4_at_0: Vec<_> = (0..result_4.regions.len())
            .filter(|&i| result_4.regions.get(i).gray_level == 0)
            .collect();
        let count_8_at_0: Vec<_> = (0..result_8.regions.len())
            .filter(|&i| result_8.regions.get(i).gray_level == 0)
            .collect();

        // 4-connected: 5 separate regions at level 0 (each diagonal pixel isolated)
        // 8-connected: 1 region at level 0 (all connected diagonally)
        assert!(
            count_4_at_0.len() > count_8_at_0.len(),
            "4-connected should have more level-0 regions ({}) than 8-connected ({})",
            count_4_at_0.len(),
            count_8_at_0.len()
        );
    }

    #[test]
    fn test_3x3_specific() {
        // 3x3 image:
        // [10, 20, 10]
        // [20, 10, 20]
        // [10, 20, 10]
        let img = [10, 20, 10, 20, 10, 20, 10, 20, 10];
        let result =
            make_tree_patch(&img, 3, 3, 3, 0, ConnectedType::FourConnected, 0);

        // Root should be at level 20 with size 9
        let root_idx = (0..result.regions.len())
            .find(|&i| result.regions.get(i).parent.is_none())
            .unwrap();
        assert_eq!(result.regions.get(root_idx).gray_level, 20);
        assert_eq!(result.regions.get(root_idx).size, 9);

        // Exactly 1 root
        assert_eq!(count_roots(&result.regions), 1);
    }

    #[test]
    fn test_single_pixel() {
        let img = [42u8];
        let result =
            make_tree_patch(&img, 1, 1, 1, 0, ConnectedType::FourConnected, 0);

        assert_eq!(result.regions.len(), 1);
        assert_eq!(result.regions.get(0).size, 1);
        assert_eq!(result.regions.get(0).gray_level, 42);
        assert!(result.regions.get(0).parent.is_none());
    }
}
