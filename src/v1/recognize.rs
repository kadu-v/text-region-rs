use crate::block_memory::BlockMemory;
use crate::v1::data::{MserRegionV1, RegionFlag};

fn get_real_parent_for_merged(
    regions: &BlockMemory<MserRegionV1>,
    idx: usize,
) -> Option<usize> {
    let mut parent_opt = regions.get(idx).parent;
    while let Some(p) = parent_opt {
        if regions.get(p).region_flag != RegionFlag::Merged {
            return Some(p);
        }
        parent_opt = regions.get(p).parent;
    }
    None
}

fn get_set_real_parent_for_merged(
    regions: &mut BlockMemory<MserRegionV1>,
    idx: usize,
) -> Option<usize> {
    let result = get_real_parent_for_merged(regions, idx);
    regions.get_mut(idx).parent = result;
    result
}

/// Compute variation for each region and mark invalid ones.
/// This ports `recognize_mser_parallel_worker`.
pub fn compute_variations(
    regions: &mut BlockMemory<MserRegionV1>,
    delta: i32,
    stable_variation: f32,
    min_point: i32,
    max_point: i32,
) {
    for i in 0..regions.len() {
        if regions.get(i).region_flag == RegionFlag::Merged {
            continue;
        }

        let gray_level_threshold = regions.get(i).gray_level as i32 + delta;
        let region_size = regions.get(i).size;

        let mut start_idx = i;
        let mut parent_opt = get_set_real_parent_for_merged(regions, start_idx);

        while let Some(parent_idx) = parent_opt {
            if (regions.get(parent_idx).gray_level as i32)
                > gray_level_threshold
            {
                break;
            }
            start_idx = parent_idx;
            parent_opt = get_real_parent_for_merged(regions, parent_idx);
        }

        let var = if parent_opt.is_some()
            || regions.get(start_idx).gray_level as i32 == gray_level_threshold
        {
            (regions.get(start_idx).size - region_size) as f32
                / region_size as f32
        } else {
            -1.0
        };

        regions.get_mut(i).var = var;

        if var > stable_variation {
            regions.get_mut(i).region_flag = RegionFlag::Invalid;
        } else if region_size < min_point
            || region_size > max_point
            || regions.get(i).parent.is_none()
        {
            regions.get_mut(i).region_flag = RegionFlag::Invalid;
        }
    }
}

/// Apply non-maximum suppression and count region level sizes.
/// Returns (region_level_size, total_unknown_count).
pub fn apply_nms_and_count(
    regions: &mut BlockMemory<MserRegionV1>,
    nms_similarity: f32,
) -> ([u32; 257], u32) {
    let mut region_level_size = [0u32; 257];
    let mut total_unknown = 0u32;

    for i in 0..regions.len() {
        if regions.get(i).region_flag == RegionFlag::Merged {
            continue;
        }

        let parent_opt = regions.get(i).parent;

        if regions.get(i).region_flag == RegionFlag::Unknown {
            region_level_size[regions.get(i).gray_level as usize] += 1;
            total_unknown += 1;
            regions.get_mut(i).calculated_var = true;
        } else if parent_opt.is_none() {
            continue;
        } else {
            let parent_idx = parent_opt.unwrap();
            if regions.get(parent_idx).region_flag == RegionFlag::Invalid {
                continue;
            }
        }

        if let Some(parent_idx) = parent_opt {
            if regions.get(parent_idx).region_flag == RegionFlag::Merged {
                continue;
            }

            if nms_similarity >= 0.0
                && regions.get(i).var >= 0.0
                && regions.get(parent_idx).var >= 0.0
                && regions.get(parent_idx).gray_level as u16
                    == regions.get(i).gray_level as u16 + 1
            {
                let sub_value = regions.get(parent_idx).var as f64
                    - regions.get(i).var as f64;
                if sub_value > nms_similarity as f64 {
                    if regions.get(parent_idx).region_flag
                        == RegionFlag::Unknown
                    {
                        if regions.get(parent_idx).calculated_var {
                            region_level_size[regions.get(parent_idx).gray_level
                                as usize] -= 1;
                            total_unknown -= 1;
                        }
                        regions.get_mut(parent_idx).region_flag =
                            RegionFlag::Invalid;
                    }
                } else if -sub_value > nms_similarity as f64 {
                    if regions.get(i).region_flag == RegionFlag::Unknown {
                        if regions.get(i).calculated_var {
                            region_level_size
                                [regions.get(i).gray_level as usize] -= 1;
                            total_unknown -= 1;
                        }
                        regions.get_mut(i).region_flag = RegionFlag::Invalid;
                    }
                }
            }
        }
    }

    (region_level_size, total_unknown)
}

/// Build gray-ordered list of region indices.
pub fn build_gray_order(
    regions: &BlockMemory<MserRegionV1>,
    region_level_size: &[u32; 257],
    total_unknown: u32,
) -> Vec<usize> {
    // Build integral array (prefix sum) for counting sort
    let mut start_indexes = [0u32; 257];
    start_indexes[0] = 0;
    for i in 1..257 {
        start_indexes[i] = start_indexes[i - 1] + region_level_size[i - 1];
    }

    let mut gray_order = vec![0usize; total_unknown as usize];

    for i in 0..regions.len() {
        if regions.get(i).region_flag == RegionFlag::Unknown {
            let level = regions.get(i).gray_level as usize;
            gray_order[start_indexes[level] as usize] = i;
            start_indexes[level] += 1;
        }
    }

    gray_order
}

/// Find duplicated regions from a stable region upward.
fn get_duplicated_regions(
    regions: &BlockMemory<MserRegionV1>,
    stable_idx: usize,
    begin_idx: usize,
    max_point: i32,
    duplicated_variation: f64,
    result: &mut Vec<usize>,
) {
    let stable_size = regions.get(stable_idx).size;
    let mut parent_opt = regions.get(begin_idx).parent;

    while let Some(parent_idx) = parent_opt {
        let parent = regions.get(parent_idx);

        if parent.size > max_point {
            break;
        }

        let variation = (parent.size - stable_size) as f64 / stable_size as f64;
        if variation > duplicated_variation {
            break;
        }

        if parent.region_flag == RegionFlag::Invalid {
            parent_opt = parent.parent;
            continue;
        }

        result.push(parent_idx);
        parent_opt = parent.parent;
    }
}

/// Remove duplicated regions, keeping the middle one.
pub fn remove_duplicates(
    regions: &mut BlockMemory<MserRegionV1>,
    gray_order: &[usize],
    max_point: i32,
    duplicated_variation: f32,
) -> Vec<usize> {
    if duplicated_variation <= 0.0 {
        // No duplicate removal: mark all as valid
        let mut result = Vec::with_capacity(gray_order.len());
        let mut _total_pixels = 0i64;
        for &idx in gray_order {
            regions.get_mut(idx).region_flag = RegionFlag::Valid;
            _total_pixels += regions.get(idx).size as i64;
            result.push(idx);
        }
        return result;
    }

    let dup_var = duplicated_variation as f64;
    let mut helper = Vec::with_capacity(100);

    for &idx in gray_order {
        if regions.get(idx).region_flag != RegionFlag::Unknown {
            continue;
        }

        helper.clear();
        helper.push(idx);
        get_duplicated_regions(
            regions,
            idx,
            idx,
            max_point,
            dup_var,
            &mut helper,
        );

        let middle_index = helper.len() / 2;

        if middle_index > 0 {
            let middle_idx = helper[middle_index];
            let last_idx = *helper.last().unwrap();
            get_duplicated_regions(
                regions,
                middle_idx,
                last_idx,
                max_point,
                dup_var,
                &mut helper,
            );
        }

        for (j, &region_idx) in helper.iter().enumerate() {
            if j != middle_index {
                regions.get_mut(region_idx).region_flag = RegionFlag::Invalid;
            } else {
                regions.get_mut(region_idx).region_flag = RegionFlag::Valid;
            }
        }
    }

    // Collect valid regions in gray order
    let mut valid_order = Vec::new();
    for &idx in gray_order {
        if regions.get(idx).region_flag == RegionFlag::Valid {
            valid_order.push(idx);
        }
    }
    valid_order
}

/// Full MSER recognition pipeline.
/// Returns gray-ordered list of valid region indices.
pub fn recognize_mser(
    regions: &mut BlockMemory<MserRegionV1>,
    delta: i32,
    stable_variation: f32,
    nms_similarity: f32,
    duplicated_variation: f32,
    min_point: i32,
    max_point: i32,
) -> Vec<usize> {
    compute_variations(regions, delta, stable_variation, min_point, max_point);
    let (region_level_size, total_unknown) =
        apply_nms_and_count(regions, nms_similarity);
    let gray_order =
        build_gray_order(regions, &region_level_size, total_unknown);
    remove_duplicates(regions, &gray_order, max_point, duplicated_variation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain(levels: &[(u8, i32)]) -> BlockMemory<MserRegionV1> {
        let mut regions = BlockMemory::<MserRegionV1>::new(4);
        let mut indices = Vec::new();
        for &(level, size) in levels {
            let mut r = MserRegionV1::new();
            r.gray_level = level;
            r.size = size;
            let idx = regions.add(r);
            indices.push(idx);
        }
        // Set up parent chain: each region's parent is the next one
        for i in 0..indices.len() - 1 {
            regions.get_mut(indices[i]).parent = Some(indices[i + 1]);
        }
        regions
    }

    #[test]
    fn test_variation_simple_chain() {
        // Chain: level 0 (size 1) -> level 1 (size 2) -> level 2 (size 4)
        let mut regions = make_chain(&[(0, 1), (1, 2), (2, 4)]);

        compute_variations(&mut regions, 1, 10.0, 0, 1000);

        // var(0) = (2-1)/1 = 1.0
        assert_eq!(regions.get(0).var, 1.0);
        // var(1) = (4-2)/2 = 1.0
        assert_eq!(regions.get(1).var, 1.0);
        // var(2): parent is None, gray_level=2, threshold=3, no ancestor → -1
        assert_eq!(regions.get(2).var, -1.0);
    }

    #[test]
    fn test_variation_no_ancestor() {
        // Single region with no parent
        let mut regions = BlockMemory::<MserRegionV1>::new(4);
        let mut r = MserRegionV1::new();
        r.gray_level = 5;
        r.size = 10;
        regions.add(r);

        compute_variations(&mut regions, 1, 10.0, 0, 1000);

        assert_eq!(regions.get(0).var, -1.0);
        assert_eq!(regions.get(0).region_flag, RegionFlag::Invalid);
    }

    #[test]
    fn test_variation_filter() {
        let mut regions = make_chain(&[(0, 1), (1, 10), (2, 100)]);
        // delta=1: var(0) = (10-1)/1 = 9.0, var(1) = (100-10)/10 = 9.0
        compute_variations(&mut regions, 1, 5.0, 0, 1000);

        // Both should be Invalid because var > stable_variation
        assert_eq!(regions.get(0).region_flag, RegionFlag::Invalid);
        assert_eq!(regions.get(1).region_flag, RegionFlag::Invalid);
    }

    #[test]
    fn test_size_filter() {
        let mut regions = make_chain(&[(0, 5), (1, 10), (2, 100)]);
        compute_variations(&mut regions, 1, 100.0, 8, 1000);

        // region 0: size=5 < min_point=8 → Invalid
        assert_eq!(regions.get(0).region_flag, RegionFlag::Invalid);
    }

    #[test]
    fn test_var_negative_one() {
        // Chain that doesn't reach gray_level + delta
        let mut regions = make_chain(&[(0, 1), (1, 2)]);
        compute_variations(&mut regions, 5, 10.0, 0, 1000);

        // delta=5, threshold=5. Region 0 at level 0, parent at level 1.
        // Can't reach level 5, and start_region.gray_level(1) != 5 → var = -1
        assert_eq!(regions.get(0).var, -1.0);
    }

    #[test]
    fn test_nms_suppress_parent() {
        let mut regions = make_chain(&[(10, 100), (11, 200), (12, 1000)]);
        // Manually set vars
        regions.get_mut(0).var = 0.1;
        regions.get_mut(1).var = 0.8;
        // NMS: parent(level 11).var - child(level 10).var = 0.7 > 0 → suppress parent
        let _ = apply_nms_and_count(&mut regions, 0.0);

        assert_eq!(regions.get(1).region_flag, RegionFlag::Invalid);
    }

    #[test]
    fn test_nms_suppress_child() {
        let mut regions = make_chain(&[(10, 100), (11, 200), (12, 1000)]);
        regions.get_mut(0).var = 0.8;
        regions.get_mut(1).var = 0.1;
        let _ = apply_nms_and_count(&mut regions, 0.0);

        assert_eq!(regions.get(0).region_flag, RegionFlag::Invalid);
    }

    #[test]
    fn test_nms_within_threshold() {
        let mut regions = make_chain(&[(10, 100), (11, 200), (12, 1000)]);
        regions.get_mut(0).var = 0.4;
        regions.get_mut(1).var = 0.5;
        // Difference = 0.1, nms_similarity = 0.5 → not suppressed
        let _ = apply_nms_and_count(&mut regions, 0.5);

        // Both should remain Unknown
        assert_eq!(regions.get(0).region_flag, RegionFlag::Unknown);
        assert_eq!(regions.get(1).region_flag, RegionFlag::Unknown);
    }

    #[test]
    fn test_nms_disabled() {
        let mut regions = make_chain(&[(10, 100), (11, 200), (12, 1000)]);
        regions.get_mut(0).var = 0.1;
        regions.get_mut(1).var = 0.8;
        let (_, _) = apply_nms_and_count(&mut regions, -1.0);

        // NMS disabled → nothing suppressed
        assert_eq!(regions.get(0).region_flag, RegionFlag::Unknown);
        assert_eq!(regions.get(1).region_flag, RegionFlag::Unknown);
    }

    #[test]
    fn test_no_duplicates() {
        let mut regions = make_chain(&[(0, 10), (1, 100), (2, 1000)]);
        // All Unknown
        let gray_order = vec![0, 1, 2];
        let result = remove_duplicates(&mut regions, &gray_order, 10000, 0.1);

        // Sizes differ significantly → no duplicates removed → all valid
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_remove_duplicates_simple() {
        // Three regions with very similar sizes
        let mut regions = make_chain(&[(0, 100), (1, 101), (2, 102)]);
        // All set to Unknown
        let gray_order = vec![0, 1, 2];
        // duplicated_variation = 0.05 → (101-100)/100 = 0.01 < 0.05 → duplicate
        let result = remove_duplicates(&mut regions, &gray_order, 10000, 0.05);

        // Should keep only the middle one
        assert!(result.len() < 3);
    }

    #[test]
    fn test_duplicate_disabled() {
        let mut regions = make_chain(&[(0, 100), (1, 101), (2, 1000)]);
        let gray_order = vec![0, 1, 2];
        let result = remove_duplicates(&mut regions, &gray_order, 10000, 0.0);

        // Disabled → all marked valid
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_gray_order_sorting() {
        let mut regions = BlockMemory::<MserRegionV1>::new(4);
        let levels = [5u8, 2, 8, 2, 5];
        for &level in &levels {
            let mut r = MserRegionV1::new();
            r.gray_level = level;
            r.size = 10;
            regions.add(r);
        }

        let mut level_size = [0u32; 257];
        for i in 0..regions.len() {
            level_size[regions.get(i).gray_level as usize] += 1;
        }

        let order = build_gray_order(&regions, &level_size, 5);

        // Should be sorted by gray level
        let ordered_levels: Vec<u8> = order
            .iter()
            .map(|&idx| regions.get(idx).gray_level)
            .collect();
        assert_eq!(ordered_levels, vec![2, 2, 5, 5, 8]);
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    fn make_chain(levels: &[(u8, i32)]) -> BlockMemory<MserRegionV1> {
        let mut regions = BlockMemory::<MserRegionV1>::new(4);
        let mut indices = Vec::new();
        for &(level, size) in levels {
            let mut r = MserRegionV1::new();
            r.gray_level = level;
            r.size = size;
            let idx = regions.add(r);
            indices.push(idx);
        }
        for i in 0..indices.len() - 1 {
            regions.get_mut(indices[i]).parent = Some(indices[i + 1]);
        }
        regions
    }

    #[test]
    fn test_nms_gray_level_255_no_overflow() {
        // Regression: gray_level + 1 used to overflow u8 when gray_level=255.
        // C++ promotes u8 to int automatically; Rust u8 wraps/panics.
        let mut regions = make_chain(&[(254, 100), (255, 200)]);
        regions.get_mut(0).var = 0.1;
        regions.get_mut(1).var = 0.8;

        // This should NOT panic and should correctly suppress parent at level 255
        let (_, total) = apply_nms_and_count(&mut regions, 0.0);

        assert_eq!(regions.get(1).region_flag, RegionFlag::Invalid);
        assert!(total <= 1);
    }

    #[test]
    fn test_nms_gray_level_255_child_at_max() {
        // When the child is at gray level 255, gray_level + 1 = 256 in C++.
        // No parent can have gray level 256, so NMS should not fire.
        let mut regions = make_chain(&[(255, 100), (255, 200)]);
        regions.get_mut(0).var = 0.1;
        regions.get_mut(1).var = 0.8;
        // parent gray (255) != child gray (255) + 1 (256) → NMS not applied
        let _ = apply_nms_and_count(&mut regions, 0.0);
        // Both remain Unknown since NMS condition (parent == child+1) fails
        assert_eq!(regions.get(0).region_flag, RegionFlag::Unknown);
        assert_eq!(regions.get(1).region_flag, RegionFlag::Unknown);
    }
}
