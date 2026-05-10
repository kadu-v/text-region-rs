use crate::mser::block_memory::BlockMemory;
use crate::mser::v1::data::RegionFlag;
use crate::mser::v2::data::MserRegionV2;

fn get_real_parent_for_merged(
    regions: &BlockMemory<MserRegionV2>,
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
    regions: &mut BlockMemory<MserRegionV2>,
    idx: usize,
) -> Option<usize> {
    let result = get_real_parent_for_merged(regions, idx);
    regions.get_mut(idx).parent = result;
    result
}

pub fn compute_variations(
    regions: &mut BlockMemory<MserRegionV2>,
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

pub fn apply_nms_and_count(
    regions: &mut BlockMemory<MserRegionV2>,
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

pub fn build_gray_order(
    regions: &BlockMemory<MserRegionV2>,
    region_level_size: &[u32; 257],
    total_unknown: u32,
) -> Vec<usize> {
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

fn get_duplicated_regions(
    regions: &BlockMemory<MserRegionV2>,
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

        if parent.region_flag == RegionFlag::Invalid
            || parent.region_flag == RegionFlag::Valid
        {
            parent_opt = parent.parent;
            continue;
        }

        result.push(parent_idx);
        parent_opt = parent.parent;
    }
}

pub fn remove_duplicates(
    regions: &mut BlockMemory<MserRegionV2>,
    gray_order: &[usize],
    max_point: i32,
    duplicated_variation: f32,
) -> Vec<usize> {
    if duplicated_variation <= 0.0 {
        let mut result = Vec::with_capacity(gray_order.len());
        for &idx in gray_order {
            regions.get_mut(idx).region_flag = RegionFlag::Valid;
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

    let mut valid_order = Vec::new();
    for &idx in gray_order {
        if regions.get(idx).region_flag == RegionFlag::Valid {
            valid_order.push(idx);
        }
    }
    valid_order
}

pub fn recognize_mser_v2(
    regions: &mut BlockMemory<MserRegionV2>,
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

    fn make_chain_v2(levels: &[(u8, i32)]) -> BlockMemory<MserRegionV2> {
        let mut regions = BlockMemory::<MserRegionV2>::new(4);
        let mut indices = Vec::new();
        for &(level, size) in levels {
            let mut r = MserRegionV2::new();
            r.gray_level = level;
            r.size = size;
            r.er_index = indices.len() as i32;
            let idx = regions.add(r);
            indices.push(idx);
        }
        for i in 0..indices.len() - 1 {
            regions.get_mut(indices[i]).parent = Some(indices[i + 1]);
        }
        regions
    }

    #[test]
    fn test_v2_variation_simple_chain() {
        let mut regions = make_chain_v2(&[(0, 1), (1, 2), (2, 4)]);
        compute_variations(&mut regions, 1, 10.0, 0, 1000);

        assert_eq!(regions.get(0).var, 1.0);
        assert_eq!(regions.get(1).var, 1.0);
        assert_eq!(regions.get(2).var, -1.0);
    }

    #[test]
    fn test_v2_variation_no_ancestor() {
        let mut regions = BlockMemory::<MserRegionV2>::new(4);
        let mut r = MserRegionV2::new();
        r.gray_level = 5;
        r.size = 10;
        regions.add(r);

        compute_variations(&mut regions, 1, 10.0, 0, 1000);

        assert_eq!(regions.get(0).var, -1.0);
        assert_eq!(regions.get(0).region_flag, RegionFlag::Invalid);
    }

    #[test]
    fn test_v2_variation_filter() {
        let mut regions = make_chain_v2(&[(0, 1), (1, 10), (2, 100)]);
        compute_variations(&mut regions, 1, 5.0, 0, 1000);

        assert_eq!(regions.get(0).region_flag, RegionFlag::Invalid);
        assert_eq!(regions.get(1).region_flag, RegionFlag::Invalid);
    }

    #[test]
    fn test_v2_size_filter() {
        let mut regions = make_chain_v2(&[(0, 5), (1, 10), (2, 100)]);
        compute_variations(&mut regions, 1, 100.0, 8, 1000);

        assert_eq!(regions.get(0).region_flag, RegionFlag::Invalid);
    }

    #[test]
    fn test_v2_nms_suppress_parent() {
        let mut regions = make_chain_v2(&[(10, 100), (11, 200), (12, 1000)]);
        regions.get_mut(0).var = 0.1;
        regions.get_mut(1).var = 0.8;
        let _ = apply_nms_and_count(&mut regions, 0.0);

        assert_eq!(regions.get(1).region_flag, RegionFlag::Invalid);
    }

    #[test]
    fn test_v2_nms_suppress_child() {
        let mut regions = make_chain_v2(&[(10, 100), (11, 200), (12, 1000)]);
        regions.get_mut(0).var = 0.8;
        regions.get_mut(1).var = 0.1;
        let _ = apply_nms_and_count(&mut regions, 0.0);

        assert_eq!(regions.get(0).region_flag, RegionFlag::Invalid);
    }

    #[test]
    fn test_v2_nms_disabled() {
        let mut regions = make_chain_v2(&[(10, 100), (11, 200), (12, 1000)]);
        regions.get_mut(0).var = 0.1;
        regions.get_mut(1).var = 0.8;
        let _ = apply_nms_and_count(&mut regions, -1.0);

        assert_eq!(regions.get(0).region_flag, RegionFlag::Unknown);
        assert_eq!(regions.get(1).region_flag, RegionFlag::Unknown);
    }

    #[test]
    fn test_v2_no_duplicates() {
        let mut regions = make_chain_v2(&[(0, 10), (1, 100), (2, 1000)]);
        let gray_order = vec![0, 1, 2];
        let result = remove_duplicates(&mut regions, &gray_order, 10000, 0.1);

        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_v2_remove_duplicates_simple() {
        let mut regions = make_chain_v2(&[(0, 100), (1, 101), (2, 102)]);
        let gray_order = vec![0, 1, 2];
        let result = remove_duplicates(&mut regions, &gray_order, 10000, 0.05);

        assert!(result.len() < 3);
    }

    #[test]
    fn test_v2_duplicate_disabled() {
        let mut regions = make_chain_v2(&[(0, 100), (1, 101), (2, 1000)]);
        let gray_order = vec![0, 1, 2];
        let result = remove_duplicates(&mut regions, &gray_order, 10000, 0.0);

        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_v2_gray_order_sorting() {
        let mut regions = BlockMemory::<MserRegionV2>::new(4);
        let levels = [5u8, 2, 8, 2, 5];
        for &level in &levels {
            let mut r = MserRegionV2::new();
            r.gray_level = level;
            r.size = 10;
            regions.add(r);
        }

        let mut level_size = [0u32; 257];
        for i in 0..regions.len() {
            level_size[regions.get(i).gray_level as usize] += 1;
        }

        let order = build_gray_order(&regions, &level_size, 5);

        let ordered_levels: Vec<u8> = order
            .iter()
            .map(|&idx| regions.get(idx).gray_level)
            .collect();
        assert_eq!(ordered_levels, vec![2, 2, 5, 5, 8]);
    }

    #[test]
    fn test_v2_full_pipeline() {
        let mut regions = make_chain_v2(&[(0, 10), (1, 12), (2, 50), (3, 200)]);
        let valid =
            recognize_mser_v2(&mut regions, 1, 10.0, -1.0, 0.0, 1, 1000);

        assert!(!valid.is_empty());
        for &idx in &valid {
            assert_eq!(regions.get(idx).region_flag, RegionFlag::Valid);
        }
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    fn make_chain_v2(levels: &[(u8, i32)]) -> BlockMemory<MserRegionV2> {
        let mut regions = BlockMemory::<MserRegionV2>::new(4);
        let mut indices = Vec::new();
        for &(level, size) in levels {
            let mut r = MserRegionV2::new();
            r.gray_level = level;
            r.size = size;
            r.er_index = indices.len() as i32;
            let idx = regions.add(r);
            indices.push(idx);
        }
        for i in 0..indices.len() - 1 {
            regions.get_mut(indices[i]).parent = Some(indices[i + 1]);
        }
        regions
    }

    #[test]
    fn test_v2_nms_gray_level_255_no_overflow() {
        let mut regions = make_chain_v2(&[(254, 100), (255, 200)]);
        regions.get_mut(0).var = 0.1;
        regions.get_mut(1).var = 0.8;

        let (_, total) = apply_nms_and_count(&mut regions, 0.0);

        assert_eq!(regions.get(1).region_flag, RegionFlag::Invalid);
        assert!(total <= 1);
    }

    #[test]
    fn test_v2_nms_gray_level_255_child_at_max() {
        let mut regions = make_chain_v2(&[(255, 100), (255, 200)]);
        regions.get_mut(0).var = 0.1;
        regions.get_mut(1).var = 0.8;
        let _ = apply_nms_and_count(&mut regions, 0.0);
        assert_eq!(regions.get(0).region_flag, RegionFlag::Unknown);
        assert_eq!(regions.get(1).region_flag, RegionFlag::Unknown);
    }
}
