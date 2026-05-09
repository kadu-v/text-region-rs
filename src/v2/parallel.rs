use rayon::prelude::*;

use crate::block_memory::BlockMemory;
use crate::error::{Result, checked_image_len, validate_image_input};
use crate::params::{ConnectedType, MserParams, ParallelConfig};
use crate::partition::*;
use crate::types::{MserRegion, MserResult};
use crate::v1::data::RegionFlag;
use crate::v2::build_tree::{TreeBuildResultV2, make_tree_patch_v2};
use crate::v2::data::*;
use crate::v2::{extract, recognize};

struct PatchTreeV2 {
    patch_info: PatchInfo,
    tree: TreeBuildResultV2,
}

struct ConsolidatedTreeV2 {
    regions: BlockMemory<MserRegionV2>,
    points: Vec<u32>,
    width: i32,
    height: i32,
    width_with_boundary: i32,
    connected_type: ConnectedType,
}

fn build_trees_parallel(
    image: &[u8],
    img_width: u32,
    img_height: u32,
    gray_mask: u8,
    connected_type: ConnectedType,
    patches: &[PatchInfo],
) -> Vec<PatchTreeV2> {
    patches
        .par_iter()
        .map(|patch| {
            debug_assert!(patch.x_start + patch.width <= img_width);
            debug_assert!(patch.y_start + patch.height <= img_height);
            let patch_offset = (patch.y_start * img_width + patch.x_start) as usize;
            let sub_image = &image[patch_offset..];
            let tree = make_tree_patch_v2(
                sub_image,
                patch.width,
                patch.height,
                img_width,
                gray_mask,
                connected_type,
                patch.patch_index,
            );
            PatchTreeV2 {
                patch_info: patch.clone(),
                tree,
            }
        })
        .collect()
}

fn consolidate_patch_trees_v2(
    patch_trees: Vec<PatchTreeV2>,
    img_width: u32,
    img_height: u32,
    connected_type: ConnectedType,
) -> ConsolidatedTreeV2 {
    let full_wb = (img_width as i32) + 2;
    let full_hb = (img_height as i32) + 2;
    let full_size = (full_wb * full_hb) as usize;

    let the_dir_mask = dir_mask_v2(connected_type);
    let the_boundary_pixel = boundary_pixel(connected_type);

    let mut full_points = vec![the_boundary_pixel; full_size];
    let mut full_regions = BlockMemory::<MserRegionV2>::new(11);

    for pt in &patch_trees {
        let p = &pt.patch_info;
        let tree = &pt.tree;
        let p_wb = tree.width_with_boundary;
        let region_offset = full_regions.len();

        for i in 0..tree.regions.len() {
            let src = tree.regions.get(i);
            let mut r = src.clone();
            r.er_index = (region_offset + i) as i32;
            if let Some(parent_idx) = r.parent {
                r.parent = Some(parent_idx + region_offset);
            }
            full_regions.add(r);
        }

        for row in 0..p.height {
            for col in 0..p.width {
                let src_idx = ((row as i32 + 1) * p_wb + (col as i32 + 1)) as usize;
                let dst_idx = ((p.y_start as i32 + row as i32 + 1) * full_wb
                    + (p.x_start as i32 + col as i32 + 1)) as usize;

                let point_val = tree.points[src_idx];
                let dir_bits = point_val & the_dir_mask;
                let er_idx = (point_val & !the_dir_mask) as usize;
                let new_er_idx = er_idx + region_offset;
                full_points[dst_idx] = dir_bits | (new_er_idx as u32);
            }
        }
    }

    // Fill border of extended_image with XORed sentinel (matching process_tree_patch_v2 behavior)
    // The boundary_pixel in full_points is already set by the vec! initialization.

    ConsolidatedTreeV2 {
        regions: full_regions,
        points: full_points,
        width: img_width as i32,
        height: img_height as i32,
        width_with_boundary: full_wb,
        connected_type,
    }
}

/// C++ get_set_real_parent_for_merged: skip Merged parents, with path compression.
fn get_real_parent_for_merged(
    regions: &mut BlockMemory<MserRegionV2>,
    idx: usize,
) -> Option<usize> {
    let mut parent = regions.get(idx).parent;
    let mut depth = 0usize;
    let max_depth = regions.len();
    while let Some(p) = parent {
        debug_assert!(p < regions.len(), "region parent index out of range");
        debug_assert!(depth <= max_depth, "cycle detected in merged parent chain");
        if depth > max_depth {
            parent = None;
            break;
        }
        if regions.get(p).region_flag == RegionFlag::Merged {
            parent = regions.get(p).parent;
        } else {
            break;
        }
        depth += 1;
    }
    regions.get_mut(idx).parent = parent;
    parent
}

/// C++ get_real_for_merged: if the endpoint itself was merged, use its real parent.
fn get_real_for_merged(regions: &BlockMemory<MserRegionV2>, idx: usize) -> Option<usize> {
    let mut cur = Some(idx);
    let mut depth = 0usize;
    let max_depth = regions.len();

    while let Some(i) = cur {
        debug_assert!(i < regions.len(), "region index out of range");
        debug_assert!(depth <= max_depth, "cycle detected in merged region chain");
        if depth > max_depth {
            return None;
        }
        if regions.get(i).region_flag != RegionFlag::Merged {
            return Some(i);
        }
        cur = regions.get(i).parent;
        depth += 1;
    }

    None
}

#[cfg(any(debug_assertions, test))]
fn validate_parent_chains(regions: &BlockMemory<MserRegionV2>) {
    let len = regions.len();
    for start in 0..len {
        let mut cur = regions.get(start).parent;
        let mut depth = 0usize;
        while let Some(idx) = cur {
            assert!(idx < len, "region {start} has out-of-range parent {idx}");
            depth += 1;
            assert!(
                depth <= len,
                "cycle or excessive parent depth starting at region {start}"
            );
            cur = regions.get(idx).parent;
        }
    }
}

/// Faithful port of C++ img_fast_mser_v2::connect().
fn connect_v2(regions: &mut BlockMemory<MserRegionV2>, idx_a: usize, idx_b: usize) {
    let mut bigger = get_real_for_merged(regions, idx_a);
    let mut smaller = get_real_for_merged(regions, idx_b);
    let mut pixel_size: i32 = 0;

    loop {
        let s = match smaller {
            Some(s) => s,
            None => break,
        };
        let b = match bigger {
            Some(b) => b,
            None => break,
        };
        if s == b {
            break;
        }

        // Ensure smaller.gray <= bigger.gray
        let (s, b) = if regions.get(s).gray_level > regions.get(b).gray_level {
            (b, s)
        } else {
            (s, b)
        };

        let smaller_parent = get_real_parent_for_merged(regions, s);

        if let Some(sp) = smaller_parent {
            if regions.get(sp).gray_level < regions.get(b).gray_level {
                regions.get_mut(s).size += pixel_size;
                smaller = Some(sp);
                bigger = Some(b);
                continue;
            }
        }

        // Merge: smaller becomes child of bigger (or merged if same gray)
        if regions.get(b).gray_level == regions.get(s).gray_level {
            regions.get_mut(s).region_flag = RegionFlag::Merged;
        }

        let temp_pixel_size = regions.get(s).size + pixel_size;
        pixel_size = regions.get(s).size;
        regions.get_mut(s).size = temp_pixel_size;
        regions.get_mut(s).parent = Some(b);

        smaller = Some(b);
        bigger = smaller_parent;
    }

    // Propagate remaining pixel_size up the tree
    if bigger.is_none() {
        let mut cur = smaller;
        while let Some(c) = cur {
            regions.get_mut(c).size += pixel_size;
            cur = get_real_parent_for_merged(regions, c);
        }
    }
}

fn merge_boundary_v2(
    consolidated: &mut ConsolidatedTreeV2,
    boundary_edges: &[BoundaryEdge],
    patches: &[PatchInfo],
) {
    let wb = consolidated.width_with_boundary;
    let the_dir_mask = dir_mask_v2(consolidated.connected_type);

    for edge in boundary_edges {
        let pa = &patches[edge.patch_a as usize];
        let pb = &patches[edge.patch_b as usize];

        if edge.is_horizontal {
            // pa is above pb: pa's bottom row connects to pb's top row
            let a_row = pa.y_start + pa.height - 1;
            let b_row = pb.y_start;
            let start_col = pa.x_start.min(pb.x_start);
            let len = edge.length;

            for i in 0..len {
                let col = start_col + i;
                let a_idx = ((a_row as i32 + 1) * wb + (col as i32 + 1)) as usize;
                let b_idx = ((b_row as i32 + 1) * wb + (col as i32 + 1)) as usize;
                let er_a = (consolidated.points[a_idx] & !the_dir_mask) as usize;
                let er_b = (consolidated.points[b_idx] & !the_dir_mask) as usize;
                connect_v2(&mut consolidated.regions, er_a, er_b);
            }
        } else {
            // pa is left of pb: pa's right col connects to pb's left col
            let a_col = pa.x_start + pa.width - 1;
            let b_col = pb.x_start;
            let start_row = pa.y_start.min(pb.y_start);
            let len = edge.length;

            for i in 0..len {
                let row = start_row + i;
                let a_idx = ((row as i32 + 1) * wb + (a_col as i32 + 1)) as usize;
                let b_idx = ((row as i32 + 1) * wb + (b_col as i32 + 1)) as usize;
                let er_a = (consolidated.points[a_idx] & !the_dir_mask) as usize;
                let er_b = (consolidated.points[b_idx] & !the_dir_mask) as usize;
                connect_v2(&mut consolidated.regions, er_a, er_b);
            }
        }
    }

    #[cfg(any(debug_assertions, test))]
    validate_parent_chains(&consolidated.regions);
}

fn run_v2_parallel_pipeline(
    image: &[u8],
    width: u32,
    height: u32,
    params: &MserParams,
    max_point: i32,
    gray_mask: u8,
    patches: &[PatchInfo],
    boundary_edges: &[BoundaryEdge],
) -> Vec<MserRegion> {
    if patches.len() <= 1 {
        return crate::v2::run_v2_pipeline(image, width, height, params, max_point, gray_mask);
    }

    let patch_trees = build_trees_parallel(
        image,
        width,
        height,
        gray_mask,
        params.connected_type,
        patches,
    );
    let mut consolidated =
        consolidate_patch_trees_v2(patch_trees, width, height, params.connected_type);
    merge_boundary_v2(&mut consolidated, boundary_edges, patches);

    let valid_order = recognize::recognize_mser_v2(
        &mut consolidated.regions,
        params.delta,
        params.stable_variation,
        params.nms_similarity,
        params.duplicated_variation,
        params.min_point,
        max_point,
    );

    extract::extract_pixels_v2(
        &mut consolidated.regions,
        &consolidated.points,
        &valid_order,
        consolidated.width,
        consolidated.height,
        consolidated.width_with_boundary,
        consolidated.connected_type,
        gray_mask,
    )
}

pub fn extract_msers_v2_partitioned(
    image: &[u8],
    width: u32,
    height: u32,
    params: &MserParams,
    config: &ParallelConfig,
) -> Result<MserResult> {
    validate_image_input(image, width, height, params)?;
    let max_point = (params.max_point_ratio * checked_image_len(width, height)? as f32) as i32;
    let grid = compute_grid_config(config.num_patches)?;
    let patches = compute_patches(width, height, &grid);
    let boundary_edges = compute_boundary_edges(&grid, &patches);

    let (from_min, from_max) = rayon::join(
        || {
            if params.from_min {
                run_v2_parallel_pipeline(
                    image,
                    width,
                    height,
                    params,
                    max_point,
                    0,
                    &patches,
                    &boundary_edges,
                )
            } else {
                vec![]
            }
        },
        || {
            if params.from_max {
                run_v2_parallel_pipeline(
                    image,
                    width,
                    height,
                    params,
                    max_point,
                    255,
                    &patches,
                    &boundary_edges,
                )
            } else {
                vec![]
            }
        },
    );

    Ok(MserResult { from_min, from_max })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ConnectedType;
    use crate::v2::extract_msers_v2;

    fn default_params() -> MserParams {
        MserParams {
            delta: 1,
            min_point: 1,
            max_point_ratio: 1.0,
            stable_variation: 10.0,
            duplicated_variation: 0.0,
            nms_similarity: -1.0,
            connected_type: ConnectedType::FourConnected,
            from_min: true,
            from_max: true,
        }
    }

    fn compare_results(single: &[MserRegion], parallel: &[MserRegion]) {
        assert_eq!(
            single.len(),
            parallel.len(),
            "Region count mismatch: single={}, parallel={}",
            single.len(),
            parallel.len()
        );

        let mut s_sorted: Vec<_> = single
            .iter()
            .map(|r| {
                let mut pts: Vec<_> = r.points.iter().map(|p| (p.x, p.y)).collect();
                pts.sort();
                (r.gray_level, pts)
            })
            .collect();
        s_sorted.sort();

        let mut p_sorted: Vec<_> = parallel
            .iter()
            .map(|r| {
                let mut pts: Vec<_> = r.points.iter().map(|p| (p.x, p.y)).collect();
                pts.sort();
                (r.gray_level, pts)
            })
            .collect();
        p_sorted.sort();

        for (i, (s, p)) in s_sorted.iter().zip(p_sorted.iter()).enumerate() {
            assert_eq!(s.0, p.0, "Gray level mismatch at region {}", i);
            assert_eq!(
                s.1.len(),
                p.1.len(),
                "Point count mismatch at region {} (gray={})",
                i,
                s.0
            );
            assert_eq!(s.1, p.1, "Points mismatch at region {} (gray={})", i, s.0);
        }
    }

    #[test]
    fn test_partition_1_matches_single() {
        let img = vec![100u8; 20 * 20];
        let params = MserParams {
            min_point: 5,
            ..default_params()
        };
        let config = ParallelConfig { num_patches: 1 };
        let single = extract_msers_v2(&img, 20, 20, &params).unwrap();
        let par = extract_msers_v2_partitioned(&img, 20, 20, &params, &config).unwrap();
        compare_results(&single.from_min, &par.from_min);
        compare_results(&single.from_max, &par.from_max);
    }

    #[test]
    fn test_uniform_4patches() {
        let img = vec![100u8; 20 * 20];
        let params = MserParams {
            min_point: 5,
            ..default_params()
        };
        let config = ParallelConfig { num_patches: 4 };
        let single = extract_msers_v2(&img, 20, 20, &params).unwrap();
        let par = extract_msers_v2_partitioned(&img, 20, 20, &params, &config).unwrap();
        compare_results(&single.from_min, &par.from_min);
    }

    #[test]
    fn test_blob_4patches() {
        let mut img = vec![200u8; 20 * 20];
        for r in 5..15 {
            for c in 5..15 {
                img[r * 20 + c] = 50;
            }
        }
        let params = MserParams {
            delta: 5,
            min_point: 5,
            stable_variation: 0.5,
            ..default_params()
        };
        let config = ParallelConfig { num_patches: 4 };
        let single = extract_msers_v2(&img, 20, 20, &params).unwrap();
        let par = extract_msers_v2_partitioned(&img, 20, 20, &params, &config).unwrap();
        compare_results(&single.from_min, &par.from_min);
        compare_results(&single.from_max, &par.from_max);
    }

    #[test]
    fn test_gradient_2patches() {
        let img: Vec<u8> = (0..100).map(|i| (i * 255 / 99) as u8).collect();
        let params = MserParams {
            min_point: 1,
            ..default_params()
        };
        let config = ParallelConfig { num_patches: 2 };
        let single = extract_msers_v2(&img, 10, 10, &params).unwrap();
        let par = extract_msers_v2_partitioned(&img, 10, 10, &params, &config).unwrap();
        compare_results(&single.from_min, &par.from_min);
    }

    #[test]
    fn test_uneven_dimensions() {
        let img = vec![128u8; 15 * 13];
        let params = MserParams {
            min_point: 3,
            ..default_params()
        };
        let config = ParallelConfig { num_patches: 4 };
        let single = extract_msers_v2(&img, 15, 13, &params).unwrap();
        let par = extract_msers_v2_partitioned(&img, 15, 13, &params, &config).unwrap();
        compare_results(&single.from_min, &par.from_min);
    }

    #[test]
    #[should_panic(expected = "cycle or excessive parent depth")]
    fn test_validate_parent_chains_detects_cycle() {
        let mut regions = BlockMemory::<MserRegionV2>::new(4);
        let a = regions.add(MserRegionV2::new());
        let b = regions.add(MserRegionV2::new());
        regions.get_mut(a).parent = Some(b);
        regions.get_mut(b).parent = Some(a);

        validate_parent_chains(&regions);
    }
}
