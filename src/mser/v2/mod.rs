pub mod build_tree;
pub mod data;
pub mod extract;
pub mod parallel;
pub mod process_patch;
pub mod recognize;

use crate::error::{Result, validate_raw_image_input};
use crate::mser::params::{MserParams, ParallelConfig};
use crate::mser::types::{MserRegion, MserRegions};
use image::GrayImage;

pub use parallel::extract_msers_v2_partitioned;

fn run_v2_pipeline(
    image: &[u8],
    width: u32,
    height: u32,
    params: &MserParams,
    max_point: i32,
    gray_mask: u8,
) -> Vec<MserRegion> {
    let tree = build_tree::make_tree_patch_v2(
        image,
        width,
        height,
        width,
        gray_mask,
        params.connected_type,
        0,
    );

    let mut regions = tree.regions;
    let valid_order = recognize::recognize_mser_v2(
        &mut regions,
        params.delta,
        params.stable_variation,
        params.nms_similarity,
        params.duplicated_variation,
        params.min_point,
        max_point,
    );

    extract::extract_pixels_v2(
        &mut regions,
        &tree.points,
        &valid_order,
        tree.width,
        tree.height,
        tree.width_with_boundary,
        tree.connected_type,
        gray_mask,
    )
}

/// Extract MSERs from a grayscale image using Fast MSER V2 (single-threaded).
pub fn extract_msers_v2(
    image: &GrayImage,
    params: &MserParams,
) -> Result<MserRegions> {
    extract_msers_v2_raw(image.as_raw(), image.width(), image.height(), params)
}

pub(crate) fn extract_msers_v2_raw(
    image: &[u8],
    width: u32,
    height: u32,
    params: &MserParams,
) -> Result<MserRegions> {
    let validated = validate_raw_image_input(image, width, height, params)?;
    let max_point = validated.max_point(params);
    let mut result = MserRegions::default();

    if params.from_min {
        result.from_min =
            run_v2_pipeline(image, width, height, params, max_point, 0);
    }
    if params.from_max {
        result.from_max =
            run_v2_pipeline(image, width, height, params, max_point, 255);
    }

    Ok(result)
}

/// Extract MSERs using Fast MSER V2 with parallel from_min/from_max execution.
pub fn extract_msers_v2_parallel(
    image: &GrayImage,
    params: &MserParams,
    config: &ParallelConfig,
) -> Result<MserRegions> {
    extract_msers_v2_parallel_raw(
        image.as_raw(),
        image.width(),
        image.height(),
        params,
        config,
    )
}

pub(crate) fn extract_msers_v2_parallel_raw(
    image: &[u8],
    width: u32,
    height: u32,
    params: &MserParams,
    _config: &ParallelConfig,
) -> Result<MserRegions> {
    let validated = validate_raw_image_input(image, width, height, params)?;
    let max_point = validated.max_point(params);

    let (from_min, from_max) = rayon::join(
        || {
            if params.from_min {
                run_v2_pipeline(image, width, height, params, max_point, 0)
            } else {
                vec![]
            }
        },
        || {
            if params.from_max {
                run_v2_pipeline(image, width, height, params, max_point, 255)
            } else {
                vec![]
            }
        },
    );

    Ok(MserRegions { from_min, from_max })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mser::params::ConnectedType;

    fn default_params_with(min_point: i32) -> MserParams {
        MserParams {
            min_point,
            ..MserParams::default()
        }
    }

    #[test]
    fn test_v2_e2e_uniform() {
        let img = [128u8; 100];
        let params = default_params_with(1);
        let result = extract_msers_v2_raw(&img, 10, 10, &params).unwrap();

        assert!(result.from_min.is_empty());
        assert!(result.from_max.is_empty());
    }

    #[test]
    fn test_v2_e2e_simple_blob() {
        let mut img = [0u8; 100];
        for r in 3..7 {
            for c in 3..7 {
                img[r * 10 + c] = 200;
            }
        }

        let params = default_params_with(1);
        let result = extract_msers_v2_raw(&img, 10, 10, &params).unwrap();

        let total = result.from_min.len() + result.from_max.len();
        assert!(total > 0, "Should detect at least one MSER");
    }

    #[test]
    fn test_v2_e2e_two_blobs() {
        let mut img = [0u8; 100];
        for r in 1..4 {
            for c in 1..4 {
                img[r * 10 + c] = 200;
            }
        }
        for r in 6..9 {
            for c in 6..9 {
                img[r * 10 + c] = 200;
            }
        }

        let params = default_params_with(1);
        let result = extract_msers_v2_raw(&img, 10, 10, &params).unwrap();

        let total = result.from_min.len() + result.from_max.len();
        assert!(
            total >= 2,
            "Should detect at least two MSERs, got {}",
            total
        );
    }

    #[test]
    fn test_v2_e2e_from_min_and_max() {
        let mut img = [128u8; 100];
        for r in 1..4 {
            for c in 1..4 {
                img[r * 10 + c] = 10;
            }
        }
        for r in 6..9 {
            for c in 6..9 {
                img[r * 10 + c] = 240;
            }
        }

        let params = default_params_with(1);
        let result = extract_msers_v2_raw(&img, 10, 10, &params).unwrap();

        assert!(
            !result.from_min.is_empty() || !result.from_max.is_empty(),
            "Should detect MSERs in at least one channel"
        );
    }

    #[test]
    fn test_v2_e2e_min_point_filter() {
        let mut img = [0u8; 100];
        img[44] = 200;
        img[45] = 200;

        let params = default_params_with(5);
        let result = extract_msers_v2_raw(&img, 10, 10, &params).unwrap();

        let small_msers: Vec<_> = result
            .from_min
            .iter()
            .filter(|m| m.points.len() < 5)
            .collect();
        assert!(
            small_msers.is_empty(),
            "Should filter out regions smaller than min_point"
        );
    }

    #[test]
    fn test_v2_e2e_max_point_filter() {
        let mut img = [200u8; 100];
        img[0] = 0;

        let mut params = MserParams::default();
        params.min_point = 1;
        params.max_point_ratio = 0.1;

        let result = extract_msers_v2_raw(&img, 10, 10, &params).unwrap();

        for mser in &result.from_min {
            assert!(
                mser.points.len() <= 10,
                "Region size {} exceeds max_point=10",
                mser.points.len()
            );
        }
    }

    #[test]
    fn test_v2_e2e_4conn_vs_8conn() {
        let mut img = [0u8; 25];
        img[0] = 100;
        img[6] = 100;
        img[12] = 100;
        img[18] = 100;
        img[24] = 100;

        let mut params_4 = default_params_with(1);
        params_4.connected_type = ConnectedType::FourConnected;

        let mut params_8 = default_params_with(1);
        params_8.connected_type = ConnectedType::EightConnected;

        let _result_4 = extract_msers_v2_raw(&img, 5, 5, &params_4).unwrap();
        let _result_8 = extract_msers_v2_raw(&img, 5, 5, &params_8).unwrap();
    }

    #[test]
    fn test_v2_matches_v1_simple_blob() {
        let mut img = [0u8; 100];
        for r in 3..7 {
            for c in 3..7 {
                img[r * 10 + c] = 200;
            }
        }

        let params = default_params_with(1);
        let v1_result =
            crate::mser::v1::extract_msers_raw(&img, 10, 10, &params).unwrap();
        let v2_result = extract_msers_v2_raw(&img, 10, 10, &params).unwrap();

        assert_eq!(
            v1_result.from_min.len(),
            v2_result.from_min.len(),
            "V1 and V2 should detect same number of from_min MSERs"
        );
        assert_eq!(
            v1_result.from_max.len(),
            v2_result.from_max.len(),
            "V1 and V2 should detect same number of from_max MSERs"
        );

        // Check that sizes match
        for (v1, v2) in v1_result.from_min.iter().zip(v2_result.from_min.iter())
        {
            assert_eq!(v1.gray_level, v2.gray_level);
            assert_eq!(v1.points.len(), v2.points.len());
        }
    }

    #[test]
    fn test_v2_matches_v1_two_blobs() {
        let mut img = [0u8; 100];
        for r in 1..4 {
            for c in 1..4 {
                img[r * 10 + c] = 200;
            }
        }
        for r in 6..9 {
            for c in 6..9 {
                img[r * 10 + c] = 200;
            }
        }

        let params = default_params_with(1);
        let v1_result =
            crate::mser::v1::extract_msers_raw(&img, 10, 10, &params).unwrap();
        let v2_result = extract_msers_v2_raw(&img, 10, 10, &params).unwrap();

        assert_eq!(
            v1_result.from_min.len(),
            v2_result.from_min.len(),
            "V1={}, V2={} from_min MSER count mismatch",
            v1_result.from_min.len(),
            v2_result.from_min.len(),
        );
    }

    #[test]
    fn test_v2_matches_v1_dark_bright() {
        let mut img = [128u8; 100];
        for r in 1..4 {
            for c in 1..4 {
                img[r * 10 + c] = 10;
            }
        }
        for r in 6..9 {
            for c in 6..9 {
                img[r * 10 + c] = 240;
            }
        }

        let params = default_params_with(1);
        let v1_result =
            crate::mser::v1::extract_msers_raw(&img, 10, 10, &params).unwrap();
        let v2_result = extract_msers_v2_raw(&img, 10, 10, &params).unwrap();

        assert_eq!(v1_result.from_min.len(), v2_result.from_min.len());
        assert_eq!(v1_result.from_max.len(), v2_result.from_max.len());
    }

    #[test]
    fn test_v2_matches_v1_gradient_image() {
        let mut img = [0u8; 400]; // 20x20
        for r in 0..20 {
            for c in 0..20 {
                img[r * 20 + c] = (r * 12 + c) as u8;
            }
        }

        let params = MserParams {
            delta: 5,
            min_point: 1,
            max_point_ratio: 0.5,
            stable_variation: 10.0,
            nms_similarity: -1.0,
            duplicated_variation: 0.0,
            ..MserParams::default()
        };
        let v1 =
            crate::mser::v1::extract_msers_raw(&img, 20, 20, &params).unwrap();
        let v2 = extract_msers_v2_raw(&img, 20, 20, &params).unwrap();

        assert_eq!(
            v1.from_min.len(),
            v2.from_min.len(),
            "gradient: V1 from_min={}, V2 from_min={}",
            v1.from_min.len(),
            v2.from_min.len()
        );
        assert_eq!(
            v1.from_max.len(),
            v2.from_max.len(),
            "gradient: V1 from_max={}, V2 from_max={}",
            v1.from_max.len(),
            v2.from_max.len()
        );

        for (v1m, v2m) in v1.from_min.iter().zip(v2.from_min.iter()) {
            assert_eq!(v1m.gray_level, v2m.gray_level);
            assert_eq!(
                v1m.points.len(),
                v2m.points.len(),
                "pixel count mismatch at gray={}",
                v1m.gray_level
            );
        }
    }

    #[test]
    fn test_v2_matches_v1_nested_regions() {
        let mut img = [200u8; 225]; // 15x15, background=200
        for r in 2..13 {
            for c in 2..13 {
                img[r * 15 + c] = 100; // outer ring
            }
        }
        for r in 5..10 {
            for c in 5..10 {
                img[r * 15 + c] = 50; // inner square
            }
        }

        let params = MserParams {
            delta: 2,
            min_point: 1,
            max_point_ratio: 0.9,
            stable_variation: 10.0,
            nms_similarity: -1.0,
            duplicated_variation: 0.0,
            ..MserParams::default()
        };
        let v1 =
            crate::mser::v1::extract_msers_raw(&img, 15, 15, &params).unwrap();
        let v2 = extract_msers_v2_raw(&img, 15, 15, &params).unwrap();

        assert_eq!(v1.from_min.len(), v2.from_min.len());

        for v1m in &v1.from_min {
            let v2m =
                v2.from_min.iter().find(|m| m.gray_level == v1m.gray_level);
            assert!(
                v2m.is_some(),
                "V2 missing region at gray={}",
                v1m.gray_level
            );
            assert_eq!(
                v1m.points.len(),
                v2m.unwrap().points.len(),
                "nested pixel count mismatch at gray={}",
                v1m.gray_level
            );
        }
    }

    #[test]
    fn test_v2_matches_v1_with_nms_and_duplicates() {
        let mut img = [0u8; 400]; // 20x20
        for r in 4..16 {
            for c in 4..16 {
                img[r * 20 + c] = 50;
            }
        }
        for r in 7..13 {
            for c in 7..13 {
                img[r * 20 + c] = 100;
            }
        }
        for r in 9..11 {
            for c in 9..11 {
                img[r * 20 + c] = 150;
            }
        }

        let params = MserParams {
            delta: 3,
            min_point: 2,
            max_point_ratio: 0.8,
            stable_variation: 0.5,
            nms_similarity: 0.3,
            duplicated_variation: 0.2,
            ..MserParams::default()
        };
        let v1 =
            crate::mser::v1::extract_msers_raw(&img, 20, 20, &params).unwrap();
        let v2 = extract_msers_v2_raw(&img, 20, 20, &params).unwrap();

        assert_eq!(
            v1.from_min.len(),
            v2.from_min.len(),
            "nms+dup: V1={}, V2={}",
            v1.from_min.len(),
            v2.from_min.len()
        );
        assert_eq!(
            v1.from_max.len(),
            v2.from_max.len(),
            "nms+dup from_max: V1={}, V2={}",
            v1.from_max.len(),
            v2.from_max.len()
        );
    }

    #[test]
    fn test_v2_gray_mask_symmetry() {
        let mut img = [128u8; 100]; // 10x10
        for r in 2..5 {
            for c in 2..5 {
                img[r * 10 + c] = 50; // dark blob
            }
        }
        for r in 6..9 {
            for c in 6..9 {
                img[r * 10 + c] = 206; // 256-50=206 = symmetric bright
            }
        }

        let params = MserParams {
            delta: 3,
            min_point: 1,
            max_point_ratio: 0.5,
            stable_variation: 10.0,
            nms_similarity: -1.0,
            duplicated_variation: 0.0,
            ..MserParams::default()
        };
        let result = extract_msers_v2_raw(&img, 10, 10, &params).unwrap();

        assert!(!result.from_min.is_empty(), "Should detect from_min MSERs");
        assert!(!result.from_max.is_empty(), "Should detect from_max MSERs");

        for m in &result.from_min {
            for pt in &m.points {
                assert!(
                    pt.x >= 0 && pt.x < 10 && pt.y >= 0 && pt.y < 10,
                    "from_min point out of bounds: ({}, {})",
                    pt.x,
                    pt.y
                );
            }
        }
        for m in &result.from_max {
            for pt in &m.points {
                assert!(
                    pt.x >= 0 && pt.x < 10 && pt.y >= 0 && pt.y < 10,
                    "from_max point out of bounds: ({}, {})",
                    pt.x,
                    pt.y
                );
            }
        }
    }

    #[test]
    fn test_v2_extreme_gray_levels() {
        let mut img = [0u8; 100]; // 10x10, all black
        for r in 3..7 {
            for c in 3..7 {
                img[r * 10 + c] = 255;
            }
        }

        let params = MserParams {
            delta: 1,
            min_point: 1,
            max_point_ratio: 0.9,
            stable_variation: 10.0,
            nms_similarity: -1.0,
            duplicated_variation: 0.0,
            ..MserParams::default()
        };
        let v1 =
            crate::mser::v1::extract_msers_raw(&img, 10, 10, &params).unwrap();
        let v2 = extract_msers_v2_raw(&img, 10, 10, &params).unwrap();

        assert_eq!(
            v1.from_min.len(),
            v2.from_min.len(),
            "extreme gray: V1={}, V2={}",
            v1.from_min.len(),
            v2.from_min.len()
        );
    }

    #[test]
    fn test_v2_pixel_propagation_correctness() {
        let mut img = [200u8; 225]; // 15x15
        for r in 1..14 {
            for c in 1..14 {
                img[r * 15 + c] = 100; // level 100: 12x12 = 144 pixels
            }
        }
        for r in 4..11 {
            for c in 4..11 {
                img[r * 15 + c] = 50; // level 50: 7x7 = 49 pixels
            }
        }
        for r in 6..9 {
            for c in 6..9 {
                img[r * 15 + c] = 10; // level 10: 3x3 = 9 pixels
            }
        }

        let params = MserParams {
            delta: 2,
            min_point: 1,
            max_point_ratio: 0.9,
            stable_variation: 10.0,
            nms_similarity: -1.0,
            duplicated_variation: 0.0,
            ..MserParams::default()
        };
        let result = extract_msers_v2_raw(&img, 15, 15, &params).unwrap();

        let r10 = result.from_min.iter().find(|m| m.gray_level == 10);
        let r50 = result.from_min.iter().find(|m| m.gray_level == 50);
        let r100 = result.from_min.iter().find(|m| m.gray_level == 100);

        if let Some(region) = r10 {
            assert_eq!(
                region.points.len(),
                9,
                "level 10: expected 3x3=9 pixels"
            );
        }
        if let Some(region) = r50 {
            assert_eq!(
                region.points.len(),
                49,
                "level 50: expected 7x7=49 pixels"
            );
        }
        if let Some(region) = r100 {
            // rows 1..14, cols 1..14 → 13x13 = 169 pixels (including inner regions)
            assert_eq!(
                region.points.len(),
                169,
                "level 100: expected 13x13=169 pixels"
            );
        }
    }

    #[test]
    fn test_v2_matches_v1_8connected() {
        let mut img = [0u8; 100]; // 10x10
        img[0] = 100;
        img[11] = 100; // diagonal
        img[22] = 100;
        img[33] = 100;
        img[44] = 100;

        let params = MserParams {
            delta: 3,
            min_point: 1,
            max_point_ratio: 0.8,
            stable_variation: 10.0,
            nms_similarity: -1.0,
            duplicated_variation: 0.0,
            connected_type: ConnectedType::EightConnected,
            ..MserParams::default()
        };
        let v1 =
            crate::mser::v1::extract_msers_raw(&img, 10, 10, &params).unwrap();
        let v2 = extract_msers_v2_raw(&img, 10, 10, &params).unwrap();

        assert_eq!(
            v1.from_min.len(),
            v2.from_min.len(),
            "8conn: V1={}, V2={}",
            v1.from_min.len(),
            v2.from_min.len()
        );
    }
}
