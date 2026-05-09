pub mod build_tree;
pub mod data;
pub mod extract;
pub mod process_patch;
pub mod recognize;

use crate::params::{MserParams, ParallelConfig};
use crate::types::{MserRegion, MserResult};

fn run_v1_pipeline(
    image: &[u8],
    width: u32,
    height: u32,
    params: &MserParams,
    max_point: i32,
    gray_mask: u8,
) -> Vec<MserRegion> {
    let mut tree = build_tree::make_tree_patch(
        image,
        width,
        height,
        width,
        gray_mask,
        params.connected_type,
        params.min_point,
    );

    let valid_order = recognize::recognize_mser(
        &mut tree.regions,
        params.delta,
        params.stable_variation,
        params.nms_similarity,
        params.duplicated_variation,
        params.min_point,
        max_point,
    );

    extract::extract_pixels(
        &tree.regions,
        &mut tree.linked_points,
        &valid_order,
        gray_mask,
    )
}

/// Extract MSERs from a grayscale image using Fast MSER V1 (single-threaded).
pub fn extract_msers(image: &[u8], width: u32, height: u32, params: &MserParams) -> MserResult {
    let max_point = (params.max_point_ratio * (width * height) as f32) as i32;
    let mut result = MserResult::default();

    if params.from_min {
        result.from_min = run_v1_pipeline(image, width, height, params, max_point, 0);
    }
    if params.from_max {
        result.from_max = run_v1_pipeline(image, width, height, params, max_point, 255);
    }

    result
}

/// Extract MSERs using Fast MSER V1 with parallel from_min/from_max execution.
pub fn extract_msers_parallel(
    image: &[u8],
    width: u32,
    height: u32,
    params: &MserParams,
    _config: &ParallelConfig,
) -> MserResult {
    let max_point = (params.max_point_ratio * (width * height) as f32) as i32;

    let (from_min, from_max) = rayon::join(
        || {
            if params.from_min {
                run_v1_pipeline(image, width, height, params, max_point, 0)
            } else {
                vec![]
            }
        },
        || {
            if params.from_max {
                run_v1_pipeline(image, width, height, params, max_point, 255)
            } else {
                vec![]
            }
        },
    );

    MserResult { from_min, from_max }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ConnectedType;

    fn default_params_with(min_point: i32) -> MserParams {
        MserParams {
            min_point,
            ..MserParams::default()
        }
    }

    #[test]
    fn test_e2e_uniform() {
        let img = [128u8; 100]; // 10x10 all same value
        let params = default_params_with(1);
        let result = extract_msers(&img, 10, 10, &params);

        // Uniform image → no MSERs (root has no parent)
        assert!(
            result.from_min.is_empty(),
            "Uniform image should produce no MSERs from_min"
        );
        assert!(
            result.from_max.is_empty(),
            "Uniform image should produce no MSERs from_max"
        );
    }

    #[test]
    fn test_e2e_simple_blob() {
        // 10x10 dark background (0) with 4x4 bright square (200) at center
        let mut img = [0u8; 100];
        for r in 3..7 {
            for c in 3..7 {
                img[r * 10 + c] = 200;
            }
        }

        let params = default_params_with(1);
        let result = extract_msers(&img, 10, 10, &params);

        // Should detect the bright blob as MSER from_min (dark → bright)
        let total = result.from_min.len() + result.from_max.len();
        assert!(total > 0, "Should detect at least one MSER");
    }

    #[test]
    fn test_e2e_two_blobs() {
        // 10x10 with two separate bright squares
        let mut img = [0u8; 100];
        // Blob 1: rows 1-3, cols 1-3
        for r in 1..4 {
            for c in 1..4 {
                img[r * 10 + c] = 200;
            }
        }
        // Blob 2: rows 6-8, cols 6-8
        for r in 6..9 {
            for c in 6..9 {
                img[r * 10 + c] = 200;
            }
        }

        let params = default_params_with(1);
        let result = extract_msers(&img, 10, 10, &params);

        let total = result.from_min.len() + result.from_max.len();
        assert!(
            total >= 2,
            "Should detect at least two MSERs, got {}",
            total
        );
    }

    #[test]
    fn test_e2e_from_min_and_max() {
        // Image with both dark and bright blobs
        let mut img = [128u8; 100]; // 10x10 mid-gray
        // Dark blob
        for r in 1..4 {
            for c in 1..4 {
                img[r * 10 + c] = 10;
            }
        }
        // Bright blob
        for r in 6..9 {
            for c in 6..9 {
                img[r * 10 + c] = 240;
            }
        }

        let params = default_params_with(1);
        let result = extract_msers(&img, 10, 10, &params);

        // from_min should detect the dark blob (dark on bright)
        // from_max should detect the bright blob (bright on dark)
        assert!(
            !result.from_min.is_empty() || !result.from_max.is_empty(),
            "Should detect MSERs in at least one channel"
        );
    }

    #[test]
    fn test_e2e_min_point_filter() {
        // Small blob that's below min_point threshold
        let mut img = [0u8; 100]; // 10x10
        img[44] = 200; // single pixel blob
        img[45] = 200; // 2 pixels

        let params = default_params_with(5); // min_point = 5
        let result = extract_msers(&img, 10, 10, &params);

        // The 2-pixel blob should be filtered out by min_point=5
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
    fn test_e2e_max_point_filter() {
        // Large blob occupying most of the image
        let mut img = [200u8; 100]; // 10x10 bright
        img[0] = 0; // tiny dark corner

        let mut params = MserParams::default();
        params.min_point = 1;
        params.max_point_ratio = 0.1; // max 10 pixels

        let result = extract_msers(&img, 10, 10, &params);

        // Regions larger than 10 pixels should be filtered
        for mser in &result.from_min {
            assert!(
                mser.points.len() <= 10,
                "Region size {} exceeds max_point=10",
                mser.points.len()
            );
        }
    }

    #[test]
    fn test_e2e_4conn_vs_8conn() {
        let mut img = [0u8; 25]; // 5x5
        img[0] = 100;
        img[6] = 100;
        img[12] = 100;
        img[18] = 100;
        img[24] = 100;

        let mut params_4 = default_params_with(1);
        params_4.connected_type = ConnectedType::FourConnected;

        let mut params_8 = default_params_with(1);
        params_8.connected_type = ConnectedType::EightConnected;

        let _result_4 = extract_msers(&img, 5, 5, &params_4);
        let _result_8 = extract_msers(&img, 5, 5, &params_8);
    }

    #[test]
    fn test_e2e_extreme_gray_levels() {
        let mut img = [0u8; 100]; // 10x10 all black
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
        let result = extract_msers(&img, 10, 10, &params);

        let total = result.from_min.len() + result.from_max.len();
        assert!(total > 0, "Should detect MSERs with extreme gray levels");

        for m in &result.from_min {
            assert!(
                m.gray_level == 0 || m.gray_level == 255,
                "Unexpected gray level: {}",
                m.gray_level
            );
        }
    }

    #[test]
    fn test_e2e_pixel_propagation_parent_includes_children() {
        let mut img = [200u8; 225]; // 15x15
        for r in 1..14 {
            for c in 1..14 {
                img[r * 15 + c] = 100;
            }
        }
        for r in 4..11 {
            for c in 4..11 {
                img[r * 15 + c] = 50;
            }
        }
        for r in 6..9 {
            for c in 6..9 {
                img[r * 15 + c] = 10;
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
        let result = extract_msers(&img, 15, 15, &params);

        let r10 = result.from_min.iter().find(|m| m.gray_level == 10);
        let r50 = result.from_min.iter().find(|m| m.gray_level == 50);
        let r100 = result.from_min.iter().find(|m| m.gray_level == 100);

        if let Some(region) = r10 {
            assert_eq!(region.points.len(), 9, "V1 level 10: expected 3x3=9 pixels");
        }
        if let Some(region) = r50 {
            assert_eq!(
                region.points.len(),
                49,
                "V1 level 50: expected 7x7=49 pixels"
            );
        }
        if let Some(region) = r100 {
            // rows 1..14, cols 1..14 → 13x13 = 169 pixels (including inner regions)
            assert_eq!(
                region.points.len(),
                169,
                "V1 level 100: expected 13x13=169 pixels"
            );
        }
    }

    #[test]
    fn test_e2e_from_max_gray_mask_255() {
        let mut img = [128u8; 100]; // 10x10
        for r in 3..7 {
            for c in 3..7 {
                img[r * 10 + c] = 200; // bright blob
            }
        }

        let params = MserParams {
            delta: 3,
            min_point: 1,
            max_point_ratio: 0.5,
            stable_variation: 10.0,
            nms_similarity: -1.0,
            duplicated_variation: 0.0,
            from_min: false,
            from_max: true,
            ..MserParams::default()
        };
        let result = extract_msers(&img, 10, 10, &params);

        assert!(
            result.from_min.is_empty(),
            "from_min should be empty when disabled"
        );
        assert!(
            !result.from_max.is_empty(),
            "Should detect bright blob via from_max"
        );

        for m in &result.from_max {
            assert!(
                m.gray_level == 128 || m.gray_level == 200,
                "from_max gray level should be original: got {}",
                m.gray_level
            );
            for pt in &m.points {
                assert!(pt.x >= 0 && pt.x < 10 && pt.y >= 0 && pt.y < 10);
            }
        }
    }

    #[test]
    fn test_e2e_single_pixel_image() {
        let img = [42u8];
        let params = default_params_with(1);
        let result = extract_msers(&img, 1, 1, &params);
        assert!(result.from_min.is_empty());
        assert!(result.from_max.is_empty());
    }

    #[test]
    fn test_e2e_1x2_image() {
        let img = [50u8, 100];
        let params = default_params_with(1);
        let result = extract_msers(&img, 2, 1, &params);

        // With default stable_variation=0.5, var(50)=(2-1)/1=1.0 > 0.5 → Invalid.
        // Root has no parent → Invalid. So no valid MSERs expected.
        // Just verify no crash and coordinates are valid if any exist.
        for m in &result.from_min {
            for pt in &m.points {
                assert!(
                    pt.x >= 0 && pt.x < 2 && pt.y == 0,
                    "1x2 point out of bounds: ({}, {})",
                    pt.x,
                    pt.y
                );
            }
        }
    }

    #[test]
    fn test_e2e_no_duplicate_pixels_in_region() {
        let mut img = [0u8; 100]; // 10x10
        for r in 3..7 {
            for c in 3..7 {
                img[r * 10 + c] = 100;
            }
        }
        for r in 4..6 {
            for c in 4..6 {
                img[r * 10 + c] = 50;
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
        let result = extract_msers(&img, 10, 10, &params);

        for m in &result.from_min {
            let mut sorted = m.points.clone();
            sorted.sort_by(|a, b| (a.y, a.x).cmp(&(b.y, b.x)));
            for w in sorted.windows(2) {
                assert!(
                    w[0] != w[1],
                    "Duplicate pixel ({}, {}) in region gray={}",
                    w[0].x,
                    w[0].y,
                    m.gray_level
                );
            }
        }
    }
}
