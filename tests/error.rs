use text_region_rs::error::MserError;
use text_region_rs::partition::compute_grid_config;
use text_region_rs::{MserParams, extract_msers, extract_msers_v2_partitioned};

#[test]
fn rejects_empty_dimensions() {
    let err = extract_msers(&[], 0, 10, &MserParams::default()).unwrap_err();

    assert!(matches!(
        err,
        MserError::EmptyImageDimensions {
            width: 0,
            height: 10
        }
    ));
}

#[test]
fn rejects_image_buffer_length_mismatch() {
    let err = extract_msers(&[0; 3], 2, 2, &MserParams::default()).unwrap_err();

    assert!(matches!(
        err,
        MserError::ImageBufferLengthMismatch {
            expected: 4,
            actual: 3,
            width: 2,
            height: 2
        }
    ));
}

#[test]
fn rejects_dimensions_too_large_for_internal_buffers() {
    let err = extract_msers(&[], 70_000, 70_000, &MserParams::default()).unwrap_err();

    assert!(matches!(
        err,
        MserError::ImageDimensionsTooLarge {
            width: 70_000,
            height: 70_000
        }
    ));
}

#[test]
fn rejects_invalid_mser_params() {
    let params = MserParams {
        delta: 0,
        ..MserParams::default()
    };
    let err = extract_msers(&[0; 4], 2, 2, &params).unwrap_err();

    assert!(matches!(
        err,
        MserError::InvalidMserParams { field: "delta", .. }
    ));
}

#[test]
fn rejects_invalid_patch_count() {
    let err = compute_grid_config(3).unwrap_err();
    assert!(matches!(
        err,
        MserError::InvalidNumPatches { num_patches: 3 }
    ));

    let cfg = text_region_rs::ParallelConfig { num_patches: 3 };
    let err =
        extract_msers_v2_partitioned(&[0; 4], 2, 2, &MserParams::default(), &cfg).unwrap_err();
    assert!(matches!(
        err,
        MserError::InvalidNumPatches { num_patches: 3 }
    ));
}
