use image::GrayImage;
use text_region_rs::error::MserError;
use text_region_rs::partition::compute_grid_config;
use text_region_rs::{MserParams, extract_msers, extract_msers_v2_partitioned};

#[test]
fn rejects_empty_dimensions() {
    let image = GrayImage::new(0, 10);
    let err = extract_msers(&image, &MserParams::default()).unwrap_err();

    assert!(matches!(
        err,
        MserError::EmptyImageDimensions {
            width: 0,
            height: 10
        }
    ));
}

#[test]
fn rejects_empty_height() {
    let image = GrayImage::new(10, 0);
    let err = extract_msers(&image, &MserParams::default()).unwrap_err();

    assert!(matches!(
        err,
        MserError::EmptyImageDimensions {
            width: 10,
            height: 0
        }
    ));
}

#[test]
fn rejects_invalid_mser_params() {
    let params = MserParams {
        delta: 0,
        ..MserParams::default()
    };
    let image = GrayImage::new(2, 2);
    let err = extract_msers(&image, &params).unwrap_err();

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
    let image = GrayImage::new(2, 2);
    let err =
        extract_msers_v2_partitioned(&image, &MserParams::default(), &cfg)
            .unwrap_err();
    assert!(matches!(
        err,
        MserError::InvalidNumPatches { num_patches: 3 }
    ));
}
