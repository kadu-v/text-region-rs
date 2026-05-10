mod chains;
mod components;
mod geometry;
mod preprocess;
mod render;
mod transform;
mod types;
mod validation;

use crate::error::Result;

pub use crate::mser::types::{Point, Rect};
pub use chains::{
    detect_text_regions_from_swt, detect_text_regions_from_swt_with_params,
};
pub use components::{
    filter_swt_components, filter_swt_components_with_params,
    normalize_and_scale, swt_connected_components,
};
pub use preprocess::{swt_preprocess_rgb, swt_preprocess_rgb_with_params};
pub use transform::stroke_width_transform;
pub use types::{
    GrayF32Image, SwtComponent, SwtDebugOutput, SwtDetections, SwtImage,
    SwtParams, SwtPreprocessed,
};

pub const INVALID_STROKE_WIDTH: f32 = -1.0;
pub(super) const SWT_COMPONENT_RATIO_THRESHOLD: f32 = 3.0;
pub(super) const COMPONENT_ROTATION_STEPS: usize = 36;

pub fn detect_text_swt(
    image: &image::RgbImage,
    params: SwtParams,
) -> Result<SwtDetections> {
    Ok(detect_text_swt_with_debug(image, params)?.detections)
}

pub fn detect_text_swt_with_debug(
    image: &image::RgbImage,
    params: SwtParams,
) -> Result<SwtDebugOutput> {
    validation::validate_image_dimensions(image.width(), image.height())?;

    let preprocessed = swt_preprocess_rgb_with_params(image, params)?;
    let swt_image = stroke_width_transform(
        &preprocessed.edge,
        &preprocessed.gradient_x,
        &preprocessed.gradient_y,
        params,
    )?;
    let detections =
        detect_text_regions_from_swt_with_params(&swt_image, image, params)?;
    let normalized_swt = normalize_and_scale(&swt_image);
    let draw_rgb = render::render_debug_rgb(
        image.width() as usize,
        image.height() as usize,
        &detections.letter_bounding_boxes,
    );

    Ok(SwtDebugOutput {
        detections,
        preprocessed,
        swt_image,
        normalized_swt,
        draw_rgb,
    })
}
