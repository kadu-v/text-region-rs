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
pub use chains::detect_text_regions_from_swt;
pub use components::{
    filter_swt_components, normalize_and_scale, swt_connected_components,
};
pub use preprocess::swt_preprocess_bgr;
pub use transform::stroke_width_transform;
pub use types::{
    SwtBgrInput, SwtComponent, SwtDebugOutput, SwtDetections, SwtImage,
    SwtInput, SwtParams, SwtPreprocessed,
};

pub const INVALID_STROKE_WIDTH: f32 = -1.0;
pub(super) const SWT_COMPONENT_RATIO_THRESHOLD: f32 = 3.0;
pub(super) const COMPONENT_ROTATION_STEPS: usize = 36;

pub fn detect_text_swt(input: SwtBgrInput<'_>) -> Result<SwtDetections> {
    Ok(detect_text_swt_with_debug(input)?.detections)
}

pub fn detect_text_swt_with_debug(
    input: SwtBgrInput<'_>,
) -> Result<SwtDebugOutput> {
    validation::validate_bgr_input(input.width, input.height, input.bgr)?;

    let preprocessed =
        swt_preprocess_bgr(input.width, input.height, input.bgr)?;
    let swt_image = stroke_width_transform(SwtInput {
        width: input.width,
        height: input.height,
        edge: &preprocessed.edge,
        gradient_x: &preprocessed.gradient_x,
        gradient_y: &preprocessed.gradient_y,
        params: input.params,
    })?;
    let detections = detect_text_regions_from_swt(&swt_image, input.bgr)?;
    let normalized_swt = normalize_and_scale(&swt_image);
    let draw_bgr = render::render_debug_bgr(
        input.width as usize,
        input.height as usize,
        &detections.letter_bounding_boxes,
    );

    Ok(SwtDebugOutput {
        detections,
        preprocessed,
        swt_image,
        normalized_swt,
        draw_bgr,
    })
}
