use image::{GrayImage, ImageBuffer, Luma, RgbImage};

use super::Rect;

pub type GrayF32Image = ImageBuffer<Luma<f32>, Vec<f32>>;
pub type SwtImage = GrayF32Image;

#[derive(Clone, Copy, Debug)]
pub struct SwtParams {
    pub dark_on_light: bool,
    pub canny_low_threshold: f32,
    pub canny_high_threshold: f32,
    pub component_variance_ratio: f32,
    pub max_component_width: f32,
    pub min_component_aspect_ratio: f32,
    pub max_component_aspect_ratio: f32,
    pub chain_pair_distance_scale: f32,
    pub chain_color_distance_threshold: f32,
    pub chain_alignment_angle_rad: f32,
    pub min_chain_components: usize,
}

impl Default for SwtParams {
    fn default() -> Self {
        Self {
            dark_on_light: true,
            canny_low_threshold: 175.0,
            canny_high_threshold: 320.0,
            component_variance_ratio: 2.0,
            max_component_width: 420.0,
            min_component_aspect_ratio: 0.1,
            max_component_aspect_ratio: 10.0,
            chain_pair_distance_scale: 9.0,
            chain_color_distance_threshold: 1600.0,
            chain_alignment_angle_rad: std::f32::consts::PI / 6.0,
            min_chain_components: 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwtPreprocessed {
    pub gray: GrayImage,
    pub edge: GrayImage,
    pub gradient_x: GrayF32Image,
    pub gradient_y: GrayF32Image,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwtComponent {
    pub bounding_rect: Rect,
    pub cx: f32,
    pub cy: f32,
    pub median: f32,
    pub mean: f32,
    pub length: i32,
    pub width: i32,
    pub points: Vec<super::Point>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SwtDetections {
    pub letter_bounding_boxes: Vec<Rect>,
    pub chain_bounding_boxes: Vec<Rect>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwtDebugOutput {
    pub detections: SwtDetections,
    pub preprocessed: SwtPreprocessed,
    pub swt_image: SwtImage,
    pub normalized_swt: GrayImage,
    pub draw_rgb: RgbImage,
}
