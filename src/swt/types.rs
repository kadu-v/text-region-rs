use image::{GrayImage, ImageBuffer, Luma, RgbImage};

use super::Rect;

pub type GrayF32Image = ImageBuffer<Luma<f32>, Vec<f32>>;
pub type SwtImage = GrayF32Image;

#[derive(Clone, Copy, Debug)]
pub struct SwtParams {
    pub dark_on_light: bool,
}

impl Default for SwtParams {
    fn default() -> Self {
        Self {
            dark_on_light: true,
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
