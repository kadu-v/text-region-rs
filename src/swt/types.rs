use super::Rect;

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

#[derive(Clone, Copy, Debug)]
pub struct SwtInput<'a> {
    pub width: u32,
    pub height: u32,
    pub edge: &'a [u8],
    pub gradient_x: &'a [f32],
    pub gradient_y: &'a [f32],
    pub params: SwtParams,
}

#[derive(Clone, Copy, Debug)]
pub struct SwtBgrInput<'a> {
    pub width: u32,
    pub height: u32,
    pub bgr: &'a [u8],
    pub params: SwtParams,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwtImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwtPreprocessed {
    pub width: u32,
    pub height: u32,
    pub gray: Vec<u8>,
    pub edge: Vec<u8>,
    pub gradient_x: Vec<f32>,
    pub gradient_y: Vec<f32>,
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
    pub normalized_swt: Vec<u8>,
    pub draw_bgr: Vec<u8>,
}
