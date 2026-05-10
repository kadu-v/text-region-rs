use crate::error::{MserError, Result};

use super::{GrayF32Image, SwtImage};

pub(super) fn validate_image_dimensions(
    width: u32,
    height: u32,
) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(MserError::EmptyImageDimensions { width, height });
    }
    (width as usize)
        .checked_mul(height as usize)
        .ok_or(MserError::ImageDimensionsTooLarge { width, height })
}

pub(super) fn validate_matching_gray_f32(
    field: &'static str,
    image: &GrayF32Image,
    width: u32,
    height: u32,
) -> Result<()> {
    if image.width() != width || image.height() != height {
        return Err(MserError::InvalidSwtInput {
            field,
            message: "image dimensions must match edge image",
        });
    }
    Ok(())
}

pub(super) fn validate_swt_image(image: &SwtImage) -> Result<()> {
    validate_image_dimensions(image.width(), image.height()).map(|_| ())
}
