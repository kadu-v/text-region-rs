use crate::error::{MserError, Result};

use super::{SwtImage, SwtInput};

pub(super) fn validate_input(input: &SwtInput<'_>) -> Result<usize> {
    if input.width == 0 || input.height == 0 {
        return Err(MserError::EmptyImageDimensions {
            width: input.width,
            height: input.height,
        });
    }

    let len = (input.width as usize)
        .checked_mul(input.height as usize)
        .ok_or(MserError::ImageDimensionsTooLarge {
            width: input.width,
            height: input.height,
        })?;

    validate_len("edge", input.edge.len(), len)?;
    validate_len("gradient_x", input.gradient_x.len(), len)?;
    validate_len("gradient_y", input.gradient_y.len(), len)?;

    Ok(len)
}

pub(super) fn validate_bgr_input(
    width: u32,
    height: u32,
    bgr: &[u8],
) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(MserError::EmptyImageDimensions { width, height });
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(MserError::ImageDimensionsTooLarge { width, height })?;
    let expected = pixels
        .checked_mul(3)
        .ok_or(MserError::ImageDimensionsTooLarge { width, height })?;
    validate_len("bgr", bgr.len(), expected)?;
    Ok(pixels)
}

pub(super) fn validate_swt_image(image: &SwtImage) -> Result<()> {
    if image.width == 0 || image.height == 0 {
        return Err(MserError::EmptyImageDimensions {
            width: image.width,
            height: image.height,
        });
    }
    let expected = (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or(MserError::ImageDimensionsTooLarge {
            width: image.width,
            height: image.height,
        })?;
    validate_len("swt.data", image.data.len(), expected)
}

pub(super) fn validate_len(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<()> {
    if actual != expected {
        return Err(MserError::InvalidSwtInput {
            field,
            message: "buffer length must match width * height",
        });
    }
    Ok(())
}
