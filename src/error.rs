use crate::params::MserParams;
use image::GrayImage;

pub type Result<T> = std::result::Result<T, MserError>;

#[derive(Debug, thiserror::Error)]
pub enum MserError {
    #[error("image dimensions must be non-zero, got {width}x{height}")]
    EmptyImageDimensions { width: u32, height: u32 },

    #[error(
        "image buffer length mismatch for {width}x{height}: expected {expected} bytes, got {actual}"
    )]
    ImageBufferLengthMismatch {
        expected: usize,
        actual: usize,
        width: u32,
        height: u32,
    },

    #[error(
        "image dimensions are too large for internal buffers: {width}x{height}"
    )]
    ImageDimensionsTooLarge { width: u32, height: u32 },

    #[error("invalid MSER parameter `{field}`: {message}")]
    InvalidMserParams {
        field: &'static str,
        message: &'static str,
    },

    #[error(
        "invalid number of patches {num_patches}; expected one of 1, 2, 4, 8, 16, or 32"
    )]
    InvalidNumPatches { num_patches: u32 },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Image(#[from] image::ImageError),
}

pub(crate) fn validate_gray_image_input(
    image: &GrayImage,
    params: &MserParams,
) -> Result<()> {
    validate_image_dimensions(image.width(), image.height())?;
    validate_params(params)
}

pub(crate) fn validate_raw_image_input(
    image: &[u8],
    width: u32,
    height: u32,
    params: &MserParams,
) -> Result<()> {
    let expected = validate_image_dimensions(width, height)?;
    if image.len() != expected {
        return Err(MserError::ImageBufferLengthMismatch {
            expected,
            actual: image.len(),
            width,
            height,
        });
    }

    validate_params(params)
}

fn validate_image_dimensions(width: u32, height: u32) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(MserError::EmptyImageDimensions { width, height });
    }

    let expected = checked_image_len(width, height)?;
    checked_padded_image_len(width, height)?;
    Ok(expected)
}

pub(crate) fn checked_image_len(width: u32, height: u32) -> Result<usize> {
    checked_len(width as u64, height as u64, width, height)
}

fn checked_padded_image_len(width: u32, height: u32) -> Result<usize> {
    let padded_width = width as u64 + 2;
    let padded_height = height as u64 + 2;
    let len = checked_len(padded_width, padded_height, width, height)?;
    let max_i32_area = i32::MAX as usize;
    if len > max_i32_area {
        return Err(MserError::ImageDimensionsTooLarge { width, height });
    }
    Ok(len)
}

fn checked_len(
    width: u64,
    height: u64,
    original_width: u32,
    original_height: u32,
) -> Result<usize> {
    let len = width.checked_mul(height).ok_or(
        MserError::ImageDimensionsTooLarge {
            width: original_width,
            height: original_height,
        },
    )?;
    if len > usize::MAX as u64 {
        return Err(MserError::ImageDimensionsTooLarge {
            width: original_width,
            height: original_height,
        });
    }
    Ok(len as usize)
}

fn validate_params(params: &MserParams) -> Result<()> {
    if params.delta <= 0 {
        return invalid_param("delta", "must be greater than zero");
    }
    if params.min_point < 0 {
        return invalid_param("min_point", "must be non-negative");
    }
    if !params.max_point_ratio.is_finite() || params.max_point_ratio <= 0.0 {
        return invalid_param(
            "max_point_ratio",
            "must be finite and greater than zero",
        );
    }
    if !params.stable_variation.is_finite() {
        return invalid_param("stable_variation", "must be finite");
    }
    if !params.duplicated_variation.is_finite() {
        return invalid_param("duplicated_variation", "must be finite");
    }
    if !params.nms_similarity.is_finite() {
        return invalid_param("nms_similarity", "must be finite");
    }

    Ok(())
}

fn invalid_param<T>(field: &'static str, message: &'static str) -> Result<T> {
    Err(MserError::InvalidMserParams { field, message })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_validation_rejects_image_buffer_length_mismatch() {
        let err =
            validate_raw_image_input(&[0; 3], 2, 2, &MserParams::default())
                .unwrap_err();

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
    fn validation_rejects_dimensions_too_large_for_internal_buffers() {
        let err = validate_raw_image_input(
            &[],
            70_000,
            70_000,
            &MserParams::default(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            MserError::ImageDimensionsTooLarge {
                width: 70_000,
                height: 70_000
            }
        ));
    }
}
