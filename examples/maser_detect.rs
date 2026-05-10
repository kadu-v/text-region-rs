use image::{ImageReader, Rgb, RgbImage};
use std::env;
use text_region_rs::error::Result;
use text_region_rs::mser::{
    ConnectedType, MserParams, MserRegion, extract_msers, extract_msers_v2,
};

fn draw_regions(base: &mut RgbImage, regions: &[MserRegion], color: Rgb<u8>) {
    for r in regions {
        let left = r.bounding_rect.x as i32;
        let top = r.bounding_rect.y as i32;
        let right = left + r.bounding_rect.width as i32 - 1;
        let bottom = top + r.bounding_rect.height as i32 - 1;

        for x in left..=right {
            if x >= 0 && x < base.width() as i32 {
                if top >= 0 && top < base.height() as i32 {
                    base.put_pixel(x as u32, top as u32, color);
                }
                if bottom >= 0 && bottom < base.height() as i32 {
                    base.put_pixel(x as u32, bottom as u32, color);
                }
            }
        }
        for y in top..=bottom {
            if y >= 0 && y < base.height() as i32 {
                if left >= 0 && left < base.width() as i32 {
                    base.put_pixel(left as u32, y as u32, color);
                }
                if right >= 0 && right < base.width() as i32 {
                    base.put_pixel(right as u32, y as u32, color);
                }
            }
        }
    }
}

fn draw_pixels(base: &mut RgbImage, regions: &[MserRegion], color: Rgb<u8>) {
    for r in regions {
        for pt in &r.points {
            if pt.x >= 0
                && pt.x < base.width() as i32
                && pt.y >= 0
                && pt.y < base.height() as i32
            {
                base.put_pixel(pt.x as u32, pt.y as u32, color);
            }
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <image_path> [--v2] [--8conn]", args[0]);
        eprintln!(
            "  Outputs: <image_path>_mser_bbox.png and <image_path>_mser_pixels.png"
        );
        std::process::exit(1);
    }

    let path = &args[1];
    let use_v2 = args.iter().any(|a| a == "--v2");
    let use_8conn = args.iter().any(|a| a == "--8conn");

    let img = ImageReader::open(path)?.decode()?;

    let gray = img.to_luma8();
    let w = gray.width();
    let h = gray.height();

    let total_pixels = (w * h) as f32;
    let min_point = (total_pixels * 0.0001) as i32;

    let params = MserParams {
        delta: 5,
        min_point: min_point.max(50),
        max_point_ratio: 0.05,
        stable_variation: 0.25,
        duplicated_variation: 0.2,
        nms_similarity: 0.5,
        connected_type: if use_8conn {
            ConnectedType::EightConnected
        } else {
            ConnectedType::FourConnected
        },
        ..MserParams::default()
    };

    let version = if use_v2 { "V2" } else { "V1" };
    eprintln!(
        "Detecting MSERs ({}, {:?}) on {}x{} image: {}",
        version, params.connected_type, w, h, path
    );

    let result = if use_v2 {
        extract_msers_v2(&gray, &params)?
    } else {
        extract_msers(&gray, &params)?
    };

    eprintln!(
        "  from_min: {} regions, from_max: {} regions",
        result.from_min.len(),
        result.from_max.len()
    );

    let rgb = img.to_rgb8();

    // --- Bounding box visualization ---
    let mut bbox_img = rgb.clone();
    draw_regions(&mut bbox_img, &result.from_min, Rgb([255, 0, 0]));
    draw_regions(&mut bbox_img, &result.from_max, Rgb([0, 0, 255]));

    let stem = path.rsplit_once('.').map_or(path.as_str(), |(s, _)| s);
    let bbox_path = format!("{}_mser_bbox.png", stem);
    bbox_img.save(&bbox_path)?;
    eprintln!("  Saved bounding boxes: {}", bbox_path);

    // --- Pixel visualization ---
    let mut pixel_img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = gray.as_raw()[(y * w + x) as usize];
            pixel_img.put_pixel(x, y, Rgb([v, v, v]));
        }
    }
    draw_pixels(&mut pixel_img, &result.from_min, Rgb([255, 50, 50]));
    draw_pixels(&mut pixel_img, &result.from_max, Rgb([50, 50, 255]));

    let pixel_path = format!("{}_mser_pixels.png", stem);
    pixel_img.save(&pixel_path)?;
    eprintln!("  Saved pixel regions: {}", pixel_path);

    Ok(())
}
