use image::{ImageReader, Rgb, RgbImage};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;
use text_region_rs::error::Result;
use text_region_rs::swt::{Rect, SwtParams, detect_text_swt_with_debug};

fn output_stem(path: &str) -> String {
    let path = Path::new(path);
    let parent = path.parent().and_then(|p| p.to_str()).unwrap_or(".");
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("swt");
    format!("{parent}/{stem}")
}

fn save_rects(path: &str, rects: &[Rect]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "x\ty\twidth\theight")?;
    for rect in rects {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}",
            rect.x, rect.y, rect.width, rect.height
        )?;
    }
    Ok(())
}

fn draw_rects(image: &mut RgbImage, rects: &[Rect]) {
    let colors = [Rgb([0, 0, 255]), Rgb([0, 255, 0]), Rgb([255, 0, 0])];
    for (i, rect) in rects.iter().enumerate() {
        draw_rect(image, *rect, colors[i % colors.len()]);
    }
}

fn draw_rect(image: &mut RgbImage, rect: Rect, color: Rgb<u8>) {
    if rect.width == 0
        || rect.height == 0
        || image.width() == 0
        || image.height() == 0
    {
        return;
    }
    let x0 = rect.x.min(image.width() - 1);
    let y0 = rect.y.min(image.height() - 1);
    let x1 = (rect.x + rect.width.saturating_sub(1)).min(image.width() - 1);
    let y1 = (rect.y + rect.height.saturating_sub(1)).min(image.height() - 1);

    for x in x0..=x1 {
        image.put_pixel(x, y0, color);
        image.put_pixel(x, y1, color);
    }
    for y in y0..=y1 {
        image.put_pixel(x0, y, color);
        image.put_pixel(x1, y, color);
    }
}

fn main() -> Result<()> {
    let image_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "resource/IMG_8364.jpeg".to_string());

    let img = ImageReader::open(&image_path)?.decode()?.to_rgb8();
    let width = img.width();
    let height = img.height();

    let start = Instant::now();
    let output = detect_text_swt_with_debug(&img, SwtParams::default())?;
    let elapsed = start.elapsed();

    let stem = output_stem(&image_path);
    let debug_path = format!("{stem}_swt_debug.png");
    let overlay_path = format!("{stem}_swt_overlay.png");
    let normalized_path = format!("{stem}_swt_normalized.png");
    let letters_path = format!("{stem}_swt_letters.tsv");
    let chains_path = format!("{stem}_swt_chains.tsv");

    output.draw_rgb.save(&debug_path)?;

    let mut overlay_img = img.clone();
    draw_rects(&mut overlay_img, &output.detections.letter_bounding_boxes);
    overlay_img.save(&overlay_path)?;

    output.normalized_swt.save(&normalized_path)?;
    save_rects(&letters_path, &output.detections.letter_bounding_boxes)?;
    save_rects(&chains_path, &output.detections.chain_bounding_boxes)?;

    println!("image: {image_path}");
    println!("size: {width}x{height}");
    println!("elapsed: {elapsed:?}");
    println!(
        "letter boxes: {}",
        output.detections.letter_bounding_boxes.len()
    );
    println!(
        "chain boxes: {}",
        output.detections.chain_bounding_boxes.len()
    );
    println!("saved debug: {debug_path}");
    println!("saved overlay: {overlay_path}");
    println!("saved normalized: {normalized_path}");
    println!("saved letters: {letters_path}");
    println!("saved chains: {chains_path}");

    Ok(())
}
