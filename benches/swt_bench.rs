use criterion::{
    BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use image::{ImageReader, RgbImage, imageops::FilterType};
use std::time::Duration;
use text_region_rs::swt::{
    SwtBgrInput, SwtInput, SwtParams, detect_text_swt,
    detect_text_swt_with_debug, filter_swt_components, stroke_width_transform,
    swt_connected_components, swt_preprocess_bgr,
};

fn load_bgr(path: &str) -> (Vec<u8>, u32, u32) {
    let img = ImageReader::open(path).unwrap().decode().unwrap().to_rgb8();
    let width = img.width();
    let height = img.height();
    (rgb_to_bgr(&img), width, height)
}

fn resize_bgr(
    bgr: &[u8],
    source_width: u32,
    source_height: u32,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let rgb = bgr_to_rgb_image(source_width, source_height, bgr);
    let resized =
        image::imageops::resize(&rgb, width, height, FilterType::Triangle);
    rgb_to_bgr(&resized)
}

fn rgb_to_bgr(rgb: &RgbImage) -> Vec<u8> {
    let mut bgr = Vec::with_capacity(rgb.as_raw().len());
    for pixel in rgb.as_raw().chunks_exact(3) {
        bgr.push(pixel[2]);
        bgr.push(pixel[1]);
        bgr.push(pixel[0]);
    }
    bgr
}

fn bgr_to_rgb_image(width: u32, height: u32, bgr: &[u8]) -> RgbImage {
    let mut rgb = Vec::with_capacity(bgr.len());
    for pixel in bgr.chunks_exact(3) {
        rgb.push(pixel[2]);
        rgb.push(pixel[1]);
        rgb.push(pixel[0]);
    }
    RgbImage::from_raw(width, height, rgb).expect("valid RGB image")
}

fn bench_pipeline_stages(c: &mut Criterion) {
    let (bgr, width, height) = load_bgr("resource/IMG_8364.jpeg");
    let params = SwtParams::default();
    let preprocessed = swt_preprocess_bgr(width, height, &bgr).unwrap();
    let swt_image = stroke_width_transform(SwtInput {
        width,
        height,
        edge: &preprocessed.edge,
        gradient_x: &preprocessed.gradient_x,
        gradient_y: &preprocessed.gradient_y,
        params,
    })
    .unwrap();
    let components = swt_connected_components(&swt_image).unwrap();

    let mut g = c.benchmark_group("swt");
    g.sample_size(10);
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(3));
    g.throughput(Throughput::Elements(width as u64 * height as u64));

    g.bench_function("preprocess_bgr", |b| {
        b.iter(|| swt_preprocess_bgr(width, height, &bgr).unwrap())
    });
    g.bench_function("stroke_width_transform", |b| {
        b.iter(|| {
            stroke_width_transform(SwtInput {
                width,
                height,
                edge: &preprocessed.edge,
                gradient_x: &preprocessed.gradient_x,
                gradient_y: &preprocessed.gradient_y,
                params,
            })
            .unwrap()
        })
    });
    g.bench_function("connected_components", |b| {
        b.iter(|| swt_connected_components(&swt_image).unwrap())
    });
    g.bench_function("filter_components", |b| {
        b.iter(|| {
            filter_swt_components(&swt_image, &components, false).unwrap()
        })
    });
    g.bench_function("detect_text_swt", |b| {
        b.iter(|| {
            detect_text_swt(SwtBgrInput {
                width,
                height,
                bgr: &bgr,
                params,
            })
            .unwrap()
        })
    });
    g.bench_function("detect_text_swt_with_debug", |b| {
        b.iter(|| {
            detect_text_swt_with_debug(SwtBgrInput {
                width,
                height,
                bgr: &bgr,
                params,
            })
            .unwrap()
        })
    });

    g.finish();
}

fn bench_image_sizes(c: &mut Criterion) {
    let (source, source_width, source_height) =
        load_bgr("resource/IMG_8364.jpeg");
    let cases = [(480, 270), (960, 540), (1440, 810), (1920, 1080)]
        .into_iter()
        .map(|(width, height)| {
            let bgr =
                resize_bgr(&source, source_width, source_height, width, height);
            (format!("{width}x{height}"), bgr, width, height)
        })
        .collect::<Vec<_>>();

    let mut g = c.benchmark_group("swt_by_image_size");
    g.sample_size(10);
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(3));

    for (name, bgr, width, height) in &cases {
        let pixels = *width as u64 * *height as u64;
        g.throughput(Throughput::Elements(pixels));
        g.bench_with_input(
            BenchmarkId::new("detect_text_swt", name),
            name,
            |b, _| {
                b.iter(|| {
                    detect_text_swt(SwtBgrInput {
                        width: *width,
                        height: *height,
                        bgr,
                        params: SwtParams::default(),
                    })
                    .unwrap()
                })
            },
        );
    }

    g.finish();
}

criterion_group!(benches, bench_pipeline_stages, bench_image_sizes);
criterion_main!(benches);
