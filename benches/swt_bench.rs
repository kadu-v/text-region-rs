use criterion::{
    BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use image::{ImageReader, RgbImage, imageops::FilterType};
use std::time::Duration;
use text_region_rs::swt::{
    SwtParams, detect_text_swt, detect_text_swt_with_debug,
    filter_swt_components, stroke_width_transform, swt_connected_components,
    swt_preprocess_rgb,
};

fn load_rgb(path: &str) -> RgbImage {
    ImageReader::open(path).unwrap().decode().unwrap().to_rgb8()
}

fn resize_rgb(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    image::imageops::resize(image, width, height, FilterType::Triangle)
}

fn bench_pipeline_stages(c: &mut Criterion) {
    let image = load_rgb("resource/IMG_8364.jpeg");
    let width = image.width();
    let height = image.height();
    let params = SwtParams::default();
    let preprocessed = swt_preprocess_rgb(&image).unwrap();
    let swt_image = stroke_width_transform(
        &preprocessed.edge,
        &preprocessed.gradient_x,
        &preprocessed.gradient_y,
        params,
    )
    .unwrap();
    let components = swt_connected_components(&swt_image).unwrap();

    let mut g = c.benchmark_group("swt");
    g.sample_size(10);
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(3));
    g.throughput(Throughput::Elements(width as u64 * height as u64));

    g.bench_function("preprocess_rgb", |b| {
        b.iter(|| swt_preprocess_rgb(&image).unwrap())
    });
    g.bench_function("stroke_width_transform", |b| {
        b.iter(|| {
            stroke_width_transform(
                &preprocessed.edge,
                &preprocessed.gradient_x,
                &preprocessed.gradient_y,
                params,
            )
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
        b.iter(|| detect_text_swt(&image, params).unwrap())
    });
    g.bench_function("detect_text_swt_with_debug", |b| {
        b.iter(|| detect_text_swt_with_debug(&image, params).unwrap())
    });

    g.finish();
}

fn bench_image_sizes(c: &mut Criterion) {
    let source = load_rgb("resource/IMG_8364.jpeg");
    let cases = [(480, 270), (960, 540), (1440, 810), (1920, 1080)]
        .into_iter()
        .map(|(width, height)| {
            let image = resize_rgb(&source, width, height);
            (format!("{width}x{height}"), image)
        })
        .collect::<Vec<_>>();

    let mut g = c.benchmark_group("swt_by_image_size");
    g.sample_size(10);
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(3));

    for (name, image) in &cases {
        let pixels = image.width() as u64 * image.height() as u64;
        g.throughput(Throughput::Elements(pixels));
        g.bench_with_input(
            BenchmarkId::new("detect_text_swt", name),
            name,
            |b, _| {
                b.iter(|| detect_text_swt(image, SwtParams::default()).unwrap())
            },
        );
    }

    g.finish();
}

criterion_group!(benches, bench_pipeline_stages, bench_image_sizes);
criterion_main!(benches);
