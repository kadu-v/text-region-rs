use criterion::{
    BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use image::{GrayImage, ImageReader, imageops::FilterType};
use std::time::Duration;
use text_region_rs::params::{MserParams, ParallelConfig};
use text_region_rs::{
    extract_msers, extract_msers_parallel, extract_msers_v2,
    extract_msers_v2_parallel, extract_msers_v2_partitioned,
};

fn load_grayscale(path: &str) -> (GrayImage, u32, u32) {
    let img = ImageReader::open(path).unwrap().decode().unwrap();
    let gray = img.to_luma8();
    let w = gray.width();
    let h = gray.height();
    (gray, w, h)
}

fn default_detect_params(w: u32, h: u32) -> MserParams {
    let total_pixels = (w * h) as f32;
    let min_point = (total_pixels * 0.0001) as i32;
    MserParams {
        delta: 5,
        min_point: min_point.max(50),
        max_point_ratio: 0.05,
        stable_variation: 0.25,
        duplicated_variation: 0.2,
        nms_similarity: 0.5,
        ..MserParams::default()
    }
}

fn resize_grayscale(image: &GrayImage, width: u32, height: u32) -> GrayImage {
    image::imageops::resize(image, width, height, FilterType::Triangle)
}

fn bench_all(c: &mut Criterion) {
    let (img_p, wp, hp) = load_grayscale("resource/IMG_8237.jpeg");
    let (img_l, wl, hl) = load_grayscale("resource/IMG_8364.jpeg");
    let pp = default_detect_params(wp, hp);
    let pl = default_detect_params(wl, hl);
    let cfg = ParallelConfig::default();

    let mut g = c.benchmark_group("mser");
    g.sample_size(10);
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(3));

    // Paper image
    g.bench_function("paper/v1_single", |b| {
        b.iter(|| extract_msers(&img_p, &pp).unwrap())
    });
    g.bench_function("paper/v1_par_minmax", |b| {
        b.iter(|| extract_msers_parallel(&img_p, &pp, &cfg).unwrap())
    });
    g.bench_function("paper/v2_single", |b| {
        b.iter(|| extract_msers_v2(&img_p, &pp).unwrap())
    });
    g.bench_function("paper/v2_par_minmax", |b| {
        b.iter(|| extract_msers_v2_parallel(&img_p, &pp, &cfg).unwrap())
    });
    for n in [2, 4, 8] {
        let c2 = ParallelConfig { num_patches: n };
        g.bench_with_input(
            BenchmarkId::new("paper/v2_partitioned", n),
            &n,
            |b, _| {
                b.iter(|| {
                    extract_msers_v2_partitioned(&img_p, &pp, &c2).unwrap()
                })
            },
        );
    }

    // Label image
    g.bench_function("label/v1_single", |b| {
        b.iter(|| extract_msers(&img_l, &pl).unwrap())
    });
    g.bench_function("label/v1_par_minmax", |b| {
        b.iter(|| extract_msers_parallel(&img_l, &pl, &cfg).unwrap())
    });
    g.bench_function("label/v2_single", |b| {
        b.iter(|| extract_msers_v2(&img_l, &pl).unwrap())
    });
    g.bench_function("label/v2_par_minmax", |b| {
        b.iter(|| extract_msers_v2_parallel(&img_l, &pl, &cfg).unwrap())
    });
    for n in [2, 4, 8] {
        let c2 = ParallelConfig { num_patches: n };
        g.bench_with_input(
            BenchmarkId::new("label/v2_partitioned", n),
            &n,
            |b, _| {
                b.iter(|| {
                    extract_msers_v2_partitioned(&img_l, &pl, &c2).unwrap()
                })
            },
        );
    }

    g.finish();
}

fn bench_image_sizes(c: &mut Criterion) {
    let (source, _, _) = load_grayscale("resource/IMG_8237.jpeg");
    let cfg = ParallelConfig { num_patches: 4 };
    let sizes = [(480, 270), (960, 540), (1440, 810), (1920, 1080)];

    let cases: Vec<_> = sizes
        .into_iter()
        .map(|(w, h)| {
            let image = resize_grayscale(&source, w, h);
            let params = default_detect_params(w, h);
            (format!("{w}x{h}"), image, params)
        })
        .collect();

    let mut g = c.benchmark_group("mser_by_image_size");
    g.sample_size(10);
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(3));

    for (name, image, params) in &cases {
        let pixels = image.width() as u64 * image.height() as u64;
        g.throughput(Throughput::Elements(pixels));

        g.bench_with_input(
            BenchmarkId::new("v2_single", name),
            name,
            |b, _| b.iter(|| extract_msers_v2(image, params).unwrap()),
        );
        g.bench_with_input(
            BenchmarkId::new("v2_partitioned_4", name),
            name,
            |b, _| {
                b.iter(|| {
                    extract_msers_v2_partitioned(image, params, &cfg).unwrap()
                })
            },
        );
    }

    g.finish();
}

criterion_group!(benches, bench_all, bench_image_sizes);
criterion_main!(benches);
