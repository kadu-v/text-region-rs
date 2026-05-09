use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use image::ImageReader;
use std::time::Duration;
use text_region_rs::params::{MserParams, ParallelConfig};
use text_region_rs::{
    extract_msers, extract_msers_parallel, extract_msers_v2, extract_msers_v2_parallel,
    extract_msers_v2_partitioned,
};

fn load_grayscale(path: &str) -> (Vec<u8>, u32, u32) {
    let img = ImageReader::open(path).unwrap().decode().unwrap();
    let gray = img.to_luma8();
    let w = gray.width();
    let h = gray.height();
    (gray.into_raw(), w, h)
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
        b.iter(|| extract_msers(&img_p, wp, hp, &pp).unwrap())
    });
    g.bench_function("paper/v1_par_minmax", |b| {
        b.iter(|| extract_msers_parallel(&img_p, wp, hp, &pp, &cfg).unwrap())
    });
    g.bench_function("paper/v2_single", |b| {
        b.iter(|| extract_msers_v2(&img_p, wp, hp, &pp).unwrap())
    });
    g.bench_function("paper/v2_par_minmax", |b| {
        b.iter(|| extract_msers_v2_parallel(&img_p, wp, hp, &pp, &cfg).unwrap())
    });
    for n in [2, 4, 8] {
        let c2 = ParallelConfig { num_patches: n };
        g.bench_with_input(BenchmarkId::new("paper/v2_partitioned", n), &n, |b, _| {
            b.iter(|| extract_msers_v2_partitioned(&img_p, wp, hp, &pp, &c2).unwrap())
        });
    }

    // Label image
    g.bench_function("label/v1_single", |b| {
        b.iter(|| extract_msers(&img_l, wl, hl, &pl).unwrap())
    });
    g.bench_function("label/v1_par_minmax", |b| {
        b.iter(|| extract_msers_parallel(&img_l, wl, hl, &pl, &cfg).unwrap())
    });
    g.bench_function("label/v2_single", |b| {
        b.iter(|| extract_msers_v2(&img_l, wl, hl, &pl).unwrap())
    });
    g.bench_function("label/v2_par_minmax", |b| {
        b.iter(|| extract_msers_v2_parallel(&img_l, wl, hl, &pl, &cfg).unwrap())
    });
    for n in [2, 4, 8] {
        let c2 = ParallelConfig { num_patches: n };
        g.bench_with_input(BenchmarkId::new("label/v2_partitioned", n), &n, |b, _| {
            b.iter(|| extract_msers_v2_partitioned(&img_l, wl, hl, &pl, &c2).unwrap())
        });
    }

    g.finish();
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
