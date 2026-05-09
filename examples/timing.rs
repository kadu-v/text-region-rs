use image::ImageReader;
use std::env;
use std::time::Instant;
use text_region_rs::error::Result;
use text_region_rs::params::{MserParams, ParallelConfig};
use text_region_rs::{extract_msers_v2, extract_msers_v2_partitioned};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Variant {
    All,
    Single,
    Partitioned,
}

struct Args {
    image_path: String,
    variant: Variant,
    patches: u32,
    runs: u32,
}

fn parse_args() -> Args {
    let mut image_path = String::from("resource/IMG_8237.jpeg");
    let mut variant = Variant::All;
    let mut patches = 4;
    let mut runs = 1;

    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--image" => {
                i += 1;
                image_path = args.get(i).expect("--image requires a path").clone();
            }
            "--variant" => {
                i += 1;
                variant = match args.get(i).expect("--variant requires a value").as_str() {
                    "all" => Variant::All,
                    "single" | "v2_single" => Variant::Single,
                    "partitioned" | "part" | "v2_partitioned" => Variant::Partitioned,
                    other => panic!("unknown variant: {other}"),
                };
            }
            "--patches" => {
                i += 1;
                patches = args
                    .get(i)
                    .expect("--patches requires a number")
                    .parse()
                    .expect("--patches must be a number");
            }
            "--runs" => {
                i += 1;
                runs = args
                    .get(i)
                    .expect("--runs requires a number")
                    .parse()
                    .expect("--runs must be a number");
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release --example timing -- [IMAGE] [--variant all|single|partitioned] [--patches N] [--runs N]"
                );
                std::process::exit(0);
            }
            value if !value.starts_with('-') => {
                image_path = value.to_string();
            }
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }

    Args {
        image_path,
        variant,
        patches,
        runs,
    }
}

fn main() -> Result<()> {
    let args = parse_args();

    let img = ImageReader::open(&args.image_path)?.decode()?;
    let gray = img.to_luma8();
    let w = gray.width();
    let h = gray.height();
    let total = (w * h) as f32;
    let params = MserParams {
        delta: 5,
        min_point: (total * 0.0001).max(50.0) as i32,
        max_point_ratio: 0.05,
        stable_variation: 0.25,
        duplicated_variation: 0.2,
        nms_similarity: 0.5,
        ..MserParams::default()
    };

    println!("image: {}  size: {}x{}", args.image_path, w, h);

    for run in 0..args.runs {
        if args.runs > 1 {
            println!("run: {}", run + 1);
        }

        if args.variant == Variant::All || args.variant == Variant::Single {
            let t = Instant::now();
            let result = extract_msers_v2(&gray, &params)?;
            println!(
                "v2_single: {:?}  regions: {}/{}",
                t.elapsed(),
                result.from_min.len(),
                result.from_max.len()
            );
        }

        if args.variant == Variant::All {
            for num_patches in [2, 4] {
                let cfg = ParallelConfig { num_patches };
                let t = Instant::now();
                let result = extract_msers_v2_partitioned(&gray, &params, &cfg)?;
                println!(
                    "v2_part/{}: {:?}  regions: {}/{}",
                    num_patches,
                    t.elapsed(),
                    result.from_min.len(),
                    result.from_max.len()
                );
            }
        } else if args.variant == Variant::Partitioned {
            let cfg = ParallelConfig {
                num_patches: args.patches,
            };
            let t = Instant::now();
            let result = extract_msers_v2_partitioned(&gray, &params, &cfg)?;
            println!(
                "v2_part/{}: {:?}  regions: {}/{}",
                args.patches,
                t.elapsed(),
                result.from_min.len(),
                result.from_max.len()
            );
        }
    }

    Ok(())
}
