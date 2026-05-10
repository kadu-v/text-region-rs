use crate::error::{MserError, Result};

use super::components::{
    component_bounding_boxes, filter_swt_components_with_params,
    swt_connected_components,
};
use super::geometry::rect_from_opencv_points;
use super::validation::{validate_image_dimensions, validate_swt_image};
use super::{SwtComponent, SwtDetections, SwtImage, SwtParams};

#[derive(Clone, Copy, Debug)]
struct ChannelAverage {
    c0: f32,
    c1: f32,
    c2: f32,
}

#[derive(Clone, Copy, Debug)]
struct Direction {
    x: f32,
    y: f32,
}

#[derive(Clone, Debug)]
struct ChainedComponent {
    chain_index_a: usize,
    chain_index_b: usize,
    component_indices: Vec<usize>,
    chain_dist: f32,
    dir: Direction,
    merged: bool,
}

pub fn detect_text_regions_from_swt(
    image: &SwtImage,
    rgb: &image::RgbImage,
) -> Result<SwtDetections> {
    detect_text_regions_from_swt_with_params(image, rgb, SwtParams::default())
}

pub fn detect_text_regions_from_swt_with_params(
    image: &SwtImage,
    rgb: &image::RgbImage,
    params: SwtParams,
) -> Result<SwtDetections> {
    validate_swt_image(image)?;
    validate_rgb_matches_swt_image(image, rgb)?;

    let components = swt_connected_components(image)?;
    let valid_components =
        filter_swt_components_with_params(image, &components, false, params)?;
    Ok(find_valid_chains(image, rgb, &valid_components, params))
}

fn find_valid_chains(
    image: &SwtImage,
    rgb: &image::RgbImage,
    components: &[SwtComponent],
    params: SwtParams,
) -> SwtDetections {
    let width = image.width() as usize;
    let pixels = rgb.as_raw();
    let mut color_averages = Vec::with_capacity(components.len());
    for component in components {
        let mut avg = ChannelAverage {
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
        };
        for point in &component.points {
            let idx = (point.y as usize * width + point.x as usize) * 3;
            avg.c0 += pixels[idx] as f32;
            avg.c1 += pixels[idx + 1] as f32;
            avg.c2 += pixels[idx + 2] as f32;
        }
        let len = component.points.len() as f32;
        avg.c0 /= len;
        avg.c1 /= len;
        avg.c2 /= len;
        color_averages.push(avg);
    }

    let mut chains = Vec::new();
    for i in 0..components.len() {
        for j in (i + 1)..components.len() {
            let comp_i = &components[i];
            let comp_j = &components[j];
            if !opencv_ratio_accepts(comp_i.median, comp_j.median)
                || !opencv_ratio_accepts(
                    comp_i.width as f32,
                    comp_j.width as f32,
                )
            {
                continue;
            }

            let dx = comp_i.cx - comp_j.cx;
            let dy = comp_i.cy - comp_j.cy;
            let dist = dx * dx + dy * dy;
            let color_dist =
                color_distance(color_averages[i], color_averages[j]);
            let scale = comp_i
                .length
                .min(comp_i.width)
                .max(comp_j.length.min(comp_j.width))
                as f32;
            if dist < params.chain_pair_distance_scale * scale * scale
                && color_dist < params.chain_color_distance_threshold
            {
                let mag = dx.hypot(dy);
                if mag <= f32::EPSILON {
                    continue;
                }
                chains.push(ChainedComponent {
                    chain_index_a: i,
                    chain_index_b: j,
                    component_indices: vec![i, j],
                    chain_dist: dist,
                    dir: Direction {
                        x: dx / mag,
                        y: dy / mag,
                    },
                    merged: false,
                });
            }
        }
    }

    chains.sort_by(|a, b| a.chain_dist.total_cmp(&b.chain_dist));

    let alignment_threshold_cos = params.chain_alignment_angle_rad.cos();
    let mut merges = 1;
    while merges > 0 {
        for chain in &mut chains {
            chain.merged = false;
        }
        merges = 0;

        for i in 0..chains.len() {
            for j in 0..chains.len() {
                if i == j || chains[i].merged || chains[j].merged {
                    continue;
                }

                let mut new_a = chains[i].chain_index_a;
                let mut new_b = chains[i].chain_index_b;
                let should_merge = chain_merge_candidate(
                    &chains[i],
                    &chains[j],
                    alignment_threshold_cos,
                    &mut new_a,
                    &mut new_b,
                );
                if should_merge {
                    chains[i].chain_index_a = new_a;
                    chains[i].chain_index_b = new_b;
                    let other_indices = chains[j].component_indices.clone();
                    chains[i].component_indices.extend(other_indices);
                    update_chain_geometry(&mut chains[i], components);
                    chains[j].merged = true;
                    merges += 1;
                }
            }
        }

        chains.retain(|chain| !chain.merged);
        chains.sort_by_key(|chain| chain.component_indices.len());
    }

    let mut component_included = vec![false; components.len()];
    let mut component_groups = Vec::new();
    let mut chain_bounding_boxes = Vec::new();

    for chain in &chains {
        if chain.component_indices.len() < params.min_chain_components {
            continue;
        }

        let mut xmin = i32::MAX;
        let mut ymin = i32::MAX;
        let mut xmax = 0_i32;
        let mut ymax = 0_i32;

        for &idx in &chain.component_indices {
            if component_included[idx] {
                continue;
            }
            component_included[idx] = true;
            let accepted_component = &components[idx];
            let mut component_points =
                Vec::with_capacity(accepted_component.points.len());
            for &point in &accepted_component.points {
                component_points.push(point);
                xmin = xmin.min(point.x);
                ymin = ymin.min(point.y);
                xmax = xmax.max(point.x);
                ymax = ymax.max(point.y);
            }
            component_groups.push(component_points);
        }

        if xmin <= xmax && ymin <= ymax {
            chain_bounding_boxes
                .push(rect_from_opencv_points(xmin, ymin, xmax, ymax, false));
        }
    }

    let final_components = filter_swt_components_with_params(
        image,
        &component_groups,
        true,
        params,
    )
    .unwrap_or_default();
    SwtDetections {
        letter_bounding_boxes: component_bounding_boxes(&final_components),
        chain_bounding_boxes,
    }
}

fn validate_rgb_matches_swt_image(
    swt: &SwtImage,
    rgb: &image::RgbImage,
) -> Result<()> {
    validate_image_dimensions(rgb.width(), rgb.height())?;
    if rgb.width() != swt.width() || rgb.height() != swt.height() {
        return Err(MserError::InvalidSwtInput {
            field: "rgb",
            message: "image dimensions must match SWT image",
        });
    }
    Ok(())
}

fn opencv_ratio_accepts(a: f32, b: f32) -> bool {
    a / b <= 2.0 || b / a <= 2.0
}

fn color_distance(a: ChannelAverage, b: ChannelAverage) -> f32 {
    let d0 = a.c0 - b.c0;
    let d1 = a.c1 - b.c1;
    let d2 = a.c2 - b.c2;
    d0 * d0 + d1 * d1 + d2 * d2
}

fn chain_merge_candidate(
    chain_i: &ChainedComponent,
    chain_j: &ChainedComponent,
    alignment_threshold_cos: f32,
    new_a: &mut usize,
    new_b: &mut usize,
) -> bool {
    if chain_i.chain_index_a == chain_j.chain_index_a {
        if chain_i.dir.x * -chain_j.dir.x + chain_i.dir.y * -chain_j.dir.y
            > alignment_threshold_cos
        {
            *new_a = chain_j.chain_index_b;
            return true;
        }
    } else if chain_i.chain_index_a == chain_j.chain_index_b {
        if chain_i.dir.x * chain_j.dir.x + chain_i.dir.y * chain_j.dir.y
            > alignment_threshold_cos
        {
            *new_a = chain_j.chain_index_a;
            return true;
        }
    } else if chain_i.chain_index_b == chain_j.chain_index_a {
        if chain_i.dir.x * chain_j.dir.x + chain_i.dir.y * chain_j.dir.y
            > alignment_threshold_cos
        {
            *new_b = chain_j.chain_index_b;
            return true;
        }
    } else if chain_i.chain_index_b == chain_j.chain_index_b
        && chain_i.dir.x * -chain_j.dir.x + chain_i.dir.y * -chain_j.dir.y
            > alignment_threshold_cos
    {
        *new_b = chain_j.chain_index_a;
        return true;
    }
    false
}

fn update_chain_geometry(
    chain: &mut ChainedComponent,
    components: &[SwtComponent],
) {
    let a = &components[chain.chain_index_a];
    let b = &components[chain.chain_index_b];
    let dx = a.cx - b.cx;
    let dy = a.cy - b.cy;
    chain.chain_dist = dx * dx + dy * dy;
    let mag = dx.hypot(dy);
    if mag > f32::EPSILON {
        chain.dir = Direction {
            x: dx / mag,
            y: dy / mag,
        };
    }
}
