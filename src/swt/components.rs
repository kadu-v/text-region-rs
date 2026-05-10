use crate::error::Result;

use super::geometry::{point_index_from_xy, rect_from_opencv_points};
use super::{
    COMPONENT_ROTATION_STEPS, Point, Rect, SWT_COMPONENT_RATIO_THRESHOLD,
    SwtComponent, SwtImage,
};

#[derive(Clone, Copy, Debug)]
struct ComponentAttr {
    mean: f32,
    variance: f32,
    median: f32,
    xmin: i32,
    ymin: i32,
    xmax: i32,
    ymax: i32,
    length: f32,
    width: f32,
}

pub fn normalize_and_scale(image: &SwtImage) -> image::GrayImage {
    let mut min_swt = f32::MAX;
    let mut max_swt = 0.0_f32;
    for &value in image.as_raw() {
        if value < 0.0 {
            continue;
        }
        min_swt = min_swt.min(value);
        max_swt = max_swt.max(value);
    }

    let data = if min_swt == f32::MAX {
        vec![255; image.as_raw().len()]
    } else {
        let amplitude = max_swt - min_swt;
        image
            .as_raw()
            .iter()
            .map(|&value| {
                if value < 0.0 {
                    255
                } else if amplitude <= f32::EPSILON {
                    0
                } else {
                    (((value - min_swt) / amplitude) * 255.0) as u8
                }
            })
            .collect()
    };
    image::GrayImage::from_vec(image.width(), image.height(), data)
        .expect("valid normalized SWT image")
}

pub fn swt_connected_components(image: &SwtImage) -> Result<Vec<Vec<Point>>> {
    super::validation::validate_swt_image(image)?;

    let width = image.width() as usize;
    let height = image.height() as usize;
    let swt = image.as_raw();
    let mut visited = vec![false; swt.len()];
    let mut components = Vec::new();

    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if visited[idx] || swt[idx] < 0.0 {
                continue;
            }

            let mut stack = vec![Point {
                x: col as i32,
                y: row as i32,
            }];
            visited[idx] = true;
            let mut component = Vec::new();

            while let Some(point) = stack.pop() {
                component.push(point);

                for (nx, ny) in positive_forward_neighbors(
                    point.x as usize,
                    point.y as usize,
                    width,
                    height,
                ) {
                    let nidx = ny * width + nx;
                    if visited[nidx] || swt[nidx] < 0.0 {
                        continue;
                    }
                    if swt_neighbors_match(
                        swt[point_index_from_xy(width, point.x, point.y)],
                        swt[nidx],
                    ) {
                        visited[nidx] = true;
                        stack.push(Point {
                            x: nx as i32,
                            y: ny as i32,
                        });
                    }
                }

                for (nx, ny) in positive_backward_neighbors(
                    point.x as usize,
                    point.y as usize,
                    width,
                    height,
                ) {
                    let nidx = ny * width + nx;
                    if visited[nidx] || swt[nidx] < 0.0 {
                        continue;
                    }
                    if swt_neighbors_match(
                        swt[point_index_from_xy(width, point.x, point.y)],
                        swt[nidx],
                    ) {
                        visited[nidx] = true;
                        stack.push(Point {
                            x: nx as i32,
                            y: ny as i32,
                        });
                    }
                }
            }

            component.sort_by_key(|point| (point.y, point.x));
            components.push(component);
        }
    }

    Ok(components)
}

pub fn filter_swt_components(
    image: &SwtImage,
    components: &[Vec<Point>],
    skip_checks: bool,
) -> Result<Vec<SwtComponent>> {
    super::validation::validate_swt_image(image)?;

    let mut filtered = Vec::with_capacity(components.len());
    for component in components {
        if component.is_empty() {
            continue;
        }

        let mut attr = component_attributes(image, component);
        if !skip_checks && attr.variance > 0.5 * attr.mean {
            continue;
        }
        if !skip_checks && attr.width > 300.0 {
            continue;
        }

        let mut area = attr.length * attr.width;
        for theta_i in 0..(COMPONENT_ROTATION_STEPS / 2) {
            let theta = theta_i as f32
                * (std::f32::consts::PI / COMPONENT_ROTATION_STEPS as f32);
            let mut xmin = f32::MAX;
            let mut ymin = f32::MAX;
            let mut xmax = 0.0_f32;
            let mut ymax = 0.0_f32;

            for point in component {
                let x = point.x as f32;
                let y = point.y as f32;
                let xtemp = x * theta.cos() + y * -theta.sin();
                let ytemp = x * theta.sin() + y * theta.cos();
                xmin = xmin.min(xtemp);
                xmax = xmax.max(xtemp);
                ymin = ymin.min(ytemp);
                ymax = ymax.max(ytemp);
            }

            let ltemp = xmax - xmin + 1.0;
            let wtemp = ymax - ymin + 1.0;
            if ltemp * wtemp < area {
                area = ltemp * wtemp;
                attr.length = ltemp;
                attr.width = wtemp;
            }
        }

        let aspect = attr.length / attr.width;
        if !skip_checks && !(0.1..=10.0).contains(&aspect) {
            continue;
        }

        filtered.push(component_from_attr(component, attr));
    }

    if !skip_checks {
        let mut temp = Vec::with_capacity(filtered.len());
        for i in 0..filtered.len() {
            let mut count = 0;
            let comp_i = &filtered[i];
            for (j, comp_j) in filtered.iter().enumerate() {
                if i == j {
                    continue;
                }
                if contains_center(comp_i, comp_j.cx, comp_j.cy) {
                    count += 1;
                }
            }
            if count < 2 {
                temp.push(comp_i.clone());
            }
        }
        filtered = temp;
    }

    Ok(filtered)
}

pub(super) fn component_bounding_boxes(
    components: &[SwtComponent],
) -> Vec<Rect> {
    components
        .iter()
        .map(|component| {
            let x0 = component.bounding_rect.x as i32;
            let y0 = component.bounding_rect.y as i32;
            let x1 =
                x0 + component.bounding_rect.width.saturating_sub(1) as i32;
            let y1 =
                y0 + component.bounding_rect.height.saturating_sub(1) as i32;
            rect_from_opencv_points(x0, y0, x1, y1, false)
        })
        .collect()
}

fn positive_forward_neighbors(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let mut neighbors = Vec::with_capacity(4);
    if x + 1 < width {
        neighbors.push((x + 1, y));
    }
    if y + 1 < height {
        if x + 1 < width {
            neighbors.push((x + 1, y + 1));
        }
        neighbors.push((x, y + 1));
        if x > 0 {
            neighbors.push((x - 1, y + 1));
        }
    }
    neighbors.into_iter()
}

fn positive_backward_neighbors(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let mut neighbors = Vec::with_capacity(4);
    if x > 0 {
        neighbors.push((x - 1, y));
    }
    if y > 0 {
        if x > 0 {
            neighbors.push((x - 1, y - 1));
        }
        neighbors.push((x, y - 1));
        if x + 1 < width {
            neighbors.push((x + 1, y - 1));
        }
    }
    let _ = height;
    neighbors.into_iter()
}

fn swt_neighbors_match(a: f32, b: f32) -> bool {
    b > 0.0
        && (a / b <= SWT_COMPONENT_RATIO_THRESHOLD
            || b / a <= SWT_COMPONENT_RATIO_THRESHOLD)
}

fn component_attributes(
    image: &SwtImage,
    component: &[Point],
) -> ComponentAttr {
    let width = image.width() as usize;
    let swt = image.as_raw();
    let mut values = Vec::with_capacity(component.len());
    let mut sum = 0.0_f32;
    let mut xmin = i32::MAX;
    let mut ymin = i32::MAX;
    let mut xmax = 0_i32;
    let mut ymax = 0_i32;

    for point in component {
        let value = swt[point_index_from_xy(width, point.x, point.y)];
        sum += value;
        values.push(value);
        xmin = xmin.min(point.x);
        ymin = ymin.min(point.y);
        xmax = xmax.max(point.x);
        ymax = ymax.max(point.y);
    }

    let mean = sum / component.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / component.len() as f32;
    values.sort_by(|a, b| a.total_cmp(b));
    let median = values[values.len() / 2];

    ComponentAttr {
        mean,
        variance,
        median,
        xmin,
        ymin,
        xmax,
        ymax,
        length: (xmax - xmin + 1) as f32,
        width: (ymax - ymin + 1) as f32,
    }
}

fn component_from_attr(
    component: &[Point],
    attr: ComponentAttr,
) -> SwtComponent {
    SwtComponent {
        bounding_rect: rect_from_opencv_points(
            attr.xmin, attr.ymin, attr.xmax, attr.ymax, true,
        ),
        cx: (attr.xmax + attr.xmin) as f32 / 2.0,
        cy: (attr.ymax + attr.ymin) as f32 / 2.0,
        median: attr.median,
        mean: attr.mean,
        length: attr.xmax - attr.xmin + 1,
        width: attr.ymax - attr.ymin + 1,
        points: component.to_vec(),
    }
}

fn contains_center(component: &SwtComponent, cx: f32, cy: f32) -> bool {
    let x0 = component.bounding_rect.x as f32;
    let y0 = component.bounding_rect.y as f32;
    let x1 = x0 + component.bounding_rect.width.saturating_sub(1) as f32;
    let y1 = y0 + component.bounding_rect.height.saturating_sub(1) as f32;
    x0 <= cx && x1 >= cx && y0 <= cy && y1 >= cy
}
