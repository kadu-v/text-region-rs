use crate::error::{MserError, Result};
pub use crate::mser::types::{Point, Rect};

pub const INVALID_STROKE_WIDTH: f32 = -1.0;
const SWT_COMPONENT_RATIO_THRESHOLD: f32 = 3.0;
const COMPONENT_ROTATION_STEPS: usize = 36;

#[derive(Clone, Copy, Debug)]
pub struct SwtParams {
    pub dark_on_light: bool,
}

impl Default for SwtParams {
    fn default() -> Self {
        Self {
            dark_on_light: true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SwtInput<'a> {
    pub width: u32,
    pub height: u32,
    pub edge: &'a [u8],
    pub gradient_x: &'a [f32],
    pub gradient_y: &'a [f32],
    pub params: SwtParams,
}

#[derive(Clone, Copy, Debug)]
pub struct SwtBgrInput<'a> {
    pub width: u32,
    pub height: u32,
    pub bgr: &'a [u8],
    pub params: SwtParams,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwtImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwtPreprocessed {
    pub width: u32,
    pub height: u32,
    pub gray: Vec<u8>,
    pub edge: Vec<u8>,
    pub gradient_x: Vec<f32>,
    pub gradient_y: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwtComponent {
    pub bounding_rect: Rect,
    pub cx: f32,
    pub cy: f32,
    pub median: f32,
    pub mean: f32,
    pub length: i32,
    pub width: i32,
    pub points: Vec<Point>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SwtDetections {
    pub letter_bounding_boxes: Vec<Rect>,
    pub chain_bounding_boxes: Vec<Rect>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwtDebugOutput {
    pub detections: SwtDetections,
    pub preprocessed: SwtPreprocessed,
    pub swt_image: SwtImage,
    pub normalized_swt: Vec<u8>,
    pub draw_bgr: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
struct SwtPoint {
    x: i32,
    y: i32,
}

#[derive(Clone, Debug)]
struct Ray {
    p: SwtPoint,
    q: SwtPoint,
    points: Vec<SwtPoint>,
}

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

pub fn stroke_width_transform(input: SwtInput<'_>) -> Result<SwtImage> {
    let len = validate_input(&input)?;
    let mut swt = vec![INVALID_STROKE_WIDTH; len];
    let mut rays = Vec::new();

    swt_first_pass(&input, &mut swt, &mut rays);
    swt_second_pass(input.width as usize, &mut swt, &rays);

    Ok(SwtImage {
        width: input.width,
        height: input.height,
        data: swt,
    })
}

pub fn detect_text_swt(input: SwtBgrInput<'_>) -> Result<SwtDetections> {
    Ok(detect_text_swt_with_debug(input)?.detections)
}

pub fn detect_text_swt_with_debug(
    input: SwtBgrInput<'_>,
) -> Result<SwtDebugOutput> {
    validate_bgr_input(input.width, input.height, input.bgr)?;

    let preprocessed =
        swt_preprocess_bgr(input.width, input.height, input.bgr)?;
    let swt_image = stroke_width_transform(SwtInput {
        width: input.width,
        height: input.height,
        edge: &preprocessed.edge,
        gradient_x: &preprocessed.gradient_x,
        gradient_y: &preprocessed.gradient_y,
        params: input.params,
    })?;
    let detections = detect_text_regions_from_swt(&swt_image, input.bgr)?;
    let normalized_swt = normalize_and_scale(&swt_image);
    let draw_bgr = render_debug_bgr(
        input.width as usize,
        input.height as usize,
        &normalized_swt,
        &detections.letter_bounding_boxes,
    );

    Ok(SwtDebugOutput {
        detections,
        preprocessed,
        swt_image,
        normalized_swt,
        draw_bgr,
    })
}

pub fn swt_preprocess_bgr(
    width: u32,
    height: u32,
    bgr: &[u8],
) -> Result<SwtPreprocessed> {
    validate_bgr_input(width, height, bgr)?;

    let gray = bgr_to_gray(bgr);
    let edge =
        canny_3x3_l1(&gray, width as usize, height as usize, 175.0, 320.0);

    let gaussian = gray
        .iter()
        .map(|&value| value as f32 / 255.0)
        .collect::<Vec<_>>();
    let gaussian = gaussian_blur(&gaussian, width as usize, height as usize, 5);
    let (gradient_x, gradient_y) =
        scharr_gradients(&gaussian, width as usize, height as usize);
    let gradient_x =
        gaussian_blur(&gradient_x, width as usize, height as usize, 3);
    let gradient_y =
        gaussian_blur(&gradient_y, width as usize, height as usize, 3);

    Ok(SwtPreprocessed {
        width,
        height,
        gray,
        edge,
        gradient_x,
        gradient_y,
    })
}

pub fn normalize_and_scale(image: &SwtImage) -> Vec<u8> {
    let mut min_swt = f32::MAX;
    let mut max_swt = 0.0_f32;
    for &value in &image.data {
        if value < 0.0 {
            continue;
        }
        min_swt = min_swt.min(value);
        max_swt = max_swt.max(value);
    }

    if min_swt == f32::MAX {
        return vec![255; image.data.len()];
    }

    let amplitude = max_swt - min_swt;
    image
        .data
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
}

pub fn swt_connected_components(image: &SwtImage) -> Result<Vec<Vec<Point>>> {
    validate_swt_image(image)?;

    let width = image.width as usize;
    let height = image.height as usize;
    let mut visited = vec![false; image.data.len()];
    let mut components = Vec::new();

    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if visited[idx] || image.data[idx] < 0.0 {
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
                    if visited[nidx] || image.data[nidx] < 0.0 {
                        continue;
                    }
                    if swt_neighbors_match(
                        image.data
                            [point_index_from_xy(width, point.x, point.y)],
                        image.data[nidx],
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
                    if visited[nidx] || image.data[nidx] < 0.0 {
                        continue;
                    }
                    if swt_neighbors_match(
                        image.data
                            [point_index_from_xy(width, point.x, point.y)],
                        image.data[nidx],
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
    validate_swt_image(image)?;

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

pub fn detect_text_regions_from_swt(
    image: &SwtImage,
    bgr: &[u8],
) -> Result<SwtDetections> {
    validate_swt_image(image)?;
    let expected_bgr_len = image.data.len().checked_mul(3).ok_or(
        MserError::ImageDimensionsTooLarge {
            width: image.width,
            height: image.height,
        },
    )?;
    validate_len("bgr", bgr.len(), expected_bgr_len)?;

    let components = swt_connected_components(image)?;
    let valid_components = filter_swt_components(image, &components, false)?;
    Ok(find_valid_chains(image, bgr, &valid_components))
}

fn validate_input(input: &SwtInput<'_>) -> Result<usize> {
    if input.width == 0 || input.height == 0 {
        return Err(MserError::EmptyImageDimensions {
            width: input.width,
            height: input.height,
        });
    }

    let len = (input.width as usize)
        .checked_mul(input.height as usize)
        .ok_or(MserError::ImageDimensionsTooLarge {
            width: input.width,
            height: input.height,
        })?;

    validate_len("edge", input.edge.len(), len)?;
    validate_len("gradient_x", input.gradient_x.len(), len)?;
    validate_len("gradient_y", input.gradient_y.len(), len)?;

    Ok(len)
}

fn validate_bgr_input(width: u32, height: u32, bgr: &[u8]) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(MserError::EmptyImageDimensions { width, height });
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(MserError::ImageDimensionsTooLarge { width, height })?;
    let expected = pixels
        .checked_mul(3)
        .ok_or(MserError::ImageDimensionsTooLarge { width, height })?;
    validate_len("bgr", bgr.len(), expected)?;
    Ok(pixels)
}

fn validate_swt_image(image: &SwtImage) -> Result<()> {
    if image.width == 0 || image.height == 0 {
        return Err(MserError::EmptyImageDimensions {
            width: image.width,
            height: image.height,
        });
    }
    let expected = (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or(MserError::ImageDimensionsTooLarge {
            width: image.width,
            height: image.height,
        })?;
    validate_len("swt.data", image.data.len(), expected)
}

fn validate_len(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<()> {
    if actual != expected {
        return Err(MserError::InvalidSwtInput {
            field,
            message: "buffer length must match width * height",
        });
    }
    Ok(())
}

fn swt_first_pass(input: &SwtInput<'_>, swt: &mut [f32], rays: &mut Vec<Ray>) {
    let width = input.width as usize;
    let height = input.height as usize;

    for row in 0..height {
        for col in 0..width {
            let start_idx = row * width + col;
            if input.edge[start_idx] == 0 {
                continue;
            }

            let Some((mut dx, mut dy)) = normalized_gradient(
                input.gradient_x[start_idx],
                input.gradient_y[start_idx],
            ) else {
                continue;
            };

            if input.params.dark_on_light {
                dx = -dx;
                dy = -dy;
            }

            let p = SwtPoint {
                x: col as i32,
                y: row as i32,
            };
            let mut points = vec![p];
            let mut cur_pos_x = col as f32 + 0.5;
            let mut cur_pos_y = row as f32 + 0.5;
            let mut cur_pix_x = col as i32;
            let mut cur_pix_y = row as i32;
            let inc = 0.05_f32;

            loop {
                cur_pos_x += inc * dx;
                cur_pos_y += inc * dy;

                let next_x = cur_pos_x.floor() as i32;
                let next_y = cur_pos_y.floor() as i32;
                if next_x == cur_pix_x && next_y == cur_pix_y {
                    continue;
                }

                cur_pix_x = next_x;
                cur_pix_y = next_y;
                if cur_pix_x < 0
                    || cur_pix_y < 0
                    || cur_pix_x >= input.width as i32
                    || cur_pix_y >= input.height as i32
                {
                    break;
                }

                let pt = SwtPoint {
                    x: cur_pix_x,
                    y: cur_pix_y,
                };
                points.push(pt);

                let idx = point_index(width, pt);
                if input.edge[idx] == 0 {
                    continue;
                }

                let Some((mut stop_dx, mut stop_dy)) = normalized_gradient(
                    input.gradient_x[idx],
                    input.gradient_y[idx],
                ) else {
                    break;
                };

                if input.params.dark_on_light {
                    stop_dx = -stop_dx;
                    stop_dy = -stop_dy;
                }

                let opposite_gradient_dot = dx * -stop_dx + dy * -stop_dy;
                if opposite_gradient_dot > 0.0 {
                    let q = pt;
                    let length = ((q.x - p.x) as f32).hypot((q.y - p.y) as f32);
                    for &ray_pt in &points {
                        let ray_idx = point_index(width, ray_pt);
                        let current = swt[ray_idx];
                        swt[ray_idx] = if current < 0.0 {
                            length
                        } else {
                            current.min(length)
                        };
                    }
                    rays.push(Ray { p, q, points });
                }
                break;
            }
        }
    }
}

fn normalized_gradient(dx: f32, dy: f32) -> Option<(f32, f32)> {
    let mag = dx.hypot(dy);
    if !mag.is_finite() || mag <= f32::EPSILON {
        return None;
    }
    Some((dx / mag, dy / mag))
}

fn swt_second_pass(width: usize, swt: &mut [f32], rays: &[Ray]) {
    for ray in rays {
        debug_assert!(ray.points.len() >= 2);
        debug_assert!(ray.p.x >= 0 && ray.q.x >= 0);

        let mut values = ray
            .points
            .iter()
            .map(|&point| swt[point_index(width, point)])
            .collect::<Vec<_>>();
        values.sort_by(|a, b| a.total_cmp(b));
        let median = values[values.len() / 2];

        for &point in &ray.points {
            let idx = point_index(width, point);
            swt[idx] = swt[idx].min(median);
        }
    }
}

fn point_index(width: usize, point: SwtPoint) -> usize {
    point.y as usize * width + point.x as usize
}

fn point_index_from_xy(width: usize, x: i32, y: i32) -> usize {
    y as usize * width + x as usize
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
    let width = image.width as usize;
    let mut values = Vec::with_capacity(component.len());
    let mut sum = 0.0_f32;
    let mut xmin = i32::MAX;
    let mut ymin = i32::MAX;
    let mut xmax = 0_i32;
    let mut ymax = 0_i32;

    for point in component {
        let value = image.data[point_index_from_xy(width, point.x, point.y)];
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

fn find_valid_chains(
    image: &SwtImage,
    bgr: &[u8],
    components: &[SwtComponent],
) -> SwtDetections {
    let width = image.width as usize;
    let mut color_averages = Vec::with_capacity(components.len());
    for component in components {
        let mut avg = ChannelAverage {
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
        };
        for point in &component.points {
            let idx = (point.y as usize * width + point.x as usize) * 3;
            avg.c0 += bgr[idx] as f32;
            avg.c1 += bgr[idx + 1] as f32;
            avg.c2 += bgr[idx + 2] as f32;
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
            if dist < 9.0 * scale * scale && color_dist < 1600.0 {
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

    let alignment_threshold_cos = (std::f32::consts::PI / 6.0).cos();
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
        if chain.component_indices.len() < 3 {
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

    let final_components =
        filter_swt_components(image, &component_groups, true)
            .unwrap_or_default();
    SwtDetections {
        letter_bounding_boxes: component_bounding_boxes(&final_components),
        chain_bounding_boxes,
    }
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

fn component_bounding_boxes(components: &[SwtComponent]) -> Vec<Rect> {
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

fn rect_from_opencv_points(
    xmin: i32,
    ymin: i32,
    xmax: i32,
    ymax: i32,
    inclusive: bool,
) -> Rect {
    let extra = if inclusive { 1 } else { 0 };
    Rect {
        x: xmin.min(xmax) as u32,
        y: ymin.min(ymax) as u32,
        width: (xmax - xmin).unsigned_abs() + extra,
        height: (ymax - ymin).unsigned_abs() + extra,
    }
}

fn bgr_to_gray(bgr: &[u8]) -> Vec<u8> {
    bgr.chunks_exact(3)
        .map(|pixel| {
            let b = pixel[0] as f32;
            let g = pixel[1] as f32;
            let r = pixel[2] as f32;
            (0.114 * b + 0.587 * g + 0.299 * r).round() as u8
        })
        .collect()
}

fn gaussian_blur(
    input: &[f32],
    width: usize,
    height: usize,
    ksize: usize,
) -> Vec<f32> {
    debug_assert!(ksize == 3 || ksize == 5);
    let kernel = gaussian_kernel(ksize);
    let radius = ksize as isize / 2;

    let mut temp = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (k, &weight) in kernel.iter().enumerate() {
                let dx = k as isize - radius;
                let sx = reflect101(x as isize + dx, width);
                sum += input[y * width + sx] * weight;
            }
            temp[y * width + x] = sum;
        }
    }

    let mut output = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (k, &weight) in kernel.iter().enumerate() {
                let dy = k as isize - radius;
                let sy = reflect101(y as isize + dy, height);
                sum += temp[sy * width + x] * weight;
            }
            output[y * width + x] = sum;
        }
    }
    output
}

fn gaussian_kernel(ksize: usize) -> Vec<f32> {
    let sigma = if ksize == 3 { 0.8 } else { 1.1 };
    let radius = ksize as i32 / 2;
    let mut kernel = Vec::with_capacity(ksize);
    let mut sum = 0.0;
    for i in -radius..=radius {
        let x = i as f32;
        let value = (-(x * x) / (2.0 * sigma * sigma)).exp();
        kernel.push(value);
        sum += value;
    }
    for value in &mut kernel {
        *value /= sum;
    }
    kernel
}

fn scharr_gradients(
    input: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0; input.len()];
    let mut gy = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let xm = reflect101(x as isize - 1, width);
            let xp = reflect101(x as isize + 1, width);
            let ym = reflect101(y as isize - 1, height);
            let yp = reflect101(y as isize + 1, height);

            let top_left = input[ym * width + xm];
            let top = input[ym * width + x];
            let top_right = input[ym * width + xp];
            let left = input[y * width + xm];
            let right = input[y * width + xp];
            let bottom_left = input[yp * width + xm];
            let bottom = input[yp * width + x];
            let bottom_right = input[yp * width + xp];

            gx[y * width + x] = 3.0 * (top_right - top_left)
                + 10.0 * (right - left)
                + 3.0 * (bottom_right - bottom_left);
            gy[y * width + x] = 3.0 * (bottom_left - top_left)
                + 10.0 * (bottom - top)
                + 3.0 * (bottom_right - top_right);
        }
    }
    (gx, gy)
}

fn canny_3x3_l1(
    gray: &[u8],
    width: usize,
    height: usize,
    low_threshold: f32,
    high_threshold: f32,
) -> Vec<u8> {
    let (gx, gy) = sobel_3x3_i16(gray, width, height);
    let mut magnitude = vec![0.0; gray.len()];
    for i in 0..gray.len() {
        magnitude[i] = gx[i].abs() as f32 + gy[i].abs() as f32;
    }

    let mut nms = vec![0.0; gray.len()];
    if width >= 3 && height >= 3 {
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let idx = y * width + x;
                let ax = gx[idx].abs();
                let ay = gy[idx].abs();
                let (a, b) = if ay * 2 <= ax {
                    (idx - 1, idx + 1)
                } else if ax * 2 <= ay {
                    (idx - width, idx + width)
                } else if gx[idx].signum() == gy[idx].signum() {
                    (idx - width - 1, idx + width + 1)
                } else {
                    (idx - width + 1, idx + width - 1)
                };
                let mag = magnitude[idx];
                if mag >= magnitude[a] && mag >= magnitude[b] {
                    nms[idx] = mag;
                }
            }
        }
    }

    let mut edge = vec![0_u8; gray.len()];
    let mut stack = Vec::new();
    for (idx, &mag) in nms.iter().enumerate() {
        if mag >= high_threshold {
            edge[idx] = 255;
            stack.push(idx);
        }
    }

    while let Some(idx) = stack.pop() {
        let x = idx % width;
        let y = idx / width;
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(height.saturating_sub(1));
        let x0 = x.saturating_sub(1);
        let x1 = (x + 1).min(width.saturating_sub(1));
        for ny in y0..=y1 {
            for nx in x0..=x1 {
                let nidx = ny * width + nx;
                if edge[nidx] == 0 && nms[nidx] >= low_threshold {
                    edge[nidx] = 255;
                    stack.push(nidx);
                }
            }
        }
    }

    edge
}

fn sobel_3x3_i16(
    gray: &[u8],
    width: usize,
    height: usize,
) -> (Vec<i16>, Vec<i16>) {
    let mut gx = vec![0_i16; gray.len()];
    let mut gy = vec![0_i16; gray.len()];
    for y in 0..height {
        for x in 0..width {
            let xm = reflect101(x as isize - 1, width);
            let xp = reflect101(x as isize + 1, width);
            let ym = reflect101(y as isize - 1, height);
            let yp = reflect101(y as isize + 1, height);

            let top_left = gray[ym * width + xm] as i16;
            let top = gray[ym * width + x] as i16;
            let top_right = gray[ym * width + xp] as i16;
            let left = gray[y * width + xm] as i16;
            let right = gray[y * width + xp] as i16;
            let bottom_left = gray[yp * width + xm] as i16;
            let bottom = gray[yp * width + x] as i16;
            let bottom_right = gray[yp * width + xp] as i16;

            gx[y * width + x] = -top_left + top_right - 2 * left + 2 * right
                - bottom_left
                + bottom_right;
            gy[y * width + x] = -top_left - 2 * top - top_right
                + bottom_left
                + 2 * bottom
                + bottom_right;
        }
    }
    (gx, gy)
}

fn render_debug_bgr(
    width: usize,
    height: usize,
    _normalized: &[u8],
    rects: &[Rect],
) -> Vec<u8> {
    let mut output = vec![255_u8; width * height * 3];

    let colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]];
    for (i, rect) in rects.iter().enumerate() {
        let color = colors[i % colors.len()];
        draw_rect_bgr(&mut output, width, height, *rect, color);
    }
    output
}

fn draw_rect_bgr(
    output: &mut [u8],
    width: usize,
    height: usize,
    rect: Rect,
    color: [u8; 3],
) {
    if rect.width == 0 || rect.height == 0 || width == 0 || height == 0 {
        return;
    }
    let x0 = rect.x as usize;
    let y0 = rect.y as usize;
    let x1 = (rect.x as usize + rect.width.saturating_sub(1) as usize)
        .min(width - 1);
    let y1 = (rect.y as usize + rect.height.saturating_sub(1) as usize)
        .min(height - 1);
    if x0 >= width || y0 >= height {
        return;
    }

    for x in x0..=x1 {
        set_bgr(output, width, x, y0, color);
        set_bgr(output, width, x, y1, color);
    }
    for y in y0..=y1 {
        set_bgr(output, width, x0, y, color);
        set_bgr(output, width, x1, y, color);
    }
}

fn set_bgr(
    output: &mut [u8],
    width: usize,
    x: usize,
    y: usize,
    color: [u8; 3],
) {
    let idx = (y * width + x) * 3;
    output[idx] = color[0];
    output[idx + 1] = color[1];
    output[idx + 2] = color[2];
}

fn reflect101(coord: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let mut value = coord;
    let max = len as isize - 1;
    while value < 0 || value > max {
        if value < 0 {
            value = -value;
        } else {
            value = 2 * max - value;
        }
    }
    value as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_pass_clamps_ray_to_upper_median() {
        let mut swt = vec![4.0, 2.0, 8.0];
        let ray = Ray {
            p: SwtPoint { x: 0, y: 0 },
            q: SwtPoint { x: 2, y: 0 },
            points: vec![
                SwtPoint { x: 0, y: 0 },
                SwtPoint { x: 1, y: 0 },
                SwtPoint { x: 2, y: 0 },
            ],
        };

        swt_second_pass(3, &mut swt, &[ray]);

        assert_eq!(swt, vec![4.0, 2.0, 4.0]);
    }
}
