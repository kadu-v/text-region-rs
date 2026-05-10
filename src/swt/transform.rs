use crate::error::Result;

use super::geometry::point_index_from_xy;
use super::{INVALID_STROKE_WIDTH, SwtImage, SwtInput};

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

pub fn stroke_width_transform(input: SwtInput<'_>) -> Result<SwtImage> {
    let len = super::validation::validate_input(&input)?;
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
    point_index_from_xy(width, point.x, point.y)
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
