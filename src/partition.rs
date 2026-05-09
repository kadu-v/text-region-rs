use crate::error::{MserError, Result};

#[derive(Debug, Clone)]
pub struct PatchInfo {
    pub patch_index: u8,
    pub x_start: u32,
    pub y_start: u32,
    pub width: u32,
    pub height: u32,
    pub grid_col: u32,
    pub grid_row: u32,
}

#[derive(Debug, Clone)]
pub struct GridConfig {
    pub cols: u32,
    pub rows: u32,
}

#[derive(Debug, Clone)]
pub struct BoundaryEdge {
    pub patch_a: u8,
    pub patch_b: u8,
    pub is_horizontal: bool,
    pub length: u32,
}

pub fn compute_grid_config(num_patches: u32) -> Result<GridConfig> {
    match num_patches {
        1 => Ok(GridConfig { cols: 1, rows: 1 }),
        2 => Ok(GridConfig { cols: 2, rows: 1 }),
        4 => Ok(GridConfig { cols: 2, rows: 2 }),
        8 => Ok(GridConfig { cols: 4, rows: 2 }),
        16 => Ok(GridConfig { cols: 4, rows: 4 }),
        32 => Ok(GridConfig { cols: 8, rows: 4 }),
        _ => Err(MserError::InvalidNumPatches { num_patches }),
    }
}

pub fn compute_patches(width: u32, height: u32, grid: &GridConfig) -> Vec<PatchInfo> {
    let mut patches = Vec::with_capacity((grid.cols * grid.rows) as usize);

    for row in 0..grid.rows {
        let y_start = row * height / grid.rows;
        let y_end = (row + 1) * height / grid.rows;
        let patch_h = y_end - y_start;

        for col in 0..grid.cols {
            let x_start = col * width / grid.cols;
            let x_end = (col + 1) * width / grid.cols;
            let patch_w = x_end - x_start;

            patches.push(PatchInfo {
                patch_index: (row * grid.cols + col) as u8,
                x_start,
                y_start,
                width: patch_w,
                height: patch_h,
                grid_col: col,
                grid_row: row,
            });
        }
    }

    patches
}

pub fn compute_boundary_edges(grid: &GridConfig, patches: &[PatchInfo]) -> Vec<BoundaryEdge> {
    let mut edges = Vec::new();

    for row in 0..grid.rows {
        for col in 0..grid.cols {
            let idx = (row * grid.cols + col) as usize;
            let patch = &patches[idx];

            if col + 1 < grid.cols {
                let right_idx = (row * grid.cols + col + 1) as usize;
                edges.push(BoundaryEdge {
                    patch_a: patch.patch_index,
                    patch_b: patches[right_idx].patch_index,
                    is_horizontal: false,
                    length: patch.height,
                });
            }

            if row + 1 < grid.rows {
                let bottom_idx = ((row + 1) * grid.cols + col) as usize;
                edges.push(BoundaryEdge {
                    patch_a: patch.patch_index,
                    patch_b: patches[bottom_idx].patch_index,
                    is_horizontal: true,
                    length: patch.width,
                });
            }
        }
    }

    edges
}

pub fn extract_patch_image(image: &[u8], img_width: u32, patch: &PatchInfo) -> Vec<u8> {
    let mut sub = Vec::with_capacity((patch.width * patch.height) as usize);
    for row in 0..patch.height {
        let src_offset = ((patch.y_start + row) * img_width + patch.x_start) as usize;
        sub.extend_from_slice(&image[src_offset..src_offset + patch.width as usize]);
    }
    sub
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_1x1() {
        let g = compute_grid_config(1).unwrap();
        assert_eq!(g.cols, 1);
        assert_eq!(g.rows, 1);
    }

    #[test]
    fn test_grid_2x2() {
        let g = compute_grid_config(4).unwrap();
        assert_eq!(g.cols, 2);
        assert_eq!(g.rows, 2);
    }

    #[test]
    fn test_grid_configs() {
        let cases = [(2, 2, 1), (8, 4, 2), (16, 4, 4), (32, 8, 4)];
        for (n, c, r) in cases {
            let g = compute_grid_config(n).unwrap();
            assert_eq!(g.cols, c, "num_patches={n}");
            assert_eq!(g.rows, r, "num_patches={n}");
        }
    }

    #[test]
    fn test_patches_even_split() {
        let g = compute_grid_config(4).unwrap();
        let patches = compute_patches(100, 100, &g);
        assert_eq!(patches.len(), 4);

        assert_eq!(patches[0].x_start, 0);
        assert_eq!(patches[0].y_start, 0);
        assert_eq!(patches[0].width, 50);
        assert_eq!(patches[0].height, 50);

        assert_eq!(patches[1].x_start, 50);
        assert_eq!(patches[1].width, 50);

        assert_eq!(patches[2].y_start, 50);
        assert_eq!(patches[2].height, 50);

        assert_eq!(patches[3].x_start, 50);
        assert_eq!(patches[3].y_start, 50);

        let total: u32 = patches.iter().map(|p| p.width * p.height).sum();
        assert_eq!(total, 10000);
    }

    #[test]
    fn test_patches_uneven_split() {
        let g = compute_grid_config(4).unwrap();
        let patches = compute_patches(101, 99, &g);
        assert_eq!(patches.len(), 4);

        let total: u32 = patches.iter().map(|p| p.width * p.height).sum();
        assert_eq!(total, 101 * 99);

        for p in &patches {
            assert!(p.width > 0);
            assert!(p.height > 0);
        }
    }

    #[test]
    fn test_boundary_edges_4patches() {
        let g = compute_grid_config(4).unwrap();
        let patches = compute_patches(100, 100, &g);
        let edges = compute_boundary_edges(&g, &patches);

        assert_eq!(edges.len(), 4);

        let vertical: Vec<_> = edges.iter().filter(|e| !e.is_horizontal).collect();
        let horizontal: Vec<_> = edges.iter().filter(|e| e.is_horizontal).collect();
        assert_eq!(vertical.len(), 2);
        assert_eq!(horizontal.len(), 2);
    }

    #[test]
    fn test_boundary_edges_2patches() {
        let g = compute_grid_config(2).unwrap();
        let patches = compute_patches(100, 50, &g);
        let edges = compute_boundary_edges(&g, &patches);
        assert_eq!(edges.len(), 1);
        assert!(!edges[0].is_horizontal);
        assert_eq!(edges[0].length, 50);
    }

    #[test]
    fn test_extract_patch_image() {
        let image: Vec<u8> = (0..20).collect();
        let patch = PatchInfo {
            patch_index: 0,
            x_start: 1,
            y_start: 1,
            width: 2,
            height: 2,
            grid_col: 0,
            grid_row: 0,
        };
        let sub = extract_patch_image(&image, 5, &patch);
        assert_eq!(sub, vec![6, 7, 11, 12]);
    }

    #[test]
    fn test_patches_cover_image() {
        for n in [1, 2, 4, 8, 16, 32] {
            let g = compute_grid_config(n).unwrap();
            let patches = compute_patches(640, 480, &g);
            assert_eq!(patches.len(), n as usize);

            let total: u32 = patches.iter().map(|p| p.width * p.height).sum();
            assert_eq!(total, 640 * 480, "num_patches={n}");

            for (i, p) in patches.iter().enumerate() {
                assert_eq!(p.patch_index, i as u8);
                assert!(p.x_start + p.width <= 640);
                assert!(p.y_start + p.height <= 480);
            }
        }
    }
}
