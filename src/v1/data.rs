use crate::params::ConnectedType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionFlag {
    Unknown,
    Invalid,
    Valid,
    Merged,
}

#[derive(Debug, Clone, Copy)]
pub struct LinkedPoint {
    pub x: u16,
    pub y: u16,
    pub next: i32,
    pub prev: i32,
    pub ref_: i32,
}

impl Default for LinkedPoint {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            next: -1,
            prev: -1,
            ref_: -1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MserRegionV1 {
    pub gray_level: u8,
    pub region_flag: RegionFlag,
    pub calculated_var: bool,
    pub boundary_region: bool,
    pub patch_index: u8,
    pub size: i32,
    pub unmerged_size: u32,
    pub var: f32,
    pub parent: Option<usize>,
    pub head: i32,
    pub tail: i32,
    pub left: u16,
    pub right: u16,
    pub top: u16,
    pub bottom: u16,
}

impl MserRegionV1 {
    pub fn new() -> Self {
        Self {
            gray_level: 0,
            region_flag: RegionFlag::Unknown,
            calculated_var: false,
            boundary_region: false,
            patch_index: 0,
            size: 0,
            unmerged_size: 0,
            var: 0.0,
            parent: None,
            head: -1,
            tail: -1,
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConnectedCompV1 {
    pub head: i32,
    pub tail: i32,
    pub region_idx: usize,
    pub gray_level: i16,
    pub size: i32,
    pub left: u16,
    pub right: u16,
    pub top: u16,
    pub bottom: u16,
}

impl ConnectedCompV1 {
    pub fn new() -> Self {
        Self {
            head: -1,
            tail: -1,
            region_idx: 0,
            gray_level: 0,
            size: 0,
            left: 0,
            right: 0,
            top: 0,
            bottom: 0,
        }
    }
}

pub const BOUNDARY_YES_MASK: i16 = 0x4000;
pub const GRAY_MASK_BITS: i16 = 0x01ff;
pub const VISITED_FLAG: i16 = -0x8000i16; // 0x8000 as signed

pub fn compute_dir_offsets(
    connected_type: ConnectedType,
    row_step: i32,
) -> Vec<i32> {
    match connected_type {
        ConnectedType::FourConnected => vec![1, -row_step, -1, row_step],
        ConnectedType::EightConnected => vec![
            1,
            1 - row_step,
            -row_step,
            -1 - row_step,
            -1,
            -1 + row_step,
            row_step,
            1 + row_step,
        ],
    }
}

pub fn dir_mask(connected_type: ConnectedType) -> i16 {
    match connected_type {
        ConnectedType::FourConnected => 0x0e00,
        ConnectedType::EightConnected => 0x1e00,
    }
}

pub fn dir_max(connected_type: ConnectedType) -> i16 {
    match connected_type {
        ConnectedType::FourConnected => 0x0800,
        ConnectedType::EightConnected => 0x1000,
    }
}

pub const DIR_SHIFT: i16 = 0x0200;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dir_4connected() {
        let row_step = 7; // width=5, padded width = 5+2 = 7
        let dirs = compute_dir_offsets(ConnectedType::FourConnected, row_step);
        assert_eq!(dirs, vec![1, -7, -1, 7]);
    }

    #[test]
    fn test_dir_8connected() {
        let row_step = 7;
        let dirs = compute_dir_offsets(ConnectedType::EightConnected, row_step);
        assert_eq!(dirs, vec![1, 1 - 7, -7, -1 - 7, -1, -1 + 7, 7, 1 + 7]);
        assert_eq!(dirs, vec![1, -6, -7, -8, -1, 6, 7, 8]);
    }
}
