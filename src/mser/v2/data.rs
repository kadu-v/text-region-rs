use crate::mser::params::ConnectedType;
use crate::mser::v1::data::RegionFlag;

#[derive(Debug, Clone)]
pub struct MserRegionV2 {
    pub gray_level: u8,
    pub region_flag: RegionFlag,
    pub calculated_var: bool,
    pub assigned_pointer: bool,
    pub patch_index: u8,
    pub size: i32,
    pub unmerged_size: u32,
    pub var: f32,
    pub er_index: i32,
    pub parent: Option<usize>,
}

impl MserRegionV2 {
    pub fn new() -> Self {
        Self {
            gray_level: 0,
            region_flag: RegionFlag::Unknown,
            calculated_var: false,
            assigned_pointer: false,
            patch_index: 0,
            size: 0,
            unmerged_size: 0,
            var: 0.0,
            er_index: 0,
            parent: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConnectedCompV2 {
    pub region_idx: usize,
    pub gray_level: i16,
    pub size: i32,
}

impl ConnectedCompV2 {
    pub fn new() -> Self {
        Self {
            region_idx: 0,
            gray_level: 0,
            size: 0,
        }
    }
}

pub fn dir_shift(connected_type: ConnectedType) -> u32 {
    match connected_type {
        ConnectedType::FourConnected => 29,
        ConnectedType::EightConnected => 28,
    }
}

pub fn dir_mask_v2(connected_type: ConnectedType) -> u32 {
    match connected_type {
        ConnectedType::FourConnected => 0xe0000000,
        ConnectedType::EightConnected => 0xF0000000,
    }
}

pub fn boundary_pixel(connected_type: ConnectedType) -> u32 {
    match connected_type {
        ConnectedType::FourConnected => 5 << 29,
        ConnectedType::EightConnected => 9 << 28,
    }
}

pub fn compute_dir_offsets_v2(
    connected_type: ConnectedType,
    row_step: i32,
) -> Vec<i32> {
    // V2 uses 1-indexed directions (dir[1]..dir[4] or dir[1]..dir[8])
    // We store index 0 as unused, directions start at index 1
    match connected_type {
        ConnectedType::FourConnected => vec![
            0, // unused index 0
            1, -row_step, -1, row_step,
        ],
        ConnectedType::EightConnected => vec![
            0, // unused index 0
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

pub fn max_dir(connected_type: ConnectedType) -> u32 {
    match connected_type {
        ConnectedType::FourConnected => 4,
        ConnectedType::EightConnected => 8,
    }
}
