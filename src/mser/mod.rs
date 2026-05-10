//! MSER text-region extraction API.
//!
//! This module groups the MSER-specific public surface so callers can use
//! `text_region_rs::mser::...` separately from `text_region_rs::swt::...`.

pub(crate) mod block_memory;
pub(crate) mod heap;
pub mod params;
pub mod partition;
pub mod types;
pub mod v1;
pub mod v2;

pub use params::{ConnectedType, MserParams, ParallelConfig};
pub use types::{MserRegion, MserRegions, Point, Rect};
pub use v1::{extract_msers, extract_msers_parallel};
pub use v2::{
    extract_msers_v2, extract_msers_v2_parallel, extract_msers_v2_partitioned,
};
