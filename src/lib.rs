pub mod error;
pub mod mser;
pub mod swt;

pub use mser::{
    MserParams, MserRegions, ParallelConfig, extract_msers,
    extract_msers_parallel, extract_msers_v2, extract_msers_v2_parallel,
    extract_msers_v2_partitioned,
};
