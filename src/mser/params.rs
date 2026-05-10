#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectedType {
    FourConnected,
    EightConnected,
}

#[derive(Debug, Clone)]
pub struct MserParams {
    pub delta: i32,
    pub stable_variation: f32,
    pub duplicated_variation: f32,
    pub nms_similarity: f32,
    pub connected_type: ConnectedType,
    pub min_point: i32,
    pub max_point_ratio: f32,
    pub from_min: bool,
    pub from_max: bool,
}

impl Default for MserParams {
    fn default() -> Self {
        Self {
            delta: 1,
            stable_variation: 0.5,
            duplicated_variation: 0.1,
            nms_similarity: 0.0,
            connected_type: ConnectedType::FourConnected,
            min_point: 20,
            max_point_ratio: 0.25,
            from_min: true,
            from_max: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParallelConfig {
    pub num_patches: u32,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self { num_patches: 4 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let p = MserParams::default();
        assert_eq!(p.delta, 1);
        assert_eq!(p.stable_variation, 0.5);
        assert_eq!(p.duplicated_variation, 0.1);
        assert_eq!(p.nms_similarity, 0.0);
        assert_eq!(p.connected_type, ConnectedType::FourConnected);
        assert_eq!(p.min_point, 20);
        assert_eq!(p.max_point_ratio, 0.25);
        assert!(p.from_min);
        assert!(p.from_max);
    }
}
