#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

pub type Rect = image::math::Rect;

pub(crate) fn rect_from_points(min: Point, max: Point) -> Rect {
    debug_assert!(min.x >= 0);
    debug_assert!(min.y >= 0);
    debug_assert!(max.x >= min.x);
    debug_assert!(max.y >= min.y);

    Rect {
        x: min.x as u32,
        y: min.y as u32,
        width: (max.x - min.x + 1) as u32,
        height: (max.y - min.y + 1) as u32,
    }
}

#[derive(Debug, Clone)]
pub struct MserRegion {
    pub gray_level: u8,
    pub points: Vec<Point>,
    pub bounding_rect: Rect,
}

#[derive(Debug, Clone, Default)]
pub struct MserResult {
    pub from_min: Vec<MserRegion>,
    pub from_max: Vec<MserRegion>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let p = Point { x: 10, y: 20 };
        assert_eq!(p.x, 10);
        assert_eq!(p.y, 20);
    }

    #[test]
    fn test_rect_creation() {
        let r = Rect {
            x: 1,
            y: 2,
            width: 10,
            height: 20,
        };
        assert_eq!(r.x, 1);
        assert_eq!(r.y, 2);
        assert_eq!(r.width, 10);
        assert_eq!(r.height, 20);
    }

    #[test]
    fn test_rect_from_points() {
        let r = rect_from_points(Point { x: 3, y: 5 }, Point { x: 7, y: 9 });
        assert_eq!(r.x, 3);
        assert_eq!(r.y, 5);
        assert_eq!(r.width, 5);
        assert_eq!(r.height, 5);
    }

    #[test]
    fn test_mser_result_default() {
        let r = MserResult::default();
        assert!(r.from_min.is_empty());
        assert!(r.from_max.is_empty());
    }
}
