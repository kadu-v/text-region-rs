#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}

impl Rect {
    pub fn from_points(min: Point, max: Point) -> Self {
        Self {
            left: min.x,
            top: min.y,
            width: max.x - min.x + 1,
            height: max.y - min.y + 1,
        }
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
            left: 1,
            top: 2,
            width: 10,
            height: 20,
        };
        assert_eq!(r.left, 1);
        assert_eq!(r.top, 2);
        assert_eq!(r.width, 10);
        assert_eq!(r.height, 20);
    }

    #[test]
    fn test_rect_from_points() {
        let r = Rect::from_points(Point { x: 3, y: 5 }, Point { x: 7, y: 9 });
        assert_eq!(r.left, 3);
        assert_eq!(r.top, 5);
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
