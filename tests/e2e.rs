use image::ImageReader;
use std::collections::HashSet;
use text_region_rs::params::{ConnectedType, MserParams};
use text_region_rs::types::MserRegion;
use text_region_rs::{extract_msers, extract_msers_v2};

fn load_grayscale(path: &str) -> (Vec<u8>, u32, u32) {
    let img = ImageReader::open(path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {}", path, e))
        .decode()
        .unwrap_or_else(|e| panic!("Failed to decode {}: {}", path, e))
        .into_luma8();
    let w = img.width();
    let h = img.height();
    (img.into_raw(), w, h)
}

const IMG_PAPER: &str = "resource/IMG_8237.jpeg";
const IMG_LABEL: &str = "resource/IMG_8364.jpeg";

// ---------------------------------------------------------------------------
// 不変条件チェックヘルパー
// ---------------------------------------------------------------------------

fn assert_invariants(regions: &[MserRegion], w: u32, h: u32, label: &str) {
    for (i, r) in regions.iter().enumerate() {
        // ピクセル座標が画像範囲内
        for pt in &r.points {
            assert!(
                pt.x >= 0 && pt.x < w as i32 && pt.y >= 0 && pt.y < h as i32,
                "{} region {}: point ({},{}) out of image {}x{}",
                label, i, pt.x, pt.y, w, h,
            );
        }

        // bounding rect が正
        assert!(r.bounding_rect.width > 0, "{} region {} has zero width", label, i);
        assert!(r.bounding_rect.height > 0, "{} region {} has zero height", label, i);
        assert!(r.bounding_rect.left >= 0, "{} region {} has negative left", label, i);
        assert!(r.bounding_rect.top >= 0, "{} region {} has negative top", label, i);

        // 全ピクセルが bounding rect 内
        for pt in &r.points {
            assert!(
                pt.x >= r.bounding_rect.left
                    && pt.x < r.bounding_rect.left + r.bounding_rect.width
                    && pt.y >= r.bounding_rect.top
                    && pt.y < r.bounding_rect.top + r.bounding_rect.height,
                "{} region {}: point ({},{}) outside bbox ({},{},{},{})",
                label, i, pt.x, pt.y,
                r.bounding_rect.left, r.bounding_rect.top,
                r.bounding_rect.width, r.bounding_rect.height,
            );
        }

        // リージョン内でピクセル重複なし
        let mut seen = HashSet::new();
        for pt in &r.points {
            assert!(
                seen.insert((pt.x, pt.y)),
                "{} region {}: duplicate pixel ({},{})",
                label, i, pt.x, pt.y,
            );
        }
    }
}

fn assert_size_filter(regions: &[MserRegion], min_point: usize, max_point: usize, label: &str) {
    for (i, r) in regions.iter().enumerate() {
        assert!(
            r.points.len() >= min_point,
            "{} region {}: size {} < min_point {}",
            label, i, r.points.len(), min_point,
        );
        assert!(
            r.points.len() <= max_point,
            "{} region {}: size {} > max_point {}",
            label, i, r.points.len(), max_point,
        );
    }
}

// ===========================================================================
// 1. 画像読み込み
// ===========================================================================

#[test]
fn e2e_load_images() {
    let (d1, w1, h1) = load_grayscale(IMG_PAPER);
    assert!(w1 > 0 && h1 > 0);
    assert_eq!(d1.len(), (w1 * h1) as usize);

    let (d2, w2, h2) = load_grayscale(IMG_LABEL);
    assert!(w2 > 0 && h2 > 0);
    assert_eq!(d2.len(), (w2 * h2) as usize);
}

// ===========================================================================
// 2. 論文画像 — MSER検出
// ===========================================================================

#[test]
fn e2e_paper_v1_detects_msers() {
    let (img, w, h) = load_grayscale(IMG_PAPER);
    let params = MserParams::default();
    let result = extract_msers(&img, w, h, &params);

    let total = result.from_min.len() + result.from_max.len();
    assert!(total > 0, "V1: paper image should have MSERs, got 0");

    assert!(!result.from_min.is_empty(), "V1: paper should have from_min (dark text)");
    assert!(!result.from_max.is_empty(), "V1: paper should have from_max");
}

#[test]
fn e2e_paper_v2_detects_msers() {
    let (img, w, h) = load_grayscale(IMG_PAPER);
    let params = MserParams::default();
    let result = extract_msers_v2(&img, w, h, &params);

    let total = result.from_min.len() + result.from_max.len();
    assert!(total > 0, "V2: paper image should have MSERs, got 0");

    assert!(!result.from_min.is_empty(), "V2: paper should have from_min (dark text)");
    assert!(!result.from_max.is_empty(), "V2: paper should have from_max");
}

// ===========================================================================
// 3. ラベル画像 — MSER検出
// ===========================================================================

#[test]
fn e2e_label_v1_detects_msers() {
    let (img, w, h) = load_grayscale(IMG_LABEL);
    let params = MserParams::default();
    let result = extract_msers(&img, w, h, &params);

    let total = result.from_min.len() + result.from_max.len();
    assert!(total > 0, "V1: label image should have MSERs, got 0");
}

#[test]
fn e2e_label_v2_detects_msers() {
    let (img, w, h) = load_grayscale(IMG_LABEL);
    let params = MserParams::default();
    let result = extract_msers_v2(&img, w, h, &params);

    let total = result.from_min.len() + result.from_max.len();
    assert!(total > 0, "V2: label image should have MSERs, got 0");
}

// ===========================================================================
// 4. 論文画像 — 不変条件
// ===========================================================================

#[test]
fn e2e_paper_v1_invariants() {
    let (img, w, h) = load_grayscale(IMG_PAPER);
    let params = MserParams::default();
    let result = extract_msers(&img, w, h, &params);

    assert_invariants(&result.from_min, w, h, "V1 from_min");
    assert_invariants(&result.from_max, w, h, "V1 from_max");
}

#[test]
fn e2e_paper_v2_invariants() {
    let (img, w, h) = load_grayscale(IMG_PAPER);
    let params = MserParams::default();
    let result = extract_msers_v2(&img, w, h, &params);

    assert_invariants(&result.from_min, w, h, "V2 from_min");
    assert_invariants(&result.from_max, w, h, "V2 from_max");
}

// ===========================================================================
// 5. ラベル画像 — 不変条件
// ===========================================================================

#[test]
fn e2e_label_v1_invariants() {
    let (img, w, h) = load_grayscale(IMG_LABEL);
    let params = MserParams::default();
    let result = extract_msers(&img, w, h, &params);

    assert_invariants(&result.from_min, w, h, "V1 from_min");
    assert_invariants(&result.from_max, w, h, "V1 from_max");
}

#[test]
fn e2e_label_v2_invariants() {
    let (img, w, h) = load_grayscale(IMG_LABEL);
    let params = MserParams::default();
    let result = extract_msers_v2(&img, w, h, &params);

    assert_invariants(&result.from_min, w, h, "V2 from_min");
    assert_invariants(&result.from_max, w, h, "V2 from_max");
}

// ===========================================================================
// 6. from_min のみ / from_max のみ
// ===========================================================================

#[test]
fn e2e_paper_from_min_only() {
    let (img, w, h) = load_grayscale(IMG_PAPER);
    let params = MserParams { from_min: true, from_max: false, ..MserParams::default() };

    let v1 = extract_msers(&img, w, h, &params);
    let v2 = extract_msers_v2(&img, w, h, &params);

    assert!(v1.from_max.is_empty());
    assert!(v2.from_max.is_empty());
    assert!(!v1.from_min.is_empty(), "V1: paper should have from_min MSERs");
    assert!(!v2.from_min.is_empty(), "V2: paper should have from_min MSERs");
}

#[test]
fn e2e_paper_from_max_only() {
    let (img, w, h) = load_grayscale(IMG_PAPER);
    let params = MserParams { from_min: false, from_max: true, ..MserParams::default() };

    let v1 = extract_msers(&img, w, h, &params);
    let v2 = extract_msers_v2(&img, w, h, &params);

    assert!(v1.from_min.is_empty());
    assert!(v2.from_min.is_empty());
}

#[test]
fn e2e_label_from_min_only() {
    let (img, w, h) = load_grayscale(IMG_LABEL);
    let params = MserParams { from_min: true, from_max: false, ..MserParams::default() };

    let v1 = extract_msers(&img, w, h, &params);
    let v2 = extract_msers_v2(&img, w, h, &params);

    assert!(v1.from_max.is_empty());
    assert!(v2.from_max.is_empty());
}

#[test]
fn e2e_label_from_max_only() {
    let (img, w, h) = load_grayscale(IMG_LABEL);
    let params = MserParams { from_min: false, from_max: true, ..MserParams::default() };

    let v1 = extract_msers(&img, w, h, &params);
    let v2 = extract_msers_v2(&img, w, h, &params);

    assert!(v1.from_min.is_empty());
    assert!(v2.from_min.is_empty());
}

// ===========================================================================
// 7. min_point フィルタリング
// ===========================================================================

#[test]
fn e2e_paper_min_point_monotonic() {
    let (img, w, h) = load_grayscale(IMG_PAPER);

    let count = |min_pt: i32| {
        let params = MserParams { min_point: min_pt, ..MserParams::default() };
        let r = extract_msers(&img, w, h, &params);
        r.from_min.len() + r.from_max.len()
    };

    let c10 = count(10);
    let c50 = count(50);
    let c200 = count(200);

    assert!(c50 <= c10, "min_point=50 should give <= MSERs than 10: {} > {}", c50, c10);
    assert!(c200 <= c50, "min_point=200 should give <= MSERs than 50: {} > {}", c200, c50);
}

#[test]
fn e2e_label_min_point_respects_filter() {
    let (img, w, h) = load_grayscale(IMG_LABEL);
    let min_pt = 100;
    let params = MserParams { min_point: min_pt, ..MserParams::default() };

    let v1 = extract_msers(&img, w, h, &params);
    let v2 = extract_msers_v2(&img, w, h, &params);

    assert_size_filter(&v1.from_min, min_pt as usize, usize::MAX, "V1 from_min");
    assert_size_filter(&v1.from_max, min_pt as usize, usize::MAX, "V1 from_max");
    assert_size_filter(&v2.from_min, min_pt as usize, usize::MAX, "V2 from_min");
    assert_size_filter(&v2.from_max, min_pt as usize, usize::MAX, "V2 from_max");
}

// ===========================================================================
// 8. max_point_ratio フィルタリング
// ===========================================================================

#[test]
fn e2e_label_max_point_respects_filter() {
    let (img, w, h) = load_grayscale(IMG_LABEL);
    let ratio = 0.01f32;
    let max_point = (ratio * (w * h) as f32) as usize;
    let params = MserParams { max_point_ratio: ratio, ..MserParams::default() };

    let v1 = extract_msers(&img, w, h, &params);
    let v2 = extract_msers_v2(&img, w, h, &params);

    assert_size_filter(&v1.from_min, 0, max_point, "V1 from_min");
    assert_size_filter(&v1.from_max, 0, max_point, "V1 from_max");
    assert_size_filter(&v2.from_min, 0, max_point, "V2 from_min");
    assert_size_filter(&v2.from_max, 0, max_point, "V2 from_max");
}

#[test]
fn e2e_paper_max_point_monotonic() {
    let (img, w, h) = load_grayscale(IMG_PAPER);

    let count = |ratio: f32| {
        let params = MserParams { max_point_ratio: ratio, ..MserParams::default() };
        let r = extract_msers(&img, w, h, &params);
        r.from_min.len() + r.from_max.len()
    };

    let c01 = count(0.01);
    let c10 = count(0.10);
    let c50 = count(0.50);

    assert!(c01 <= c10, "ratio=0.01 should give <= MSERs than 0.10: {} > {}", c01, c10);
    assert!(c10 <= c50, "ratio=0.10 should give <= MSERs than 0.50: {} > {}", c10, c50);
}

// ===========================================================================
// 9. delta パラメータ
// ===========================================================================

#[test]
fn e2e_paper_delta_monotonic() {
    let (img, w, h) = load_grayscale(IMG_PAPER);

    let count = |delta: i32| {
        let params = MserParams { delta, ..MserParams::default() };
        let r = extract_msers(&img, w, h, &params);
        r.from_min.len() + r.from_max.len()
    };

    let d1 = count(1);
    let d5 = count(5);
    let d10 = count(10);

    assert!(d5 <= d1, "delta=5 should give <= MSERs than 1: {} > {}", d5, d1);
    assert!(d10 <= d5, "delta=10 should give <= MSERs than 5: {} > {}", d10, d5);
}

#[test]
fn e2e_label_delta_monotonic() {
    let (img, w, h) = load_grayscale(IMG_LABEL);

    let count = |delta: i32| {
        let params = MserParams { delta, ..MserParams::default() };
        let r = extract_msers_v2(&img, w, h, &params);
        r.from_min.len() + r.from_max.len()
    };

    let d1 = count(1);
    let d5 = count(5);
    let d10 = count(10);

    assert!(d5 <= d1, "V2 delta=5 should give <= MSERs than 1: {} > {}", d5, d1);
    assert!(d10 <= d5, "V2 delta=10 should give <= MSERs than 5: {} > {}", d10, d5);
}

// ===========================================================================
// 10. stable_variation パラメータ
// ===========================================================================

#[test]
fn e2e_paper_stable_variation_monotonic() {
    let (img, w, h) = load_grayscale(IMG_PAPER);

    let count = |sv: f32| {
        let params = MserParams { stable_variation: sv, ..MserParams::default() };
        let r = extract_msers(&img, w, h, &params);
        r.from_min.len() + r.from_max.len()
    };

    let s01 = count(0.1);
    let s05 = count(0.5);
    let s20 = count(2.0);

    assert!(s01 <= s05, "sv=0.1 should give <= MSERs than 0.5: {} > {}", s01, s05);
    assert!(s05 <= s20, "sv=0.5 should give <= MSERs than 2.0: {} > {}", s05, s20);
}

#[test]
fn e2e_label_stable_variation_monotonic() {
    let (img, w, h) = load_grayscale(IMG_LABEL);

    let count = |sv: f32| {
        let params = MserParams { stable_variation: sv, ..MserParams::default() };
        let r = extract_msers_v2(&img, w, h, &params);
        r.from_min.len() + r.from_max.len()
    };

    let s01 = count(0.1);
    let s05 = count(0.5);
    let s20 = count(2.0);

    assert!(s01 <= s05, "V2 sv=0.1 should give <= MSERs than 0.5: {} > {}", s01, s05);
    assert!(s05 <= s20, "V2 sv=0.5 should give <= MSERs than 2.0: {} > {}", s05, s20);
}

// ===========================================================================
// 11. NMS
// ===========================================================================

#[test]
fn e2e_paper_nms_reduces_count() {
    let (img, w, h) = load_grayscale(IMG_PAPER);

    let params_off = MserParams { nms_similarity: -1.0, ..MserParams::default() };
    let params_on = MserParams { nms_similarity: 0.0, ..MserParams::default() };

    let off = extract_msers(&img, w, h, &params_off);
    let on = extract_msers(&img, w, h, &params_on);

    let off_total = off.from_min.len() + off.from_max.len();
    let on_total = on.from_min.len() + on.from_max.len();

    assert!(on_total <= off_total, "NMS should reduce count: off={}, on={}", off_total, on_total);
}

#[test]
fn e2e_label_nms_reduces_count() {
    let (img, w, h) = load_grayscale(IMG_LABEL);

    let params_off = MserParams { nms_similarity: -1.0, ..MserParams::default() };
    let params_on = MserParams { nms_similarity: 0.0, ..MserParams::default() };

    let off = extract_msers_v2(&img, w, h, &params_off);
    let on = extract_msers_v2(&img, w, h, &params_on);

    let off_total = off.from_min.len() + off.from_max.len();
    let on_total = on.from_min.len() + on.from_max.len();

    assert!(on_total <= off_total, "V2 NMS should reduce count: off={}, on={}", off_total, on_total);
}

// ===========================================================================
// 12. duplicated_variation
// ===========================================================================

#[test]
fn e2e_paper_duplicated_variation() {
    let (img, w, h) = load_grayscale(IMG_PAPER);

    let params_off = MserParams { duplicated_variation: 0.0, ..MserParams::default() };
    let params_on = MserParams { duplicated_variation: 0.2, ..MserParams::default() };

    let off = extract_msers(&img, w, h, &params_off);
    let on = extract_msers(&img, w, h, &params_on);

    let off_total = off.from_min.len() + off.from_max.len();
    let on_total = on.from_min.len() + on.from_max.len();

    assert!(
        on_total <= off_total,
        "Dup removal should reduce count: off={}, on={}",
        off_total, on_total,
    );
}

// ===========================================================================
// 13. 4-connected vs 8-connected
// ===========================================================================

#[test]
fn e2e_label_4conn_v1_invariants() {
    let (img, w, h) = load_grayscale(IMG_LABEL);
    let params = MserParams {
        connected_type: ConnectedType::FourConnected,
        ..MserParams::default()
    };

    let result = extract_msers(&img, w, h, &params);
    assert_invariants(&result.from_min, w, h, "V1 4conn from_min");
    assert_invariants(&result.from_max, w, h, "V1 4conn from_max");
}

#[test]
fn e2e_label_8conn_v1_invariants() {
    let (img, w, h) = load_grayscale(IMG_LABEL);
    let params = MserParams {
        connected_type: ConnectedType::EightConnected,
        ..MserParams::default()
    };

    let result = extract_msers(&img, w, h, &params);
    assert_invariants(&result.from_min, w, h, "V1 8conn from_min");
    assert_invariants(&result.from_max, w, h, "V1 8conn from_max");
}

#[test]
fn e2e_label_8conn_v2_invariants() {
    let (img, w, h) = load_grayscale(IMG_LABEL);
    let params = MserParams {
        connected_type: ConnectedType::EightConnected,
        ..MserParams::default()
    };

    let result = extract_msers_v2(&img, w, h, &params);
    assert_invariants(&result.from_min, w, h, "V2 8conn from_min");
    assert_invariants(&result.from_max, w, h, "V2 8conn from_max");
}

// ===========================================================================
// 14. 決定性
// ===========================================================================

#[test]
fn e2e_paper_deterministic() {
    let (img, w, h) = load_grayscale(IMG_PAPER);
    let params = MserParams::default();

    let r1 = extract_msers(&img, w, h, &params);
    let r2 = extract_msers(&img, w, h, &params);

    assert_eq!(r1.from_min.len(), r2.from_min.len());
    assert_eq!(r1.from_max.len(), r2.from_max.len());

    for (a, b) in r1.from_min.iter().zip(r2.from_min.iter()) {
        assert_eq!(a.gray_level, b.gray_level);
        assert_eq!(a.points.len(), b.points.len());
    }
}

#[test]
fn e2e_label_deterministic_v2() {
    let (img, w, h) = load_grayscale(IMG_LABEL);
    let params = MserParams::default();

    let r1 = extract_msers_v2(&img, w, h, &params);
    let r2 = extract_msers_v2(&img, w, h, &params);

    assert_eq!(r1.from_min.len(), r2.from_min.len());
    assert_eq!(r1.from_max.len(), r2.from_max.len());

    for (a, b) in r1.from_min.iter().zip(r2.from_min.iter()) {
        assert_eq!(a.gray_level, b.gray_level);
        assert_eq!(a.points.len(), b.points.len());
    }
}

// ===========================================================================
// 15. V1/V2 検出数の近さ（完全一致ではなく1%以内）
// ===========================================================================

fn assert_v1_v2_close(
    v1_count: usize,
    v2_count: usize,
    tolerance_pct: f64,
    label: &str,
) {
    if v1_count == 0 && v2_count == 0 {
        return;
    }
    let max = v1_count.max(v2_count) as f64;
    let diff = (v1_count as f64 - v2_count as f64).abs();
    let pct = diff / max * 100.0;
    assert!(
        pct <= tolerance_pct,
        "{}: V1={} V2={} differ by {:.1}% (tolerance {:.1}%)",
        label, v1_count, v2_count, pct, tolerance_pct,
    );
}

#[test]
fn e2e_paper_v1_v2_count_close() {
    let (img, w, h) = load_grayscale(IMG_PAPER);
    let params = MserParams::default();

    let v1 = extract_msers(&img, w, h, &params);
    let v2 = extract_msers_v2(&img, w, h, &params);

    assert_v1_v2_close(v1.from_min.len(), v2.from_min.len(), 1.0, "paper from_min");
    assert_v1_v2_close(v1.from_max.len(), v2.from_max.len(), 1.0, "paper from_max");
}

#[test]
fn e2e_label_v1_v2_count_close() {
    let (img, w, h) = load_grayscale(IMG_LABEL);
    let params = MserParams::default();

    let v1 = extract_msers(&img, w, h, &params);
    let v2 = extract_msers_v2(&img, w, h, &params);

    assert_v1_v2_close(v1.from_min.len(), v2.from_min.len(), 1.0, "label from_min");
    assert_v1_v2_close(v1.from_max.len(), v2.from_max.len(), 1.0, "label from_max");
}

// ===========================================================================
// 16. 複数パラメータでの V1/V2 近似一致
// ===========================================================================

#[test]
fn e2e_paper_multi_params_v1_v2_close() {
    let (img, w, h) = load_grayscale(IMG_PAPER);

    let configs = [
        MserParams { delta: 5, ..MserParams::default() },
        MserParams { min_point: 50, max_point_ratio: 0.1, ..MserParams::default() },
        MserParams { stable_variation: 1.0, nms_similarity: -1.0, ..MserParams::default() },
        MserParams {
            connected_type: ConnectedType::EightConnected,
            ..MserParams::default()
        },
    ];

    for (i, params) in configs.iter().enumerate() {
        let v1 = extract_msers(&img, w, h, params);
        let v2 = extract_msers_v2(&img, w, h, params);

        let label = format!("paper config {}", i);
        assert_v1_v2_close(
            v1.from_min.len() + v1.from_max.len(),
            v2.from_min.len() + v2.from_max.len(),
            2.0,
            &label,
        );
    }
}

#[test]
fn e2e_label_multi_params_v1_v2_close() {
    let (img, w, h) = load_grayscale(IMG_LABEL);

    let configs = [
        MserParams { delta: 3, ..MserParams::default() },
        MserParams { min_point: 30, max_point_ratio: 0.05, ..MserParams::default() },
        MserParams { stable_variation: 0.8, duplicated_variation: 0.05, ..MserParams::default() },
        MserParams {
            connected_type: ConnectedType::EightConnected,
            ..MserParams::default()
        },
    ];

    for (i, params) in configs.iter().enumerate() {
        let v1 = extract_msers(&img, w, h, params);
        let v2 = extract_msers_v2(&img, w, h, params);

        let label = format!("label config {}", i);
        assert_v1_v2_close(
            v1.from_min.len() + v1.from_max.len(),
            v2.from_min.len() + v2.from_max.len(),
            2.0,
            &label,
        );
    }
}
