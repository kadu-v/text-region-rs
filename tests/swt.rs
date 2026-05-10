use text_region_rs::swt::{
    INVALID_STROKE_WIDTH, SwtBgrInput, SwtImage, SwtInput, SwtParams,
    detect_text_regions_from_swt, detect_text_swt, detect_text_swt_with_debug,
    filter_swt_components, normalize_and_scale, stroke_width_transform,
    swt_connected_components, swt_preprocess_bgr,
};

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e == INVALID_STROKE_WIDTH {
            assert_eq!(a, e, "index {i}");
        } else {
            assert!(
                (a - e).abs() <= 1.0e-6,
                "index {i}: actual {a}, expected {e}"
            );
        }
    }
}

#[test]
fn swt_rejects_mismatched_input_buffers() {
    let err = stroke_width_transform(SwtInput {
        width: 2,
        height: 2,
        edge: &[0; 4],
        gradient_x: &[0.0; 3],
        gradient_y: &[0.0; 4],
        params: SwtParams::default(),
    })
    .unwrap_err();

    assert!(
        err.to_string().contains("gradient_x"),
        "unexpected error: {err}"
    );
}

#[test]
fn swt_keeps_non_ray_pixels_invalid() {
    let image = stroke_width_transform(SwtInput {
        width: 3,
        height: 2,
        edge: &[0; 6],
        gradient_x: &[0.0; 6],
        gradient_y: &[0.0; 6],
        params: SwtParams::default(),
    })
    .unwrap();

    assert_eq!(image.width, 3);
    assert_eq!(image.height, 2);
    assert_close(&image.data, &[INVALID_STROKE_WIDTH; 6]);
}

#[test]
fn swt_matches_opencv_first_and_second_pass_for_horizontal_stroke() {
    let width = 5;
    let height = 3;
    let mut edge = vec![0; width * height];
    let mut gradient_x = vec![0.0; width * height];
    let gradient_y = vec![0.0; width * height];

    edge[1 * width + 1] = 255;
    edge[1 * width + 3] = 255;
    gradient_x[1 * width + 1] = -1.0;
    gradient_x[1 * width + 3] = 1.0;

    let image = stroke_width_transform(SwtInput {
        width: width as u32,
        height: height as u32,
        edge: &edge,
        gradient_x: &gradient_x,
        gradient_y: &gradient_y,
        params: SwtParams::default(),
    })
    .unwrap();

    assert_close(
        &image.data,
        &[
            -1.0, -1.0, -1.0, -1.0, -1.0, //
            -1.0, 2.0, 2.0, 2.0, -1.0, //
            -1.0, -1.0, -1.0, -1.0, -1.0,
        ],
    );
}

#[test]
fn swt_rejects_same_direction_edge_gradients_like_opencv() {
    let width = 5;
    let height = 3;
    let mut edge = vec![0; width * height];
    let mut gradient_x = vec![0.0; width * height];
    let gradient_y = vec![0.0; width * height];

    edge[1 * width + 1] = 255;
    edge[1 * width + 3] = 255;
    gradient_x[1 * width + 1] = -1.0;
    gradient_x[1 * width + 3] = -1.0;

    let image = stroke_width_transform(SwtInput {
        width: width as u32,
        height: height as u32,
        edge: &edge,
        gradient_x: &gradient_x,
        gradient_y: &gradient_y,
        params: SwtParams::default(),
    })
    .unwrap();

    assert_close(&image.data, &[INVALID_STROKE_WIDTH; 15]);
}

#[test]
fn swt_normalize_and_scale_matches_opencv_reference_values() {
    let image = SwtImage {
        width: 3,
        height: 1,
        data: vec![INVALID_STROKE_WIDTH, 2.0, 6.0],
    };

    assert_eq!(normalize_and_scale(&image), vec![255, 0, 255]);
}

#[test]
fn swt_matches_opencv_reference_values_for_synthetic_rays() {
    let width = 8;
    let height = 7;
    let mut edge = vec![0; width * height];
    let mut gradient_x = vec![0.0; width * height];
    let mut gradient_y = vec![0.0; width * height];

    edge[1 * width + 1] = 255;
    edge[1 * width + 4] = 255;
    gradient_x[1 * width + 1] = -1.0;
    gradient_x[1 * width + 4] = 1.0;

    edge[5 * width + 1] = 255;
    edge[2 * width + 4] = 255;
    gradient_x[5 * width + 1] = -1.0;
    gradient_y[5 * width + 1] = 1.0;
    gradient_x[2 * width + 4] = 1.0;
    gradient_y[2 * width + 4] = -1.0;

    let actual = stroke_width_transform(SwtInput {
        width: width as u32,
        height: height as u32,
        edge: &edge,
        gradient_x: &gradient_x,
        gradient_y: &gradient_y,
        params: SwtParams::default(),
    })
    .unwrap();

    assert_close(
        &actual.data,
        &[
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 3.0, 3.0,
            3.0, 3.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.2426405,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.2426405, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, 4.2426405, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            4.2426405, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0,
        ],
    );
}

#[test]
fn swt_components_match_opencv_positive_neighbor_connectivity() {
    let image = SwtImage {
        width: 4,
        height: 3,
        data: vec![
            1.0, 10.0, -1.0, -1.0, //
            -1.0, 30.0, -1.0, 2.0, //
            -1.0, -1.0, -1.0, 2.0,
        ],
    };

    let components = swt_connected_components(&image).unwrap();
    let sizes = components
        .iter()
        .map(|component| component.len())
        .collect::<Vec<_>>();

    assert_eq!(sizes, vec![3, 2]);
}

#[test]
fn swt_component_filter_matches_opencv_variance_check() {
    let image = SwtImage {
        width: 5,
        height: 2,
        data: vec![
            2.0, 2.0, -1.0, 1.0, 9.0, //
            2.0, 2.0, -1.0, 1.0, 9.0,
        ],
    };

    let components = swt_connected_components(&image).unwrap();
    let filtered = filter_swt_components(&image, &components, false).unwrap();

    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].bounding_rect.x, 0);
    assert_eq!(filtered[0].bounding_rect.y, 0);
    assert_eq!(filtered[0].bounding_rect.width, 2);
    assert_eq!(filtered[0].bounding_rect.height, 2);
    assert_eq!(filtered[0].median, 2.0);
}

#[test]
fn swt_detect_text_regions_matches_opencv_chain_heuristics() {
    let width = 10;
    let height = 6;
    let mut data = vec![INVALID_STROKE_WIDTH; width * height];
    for &(x0, y0) in &[(1, 2), (4, 2), (7, 2)] {
        for y in y0..y0 + 2 {
            for x in x0..x0 + 2 {
                data[y * width + x] = 2.0;
            }
        }
    }
    let image = SwtImage {
        width: width as u32,
        height: height as u32,
        data,
    };
    let bgr = vec![120; width * height * 3];

    let detected = detect_text_regions_from_swt(&image, &bgr).unwrap();

    assert_eq!(detected.letter_bounding_boxes.len(), 3);
    assert_eq!(detected.chain_bounding_boxes.len(), 1);
    assert_eq!(detected.chain_bounding_boxes[0].x, 1);
    assert_eq!(detected.chain_bounding_boxes[0].y, 2);
    assert_eq!(detected.chain_bounding_boxes[0].width, 7);
    assert_eq!(detected.chain_bounding_boxes[0].height, 1);
}

#[test]
fn swt_preprocess_bgr_matches_opencv_grayscale_conversion() {
    let bgr = vec![
        0, 0, 255, //
        0, 255, 0, //
        255, 0, 0, //
    ];

    let preprocessed = swt_preprocess_bgr(3, 1, &bgr).unwrap();

    assert_eq!(preprocessed.gray, vec![76, 150, 29]);
    assert_eq!(preprocessed.edge.len(), 3);
    assert_eq!(preprocessed.gradient_x.len(), 3);
    assert_eq!(preprocessed.gradient_y.len(), 3);
}

#[test]
fn swt_detect_text_swt_rejects_bad_bgr_length() {
    let err = detect_text_swt(SwtBgrInput {
        width: 2,
        height: 2,
        bgr: &[0; 11],
        params: SwtParams::default(),
    })
    .unwrap_err();

    assert!(err.to_string().contains("bgr"), "unexpected error: {err}");
}

#[test]
fn swt_detect_text_swt_runs_full_opencv_style_pipeline() {
    let width = 48;
    let height = 24;
    let mut bgr = vec![255; width * height * 3];
    for &(x0, y0) in &[(8, 7), (18, 7), (28, 7)] {
        for y in y0..(y0 + 10) {
            for x in x0..(x0 + 5) {
                let idx = (y * width + x) * 3;
                bgr[idx] = 0;
                bgr[idx + 1] = 0;
                bgr[idx + 2] = 0;
            }
        }
    }

    let output = detect_text_swt_with_debug(SwtBgrInput {
        width: width as u32,
        height: height as u32,
        bgr: &bgr,
        params: SwtParams::default(),
    })
    .unwrap();

    assert_eq!(output.preprocessed.gray.len(), width * height);
    assert_eq!(output.preprocessed.edge.len(), width * height);
    assert_eq!(output.swt_image.data.len(), width * height);
    assert_eq!(output.normalized_swt.len(), width * height);
    assert_eq!(output.draw_bgr.len(), width * height * 3);
    assert!(output.preprocessed.edge.iter().any(|&value| value > 0));
    assert!(output.swt_image.data.iter().any(|&value| value > 0.0));

    let detections = detect_text_swt(SwtBgrInput {
        width: width as u32,
        height: height as u32,
        bgr: &bgr,
        params: SwtParams::default(),
    })
    .unwrap();
    assert_eq!(detections, output.detections);
}
