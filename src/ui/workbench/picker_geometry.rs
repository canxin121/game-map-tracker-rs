use gpui::{Bounds, Pixels, point, px, size};

use crate::config::CaptureRegion;

pub(super) fn capture_region_from_selection_bounds(
    display_bounds: Bounds<Pixels>,
    selection: Bounds<Pixels>,
) -> CaptureRegion {
    CaptureRegion {
        top: (f32::from(display_bounds.origin.y) + f32::from(selection.origin.y)).round() as i32,
        left: (f32::from(display_bounds.origin.x) + f32::from(selection.origin.x)).round() as i32,
        width: f32::from(selection.size.width).round().max(1.0) as u32,
        height: f32::from(selection.size.height).round().max(1.0) as u32,
    }
}

pub(super) fn selection_bounds_from_capture_region(
    display_bounds: Bounds<Pixels>,
    region: &CaptureRegion,
) -> Option<Bounds<Pixels>> {
    if region.width == 0 || region.height == 0 {
        return None;
    }

    Some(Bounds {
        origin: point(
            px(region.left as f32 - f32::from(display_bounds.origin.x)),
            px(region.top as f32 - f32::from(display_bounds.origin.y)),
        ),
        size: size(px(region.width as f32), px(region.height as f32)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_region_round_trips_through_selection_bounds() {
        let display_bounds = Bounds {
            origin: point(px(1200.0), px(80.0)),
            size: size(px(1600.0), px(900.0)),
        };
        let region = CaptureRegion {
            top: 116,
            left: 1660,
            width: 590,
            height: 38,
        };

        let selection =
            selection_bounds_from_capture_region(display_bounds, &region).expect("selection");
        let round_trip = capture_region_from_selection_bounds(display_bounds, selection);

        assert_eq!(round_trip, region);
    }
}
