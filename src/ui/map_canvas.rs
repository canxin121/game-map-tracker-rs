use gpui::{Bounds, Pixels, Point, point, px, size};

use crate::domain::{
    geometry::{MapCamera, MapDimensions, ViewportSize, WorldPoint},
    route::RouteDocument,
};

#[derive(Debug, Clone, Default)]
pub struct MapViewportState {
    pub camera: MapCamera,
    pub viewport: ViewportSize,
    pub needs_fit: bool,
    pub dragging_from: Option<WorldPoint>,
    pub pending_center: Option<WorldPoint>,
}

impl MapViewportState {
    pub fn update_viewport(&mut self, width: f32, height: f32) {
        self.viewport = ViewportSize { width, height };
    }

    pub fn request_fit(&mut self) {
        self.needs_fit = true;
    }

    pub fn center_on_or_queue(&mut self, point: WorldPoint) {
        if self.viewport.is_valid() {
            self.camera.center_on(point, self.viewport);
            self.pending_center = None;
        } else {
            self.pending_center = Some(point);
        }
    }

    pub fn fit_to_route_or_map(
        &mut self,
        route: Option<&RouteDocument>,
        map_dimensions: MapDimensions,
        padding: f32,
    ) {
        if !self.needs_fit || !self.viewport.is_valid() {
            return;
        }

        if let Some(bounds) = route.and_then(RouteDocument::bounds) {
            self.camera.fit_rect(bounds, self.viewport, padding);
        } else {
            self.camera
                .fit_rect(map_dimensions.as_world_rect(), self.viewport, padding);
        }

        self.needs_fit = false;
    }

    pub fn apply_pending_center(&mut self) -> bool {
        let Some(point) = self.pending_center.take() else {
            return false;
        };
        self.camera.center_on(point, self.viewport);
        true
    }
}

#[must_use]
pub fn route_points(route: &RouteDocument) -> Vec<WorldPoint> {
    route
        .points
        .iter()
        .map(|point| WorldPoint::new(point.x, point.y))
        .collect()
}

#[must_use]
pub fn screen_points(camera: MapCamera, points: &[WorldPoint]) -> Vec<WorldPoint> {
    points
        .iter()
        .copied()
        .map(|point| camera.world_to_screen(point))
        .collect()
}

#[must_use]
pub fn parse_hex_color(value: &str, fallback: u32) -> u32 {
    let hex = value.trim().trim_start_matches('#');
    if hex.len() == 6 && hex.chars().all(|ch| ch.is_ascii_hexdigit()) {
        u32::from_str_radix(hex, 16).unwrap_or(fallback)
    } else {
        fallback
    }
}

#[must_use]
pub fn marker_image_bounds(anchor: Point<Pixels>, size_px: f32) -> Bounds<Pixels> {
    let width = size_px.clamp(14.0, 64.0);
    let height = width * 1.25;
    Bounds {
        origin: point(anchor.x - px(width * 0.5), anchor.y - px(height)),
        size: size(px(width), px(height)),
    }
}

#[must_use]
pub fn inflate_bounds(bounds: Bounds<Pixels>, inset: f32) -> Bounds<Pixels> {
    Bounds {
        origin: point(bounds.origin.x - px(inset), bounds.origin.y - px(inset)),
        size: size(
            px(f32::from(bounds.size.width) + inset * 2.0),
            px(f32::from(bounds.size.height) + inset * 2.0),
        ),
    }
}

pub fn bounds_corner_radius(bounds: Bounds<Pixels>, max_radius: f32) -> Pixels {
    let min_side = f32::from(bounds.size.width.min(bounds.size.height));
    px((min_side * 0.3).min(max_radius))
}
