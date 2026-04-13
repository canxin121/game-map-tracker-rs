use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WorldPoint {
    pub x: f32,
    pub y: f32,
}

impl WorldPoint {
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldRect {
    pub min: WorldPoint,
    pub max: WorldPoint,
}

impl WorldRect {
    #[must_use]
    pub fn from_point(point: WorldPoint) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    pub fn include(&mut self, point: WorldPoint) {
        self.min.x = self.min.x.min(point.x);
        self.min.y = self.min.y.min(point.y);
        self.max.x = self.max.x.max(point.x);
        self.max.y = self.max.y.max(point.y);
    }

    #[must_use]
    pub fn width(self) -> f32 {
        self.max.x - self.min.x
    }

    #[must_use]
    pub fn height(self) -> f32 {
        self.max.y - self.min.y
    }

    #[must_use]
    pub fn center(self) -> WorldPoint {
        WorldPoint::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapDimensions {
    pub width: u32,
    pub height: u32,
}

impl MapDimensions {
    #[must_use]
    pub fn as_world_rect(self) -> WorldRect {
        WorldRect {
            min: WorldPoint::new(0.0, 0.0),
            max: WorldPoint::new(self.width as f32, self.height as f32),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ViewportSize {
    pub width: f32,
    pub height: f32,
}

impl ViewportSize {
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.width > 1.0 && self.height > 1.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapCamera {
    pub zoom: f32,
    pub offset_x: f32,
    pub offset_y: f32,
}

impl Default for MapCamera {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }
}

impl MapCamera {
    pub fn fit_rect(&mut self, rect: WorldRect, viewport: ViewportSize, padding: f32) {
        if !viewport.is_valid() {
            return;
        }

        let padded_width = (viewport.width - padding * 2.0).max(1.0);
        let padded_height = (viewport.height - padding * 2.0).max(1.0);
        let rect_width = rect.width().max(1.0);
        let rect_height = rect.height().max(1.0);

        self.zoom = (padded_width / rect_width)
            .min(padded_height / rect_height)
            .clamp(0.05, 8.0);

        let center = rect.center();
        self.offset_x = viewport.width * 0.5 - center.x * self.zoom;
        self.offset_y = viewport.height * 0.5 - center.y * self.zoom;
    }

    pub fn pan_by(&mut self, dx: f32, dy: f32) {
        self.offset_x += dx;
        self.offset_y += dy;
    }

    pub fn zoom_at(&mut self, anchor_x: f32, anchor_y: f32, delta: f32) {
        let next_zoom = (self.zoom * (1.0 + delta)).clamp(0.05, 8.0);
        let scale = next_zoom / self.zoom;
        self.offset_x = anchor_x - (anchor_x - self.offset_x) * scale;
        self.offset_y = anchor_y - (anchor_y - self.offset_y) * scale;
        self.zoom = next_zoom;
    }

    pub fn center_on(&mut self, point: WorldPoint, viewport: ViewportSize) {
        if !viewport.is_valid() {
            return;
        }

        self.offset_x = viewport.width * 0.5 - point.x * self.zoom;
        self.offset_y = viewport.height * 0.5 - point.y * self.zoom;
    }

    #[must_use]
    pub fn world_to_screen(self, point: WorldPoint) -> WorldPoint {
        WorldPoint::new(
            point.x * self.zoom + self.offset_x,
            point.y * self.zoom + self.offset_y,
        )
    }

    #[must_use]
    pub fn screen_to_world(self, point: WorldPoint) -> WorldPoint {
        WorldPoint::new(
            (point.x - self.offset_x) / self.zoom.max(0.0001),
            (point.y - self.offset_y) / self.zoom.max(0.0001),
        )
    }
}
