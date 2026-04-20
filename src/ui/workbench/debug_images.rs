use std::sync::Arc;

use gpui::{Bounds, Pixels, RenderImage, point, px, size};
use image::{Frame, GrayImage, ImageBuffer, Rgba, RgbaImage};

use crate::tracking::debug::{DebugImage, DebugImageFormat};

pub(super) fn render_image_from_debug_image(image: &DebugImage) -> Option<Arc<RenderImage>> {
    if image.width == 0 || image.height == 0 || image.pixels.is_empty() {
        return None;
    }

    let mut bgra = match image.format {
        DebugImageFormat::Gray8 => {
            let gray = GrayImage::from_raw(image.width, image.height, image.pixels.clone())?;
            RgbaImage::from_fn(gray.width(), gray.height(), |x, y| {
                let value = gray.get_pixel(x, y).0[0];
                Rgba([value, value, value, 255])
            })
        }
        DebugImageFormat::Rgba8 => {
            ImageBuffer::from_raw(image.width, image.height, image.pixels.clone())?
        }
    };

    // gpui stores RenderImage pixels in BGRA order.
    for pixel in bgra.pixels_mut() {
        pixel.0.swap(0, 2);
    }

    Some(Arc::new(RenderImage::new(vec![Frame::new(bgra)])))
}

pub(super) fn contained_image_bounds(
    bounds: Bounds<Pixels>,
    image_width: u32,
    image_height: u32,
) -> Bounds<Pixels> {
    if image_width == 0 || image_height == 0 {
        return bounds;
    }

    let available_width = f32::from(bounds.size.width).max(1.0);
    let available_height = f32::from(bounds.size.height).max(1.0);
    let width_scale = available_width / image_width as f32;
    let height_scale = available_height / image_height as f32;
    let scale = width_scale.min(height_scale).max(0.0001);
    let render_width = image_width as f32 * scale;
    let render_height = image_height as f32 * scale;
    let offset_x = (available_width - render_width) * 0.5;
    let offset_y = (available_height - render_height) * 0.5;

    Bounds {
        origin: point(
            bounds.origin.x + px(offset_x),
            bounds.origin.y + px(offset_y),
        ),
        size: size(px(render_width), px(render_height)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_debug_images_are_converted_to_bgra_for_gpui() {
        let render_image = render_image_from_debug_image(&DebugImage::rgba(
            "probe",
            1,
            1,
            crate::tracking::debug::DebugImageKind::Snapshot,
            vec![0x11, 0x22, 0x33, 0x44],
        ))
        .expect("render image");

        assert_eq!(
            render_image.as_bytes(0).expect("frame bytes"),
            &[0x33, 0x22, 0x11, 0x44]
        );
    }

    #[test]
    fn gray_debug_images_expand_to_opaque_bgra() {
        let render_image =
            render_image_from_debug_image(&DebugImage::new("probe-mask", 1, 1, vec![0x7f]))
                .expect("render image");

        assert_eq!(
            render_image.as_bytes(0).expect("frame bytes"),
            &[0x7f, 0x7f, 0x7f, 0xff]
        );
    }
}
