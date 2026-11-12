//! Image decoding and resizing: bytes in, RGB pixels at a model's input size out.
use fast_image_resize::{
    FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::Image, images::ImageRef,
};
use image::{DynamicImage, ImageDecoder, ImageReader};
use std::io::Cursor;

/// Same cap as the Node version (sharp's limitInputPixels).
// ponytail: the whole image is decoded in memory (3-4 bytes per pixel, so up to
// ~400 MB at the cap); decode JPEGs at a reduced DCT scale if big uploads hurt.
const MAX_PIXELS: u64 = 100_000_000;

#[derive(Debug)]
pub struct DecodeError;

/// A decoded image: 8-bit RGB, EXIF rotation applied, transparency flattened onto white.
pub struct Rgb {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

pub enum Fit {
    /// Stretch to the target, ignoring the aspect ratio.
    Fill,
    /// Scale the shortest edge to the target and crop the centre.
    Cover,
}

impl Rgb {
    /// A mid-grey square, for warm-up inferences.
    pub fn blank(size: u32) -> Rgb {
        Rgb {
            data: vec![128; (size * size * 3) as usize],
            width: size,
            height: size,
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<Rgb, DecodeError> {
        match image::guess_format(bytes) {
            Ok(format) => raster(bytes, format).map_err(|_| DecodeError),
            Err(_) => svg(bytes).ok_or(DecodeError),
        }
    }

    /// Resized to `size`×`size`, as interleaved RGB bytes.
    pub fn resize(&self, size: u32, fit: Fit, filter: FilterType) -> Vec<u8> {
        let src = ImageRef::new(self.width, self.height, &self.data, PixelType::U8x3)
            .expect("buffer matches its size");
        let mut dst = Image::new(size, size, PixelType::U8x3);
        let mut options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(filter));
        if let Fit::Cover = fit {
            options = options.fit_into_destination(Some((0.5, 0.5)));
        }
        Resizer::new()
            .resize(&src, &mut dst, &options)
            .expect("same pixel type on both sides");
        dst.into_vec()
    }
}

fn raster(bytes: &[u8], format: image::ImageFormat) -> image::ImageResult<Rgb> {
    let mut decoder = ImageReader::with_format(Cursor::new(bytes), format).into_decoder()?;
    let (width, height) = decoder.dimensions();
    if u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(image::ImageError::Limits(
            image::error::LimitError::from_kind(image::error::LimitErrorKind::DimensionError),
        ));
    }
    let orientation = decoder.orientation()?;
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);

    let data = if img.color().has_alpha() {
        let rgba = img.into_rgba8();
        let mut data = Vec::with_capacity(rgba.len() / 4 * 3);
        for px in rgba.chunks_exact(4) {
            let a = u32::from(px[3]);
            // Blend onto white, rounded.
            data.extend(
                px[..3]
                    .iter()
                    .map(|&c| ((u32::from(c) * a + 255 * (255 - a) + 127) / 255) as u8),
            );
        }
        return Ok(Rgb {
            width: rgba.width(),
            height: rgba.height(),
            data,
        });
    } else {
        img.into_rgb8()
    };
    Ok(Rgb {
        width: data.width(),
        height: data.height(),
        data: data.into_raw(),
    })
}

fn svg(bytes: &[u8]) -> Option<Rgb> {
    use resvg::{tiny_skia, usvg};
    let tree = usvg::Tree::from_data(bytes, &usvg::Options::default()).ok()?;
    let size = tree.size().to_int_size();
    if u64::from(size.width()) * u64::from(size.height()) > MAX_PIXELS {
        return None;
    }
    let mut pixmap = tiny_skia::Pixmap::new(size.width(), size.height())?;
    pixmap.fill(tiny_skia::Color::WHITE);
    resvg::render(
        &tree,
        tiny_skia::Transform::identity(),
        &mut pixmap.as_mut(),
    );
    // Opaque white underneath, so every pixel ends up opaque: dropping alpha is exact.
    let data = pixmap
        .data()
        .chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    Some(Rgb {
        width: size.width(),
        height: size.height(),
        data,
    })
}
