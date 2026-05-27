//! Pure-Rust 16-bit PPM writer (P6 binary or P3 ASCII).
//!
//! Byte-for-byte compatible with the legacy `x3f_output_ppm.c`:
//!  - Header: `P6\n<cols> <rows>\n65535\n` (or `P3` for ASCII).
//!  - Binary samples are big-endian uint16.
//!  - ASCII samples are decimal with a trailing space *after every value*
//!    (including the last in a row), then `\n` to end the row.
//!
//! The trailing-space quirk is load-bearing for tier-2 MD5 stability.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::Image;

/// Write `image` as a PPM file. `binary=true` selects P6, false selects P3.
pub fn write(image: &Image, out: impl AsRef<Path>, binary: bool) -> io::Result<()> {
    if image.channels < 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "PPM requires at least 3 channels",
        ));
    }
    let f = File::create(out)?;
    let mut w = BufWriter::new(f);

    let magic = if binary { "P6" } else { "P3" };
    write!(w, "{magic}\n{} {}\n65535\n", image.columns, image.rows)?;

    let stride = image.row_stride as usize;
    let chans = image.channels as usize;
    let cols = image.columns as usize;
    let rows = image.rows as usize;

    for row in 0..rows {
        let row_off = row * stride;
        for col in 0..cols {
            let pix_off = row_off + col * chans;
            let r = image.data[pix_off];
            let g = image.data[pix_off + 1];
            let b = image.data[pix_off + 2];
            if binary {
                w.write_all(&r.to_be_bytes())?;
                w.write_all(&g.to_be_bytes())?;
                w.write_all(&b.to_be_bytes())?;
            } else {
                writeln!(w, "{r} {g} {b} ")?;
            }
        }
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn fake_image(cols: u32, rows: u32, stride_pad: u32) -> Image {
        let chans = 3u32;
        let row_stride = cols * chans + stride_pad;
        let mut data = vec![0u16; (rows * row_stride) as usize];
        // Write a recognisable pattern: pixel(r,c) = (r+1, c+1, r*c + 1).
        for r in 0..rows {
            for c in 0..cols {
                let off = (r * row_stride + c * chans) as usize;
                data[off] = (r + 1) as u16;
                data[off + 1] = (c + 1) as u16;
                data[off + 2] = (r * c + 1) as u16;
            }
        }
        Image {
            data,
            rows,
            columns: cols,
            channels: chans,
            row_stride,
            levels: crate::ImageLevels::default(),
            dng_highlight_scale: 1.0,
        }
    }

    fn read_all(path: &Path) -> Vec<u8> {
        let mut f = File::open(path).unwrap();
        let mut v = Vec::new();
        f.read_to_end(&mut v).unwrap();
        v
    }

    #[test]
    fn p6_header_and_be_payload() {
        let dir = tempdir();
        let img = fake_image(2, 1, 0);
        let path = dir.join("a.ppm");
        write(&img, &path, true).unwrap();
        let bytes = read_all(&path);
        // header
        let header = b"P6\n2 1\n65535\n";
        assert!(bytes.starts_with(header), "bad header: {bytes:?}");
        // payload: pixel0 = (1,1,1) BE; pixel1 = (1,2,1) BE
        let payload = &bytes[header.len()..];
        assert_eq!(payload, &[0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1]);
    }

    #[test]
    fn p3_header_and_ascii_payload() {
        let dir = tempdir();
        let img = fake_image(2, 1, 0);
        let path = dir.join("a.ppm");
        write(&img, &path, false).unwrap();
        let s = String::from_utf8(read_all(&path)).unwrap();
        assert_eq!(s, "P3\n2 1\n65535\n1 1 1 \n1 2 1 \n");
    }

    #[test]
    fn stride_padding_is_skipped() {
        // row_stride = cols*chans + 4: an extra 4 u16s of padding per row
        // that must NOT appear in the output.
        let dir = tempdir();
        let img = fake_image(1, 2, 4);
        let path = dir.join("a.ppm");
        write(&img, &path, true).unwrap();
        let bytes = read_all(&path);
        let header = b"P6\n1 2\n65535\n";
        let payload = &bytes[header.len()..];
        // Two pixels of three u16 BE each = 12 bytes. No padding bytes.
        assert_eq!(payload.len(), 12);
        assert_eq!(payload, &[0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1]);
    }

    #[test]
    fn channels_under_three_errors() {
        let dir = tempdir();
        let mut img = fake_image(1, 1, 0);
        img.channels = 1;
        img.row_stride = 1;
        img.data = vec![42];
        let err = write(&img, dir.join("a.ppm"), true).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    fn tempdir() -> std::path::PathBuf {
        let p =
            std::env::temp_dir().join(format!("x3f-ppm-test-{}-{}", std::process::id(), uniq()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn uniq() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        N.fetch_add(1, Ordering::Relaxed)
    }
}
