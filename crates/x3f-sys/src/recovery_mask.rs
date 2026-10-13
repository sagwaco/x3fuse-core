//! Write a separate Portable Graymap (PGM) mask of unreliable source measurements.
//! A value of 255 marks pixels with any layer below full reliability.
//! Zero means all layers are fully reliable. Neither value describes changes
//! to the output image or how much detail can be recovered.
use super::{DngCtx, LocalRecovery};
use crate::Control;
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Return the export result directly to this conversion's ProcessingInfo.
pub(super) fn write(
    path: &Path,
    ctx: &DngCtx<'_>,
    model: Option<&LocalRecovery>,
    bounds: [usize; 4],
    control: Control<'_>,
) -> io::Result<()> {
    check_cancel(control)?;
    if !ctx.recovery {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "mask export requires DNG recovery",
        ));
    }
    let model = model.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "mask export requires native sensor reliability",
        )
    })?;
    if bounds[0] >= bounds[2]
        || bounds[1] >= bounds[3]
        || bounds[2] > ctx.rows as usize
        || bounds[3] > ctx.cols as usize
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid mask bounds",
        ));
    }
    write_file(path, bounds, |r, c| model.mask(r, c), control)
}

fn check_cancel(control: Control<'_>) -> io::Result<()> {
    control
        .check()
        .map_err(|error| io::Error::new(io::ErrorKind::Interrupted, error))
}

fn write_file(
    path: &Path,
    bounds: [usize; 4],
    mask: impl FnMut(usize, usize) -> [u8; 3],
    control: Control<'_>,
) -> io::Result<()> {
    check_cancel(control)?;
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut out = BufWriter::new(file);
    write_mask(&mut out, bounds, mask, control)?;
    out.flush()?;
    out.get_ref().sync_all()?;
    check_cancel(control)
}

fn write_mask(
    out: &mut impl Write,
    bounds: [usize; 4],
    mut mask: impl FnMut(usize, usize) -> [u8; 3],
    control: Control<'_>,
) -> io::Result<()> {
    check_cancel(control)?;
    let [top, left, bottom, right] = bounds;
    assert!(top < bottom && left < right);
    let width = right - left;
    writeln!(out, "P5\n{width} {}\n255", bottom - top)?;
    let mut row = vec![0; width];
    for r in top..bottom {
        check_cancel(control)?;
        for (i, c) in (left..right).enumerate() {
            row[i] = if mask(r, c) == [255; 3] { 0 } else { 255 };
        }
        out.write_all(&row)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_export_does_not_create_a_file() {
        let path = std::env::temp_dir().join(format!(
            "x3f-cancelled-mask-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let cancel = std::sync::atomic::AtomicBool::new(true);
        assert_eq!(
            write_file(&path, [0, 0, 1, 1], |_, _| [255; 3], Control::new(&cancel))
                .unwrap_err()
                .kind(),
            io::ErrorKind::Interrupted
        );
        assert!(!path.exists());
    }

    #[test]
    fn cancellation_between_rows_stops_a_partial_mask() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let cancel = AtomicBool::new(false);
        let mut out = Vec::new();
        let error = write_mask(
            &mut out,
            [0, 0, 2, 1],
            |r, _| {
                assert_eq!(r, 0);
                cancel.store(true, Ordering::Relaxed);
                [255; 3]
            },
            Control::new(&cancel),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert_eq!(out, b"P5\n1 2\n255\n\x00");
    }

    #[test]
    fn exclusive_mask_creation_preserves_existing_bytes_and_reports_io_errors() {
        let name = format!(
            "x3f-mask-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let directory = std::env::temp_dir().join(name);
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("mask.pgm");
        write_file(
            &path,
            [0, 0, 1, 2],
            |_, c| if c == 0 { [255; 3] } else { [0; 3] },
            Control::none(),
        )
        .unwrap();
        let expected = b"P5\n2 1\n255\n\x00\xff";
        assert_eq!(std::fs::read(&path).unwrap(), expected);
        assert_eq!(
            write_file(&path, [0, 0, 1, 1], |_, _| [255; 3], Control::none())
                .unwrap_err()
                .kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(std::fs::read(&path).unwrap(), expected);
        assert_eq!(
            write_file(
                &directory.join("missing/mask.pgm"),
                [0, 0, 1, 1],
                |_, _| [255; 3],
                Control::none(),
            )
            .unwrap_err()
            .kind(),
            io::ErrorKind::NotFound
        );
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn healthy_partial_and_fully_clipped_are_distinguished() {
        let source = [[255; 3], [254, 255, 255], [255, 0, 255], [0; 3]];
        let mut out = Vec::new();
        write_mask(&mut out, [0, 0, 1, 4], |_, c| source[c], Control::none()).unwrap();
        assert_eq!(out, b"P5\n4 1\n255\n\x00\xff\xff\xff");
        assert_eq!(source[0], [255; 3]);
    }

    #[test]
    fn active_bounds_are_cropped_without_rotation_or_coordinate_shift() {
        let mut visited = Vec::new();
        let mut out = Vec::new();
        write_mask(
            &mut out,
            [2, 3, 4, 5],
            |r, c| {
                visited.push((r, c));
                if (r, c) == (3, 4) {
                    [255, 254, 255]
                } else {
                    [255; 3]
                }
            },
            Control::none(),
        )
        .unwrap();
        assert_eq!(visited, [(2, 3), (2, 4), (3, 3), (3, 4)]);
        assert_eq!(out, b"P5\n2 2\n255\n\x00\x00\x00\xff");
    }

    #[test]
    fn io_failure_is_not_silently_accepted() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("fixture"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        assert!(write_mask(&mut Broken, [0, 0, 1, 1], |_, _| [255; 3], Control::none()).is_err());
    }
}
