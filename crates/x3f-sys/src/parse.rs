//! Bounded reads and allocations shared by the X3F container and section parsers.

use crate::control::{Control, Error, Result};
use crate::sysabi as libc;

// The largest supported Quattro raster is under 400 MiB. A single parser
// allocation above 1 GiB indicates hostile dimensions, not another camera.
pub(crate) const MAX_ALLOCATION: usize = 1024 * 1024 * 1024;

pub(crate) unsafe fn alloc<T>(count: usize) -> Result<*mut T> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(Error::Allocation)?;
    if bytes > MAX_ALLOCATION {
        return Err(Error::Allocation);
    }
    if count == 0 {
        return Ok(std::ptr::null_mut());
    }
    let pointer = unsafe { libc::calloc(count, std::mem::size_of::<T>()) }.cast::<T>();
    if pointer.is_null() {
        Err(Error::Allocation)
    } else {
        Ok(pointer)
    }
}

pub(crate) unsafe fn copy_alloc<T: Copy>(data: &[T]) -> Result<*mut T> {
    let pointer = unsafe { alloc::<T>(data.len()) }?;
    if !data.is_empty() {
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), pointer, data.len()) };
    }
    Ok(pointer)
}

pub(crate) fn checked_size(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&n| n <= MAX_ALLOCATION)
        .ok_or(Error::Allocation)
}

pub(crate) struct Input<'a> {
    file: *mut libc::FILE,
    base: u64,
    position: u64,
    end: u64,
    control: Control<'a>,
}

impl<'a> Input<'a> {
    pub(crate) unsafe fn new(file: *mut libc::FILE, control: Control<'a>) -> Result<Self> {
        control.check()?;
        if file.is_null() {
            return Err(Error::InvalidData("missing input stream"));
        }
        if unsafe { libc::fseek(file, 0, libc::SEEK_END) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let length = unsafe { libc::ftell(file) };
        if length < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let (base, end) = control.window().unwrap_or((0, length as u64));
        if base.checked_add(end).is_none_or(|end| end > length as u64) {
            return Err(Error::InvalidData("window outside input"));
        }
        let mut result = Self {
            file,
            base,
            position: end,
            end,
            control,
        };
        result.seek(0)?;
        Ok(result)
    }

    pub(crate) fn position(&self) -> u64 {
        self.position
    }
    pub(crate) fn end(&self) -> u64 {
        self.end
    }
    pub(crate) fn remaining(&self) -> usize {
        (self.end - self.position) as usize
    }

    pub(crate) fn seek(&mut self, position: u64) -> Result<()> {
        self.control.check()?;
        if position > self.end {
            return Err(Error::InvalidData("offset outside input"));
        }
        let offset = libc::c_long::try_from(self.base + position)
            .map_err(|_| Error::InvalidData("input offset exceeds platform limits"))?;
        if unsafe { libc::fseek(self.file, offset, libc::SEEK_SET) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        self.position = position;
        Ok(())
    }

    pub(crate) fn limit(&mut self, end: u64) -> Result<()> {
        if end < self.position || end > self.end {
            return Err(Error::InvalidData("section outside input"));
        }
        self.end = end;
        Ok(())
    }

    pub(crate) fn read(&mut self, bytes: &mut [u8]) -> Result<()> {
        if bytes.len() as u64 > self.end - self.position {
            return Err(Error::InvalidData("truncated input"));
        }
        for chunk in bytes.chunks_mut(64 * 1024) {
            self.control.check()?;
            let mut read = 0;
            while read < chunk.len() {
                self.control.check()?;
                let count = unsafe {
                    libc::fread(
                        chunk[read..].as_mut_ptr().cast(),
                        1,
                        chunk.len() - read,
                        self.file,
                    )
                };
                if count == 0 {
                    return Err(Error::InvalidData("truncated input"));
                }
                read += count;
                self.position += count as u64;
            }
        }
        self.control.check()
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        let mut b = [0];
        self.read(&mut b)?;
        Ok(b[0])
    }
    pub(crate) fn u16(&mut self) -> Result<u16> {
        let mut b = [0; 2];
        self.read(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    pub(crate) fn u32(&mut self) -> Result<u32> {
        let mut b = [0; 4];
        self.read(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    pub(crate) unsafe fn read_alloc(&mut self, size: usize) -> Result<*mut u8> {
        if size > self.remaining() {
            return Err(Error::InvalidData("truncated section"));
        }
        let pointer = unsafe { alloc::<u8>(size) }?;
        if size == 0 {
            return Ok(pointer);
        }
        if let Err(error) = self.read(unsafe { std::slice::from_raw_parts_mut(pointer, size) }) {
            unsafe { libc::free(pointer.cast()) };
            return Err(error);
        }
        Ok(pointer)
    }
}

pub(crate) fn bytes_at(data: &[u8], offset: usize, size: usize) -> Result<&[u8]> {
    let end = offset
        .checked_add(size)
        .ok_or(Error::InvalidData("offset overflow"))?;
    data.get(offset..end)
        .ok_or(Error::InvalidData("value outside section"))
}

pub(crate) fn u32_at(data: &[u8], offset: usize) -> Result<u32> {
    let bytes: [u8; 4] = bytes_at(data, offset, 4)?.try_into().unwrap();
    Ok(u32::from_le_bytes(bytes))
}

pub(crate) fn cstr_at(data: &[u8], offset: usize) -> Result<()> {
    let string = data
        .get(offset..)
        .ok_or(Error::InvalidData("string outside section"))?;
    if string.contains(&0) {
        Ok(())
    } else {
        Err(Error::InvalidData("unterminated string"))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn windows_bound_positions_reads_and_section_limits() {
        unsafe {
            let file = libc::tmpfile();
            assert!(!file.is_null());
            let bytes = b"beforeFRAMEafter";
            assert_eq!(
                libc::fwrite(bytes.as_ptr().cast(), 1, bytes.len(), file),
                bytes.len()
            );
            let mut input = Input::new(file, Control::none().with_window(6, 5)).unwrap();
            assert_eq!(
                (input.position(), input.end(), input.remaining()),
                (0, 5, 5)
            );
            assert!(matches!(input.seek(6), Err(Error::InvalidData(_))));
            assert!(matches!(input.limit(6), Err(Error::InvalidData(_))));
            let mut data = [0; 6];
            assert!(input.read(&mut data).is_err());
            assert_eq!(input.position(), 0);
            assert_eq!(libc::ftell(file), 6);
            input.read(&mut data[..5]).unwrap();
            assert_eq!(&data[..5], b"FRAME");
            assert_eq!(input.remaining(), 0);
            assert!(input.read(&mut data[..1]).is_err());
            assert_eq!(libc::ftell(file), 11);
            input.seek(1).unwrap();
            input.limit(3).unwrap();
            assert!(input.seek(4).is_err());
            assert!(input.read(&mut data[..3]).is_err());
            for (base, length) in [(6, 100), (u64::MAX, 1), (1, u64::MAX)] {
                assert!(matches!(
                    Input::new(file, Control::none().with_window(base, length)),
                    Err(Error::InvalidData(_))
                ));
            }
            let mut input = Input::new(file, Control::none()).unwrap();
            assert_eq!(input.end(), bytes.len() as u64);
            input.read(&mut data).unwrap();
            assert_eq!(&data, b"before");
            libc::fclose(file);
        }
    }
}
