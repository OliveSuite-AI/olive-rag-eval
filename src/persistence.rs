//! Portable publication of immutable snapshots and replacement of run artifacts.
use std::{
    io::{self, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

pub fn sync_file(path: &Path) -> io::Result<()> {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?
        .sync_all()
}

pub fn sync_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        std::fs::File::open(path)?.sync_all()
    }
    // Windows has no equivalent directory fsync. Files are flushed before
    // publication; file replacement below requests write-through semantics.
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    #[cfg(not(windows))]
    {
        std::fs::rename(source, destination)
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::{
            MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
        };
        let source = source.canonicalize()?;
        let destination = destination
            .parent()
            .unwrap_or(Path::new("."))
            .canonicalize()?
            .join(destination.file_name().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "destination must be a file")
            })?);
        let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
        let destination: Vec<u16> = destination
            .as_os_str()
            .encode_wide()
            .chain(Some(0))
            .collect();
        // SAFETY: both buffers are NUL-terminated and remain alive for the call.
        if unsafe {
            MoveFileExW(
                source.as_ptr(),
                destination.as_ptr(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
            )
        } == 0
        {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}
pub fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let temporary = path.with_extension(format!(
        "{}-{}.tmp",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        replace_file(&temporary, path)?;
        sync_directory(path.parent().unwrap_or(Path::new(".")))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temporary);
    }
    result
}
