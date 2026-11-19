//! Best-effort OS allocator tuning and page trimming.
//!
//! Glibc retains freed pages in allocator arenas after burst allocations.
//! Call [`release_freed_memory`] at the boundary that owns and drops a
//! transient working set, and [`limit_allocator_arenas`] before worker
//! threads start.

/// Cap per-thread glibc arenas when `MALLOC_ARENA_MAX` is unset.
///
/// Glibc reads `MALLOC_ARENA_MAX` before `main`, so setting the environment
/// variable inside `main` cannot limit arenas. Applying `M_ARENA_MAX` before
/// spawning workers keeps concurrent file reads from retaining an arena per
/// thread.
pub fn limit_allocator_arenas() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    if std::env::var_os("MALLOC_ARENA_MAX").is_none() {
        const M_ARENA_MAX: std::ffi::c_int = -8;
        #[allow(unsafe_code)]
        unsafe {
            // SAFETY: mallopt locks the allocator while changing its arena
            // limit. Existing allocations remain valid.
            unsafe extern "C" {
                fn mallopt(param: std::ffi::c_int, value: std::ffi::c_int) -> std::ffi::c_int;
            }
            let _ = mallopt(M_ARENA_MAX, 2);
        }
    }
}

/// Release freed heap pages back to the OS when supported.
///
/// On glibc this calls `malloc_trim(0)`. Other allocators and platforms
/// either do not expose an equivalent or handle decay internally.
pub fn release_freed_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[allow(unsafe_code)]
    unsafe {
        // SAFETY: malloc_trim is thread-safe per glibc documentation and has
        // no visible effect except returning unused allocator pages to the OS.
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> std::ffi::c_int;
        }
        let _ = malloc_trim(0);
    }
}

/// Release freed memory when a transient allocation scope exits.
#[must_use = "dropping immediately trims before the transient scope exits"]
pub struct ReleaseFreedMemoryOnDrop;

impl Drop for ReleaseFreedMemoryOnDrop {
    fn drop(&mut self) {
        release_freed_memory();
    }
}
