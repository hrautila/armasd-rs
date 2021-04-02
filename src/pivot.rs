
use libarmasd_sys as ffi;
use std::mem;
use std::convert::TryInto;

pub struct Pivot {
    pivots: ffi::armas_pivot
}

impl Pivot {

    pub fn new(size: u32) -> Pivot {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_pivot>::zeroed();
            ffi::armas_pivot_init(m.as_mut_ptr(), size.try_into().unwrap_or(0));
            Pivot { pivots: m.assume_init() }
        }
    }

    pub fn as_ptr(&self) -> *const ffi::armas_pivot {
        &self.pivots
    }

    pub fn as_mut_ptr(&mut self) -> *mut ffi::armas_pivot {
        &mut self.pivots
    }
}

impl Drop for Pivot {
    fn drop(&mut self) {
        unsafe {
            ffi::armas_pivot_release(&mut self.pivots);
        }
    }

}
