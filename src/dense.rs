
use libarmasd_sys as ffi;

use std::mem;
use std::convert::TryInto;
use super::{CopyOps};

#[derive(Debug)]
pub struct Matrix {
    data: ffi::armas_dense
}

impl Matrix {
    pub fn as_ptr(&self) -> *const ffi::armas_dense {
        &self.data
    }

    pub fn as_mut_ptr(&mut self) -> *mut ffi::armas_dense {
        &mut self.data
    }

    pub fn new(rows: u32, cols: u32) -> Matrix {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_init(m.as_mut_ptr(), rows.try_into().unwrap_or(0), cols.try_into().unwrap_or(0));
            Matrix { data: m.assume_init() }
        }
    }

    /// Create new matrix with provided array as data. If array too small to hold rows*cols elements
    /// then zero size matrix is returned.
    pub fn from_array(rows: u32, cols: u32, array: &mut [f64]) -> Matrix {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            if ((rows * cols) as usize) < array.len() {
                ffi::armas_make(m.as_mut_ptr(), rows as i32, cols as i32, rows as i32, array.as_mut_ptr());
            }
            Matrix { data: m.assume_init() }
        }
    }

    /// Create new matrix with provided array as data. If vector too small to hold rows*cols elements
    /// then zero size matrix is returned.
    pub fn from_vector(rows: u32, cols: u32, mut vec: Vec<f64>) -> Matrix {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            if ((rows * cols) as usize) < vec.len() {
                ffi::armas_make(m.as_mut_ptr(), rows as i32, cols as i32, rows as i32, vec.as_mut_ptr());
            }
            Matrix { data: m.assume_init() }
        }
    }

    pub fn uniform(rows: u32, cols: u32) -> Matrix {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_init(m.as_mut_ptr(), rows.try_into().unwrap_or(0), cols.try_into().unwrap_or(0));
            ffi::armas_set_all(m.as_mut_ptr(), ffi::armas_uniform, 0);
            Matrix { data: m.assume_init() }
        }
    }

    pub fn normal(rows: u32, cols: u32) -> Matrix {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_init(m.as_mut_ptr(), rows.try_into().unwrap_or(0), cols.try_into().unwrap_or(0));
            ffi::armas_set_all(m.as_mut_ptr(), ffi::armas_normal, 0);
            Matrix { data: m.assume_init() }
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.data.rows.try_into().unwrap_or(0), self.data.cols.try_into().unwrap_or(0))
    }

    pub fn submatrix(&self, row: u32, col: u32, nrows: u32, ncols: u32) -> Matrix {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_submatrix_unsafe(m.as_mut_ptr(), &self.data,
                row.try_into().unwrap_or(0),
                col.try_into().unwrap_or(0),
                nrows.try_into().unwrap_or(0),
                ncols.try_into().unwrap_or(0));
            Matrix { data: m.assume_init() }
        }
    }

    pub fn copy_to(&self, dst: &mut Matrix, opts: CopyOps) -> &Matrix {
        if self.size() != dst.size() {
            return &self;
        }
        unsafe {
            ffi::armas_mcopy(&mut dst.data, &self.data, opts as i32);
        }
        self
    }

    pub fn diagonal(&self, n: u32) -> Vector {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_diag_unsafe(m.as_mut_ptr(), &self.data, n.try_into().unwrap_or(0));
            Vector { data: m.assume_init() }
        }
    }

    pub fn row(&self, n: u32) -> Vector {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_row_unsafe(m.as_mut_ptr(), &self.data, n.try_into().unwrap_or(0));
            Vector { data: m.assume_init() }
        }
    }

    /// Copy self to destination.
    /// TODO: return self or dest? With error as Result<T, E>?
    pub fn column(&self, n: i32) -> Vector {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_row_unsafe(m.as_mut_ptr(), &self.data, n);
            Vector { data: m.assume_init() }
        }
    }

    pub fn get(&self, row: u32, col: u32) -> f64 {
        unsafe {
            ffi::armas_get_unsafe(&self.data, row.try_into().unwrap_or(0), col.try_into().unwrap_or(0))
        }
    }

    pub fn set(&mut self, row: u32, col: u32, value: f64) {
        unsafe {
            ffi::armas_set_unsafe(&mut self.data, row.try_into().unwrap_or(0), col.try_into().unwrap_or(0), value);
        }
    }

    pub fn set_all(&mut self, func: extern fn() -> f64) {
        unsafe {
            ffi::armas_set_all(&mut self.data, func, 0);
        }
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe {
            ffi::armas_release(&mut self.data);
        }
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        let (rows, cols) = self.size();
        let mut mat = Matrix::new(rows, cols);
        self.copy_to(&mut mat, CopyOps::All);
        return mat;
    }
}

#[derive(Debug)]
pub struct Vector {
    pub data: ffi::armas_dense
}

impl Vector {
    pub fn as_ptr(&self) -> *const ffi::armas_dense {
        &self.data
    }

    pub fn as_mut_ptr(&mut self) -> *mut ffi::armas_dense {
        &mut self.data
    }

    pub fn new(n: u32) -> Vector {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_init(m.as_mut_ptr(), n.try_into().unwrap_or(0), 1);
            Vector { data: m.assume_init() }
        }
    }

    pub fn size(&self) -> u32 {
        let rows: u32 = self.data.rows.try_into().unwrap_or(0);
        let cols: u32 = self.data.cols.try_into().unwrap_or(0);
        // note: either cols or rows is 1
        return rows*cols;
    }

    pub fn get(&self, index: u32) -> f64 {
        unsafe {
            ffi::armas_get_at_unsafe(&self.data, index.try_into().unwrap_or(0))
        }
    }

    pub fn set(&mut self, index: u32, value: f64) {
        unsafe {
            ffi::armas_set_at_unsafe(&mut self.data, index.try_into().unwrap_or(0), value)
        }
    }

    /// Copy self to destination.
    /// TODO: return self or dest? With error as Result<T, E>?
    pub fn copy_to(&self, dst: &mut Vector) -> &Vector {
        if self.size() != dst.size() {
            return &self;
        }
        unsafe {
            ffi::armas_mcopy(&mut dst.data, &self.data, 0);
        }
        self
    }
}

impl Drop for Vector {
    fn drop(&mut self) {
        unsafe {
            ffi::armas_release(&mut self.data);
        }
    }
}

impl Clone for Vector {
    fn clone(&self) -> Self {
        let mut vec = Vector::new(self.size());
        self.copy_to(&mut vec);
        vec
    }
}

