
use libarmasd_sys as ffi;

use std::mem;
use std::convert::TryInto;
use super::{CopyOps};
use super::vec::{Vector};

use serde::{Serialize, Serializer, Deserialize};
use serde::ser::{SerializeStruct, SerializeSeq};

#[derive(Deserialize)]
struct MatrixShadow {
    rows: u32,
    cols: u32,
    data: Vec<f64>
}

#[derive(Debug, Deserialize)]
#[serde(from = "MatrixShadow")]
pub struct Matrix {
    data: ffi::armas_dense,
    vec: Box<Vec<f64>>
}

pub struct MatrixIterator<'a> {
    source: &'a Matrix,
    index: u32,
    size: u32,
    rows: u32
}

impl Matrix {
    pub fn as_ptr(&self) -> *const ffi::armas_dense {
        &self.data
    }

    pub fn as_mut_ptr(&mut self) -> *mut ffi::armas_dense {
        &mut self.data
    }

    /// Create new matrix of spesificed size.
    pub fn new(rows: u32, cols: u32) -> Matrix {
        unsafe {
            let count: usize = (rows * cols) as usize;
            let mut vec: Vec<f64> = Vec::with_capacity(count);
            vec.set_len(count);
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_make(m.as_mut_ptr(), rows as i32, cols as i32, rows as i32, vec.as_mut_ptr());
            Matrix { data: m.assume_init(), vec: Box::new(vec) }
        }
    }

    pub fn new_from(rows: u32, cols: u32, mut vec: Vec<f64>) -> Matrix {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            if ((rows * cols) as usize) < vec.len() {
                ffi::armas_make(m.as_mut_ptr(), rows as i32, cols as i32, rows as i32, vec.as_mut_ptr());
            }
            Matrix { data: m.assume_init(), vec: Box::new(vec) }
        }
    }

    /// Create new matrix with provided array as data. If vector too small to hold rows*cols elements
    /// then zero size matrix is returned.
    pub fn from_vector(rows: u32, cols: u32, vec: &mut Vec<f64>) -> Matrix {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            if ((rows * cols) as usize) < vec.len() {
                ffi::armas_make(m.as_mut_ptr(), rows as i32, cols as i32, rows as i32, vec.as_mut_ptr());
            }
            Matrix { data: m.assume_init(), vec: Box::new(Vec::new()) }
        }
    }

    pub fn uniform(rows: u32, cols: u32) -> Matrix {
        let mut m = Matrix::new(rows, cols);
        unsafe {
            ffi::armas_set_all(m.as_mut_ptr(), ffi::armas_uniform, 0);
        }
        m
    }

    pub fn normal(rows: u32, cols: u32) -> Matrix {
        let mut m = Matrix::new(rows, cols);
        unsafe {
            ffi::armas_set_all(m.as_mut_ptr(), ffi::armas_normal, 0);
        }
        m
    }

    pub fn size(&self) -> (u32, u32) {
        (self.data.rows.try_into().unwrap_or(0), self.data.cols.try_into().unwrap_or(0))
    }

    /// Create spesified submatrix view over  matrix. Result matrix shares storage
    /// with the original matrix and changes on submatrix are visible in the original.
    pub fn submatrix(&self, row: u32, col: u32, nrows: u32, ncols: u32) -> Matrix {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_submatrix_unsafe(m.as_mut_ptr(), &self.data,
                row.try_into().unwrap_or(0),
                col.try_into().unwrap_or(0),
                nrows.try_into().unwrap_or(0),
                ncols.try_into().unwrap_or(0));
            Matrix { data: m.assume_init(), vec: Box::new(Vec::new()) }
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

    /// Create diagonal view over original matrix. Negative n means n'th subdiagonal and
    /// positive n meahs n'th superdiagonal. Zero n mean main diagonal.
    pub fn diagonal(&self, n: i32) -> Vector {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_diag_unsafe(m.as_mut_ptr(), &self.data, n);
            Vector { data: m.assume_init(), vec: Box::new(Vec::new()) }
        }
    }

    /// Create a row vector view of n'th row  in the original matrix.
    pub fn row(&self, n: u32) -> Vector {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_row_unsafe(m.as_mut_ptr(), &self.data, n.try_into().unwrap_or(0));
            Vector { data: m.assume_init(), vec: Box::new(Vec::new()) }
        }
    }

    /// Create a column vector view of n'th column in the original matrix.
    pub fn column(&self, n: i32) -> Vector {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_row_unsafe(m.as_mut_ptr(), &self.data, n);
            Vector { data: m.assume_init(), vec: Box::new(Vec::new()) }
        }
    }

    /// Get element at [i, j]
    pub fn get(&self, i: u32, j: u32) -> f64 {
        unsafe {
            ffi::armas_get_unsafe(&self.data, i.try_into().unwrap_or(0), j.try_into().unwrap_or(0))
        }
    }

    /// Set element at [i, j]
    pub fn set(&mut self, i: u32, j: u32, value: f64) {
        unsafe {
            ffi::armas_set_unsafe(&mut self.data, i.try_into().unwrap_or(0), j.try_into().unwrap_or(0), value);
        }
    }

    pub fn set_all(&mut self, func: extern fn() -> f64) {
        unsafe {
            ffi::armas_set_all(&mut self.data, func, 0);
        }
    }

    pub fn iter(&self) -> MatrixIterator {
        let (rows, cols) = self.size();
        MatrixIterator { source: self, index: 0, rows: rows, size: rows*cols }
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

impl<'a> Iterator for MatrixIterator<'a> {
    type Item = (u32, u32, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.size {
            return None;
        }
        let i = self.index % self.rows;
        let j = self.index / self.rows;
        self.index += 1;
        Some((i, j, self.source.get(i, j)))
    }
}

impl<'a> IntoIterator for &'a Matrix {
    type Item = (u32, u32, f64);
    type IntoIter = MatrixIterator<'a>;

    fn into_iter(self) -> MatrixIterator<'a> {
        self.iter()
    }
}

/// Serialize matrix elements thought MatrixIterator as the source
/// matrix may be submatrix view of underlying matrix. Therefore the
/// matrix elements are not necessary in sequential memory addresses.
impl<'a> Serialize for MatrixIterator<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let (rows, _) = self.source.size();
        let mut s = serializer.serialize_seq(Some(self.size as usize))?;
        for index in 0..self.size {
            let i = index % rows;
            let j = index / rows;
            let val = self.source.get(i, j);
            s.serialize_element(&val)?;
        }
        s.end()
    }
}

impl Serialize for Matrix {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let (rows, cols) = self.size();
        let mut state = serializer.serialize_struct("Matrix", 3)?;
        let iter = self.iter();
        state.serialize_field("rows", &rows)?;
        state.serialize_field("cols", &cols)?;
        state.serialize_field("data", &iter)?;
        state.end()
    }
}

impl From<MatrixShadow> for Matrix {
    fn from(m: MatrixShadow) -> Self {
        Matrix::new_from(m.rows, m.cols, m.data)
    }
}
