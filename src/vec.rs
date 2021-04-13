use libarmasd_sys as ffi;

use std::mem;
// use std::fmt;
use std::convert::TryInto;
use serde::{Serialize, Serializer, Deserialize};
use serde::ser::{SerializeStruct, SerializeSeq};

#[derive(Deserialize)]
struct VectorShadow {
    vec: Vec<f64>
}

#[derive(Debug, Deserialize)]
#[serde(from = "VectorShadow")]
pub struct Vector {
    pub data: ffi::armas_dense,
    pub vec: Box<Vec<f64>>
}

pub struct VectorIterator<'a> {
    source: &'a Vector,
    index: u32,
    size: u32,
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
            let mut x: Vec<f64> = Vec::with_capacity(n as usize);
            x.set_len(n as usize);
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            ffi::armas_make(m.as_mut_ptr(), n as i32, 1, n as i32, x.as_mut_ptr());
            Vector { data: m.assume_init(), vec: Box::new(x) }
        }
    }

    pub fn new_from(mut v: Vec<f64>) -> Self {
        unsafe {
            let mut m = mem::MaybeUninit::<ffi::armas_dense>::zeroed();
            let n: u32 = v.len() as u32;
            ffi::armas_make(m.as_mut_ptr(), n as i32, 1, n as i32, v.as_mut_ptr());
            Vector { data: m.assume_init(), vec: Box::new(v) }
        }
    }

    pub fn uniform(n: u32) -> Vector {
        let mut v = Vector::new(n);
        unsafe {
            ffi::armas_set_all(v.as_mut_ptr(), ffi::armas_uniform, 0);
        }
        v
    }

    pub fn normal(n: u32) -> Vector {
        let mut v = Vector::new(n);
        unsafe {
            ffi::armas_set_all(v.as_mut_ptr(), ffi::armas_normal, 0);
        }
        v
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

    pub fn iter(&self) -> VectorIterator {
        VectorIterator { source: self, index: 0, size: self.size() }
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

impl<'a> Iterator for VectorIterator<'a> {
    type Item = (u32, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.size {
            return None;
        }
        let val = self.source.get(self.index);
        let i = self.index;
        self.index += 1;
        Some((i, val))
    }
}

impl<'a> IntoIterator for &'a Vector {
    type Item = (u32, f64);
    type IntoIter = VectorIterator<'a>;

    fn into_iter(self) -> VectorIterator<'a> {
        self.iter()
    }
}

/// Serialize vector elements thought VectorIterator as the source
/// vector may be vector view of underlying matrix. Therefore the
/// vector elements are not necessary in sequential memory addresses.
impl<'a> Serialize for VectorIterator<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_seq(Some(self.size as usize))?;
        for i in 0..self.size {
            let val = self.source.get(i);
            s.serialize_element(&val)?;
        }
        s.end()
    }
}

impl Serialize for Vector {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Vector", 1)?;
        let iter = self.iter();
        state.serialize_field("vec", &iter)?;
        state.end()
    }
}

impl From<VectorShadow> for Vector {
    fn from(v: VectorShadow) -> Self {
        Vector::new_from(v.vec)
    }
}
