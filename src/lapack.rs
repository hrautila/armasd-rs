
// Allow non_snake_case variables to use upper case characters as identifier for Matrix type arguments.
#![allow(non_snake_case)]

use libarmasd_sys as ffi;

use super::{OpCodes};
use super::dense::{Matrix};
use super::vec::{Vector};
use super::pivot::*;

/// Compute QR factorization of matrix.
pub fn qrfactor(A: &mut Matrix, tau: &mut Vector) -> Result<(), i32> {
    unsafe {
        match ffi::armas_qrfactor(A.as_mut_ptr(), tau.as_mut_ptr(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Build the Q matrix of QR factorization.
pub fn qrbuild(A: &mut Matrix, tau: &Vector, k: u32) -> Result<(), i32> {
    unsafe {
        match ffi::armas_qrbuild(A.as_mut_ptr(), tau.as_ptr(), k as i32, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Multiply matrix with Q matrix of QR factorization.
pub fn qrmult(C: &mut Matrix, A: &Matrix, tau: &Vector, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_qrmult(C.as_mut_ptr(), A.as_ptr(), tau.as_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

pub fn qrsolve(C: &mut Matrix, A: &Matrix, tau: &Vector, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_qrsolve(C.as_mut_ptr(), A.as_ptr(), tau.as_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute LQ factorization of matrix.
pub fn lqfactor(A: &mut Matrix, tau: &mut Vector) -> Result<(), i32> {
    unsafe {
        match ffi::armas_lqfactor(A.as_mut_ptr(), tau.as_mut_ptr(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Build the Q matrix of LQ factorization.
pub fn lqbuild(A: &mut Matrix, tau: &Vector, k: u32) -> Result<(), i32> {
    unsafe {
        match ffi::armas_lqbuild(A.as_mut_ptr(), tau.as_ptr(), k as i32, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Multiply matrix with Q matrix of LQ factorization.
pub fn lqmult(C: &mut Matrix, A: &Matrix, tau: &Vector, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_lqmult(C.as_mut_ptr(), A.as_ptr(), tau.as_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

pub fn lqsolve(C: &mut Matrix, A: &Matrix, tau: &Vector, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_lqsolve(C.as_mut_ptr(), A.as_ptr(), tau.as_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute LDL^T factorization of symmetric matrix.
pub fn ldlfactor(A: &mut Matrix, pivot: &mut Pivot, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_ldlfactor(A.as_mut_ptr(), pivot.as_mut_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Solve X = A^{-1}*B with LDL^T factorized symmetric matrix A.
pub fn ldlsolve(B: &mut Matrix, A: &Matrix, pivot: &Pivot, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_ldlsolve(B.as_mut_ptr(), A.as_ptr(), pivot.as_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Inverse LDL^T factorized matrix.
pub fn ldlinverse(A: &mut Matrix, pivot: &Pivot, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_ldlinverse(A.as_mut_ptr(), pivot.as_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute Bunch-Kauffman factorization of symmetric matrix.
pub fn bkfactor(A: &mut Matrix, pivot: &mut Pivot, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_ldlfactor(A.as_mut_ptr(), pivot.as_mut_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Solve X = A^{-1}*B with LDL factorized symmetric matrix A.
pub fn bksolve(B: &mut Matrix, A: &Matrix, pivot: &Pivot, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_ldlsolve(B.as_mut_ptr(), A.as_ptr(), pivot.as_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute LU factorization of  matrix.
pub fn lufactor(A: &mut Matrix, pivot: &mut Pivot) -> Result<(), i32> {
    unsafe {
        match ffi::armas_lufactor(A.as_mut_ptr(), pivot.as_mut_ptr(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Solve X = A^{-1}*B with LU factorized  matrix A.
pub fn lusolve(B: &mut Matrix, A: &mut Matrix, pivot: &mut Pivot, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_lusolve(B.as_mut_ptr(), A.as_mut_ptr(), pivot.as_mut_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Inverse LU factorized matrix.
pub fn luinverse(A: &mut Matrix, pivot: &Pivot) -> Result<(), i32> {
    unsafe {
        match ffi::armas_luinverse(A.as_mut_ptr(), pivot.as_ptr(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute Cholesky factorization of  matrix.
pub fn cholfactor(A: &mut Matrix, pivot: &mut Pivot, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_cholfactor(A.as_mut_ptr(), pivot.as_mut_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Solve X = A^{-1}*B with LU factorized  matrix A.
pub fn cholsolve(B: &mut Matrix, A: &Matrix, pivot: &Pivot, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_cholsolve(B.as_mut_ptr(), A.as_ptr(), pivot.as_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute unpivoted Cholesky factorization of matrix.
pub fn cholesky(A: &mut Matrix, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_cholesky(A.as_mut_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Rank update unpivoted Cholesky factorization of matrix.
pub fn cholupdate(A: &mut Matrix, x: &mut Vector, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_cholupdate(A.as_mut_ptr(), x.as_mut_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Inverse update unpivoted Cholesky factorized matrix.
pub fn cholinverse(A: &mut Matrix, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_cholinverse(A.as_mut_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute Hessenberg reduction of matrix.
pub fn hessreduce(A: &mut Matrix, tau: &mut Vector) -> Result<(), i32> {
    unsafe {
        match ffi::armas_hessreduce(A.as_mut_ptr(), tau.as_mut_ptr(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

pub fn hessmult(B: &mut Matrix, A: &Matrix, tau: &Vector, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_hessmult(B.as_mut_ptr(), A.as_ptr(), tau.as_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute bidiagonal reduction A = Q*B*P^T of matrix.
pub fn bdreduce(A: &mut Matrix, tauq: &mut Vector, taup: &mut Vector) -> Result<(), i32> {
    unsafe {
        match ffi::armas_bdreduce(A.as_mut_ptr(), tauq.as_mut_ptr(), taup.as_mut_ptr(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

///
pub fn bdbuild(A: &mut Matrix, tau: &Vector, k: u32, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_bdbuild(A.as_mut_ptr(), tau.as_ptr(), k as i32, bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

///
pub fn bdmult(B: &mut Matrix, A: &Matrix, tau: &Vector, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_bdmult(B.as_mut_ptr(), A.as_ptr(), tau.as_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute tridiagonal reduction A = Q*T*Q^T of symmetric matrix.
pub fn trdreduce(A: &mut Matrix, tau: &mut Vector, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_trdreduce(A.as_mut_ptr(), tau.as_mut_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

///
pub fn trdbuild(A: &mut Matrix, tau: &Vector, k: u32, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_trdbuild(A.as_mut_ptr(), tau.as_ptr(), k as i32, bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

///
pub fn trdmult(B: &mut Matrix, A: &Matrix, tau: &Vector, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_trdmult(B.as_mut_ptr(), A.as_ptr(), tau.as_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

pub fn trdeigen(d: &mut Vector, e: &mut Vector, V: &mut Matrix, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_trdeigen(d.as_mut_ptr(), e.as_mut_ptr(), V.as_mut_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

// Compute singular value  decomposition  B = U*S*V^T of bidiagonal matrix.
pub fn bdsvd(d: &mut Vector, e: &mut Vector, U: &mut Matrix, V: &mut Matrix, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_bdsvd(d.as_mut_ptr(), e.as_mut_ptr(), U.as_mut_ptr(), V.as_mut_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute singular value  decomposition  A = U*S*V^T of matrix.
pub fn svd(s: &mut Vector, U: &mut Matrix, V: &mut Matrix, A: &mut Matrix, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_svd(s.as_mut_ptr(), U.as_mut_ptr(), V.as_mut_ptr(), A.as_mut_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

pub fn eigen_sym(d: &mut Vector, A: &mut Matrix, ops: Option<OpCodes>) -> Result<(), i32> {
    unsafe {
        let bits = ops.unwrap_or(OpCodes::NOTRANS).bits();
        match ffi::armas_eigen_sym(d.as_mut_ptr(), A.as_mut_ptr(), bits, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}
