
use libarmasd_sys as ffi;

use super::{OpCodes};
use super::dense::*;
use super::pivot::*;

/// Compute QR factorization of matrix.
pub fn qrfactor(a_mat: &mut Matrix, tau: &mut Vector) -> Result<(), i32> {
    unsafe {
        match ffi::armas_qrfactor(a_mat.as_mut_ptr(), tau.as_mut_ptr(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Build the Q matrix of QR factorization.
pub fn qrbuild(a_mat: &mut Matrix, tau: &Vector, k: u32) -> Result<(), i32> {
    unsafe {
        match ffi::armas_qrbuild(a_mat.as_mut_ptr(), tau.as_ptr(), k as i32, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Multiply matrix with Q matrix of QR factorization.
pub fn qrmult(c_mat: &mut Matrix, a_mat: &Matrix, tau: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_qrmult(c_mat.as_mut_ptr(), a_mat.as_ptr(), tau.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

pub fn qrsolve(c_mat: &mut Matrix, a_mat: &Matrix, tau: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_qrsolve(c_mat.as_mut_ptr(), a_mat.as_ptr(), tau.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute LQ factorization of matrix.
pub fn lqfactor(a_mat: &mut Matrix, tau: &mut Vector) -> Result<(), i32> {
    unsafe {
        match ffi::armas_lqfactor(a_mat.as_mut_ptr(), tau.as_mut_ptr(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Build the Q matrix of LQ factorization.
pub fn lqbuild(a_mat: &mut Matrix, tau: &Vector, k: u32) -> Result<(), i32> {
    unsafe {
        match ffi::armas_lqbuild(a_mat.as_mut_ptr(), tau.as_ptr(), k as i32, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Multiply matrix with Q matrix of LQ factorization.
pub fn lqmult(c_mat: &mut Matrix, a_mat: &Matrix, tau: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_lqmult(c_mat.as_mut_ptr(), a_mat.as_ptr(), tau.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

pub fn lqsolve(c_mat: &mut Matrix, a_mat: &Matrix, tau: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_lqsolve(c_mat.as_mut_ptr(), a_mat.as_ptr(), tau.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute LDL^T factorization of symmetric matrix.
pub fn ldlfactor(a_mat: &mut Matrix, pivot: &mut Pivot, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_ldlfactor(a_mat.as_mut_ptr(), pivot.as_mut_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Solve X = A^{-1}*B with LDL^T factorized symmetric matrix A.
pub fn ldlsolve(b_mat: &mut Matrix, a_mat: &Matrix, pivot: &Pivot, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_ldlsolve(b_mat.as_mut_ptr(), a_mat.as_ptr(), pivot.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Inverse LDL^T factorized matrix.
pub fn ldlinverse(a_mat: &mut Matrix, pivot: &Pivot, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_ldlinverse(a_mat.as_mut_ptr(), pivot.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute Bunch-Kauffman factorization of symmetric matrix.
pub fn bkfactor(a_mat: &mut Matrix, pivot: &mut Pivot, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_ldlfactor(a_mat.as_mut_ptr(), pivot.as_mut_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Solve X = A^{-1}*B with LDL factorized symmetric matrix A.
pub fn bksolve(b_mat: &mut Matrix, a_mat: &Matrix, pivot: &Pivot, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_ldlsolve(b_mat.as_mut_ptr(), a_mat.as_ptr(), pivot.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute LU factorization of  matrix.
pub fn lufactor(a_mat: &mut Matrix, pivot: &mut Pivot) -> Result<(), i32> {
    unsafe {
        match ffi::armas_lufactor(a_mat.as_mut_ptr(), pivot.as_mut_ptr(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Solve X = A^{-1}*B with LU factorized  matrix A.
pub fn lusolve(b_mat: &mut Matrix, a_mat: &mut Matrix, pivot: &mut Pivot, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_lusolve(b_mat.as_mut_ptr(), a_mat.as_mut_ptr(), pivot.as_mut_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Inverse LU factorized matrix.
pub fn luinverse(a_mat: &mut Matrix, pivot: &Pivot) -> Result<(), i32> {
    unsafe {
        match ffi::armas_luinverse(a_mat.as_mut_ptr(), pivot.as_ptr(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute Cholesky factorization of  matrix.
pub fn cholfactor(a_mat: &mut Matrix, pivot: &mut Pivot, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_cholfactor(a_mat.as_mut_ptr(), pivot.as_mut_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Solve X = A^{-1}*B with LU factorized  matrix A.
pub fn cholsolve(b_mat: &mut Matrix, a_mat: &Matrix, pivot: &Pivot, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_cholsolve(b_mat.as_mut_ptr(), a_mat.as_ptr(), pivot.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute unpivoted Cholesky factorization of matrix.
pub fn cholesky(a_mat: &mut Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_cholesky(a_mat.as_mut_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Rank update unpivoted Cholesky factorization of matrix.
pub fn cholupdate(a_mat: &mut Matrix, x: &mut Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_cholupdate(a_mat.as_mut_ptr(), x.as_mut_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Inverse update unpivoted Cholesky factorized matrix.
pub fn cholinverse(a_mat: &mut Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_cholinverse(a_mat.as_mut_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute Hessenberg reduction of matrix.
pub fn hessreduce(a_mat: &mut Matrix, tau: &mut Vector) -> Result<(), i32> {
    unsafe {
        match ffi::armas_hessreduce(a_mat.as_mut_ptr(), tau.as_mut_ptr(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

pub fn hessmult(b_mat: &mut Matrix, a_mat: &Matrix, tau: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_hessmult(b_mat.as_mut_ptr(), a_mat.as_ptr(), tau.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute bidiagonal reduction A = Q*B*P^T of matrix.
pub fn bdreduce(a_mat: &mut Matrix, tauq: &mut Vector, taup: &mut Vector) -> Result<(), i32> {
    unsafe {
        match ffi::armas_bdreduce(a_mat.as_mut_ptr(), tauq.as_mut_ptr(), taup.as_mut_ptr(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

///
pub fn bdbuild(a_mat: &mut Matrix, tau: &Vector, k: u32, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_bdbuild(a_mat.as_mut_ptr(), tau.as_ptr(), k as i32, ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

///
pub fn bdmult(b_mat: &mut Matrix, a_mat: &Matrix, tau: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_bdmult(b_mat.as_mut_ptr(), a_mat.as_ptr(), tau.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute tridiagonal reduction A = Q*T*Q^T of symmetric matrix.
pub fn trdreduce(a_mat: &mut Matrix, tau: &mut Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_trdreduce(a_mat.as_mut_ptr(), tau.as_mut_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

///
pub fn trdbuild(a_mat: &mut Matrix, tau: &Vector, k: u32, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_trdbuild(a_mat.as_mut_ptr(), tau.as_ptr(), k as i32, ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

///
pub fn trdmult(b_mat: &mut Matrix, a_mat: &Matrix, tau: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_trdmult(b_mat.as_mut_ptr(), a_mat.as_ptr(), tau.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

pub fn trdeigen(d: &mut Vector, e: &mut Vector, v_mat: &mut Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_trdeigen(d.as_mut_ptr(), e.as_mut_ptr(), v_mat.as_mut_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

// Compute singular value  decomposition  B = U*S*V^T of bidiagonal matrix.
pub fn bdsvd(d: &mut Vector, e: &mut Vector, u_mat: &mut Matrix, v_mat: &mut Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_bdsvd(d.as_mut_ptr(), e.as_mut_ptr(), u_mat.as_mut_ptr(), v_mat.as_mut_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute singular value  decomposition  A = U*S*V^T of matrix.
pub fn svd(s: &mut Vector, u_mat: &mut Matrix, v_mat: &mut Matrix, a_mat: &mut Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_svd(s.as_mut_ptr(), u_mat.as_mut_ptr(), v_mat.as_mut_ptr(), a_mat.as_mut_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

pub fn eigen_sym(d: &mut Vector, e: &mut Vector, v_mat: &mut Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_trdeigen(d.as_mut_ptr(), e.as_mut_ptr(), v_mat.as_mut_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}
