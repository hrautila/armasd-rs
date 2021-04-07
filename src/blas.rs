
use libarmasd_sys as ffi;
use super::{OpCodes, Norms};
use super::dense::{Matrix};
use super::vec::{Vector};

/// Scale vector, x = alpha * x
pub fn scale(x: &mut Vector, alpha: f64) -> Result<&mut Vector, i32> {
    unsafe {
        match ffi::armas_scale(x.as_mut_ptr(), alpha, ffi::armas_conf_default()) {
            0 => Ok(x),
            x => Err(-x)
        }
    }
}

/// Scale matrix, A = alpha * A
pub fn mscale(a_mat: &mut Matrix, alpha: f64, ops: OpCodes) -> Result<&mut Matrix, i32> {
    unsafe {
        match ffi::armas_mscale(a_mat.as_mut_ptr(), alpha, ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(a_mat),
            x => Err(-x)
        }
    }
}

/// Add constant to matrix, A = A + alpha
pub fn madd(a_mat: &mut Matrix, alpha: f64, ops: OpCodes) -> Result<&mut Matrix, i32> {
    unsafe {
        match ffi::armas_madd(a_mat.as_mut_ptr(), alpha, ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(a_mat),
            x => Err(-x)
        }
    }
}

/// Element wise add of matrices, A = alpha*A + beta*B
pub fn mplus<'a, 'b>(alpha: f64, a_mat: &'a mut Matrix, beta: f64, b_mat: &'b Matrix, ops: OpCodes) -> Result<&'a mut Matrix, i32> {
    unsafe {
        match ffi::armas_mplus(alpha, a_mat.as_mut_ptr(), beta, b_mat.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(a_mat),
            x => Err(-x)
        }
    }
}

/// Compute inner product of two vectors.
pub fn dot(x: &Vector, y: &Vector) -> f64 {
    unsafe {
        ffi::armas_dot(&x.data, &y.data, ffi::armas_conf_default())
    }
}

/// Compute  result = initial + alpha*x^T*y
pub fn adot(initial: f64, alpha: f64, x: &Vector, y: &Vector) -> Result<f64, i32> {
    let mut value: f64 = initial;
    unsafe {
        match ffi::armas_adot(&mut value, alpha, &x.data, &y.data, ffi::armas_conf_default()) {
            0 => Ok(value),
            x => Err(-x)
        }
    }
}

/// Compute Euclidean norm of vector.
pub fn norm2(x: &Vector) -> Result<f64, i32> {
    unsafe {
        let cf: *mut ffi::armas_conf = ffi::armas_conf_default();
        let result = ffi::armas_nrm2(&x.data, cf);
        Ok(result)
    }
}

/// Compute sum(|a_i|)
pub fn asum(x: &Vector) -> Result<f64, i32> {
    unsafe {
        let cf: *mut ffi::armas_conf = ffi::armas_conf_default();
        let result = ffi::armas_asum(&x.data, cf);
        Ok(result)
    }
}

/// Index of absolute maximum value
pub fn iamax(x: &Vector) -> Result<u32, i32> {
    unsafe {
        let index = ffi::armas_iamax(&x.data, ffi::armas_conf_default());
        if index < 0 {
            return Err(-index);
        }
        Ok(index as u32)
    }
}

/// Compute y = beta * y + alpha * x
pub fn axpby(beta: f64, y: &mut Vector, alpha: f64, x: &Vector) -> Result<(), i32> {
    unsafe {
        match ffi::armas_axpby(beta, &mut y.data, alpha,  &x.data, ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute y = alpha*y + beta*A*x
pub fn mvmult(alpha: f64, y: &mut Vector, beta: f64, a_mat: &Matrix, x: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_mvmult(alpha, y.as_mut_ptr(), beta, a_mat.as_ptr(), x.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute y = alpha*y + beta*A*x where A holds either lower or upper triangular part of symmetric matrix A.
pub fn mvmult_sym(alpha: f64, y: &mut Vector, beta: f64, a_mat: &Matrix, x: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_mvmult_sym(alpha, y.as_mut_ptr(), beta, a_mat.as_ptr(), x.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute rank update of matrix, A = alpha*A + beta*x*y^T
pub fn mvupdate(alpha: f64, a_mat: &mut Matrix, beta: f64, x: &Vector, y: &Vector) -> Result<(), i32> {
    unsafe {
        match ffi::armas_mvupdate(alpha, a_mat.as_mut_ptr(), beta, x.as_ptr(),  y.as_ptr(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute rank update of symmetric matrix, A = alpha*A + beta*x*x^T
pub fn mvupdate_sym(alpha: f64, a_mat: &mut Matrix, beta: f64, x: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_mvupdate_sym(alpha, a_mat.as_mut_ptr(), beta, x.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute rank-2  update of symmetric matrix, A = alpha*A + beta*x*x^T
pub fn mvupdate2_sym(alpha: f64, a_mat: &mut Matrix, beta: f64, x: &Vector, y: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_mvupdate2_sym(alpha, a_mat.as_mut_ptr(), beta, x.as_ptr(), y.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute rank-2  update of triangular matrix, A = alpha*A + beta*x*y^T
pub fn mvupdate_trm(alpha: f64, a_mat: &mut Matrix, beta: f64, x: &Vector, y: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_mvupdate_trm(alpha, a_mat.as_mut_ptr(), beta, x.as_ptr(), y.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute x = alpha*A*x or x = alpha*A^T*x, where A is lower (upper) triangular matrix.
pub fn mvmult_trm(x: &mut Vector, alpha: f64, a_mat: &Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_mvmult_trm(x.as_mut_ptr(), alpha, a_mat.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute x = alpha*A^{-1}*x or x = alpha*A^{-T}*x, where A is lower (upper) triangular matrix.
pub fn mvsolve_trm(x: &mut Vector, alpha: f64, a_mat: &Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_mvmult_trm(x.as_mut_ptr(), alpha, a_mat.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute norm of a matrix.
pub fn mnorm(a_mat: &Matrix, ops: Norms) -> Result<f64, i32> {
    unsafe {
        let mut cf = * ffi::armas_conf_default();
        cf.error = 0;
        let res: f64 = ffi::armas_mnorm(a_mat.as_ptr(), ops as i32, &mut cf);
        match cf.error {
            0 => Ok(res),
            x => Err(x)
        }
    }
}

/// Compute C = alpha*C + beta*A*B
pub fn mult(alpha: f64, c_mat: &mut Matrix, beta: f64, a_mat: &Matrix, b_mat: &Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_mult(alpha, c_mat.as_mut_ptr(), beta, a_mat.as_ptr(), b_mat.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute C = alpha*C + beta*A*B, where A is symmetic matrix with lower (upper) triangular part set.
pub fn mult_sym(alpha: f64, c_mat: &mut Matrix, beta: f64, a_mat: &Matrix, b_mat: &Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_mult_sym(alpha, c_mat.as_mut_ptr(), beta, a_mat.as_ptr(), b_mat.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute B = alpha*A*B or B = alpha*B*A where A is lower (upper) triangular matrix.
pub fn mult_trm(b_mat: &mut Matrix, alpha: f64, a_mat: &Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_mult_trm(b_mat.as_mut_ptr(), alpha, a_mat.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute B = alpha*A^{-1}*B or B = alpha*B*A^{-1} where A is lower (upper) triangular matrix.
pub fn solve_trm(b_mat: &mut Matrix, alpha: f64, a_mat: &Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_solve_trm(b_mat.as_mut_ptr(), alpha, a_mat.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute C = alpha*C + beta*A*B  where C is lower (upper) tridiagonal matrix
pub fn update_trm(alpha: f64, c_mat: &mut Matrix, beta: f64, a_mat: &Matrix, b_mat: &Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_update_trm(alpha, c_mat.as_mut_ptr(), beta, a_mat.as_ptr(), b_mat.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute C = alpha*C + beta*A*A^T  where C is lower (upper) tridiagonal matrix
pub fn update_sym(alpha: f64, c_mat: &mut Matrix, beta: f64, a_mat: &Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_update_sym(alpha, c_mat.as_mut_ptr(), beta, a_mat.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute C = alpha*C + beta*A*B  where C is lower (upper) tridiagonal matrix
pub fn update2_sym(alpha: f64, c_mat: &mut Matrix, beta: f64, a_mat: &Matrix, b_mat: &Matrix, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_update2_sym(alpha, c_mat.as_mut_ptr(), beta, a_mat.as_ptr(), b_mat.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute B = alpha*diag(x)*B or B = alpha*B*diag(x)
pub fn mult_diag(b_mat: &mut Matrix, alpha: f64, x: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_mult_diag(b_mat.as_mut_ptr(), alpha, x.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}

/// Compute B = alpha*A^{-1}*diag(x) or B = alpha*diag(x)*A^{-1}
pub fn solve_diag(b_mat: &mut Matrix, alpha: f64, x: &Vector, ops: OpCodes) -> Result<(), i32> {
    unsafe {
        match ffi::armas_solve_diag(b_mat.as_mut_ptr(), alpha, x.as_ptr(), ops.bits(), ffi::armas_conf_default()) {
            0 => Ok(()),
            x => Err(-x)
        }
    }
}
