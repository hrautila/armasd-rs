#[macro_use]
extern crate bitflags;

use libarmasd_sys as ffi;

bitflags! {
    pub struct OpCodes: i32 {
        /// No transponse
        const NOTRANS = ffi::ARMAS_NOTRANS;
        /// Lower triangular matrix
        const LOWER = ffi::ARMAS_LOWER;
        /// Upper triangular matrix
        const UPPER = ffi::ARMAS_UPPER;
        /// Symmetric matrix
        const SYMM = ffi::ARMAS_SYMM;
        /// Unit diagonal matrix
        const UNIT = ffi::ARMAS_UNIT;
        /// Multiplication from left
        const LEFT = ffi::ARMAS_LEFT;
        /// Multiplication from right
        const RIGHT = ffi::ARMAS_RIGHT;
        /// Matrix operand is transposed
        const TRANS = ffi::ARMAS_TRANS;
        /// Operand A is transposed
        const TRANSA = ffi::ARMAS_TRANSA;
        /// Operand B is transposed
        const TRANSB = ffi::ARMAS_TRANSB;

        // 0x200, 0x400 reserved in libarmas

        /// Multiply with Q in bidiagonal
        const MULTQ = ffi::ARMAS_MULTQ;
        /// Multiply with P in bidiagonal
        const MULTP = ffi::ARMAS_MULTP;
        /// Build the Q matrix in bidiagonal
        const WANTQ = ffi::ARMAS_WANTQ;
        /// Build the P matrix in bidiagonal
        const WANTP = ffi::ARMAS_WANTP;
        /// Generate left eigenvectors
        const WANTU = ffi::ARMAS_WANTU;
        /// Generate right eigenvectors
        const WANTV = ffi::ARMAS_WANTV;
        /// Apply forward
        const FORWARD = ffi::ARMAS_FORWARD;
        /// Apply backward
        const BACKWARD = ffi::ARMAS_BACKWARD;
        // 0x80000 - 0x400000 reserved in libarmas

        /// Compute Householder for [-beta; 0]
        const HHNEGATIVE = ffi::ARMAS_HHNEGATIVE;
        /// Request non-negative result (householder)
        const NONNEG = ffi::ARMAS_NONNEG;
    }
}

pub enum CopyOps {
    All = 0,
    Lower = 0x1,
    Upper = 0x2,
    Symm = 0x4,
    Unit = 0x10,
}

pub enum Norms {
    One = 1,
    Two = 2,
    Infinity = 3,
    Frobenius = 4
}

pub enum PivotOps {
    /// Pivot forwards
    Forward = 0x0,
    /// Pivot backwards
    Backward = 0x1,
    /// Pivot rows
    Rows = 0x2,
    /// Pivot columns
    Columns = 0x4,
    /// Pivot upper triangular symmetric matrix
    Upper = 0x8,
    /// Pivot lower triangular symmetric matrix
    Lower = 0x10,
}

pub enum Error {
    ENone = 0,
    /// Operand size mismatch
    ESize = 1,
    /// Vector operand required
    ENeedVector = 2,
    /// Invalid parameter
    EInval = 3,
    /// Not implemented
    EImp = 4,
    /// Workspace too small
    EWork = 5,
    /// Singular matrix
    ESingular = 6,
    /// Negative value on diagonal
    ENegative = 7,
    /// Memory allocation failed
    EMemory = 8,
    /// Algorithm does not converge
    EConverge = 9,
    /// Svd factorization failed
    ESvdFact = 10,
    /// Svd left eigenvector error
    ESvdLeft = 11,
    /// Svd right eigenvector error
    ESvdRight = 12,
    /// Svd bidiagonal eigenvalue error
    ESvdEigen = 13,
}

pub mod vec;
pub mod dense;
pub mod pivot;
pub mod blas;
pub mod lapack;

mod tests;

