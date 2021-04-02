

bitflags! {
    pub flags OpCodes: i32 {
        /// No transponse
        const Notrans = 0,
        /// Lower triangular matrix
        const Lower = 0x1,
        /// Upper triangular matrix
        const Upper = 0x2,
        /// Symmetric matrix
        const Symm = 0x4,
        /// Hermitian matrix
        const Herm = 0x8,
        /// Unit diagonal matrix
        const Unit = 0x10,
        /// Multiplication from left
        const Left = 0x20,
        /// Multiplication from right
        const Right = 0x40,
        /// Matrix operand is transposed
        /// Operand A is transposed
        const TransA = 0x80,
        /// Operand B is transposed
        const TransB = 0x100,

        // 0x200, 0x400 reserved in libarmas

        /// Multiply with Q in bidiagonal
        const MultQ = 0x800,
        /// Multiply with P in bidiagonal
        const MultP = 0x1000,
        /// Build the Q matrix in bidiagonal
        const WantQ = 0x2000,
        /// Build the P matrix in bidiagonal
        const WantP = 0x4000,
        /// Generate left eigenvectors
        const WantU = 0x8000,
        /// Generate right eigenvectors
        const WantV = 0x10000,
        /// Apply forward
        const Forward = 0x20000,
        /// Apply backward
        const Backward = 0x40000,
        // 0x80000 - 0x400000 reserved in libarmas

        /// Compute Householder for [-beta; 0]
        const HHNegative = 0x800000,
        /// Request non-negative result (householder)
        const Nonneg = 0x1000000,
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