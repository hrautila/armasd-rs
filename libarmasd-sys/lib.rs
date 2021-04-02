
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use ::std::os::raw::{c_int, c_uint, c_void, c_double};

pub const ARMAS_NOTRANS: c_int = 0;
pub const ARMAS_NONE: c_int = 0;
pub const ARMAS_ALL: c_int = 0;
pub const ARMAS_LOWER: c_int = 1;
pub const ARMAS_UPPER: c_int = 2;
pub const ARMAS_SYMM: c_int = 4;
pub const ARMAS_HERM: c_int = 8;
pub const ARMAS_UNIT: c_int = 16;
pub const ARMAS_LEFT: c_int = 32;
pub const ARMAS_RIGHT: c_int = 64;
pub const ARMAS_TRANSA: c_int = 128;
pub const ARMAS_TRANSB: c_int = 256;
pub const ARMAS_TRANS: c_int = 128;
pub const ARMAS_CTRANSA: c_int = 512;
pub const ARMAS_CTRANSB: c_int = 1024;
pub const ARMAS_CTRANS: c_int = 512;
pub const ARMAS_MULTQ: c_int = 2048;
pub const ARMAS_MULTP: c_int = 4096;
pub const ARMAS_WANTQ: c_int = 8192;
pub const ARMAS_WANTP: c_int = 16384;
pub const ARMAS_WANTU: c_int = 32768;
pub const ARMAS_WANTV: c_int = 65536;
pub const ARMAS_FORWARD: c_int = 131072;
pub const ARMAS_BACKWARD: c_int = 262144;
pub const ARMAS_ABSA: c_int = 524288;
pub const ARMAS_ABSB: c_int = 1048576;
pub const ARMAS_ABS: c_int = 524288;
pub const ARMAS_CONJA: c_int = 2097152;
pub const ARMAS_CONJB: c_int = 4194304;
pub const ARMAS_CONJ: c_int = 2097152;
pub const ARMAS_HHNEGATIVE: c_int = 8388608;
pub const ARMAS_NONNEG: c_int = 16777216;


pub const ARMAS_ASC: c_int = 1;
pub const ARMAS_DESC: c_int = -1;

pub const ARMAS_PIVOT_FORWARD: c_uint = 0;
pub const ARMAS_PIVOT_BACKWARD: c_uint = 1;
pub const ARMAS_PIVOT_ROWS: c_uint = 2;
pub const ARMAS_PIVOT_COLS: c_uint = 4;
pub const ARMAS_PIVOT_UPPER: c_uint = 8;
pub const ARMAS_PIVOT_LOWER: c_uint = 16;

pub const ARMAS_ONAIVE: c_uint = 1;
pub const ARMAS_OKAHAN: c_uint = 2;
pub const ARMAS_OPAIRWISE: c_uint = 4;
pub const ARMAS_ORECURSIVE: c_uint = 8;
pub const ARMAS_OBLAS_RECURSIVE: c_uint = 16;
pub const ARMAS_OBLAS_BLOCKED: c_uint = 32;
pub const ARMAS_OBLAS_TILED: c_uint = 64;
pub const ARMAS_OSCHED_ROUNDROBIN: c_uint = 128;
pub const ARMAS_OSCHED_RANDOM: c_uint = 256;
pub const ARMAS_OSCHED_TWO: c_uint = 512;
pub const ARMAS_OBSVD_GOLUB: c_uint = 1024;
pub const ARMAS_OBSVD_DEMMEL: c_uint = 2048;
pub const ARMAS_OABSTOL: c_uint = 4096;
pub const ARMAS_OEXTPREC: c_uint = 8192;
pub const ARMAS_ONONNEG: c_uint = 16384;
pub const ARMAS_CBUF_THREAD: c_uint = 32768;
pub const ARMAS_CBUF_LOCAL: c_uint = 65536;

pub const ARMAS_ESIZE: c_uint = 1;
pub const ARMAS_ENEED_VECTOR: c_uint = 2;
pub const ARMAS_EINVAL: c_uint = 3;
pub const ARMAS_EIMP: c_uint = 4;
pub const ARMAS_EWORK: c_uint = 5;
pub const ARMAS_ESINGULAR: c_uint = 6;
pub const ARMAS_ENEGATIVE: c_uint = 7;
pub const ARMAS_EMEMORY: c_uint = 8;
pub const ARMAS_ECONVERGE: c_uint = 9;
pub const ARMAS_ESVD_FACT: c_uint = 10;
pub const ARMAS_ESVD_LEFT: c_uint = 11;
pub const ARMAS_ESVD_RIGHT: c_uint = 12;
pub const ARMAS_ESVD_EIGEN: c_uint = 13;

pub const ARMAS_NORM_ONE: c_uint = 1;
pub const ARMAS_NORM_TWO: c_uint = 2;
pub const ARMAS_NORM_INF: c_uint = 3;
pub const ARMAS_NORM_FRB: c_uint = 4;

#[repr(C)]
#[derive(Debug)]
pub struct armas_dense {
    pub elems: *mut f64,
    pub step: c_int,
    pub rows: c_int,
    pub cols: c_int,
    __data: *mut c_void,
    __nbytes: c_int,
}

#[repr(C)]
#[derive(Debug)]
pub struct armas_pivot {
    pub npivots: c_int,       /// Pivot storage size
    pub indexes: *mut c_int,  /// Pivot storage
    owner: c_int
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct armas_conf {
    pub error: c_int,
    pub optflags: c_int,
    pub tolmult: c_int,
    pub work: *const c_void,
    pub accel: *const c_void,
    pub maxiter: c_int,
    pub gmres_m: c_int,
    pub numiters: c_int,
    pub stop: f64,
    pub smult: f64,
    pub residual: f64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct armas_eigen_parameter {
    pub ileft: c_int,
    pub iright: c_int,
    pub left: f64,
    pub right: f64,
    pub tau: f64,
}

pub type armas_generator = unsafe extern "C" fn() -> f64;
pub type armas_valuefunc = unsafe extern "C" fn(r: c_int, c: c_int) -> f64;
pub type armas_operator = unsafe extern "C" fn(x: f64) -> f64;
pub type armas_operator2 = unsafe extern "C" fn(x: f64, y: *mut c_void) -> f64;
pub type armas_iterator = unsafe extern "C" fn(x: f64, p: *mut c_void) -> c_int;

#[link(name = "armasd")]
extern "C" {
    pub fn armas_pivot_init(p: *mut armas_pivot, n: c_int) -> *mut armas_pivot;

    pub fn armas_pivot_release(p: *mut armas_pivot);

    pub fn armas_pivot_get_unsafe(p: *mut armas_pivot, k: c_int) -> c_int;

    pub fn armas_pivot_set_unsafe(p: *mut armas_pivot, k: c_int, v: c_int);

    pub fn armas_conf_default() -> *mut armas_conf;

    pub fn armas_init(m: *mut armas_dense, r: c_int, c: c_int) -> *mut armas_dense;

    pub fn armas_make(m: *mut armas_dense, r: c_int, c: c_int, s: c_int, buf: *mut f64) -> *mut armas_dense;

    pub fn armas_alloc(r: c_int, c: c_int) -> *mut armas_dense;

    pub fn armas_release(m: *mut armas_dense);

    pub fn armas_free(m: *mut armas_dense);

    pub fn armas_set_unsafe(m: *mut armas_dense, r: c_int, c: c_int, v: c_double);

    pub fn armas_set_at_unsafe(m: *mut armas_dense, k: c_int, v: c_double);

    pub fn armas_get_unsafe(m: *const armas_dense, r: c_int, c: c_int) ->  c_double;

    pub fn armas_get_at_unsafe(m: *const armas_dense, k: c_int) ->  c_double;

    pub fn armas_row_unsafe(A: *mut armas_dense, B: *const armas_dense, r: c_int) ->  *mut armas_dense;

    pub fn armas_column_unsafe(A: *mut armas_dense, B: *const armas_dense, c: c_int) ->  *mut armas_dense;

    pub fn armas_diag_unsafe(A: *mut armas_dense, B: *const armas_dense, k: c_int) ->  *mut armas_dense;

    pub fn armas_submatrix_unsafe(
        A: *mut armas_dense, B: *const armas_dense, r: c_int,
        c: c_int, nr: c_int, nc: c_int) ->  *mut armas_dense;

    pub fn armas_mcopy(A: *mut armas_dense, B: *const armas_dense, flags: i32);

    pub fn armas_set_all(m: *mut armas_dense, func: armas_generator, flags: c_int) -> c_int;

    pub fn armas_set_values(m: *mut armas_dense, func: armas_valuefunc, flags: c_int) -> c_int;

    pub fn armas_make_trm(m: *mut armas_dense, flags: c_int);

    pub fn armas_madd(d: *mut armas_dense, alpha: f64, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_mscale(d: *mut armas_dense, alpha: f64, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_scale(d: *mut armas_dense, alpha: f64, cf: *mut armas_conf) -> c_int;

    pub fn armas_mplus(
        alpha: f64, A: *mut armas_dense, beta: f64, B: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_apply(A: *mut armas_dense, func: armas_operator, flags: c_int) -> c_int;

    pub fn armas_apply2(A: *mut armas_dense, func: armas_operator2, arg: *mut c_void, flags: c_int) -> c_int;

    pub fn armas_iterate(A: *const armas_dense, func: armas_iterator, p: *mut c_void, flags: c_int) -> c_int;

    pub fn armas_normal() -> f64;
    pub fn armas_uniform() -> f64;

    pub fn armas_mnorm(A: *const armas_dense, norm: c_int, cf: *mut armas_conf) -> f64;

    // Blas 1
    pub fn armas_iamax(X: *const armas_dense, cf: *mut armas_conf) -> c_int;

    pub fn armas_amax(X: *const armas_dense, cf: *mut armas_conf) -> f64;

    pub fn armas_asum(X: *const armas_dense, cf: *mut armas_conf) -> f64;

    pub fn armas_nrm2(X: *const armas_dense, cf: *mut armas_conf) -> f64;

    pub fn armas_dot(X: *const armas_dense, Y: *const armas_dense, cf: *mut armas_conf) -> f64;

    pub fn armas_adot(result: *mut f64, alpha: f64, X: *const armas_dense, Y: *const armas_dense, cf: *mut armas_conf) -> c_int;

    pub fn armas_axpy(Y: *mut armas_dense, alpha: f64, X: *const armas_dense, cf: *mut armas_conf) -> c_int;

    pub fn armas_axpby(beta: f64, Y: *mut armas_dense, alpha: f64, X: *const armas_dense, cf: *mut armas_conf) -> c_int;

    pub fn armas_copy(Y: *mut armas_dense, X: *const armas_dense, cf: *mut armas_conf) -> c_int;

    pub fn armas_swap(Y: *mut armas_dense, X: *mut armas_dense, cf: *mut armas_conf) -> c_int;

    pub fn armas_sum(X: *const armas_dense, cf: *mut armas_conf) -> f64;

    // BLAS2
    pub fn armas_mvmult(
        beta: f64, Y: *mut armas_dense, alpha: f64, A: *const armas_dense, X: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_mvupdate(
        beta: f64, A: *mut armas_dense, alpha: f64, X: *const armas_dense, Y: *const armas_dense, cf: *mut armas_conf) -> c_int;

    pub fn armas_mvmult_sym(
        beta: f64, Y: *mut armas_dense, alpha: f64, A: *const armas_dense, X: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_mvupdate2_sym(
        beta: f64, A: *mut armas_dense, alpha: f64, X: *const armas_dense, Y: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_mvupdate_sym(
        beta: f64, A: *mut armas_dense, alpha: f64, X: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_mvupdate_trm(
        beta: f64, A: *mut armas_dense, alpha: f64, X: *const armas_dense, Y: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_mvmult_trm(
        X: *mut armas_dense, alpha: f64, A: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_mvsolve_trm(
        X: *mut armas_dense, alpha: f64, A: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    // Blas3
    pub fn armas_mult(
        beta: f64, C: *mut armas_dense, alpha: f64, A: *const armas_dense, B: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_mult_sym(
        beta: f64, C: *mut armas_dense, alpha: f64, A: *const armas_dense, B: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_mult_trm(
        B: *mut armas_dense, alpha: f64, A: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_solve_trm(
        B: *mut armas_dense, alpha: f64, A: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_update_trm(
        beta: f64, C: *mut armas_dense, alpha: f64, A: *const armas_dense, B: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_update_sym(
        beta: f64, C: *mut armas_dense, alpha: f64, A: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_update2_sym(
        beta: f64, C: *mut armas_dense, alpha: f64, A: *const armas_dense, B: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_mult_diag(
        B: *mut armas_dense, alpha: f64, A: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_solve_diag(
        B: *mut armas_dense, alpha: f64, A: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_qrbuild(
        A: *mut armas_dense, tau: *const armas_dense, K: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_qrfactor(
        A: *mut armas_dense, tau: *mut armas_dense, cf: *mut armas_conf) -> c_int;

    pub fn armas_qrmult(
        C: *mut armas_dense, A: *const armas_dense, tau: *const armas_dense, flags: c_int,cf: *mut armas_conf) -> c_int;

    pub fn armas_qrsolve(
        B: *mut armas_dense, A: *const armas_dense, tau: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_lqbuild(
        A: *mut armas_dense, tau: *const armas_dense, K: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_lqfactor(
        A: *mut armas_dense, tau: *mut armas_dense, cf: *mut armas_conf) -> c_int;

    pub fn armas_lqmult(
        C: *mut armas_dense, A: *const armas_dense, tau: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_lqsolve(
        B: *mut armas_dense, A: *const armas_dense, tau: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_ldlfactor(
        A: *mut armas_dense, P: *mut armas_pivot, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_ldlsolve(
        B: *mut armas_dense, A: *const armas_dense, P: *const armas_pivot, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_bkfactor(
        A: *mut armas_dense, P: *mut armas_pivot, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_bksolve(
        B: *mut armas_dense, A: *const armas_dense, P: *const armas_pivot, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_lufactor(
        A: *mut armas_dense, P: *mut armas_pivot, cf: *mut armas_conf) -> c_int;

    pub fn armas_lusolve(
        B: *mut armas_dense, A: *mut armas_dense, P: *mut armas_pivot, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_hessreduce(
        A: *mut armas_dense, tau: *mut armas_dense, cf: *mut armas_conf) -> c_int;

    pub fn armas_hessmult(
        B: *mut armas_dense, A: *const armas_dense, tau: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_cholesky(
        A: *mut armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_cholfactor(
        A: *mut armas_dense, P: *mut armas_pivot, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_cholsolve(
        B: *mut armas_dense, A: *const armas_dense, P: *const armas_pivot, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_cholupdate(
        A: *mut armas_dense, X: *mut armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_bdreduce(
        A: *mut armas_dense, tauq: *mut armas_dense, taup: *mut armas_dense, cf: *mut armas_conf) -> c_int;

    pub fn armas_bdbuild(
        A: *mut armas_dense, tau: *const armas_dense, K: c_int, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_bdmult(
        B: *mut armas_dense, A: *const armas_dense, tau: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_trdreduce(
        A: *mut armas_dense, tau: *mut armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_trdbuild(
        A: *mut armas_dense, tau: *const armas_dense, K: c_int, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_trdmult(
        B: *mut armas_dense, A: *const armas_dense, tau: *const armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_trdeigen(
        D: *mut armas_dense, E: *mut armas_dense, V: *mut armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_trdbisect(
        Y: *mut armas_dense, D: *mut armas_dense, E: *mut armas_dense, params: *const armas_eigen_parameter, cf: *mut armas_conf) -> c_int;

    pub fn armas_gvcompute(c: *mut f64, s: *mut f64, r: *mut f64, a: f64, b: f64);

    pub fn armas_gvrotate(v0: *mut f64, v1: *mut f64, c: f64, s: f64, y0: f64, y1: f64);

    pub fn armas_gvleft(
        A: *mut armas_dense, c: f64, s: f64, r1: c_int, r2: c_int, col: c_int, ncol: c_int);

    pub fn armas_gvright(
        A: *mut armas_dense, c: f64, s: f64, r1: c_int, r2: c_int, col: c_int, ncol: c_int);

    pub fn armas_gvupdate(
        A: *mut armas_dense, start: c_int, C: *mut armas_dense, S: *mut armas_dense, nrot: c_int, flags: c_int) -> c_int;

    pub fn armas_gvrot_vec(
        X: *mut armas_dense, Y: *mut armas_dense, c: f64, s: f64) -> c_int;

    pub fn armas_bdsvd(
        D: *mut armas_dense, E: *mut armas_dense, U: *mut armas_dense, V: *mut armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_svd(
        S: *mut armas_dense, U: *mut armas_dense, V: *mut armas_dense, A: *mut armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_eigen_sym(
        D: *mut armas_dense, A: *mut armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_eigen_sym_selected(
        D: *mut armas_dense, A: *mut armas_dense, params: *const armas_eigen_parameter, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_luinverse(
        A: *mut armas_dense, P: *const armas_pivot, cf: *mut armas_conf) -> c_int;

    pub fn armas_cholinverse(
        A: *mut armas_dense, flags: c_int, cf: *mut armas_conf) -> c_int;

    pub fn armas_ldlinverse(
        A: *mut armas_dense, P: *const armas_pivot, flags: c_int, cf: *mut armas_conf) -> c_int;

    }
