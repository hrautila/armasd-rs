
#[cfg(test)]
use super::dense;

#[cfg(test)]
use super::blas;

#[cfg(test)]
use super::{OpCodes, Norms};

#[cfg(test)]
const M: u32 = 157;

#[cfg(test)]
const N: u32 = 137;

#[cfg(test)]
const K: u32 = 123;

#[cfg(test)]
extern fn one() -> f64 {
    1.0
}

#[test]
fn test_create() {
    let mat = dense::Matrix::new(5, 5);
    let (rows, cols) = mat.size();
    assert_eq!(rows, 5);
    assert_eq!(cols, 5);
    assert_eq!(mat.get(0, 0), 0.0);
    assert_eq!(mat.get(4, 4), 0.0);
}

#[test]
fn test_views() {
    let mut mat = dense::Matrix::new(5, 5);
    let mut d = mat.diagonal(0);
    assert_eq!(d.size(), 5);

    mat.set(3, 3, 5.0);
    assert_eq!(mat.get(3, 3), 5.0);
    assert_eq!(d.get(3), 5.0);

    let r = mat.row(3);
    assert_eq!(r.get(3), 5.0);

    let c = mat.column(3);
    assert_eq!(c.get(3), 5.0);

    d.set(3, 10.0);
    assert_eq!(mat.get(3, 3), 10.0);
}

#[test]
fn test_iter() {
    let a0 = dense::Matrix::uniform(4, 3);

    for e in &a0 {
        assert!((e.0 < 4 && e.1 < 3));
        assert!((e.2  >= 0.0 && e.2 < 1.0));
    }
}

#[test]
fn test_set_all() {
    let mut m = dense::Matrix::new(5, 4);
    m.set_all(one);
    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(4, 3), 1.0);
}

#[test]
fn test_blas_mult() {
    let a0 = dense::Matrix::uniform(M, K);
    let a1 = dense::Matrix::uniform(K, N);
    let mut r0 = dense::Matrix::new(M, N);
    let mut r1 = dense::Matrix::new(N, M);

    blas::mult(0.0, &mut r0, 1.0, &a0, &a1, OpCodes::NOTRANS).unwrap();
    blas::mult(0.0, &mut r1, 1.0, &a1, &a0, OpCodes::TRANSA|OpCodes::TRANSB).unwrap();

    // compute r1 = r1 - r0^T = B^T*A^T - (A*B)^T ; expect |r1|_inf/|r0|_inf < epsilon
    blas::mplus(1.0, &mut r1, -1.0, &r0, OpCodes::TRANSB).unwrap();
    let n0 = blas::mnorm(&r0, Norms::Infinity).unwrap();
    let n1 = blas::mnorm(&r1, Norms::Infinity).unwrap();
    assert!((n1/n0 < 2e-16));
    // println!("n0: {}, n1: {}, n1/n0: {}", n0, n1, n1/n0);
}
