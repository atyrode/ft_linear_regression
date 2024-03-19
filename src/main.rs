extern crate ndarray;
use ndarray::Array;

fn main() {
    let a = Array::from_vec(vec![1, 2, 3, 4]);
    println!("Array: {a:?}");
    let sum = a.sum();
    println!("Sum of elements: {sum}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndarray_working() {
        let a = Array::from_vec(vec![1, 2, 3, 4]);
        assert_eq!(a.sum(), 10);
    }
}
