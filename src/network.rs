use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub struct FullyConnected {
    size: usize,
    weights: Option<Array2<f32>>,
    biases: Option<Array1<f32>>,
    input: Option<Box<FullyConnected>>,
}

impl FullyConnected {
    fn new(size: usize) -> Self {
        FullyConnected {
            size,
            weights: None,
            biases: None,
            input: None,
        }
    }

    fn stack(input: FullyConnected, size: usize) -> Self {
        let distribution = Uniform::new(-0.01, 0.01);
        FullyConnected {
            size,
            weights: Some(Array2::random((size, input.size), distribution)),
            biases: Some(Array1::random(size, distribution)),
            input: Some(Box::new(input)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_layers() {
        let input = FullyConnected::new(2);
        let hidden = FullyConnected::stack(input, 3);

        assert!(hidden.weights.is_some());
        assert!(hidden.biases.is_some());
        assert_eq!(hidden.weights.clone().unwrap().len(), 6);
        assert_eq!(hidden.biases.clone().unwrap().len(), 3);

        let output = FullyConnected::stack(hidden, 1);

        assert!(output.weights.is_some());
        assert!(output.biases.is_some());
        assert_eq!(output.weights.unwrap().len(), 3);
        assert_eq!(output.biases.unwrap().len(), 1);
    }
}
