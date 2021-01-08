use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub trait Layer: Clone {
    fn forward(&self, input: &Array1<f32>) -> Array1<f32>;
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
}

#[derive(Clone)]
pub struct ReLuLayer {
    size: usize,
    inputs: usize,
    weights: Array2<f32>,
    biases: Array1<f32>,
}

impl ReLuLayer {
    pub fn new(size: usize, inputs: usize) -> Self {
        let distribution = Uniform::new(-0.01, 0.01);
        ReLuLayer {
            size,
            inputs,
            weights: Array2::random((size, inputs), distribution),
            biases: Array1::random(size, distribution),
        }
    }
}

impl Layer for ReLuLayer {
    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let relu = |&x| {
            if x > 0. {
                x
            } else {
                0.
            }
        };
        (self.weights.dot(input) + &self.biases).map(relu)
    }

    fn input_size(&self) -> usize {
        self.inputs
    }

    fn output_size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_layer() {
        let layer = ReLuLayer::new(3, 2);
        let output = layer.forward(&arr1(&[1., 0.]));
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn size_test() {
        let layer = ReLuLayer::new(3, 2);
        assert_eq!(layer.input_size(), 2);
        assert_eq!(layer.output_size(), 3);
    }
}
