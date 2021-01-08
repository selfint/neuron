use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub trait Layer: Clone + PartialEq {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn get_weights(&self) -> &Array2<f32>;
    fn get_biases(&self) -> &Array1<f32>;
    fn get_weights_mut(&mut self) -> &mut Array2<f32>;
    fn get_biases_mut(&mut self) -> &mut Array1<f32>;
    fn from_weights_and_biases(weights: Array2<f32>, biases: Array1<f32>) -> Self;
}

pub trait FeedForwardLayer: Layer + Clone + PartialEq {
    fn forward(&self, input: &Array1<f32>) -> Array1<f32>;
}

#[derive(Clone, PartialEq)]
pub struct ReLuLayer {
    output_size: usize,
    input_size: usize,
    weights: Array2<f32>,
    biases: Array1<f32>,
}

impl ReLuLayer {
    pub fn new(output_size: usize, input_size: usize) -> Self {
        let distribution = Uniform::new(-0.01, 0.01);
        ReLuLayer {
            output_size,
            input_size,
            weights: Array2::random((output_size, input_size), distribution),
            biases: Array1::random(output_size, distribution),
        }
    }
}

impl Layer for ReLuLayer {
    fn input_size(&self) -> usize {
        self.input_size
    }

    fn output_size(&self) -> usize {
        self.output_size
    }

    fn get_weights(&self) -> &Array2<f32> {
        &self.weights
    }

    fn get_biases(&self) -> &Array1<f32> {
        &self.biases
    }

    fn get_weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.weights
    }

    fn get_biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.biases
    }

    fn from_weights_and_biases(weights: Array2<f32>, biases: Array1<f32>) -> Self {
        let (input_size, output_size) = (weights.shape()[1], weights.shape()[0]);
        Self {
            input_size,
            output_size,
            weights,
            biases,
        }
    }
}

impl FeedForwardLayer for ReLuLayer {
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

    #[test]
    fn test_layer_from_weights_and_biases() {
        let weights = arr2(&[[1., 0.], [0., 1.]]);
        let biases = arr1(&[1., 0.]);

        let layer = ReLuLayer::from_weights_and_biases(weights.clone(), biases.clone());

        assert_eq!(layer.get_weights(), &weights);
        assert_eq!(layer.get_biases(), &biases);
    }
}
