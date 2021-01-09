use ndarray::prelude::*;
use std::fmt::Debug;

pub trait Layer: Clone + PartialEq + Debug + Sized {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn get_weights(&self) -> &Array2<f32>;
    fn get_biases(&self) -> &Array1<f32>;
    fn get_weights_mut(&mut self) -> &mut Array2<f32>;
    fn get_biases_mut(&mut self) -> &mut Array1<f32>;
    fn from_weights_and_biases(weights: Array2<f32>, biases: Array1<f32>) -> Self;
}

pub trait FeedForwardLayer: Layer {
    fn activate(&self, input: &Array1<f32>) -> Array1<f32>;
    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        self.activate(&(self.get_weights().dot(input) + self.get_biases()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::relu::ReLuLayer;

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
