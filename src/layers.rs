use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[derive(Debug)]
pub struct FullyConnected {
    pub size: usize,
    pub weights: Option<Array2<f32>>,
    pub biases: Option<Array1<f32>>,
    pub input: Option<Box<FullyConnected>>,
}

impl FullyConnected {
    pub fn new(size: usize) -> Self {
        FullyConnected {
            size,
            weights: None,
            biases: None,
            input: None,
        }
    }

    pub fn stack(input: FullyConnected, size: usize) -> Self {
        let distribution = Uniform::new(-0.01, 0.01);
        FullyConnected {
            size,
            weights: Some(Array2::random((size, input.size), distribution)),
            biases: Some(Array1::random(size, distribution)),
            input: Some(Box::new(input)),
        }
    }

    pub fn predict(&self, network_input: &[f32]) -> Array1<f32> {
        if let Some(input_layer) = &self.input {
            self.weights
                .as_ref()
                .unwrap()
                .dot(&input_layer.predict(network_input))
                + self.biases.as_ref().unwrap()
        } else {
            arr1(network_input)
        }
    }

    pub fn build_stack(weights: &[Array2<f32>], biases: &[Array1<f32>]) -> Self {
        weights.iter().zip(biases.iter()).fold(
            FullyConnected::new(weights[0].shape()[1]),
            |prev_layer, (layer_weights, layer_biases)| {
                assert_eq!(layer_weights.shape()[0], layer_biases.len());
                assert_eq!(layer_weights.shape()[1], prev_layer.size);

                FullyConnected {
                    size: layer_weights.shape()[0],
                    weights: Some(layer_weights.clone()),
                    biases: Some(layer_biases.clone()),
                    input: Some(Box::new(prev_layer)),
                }
            },
        )
    }

    pub fn get_weights(&self) -> Vec<&Array2<f32>> {
        if let Some(input_layer) = &self.input {
            let mut weights = input_layer.get_weights();
            weights.push(self.weights.as_ref().unwrap());
            weights
        } else {
            vec![]
        }
    }

    pub fn get_biases(&self) -> Vec<&Array1<f32>> {
        if let Some(input_layer) = &self.input {
            let mut biases = input_layer.get_biases();
            biases.push(self.biases.as_ref().unwrap());
            biases
        } else {
            vec![]
        }
    }

    pub fn clone_weights(&self) -> Vec<Array2<f32>> {
        self.get_weights().iter().map(|&w| w.clone()).collect()
    }

    pub fn clone_biases(&self) -> Vec<Array1<f32>> {
        self.get_biases().iter().map(|&b| b.clone()).collect()
    }
}

impl From<&[usize]> for FullyConnected {
    fn from(dims: &[usize]) -> Self {
        assert!(!dims.is_empty(), "layer size must be non-empty");

        dims.iter()
            .skip(1)
            .fold(FullyConnected::new(dims[0]), |prev_layer, &layer_size| {
                FullyConnected::stack(prev_layer, layer_size)
            })
    }
}

impl From<Vec<usize>> for FullyConnected {
    fn from(dims: Vec<usize>) -> Self {
        FullyConnected::from(dims.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_weights_and_biases() {
        let distribution = Uniform::new(-0.01, 0.01);
        let weights = vec![
            Array2::random((3, 2), distribution),
            Array2::random((1, 3), distribution),
        ];
        let biases = vec![
            Array1::random(3, distribution),
            Array1::random(1, distribution),
        ];

        let network = FullyConnected::build_stack(&weights, &biases);
        for (original_weights, network_weights) in weights.iter().zip(network.get_weights()) {
            assert_eq!(original_weights, network_weights);
        }
        for (original_biases, network_biases) in biases.iter().zip(network.get_biases()) {
            assert_eq!(original_biases, network_biases);
        }
    }

    #[test]
    fn test_from_weights_and_biases() {
        let distribution = Uniform::new(-0.01, 0.01);
        let weights = vec![
            Array2::random((3, 2), distribution),
            Array2::random((1, 3), distribution),
        ];
        let biases = vec![
            Array1::random(3, distribution),
            Array1::random(1, distribution),
        ];
        let network = FullyConnected::build_stack(&weights, &biases);

        // unravel inner layers
        let hidden = network.input.unwrap();
        let input = hidden.input.unwrap();

        assert!(input.weights.is_none());
        assert!(input.biases.is_none());
        assert_eq!(input.size, 2);

        assert!(hidden.weights.is_some());
        assert!(hidden.biases.is_some());
        assert_eq!(hidden.weights.unwrap().len(), 6);
        assert_eq!(hidden.biases.unwrap().len(), 3);

        assert!(network.weights.is_some());
        assert!(network.biases.is_some());
        assert_eq!(network.weights.unwrap().len(), 3);
        assert_eq!(network.biases.unwrap().len(), 1);
    }

    #[test]
    #[should_panic(expected = "layer size must be non-empty")]
    fn test_fast_layer_stacking_edge_case_empty_dims() {
        FullyConnected::from(vec![]);
    }

    #[test]
    fn test_forward_propagation() {
        let network = FullyConnected::from(vec![2, 3, 3, 1]);
        let input = [0., 1.];
        let output = network.predict(&input);

        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_layer_stacking() {
        let input = FullyConnected::new(2);
        let hidden = FullyConnected::stack(input, 3);
        let output = FullyConnected::stack(hidden, 1);

        // unravel inner layers
        let hidden = output.input.unwrap();
        let input = hidden.input.unwrap();

        assert!(input.weights.is_none());
        assert!(input.biases.is_none());
        assert_eq!(input.size, 2);

        assert!(hidden.weights.is_some());
        assert!(hidden.biases.is_some());
        assert_eq!(hidden.weights.unwrap().len(), 6);
        assert_eq!(hidden.biases.unwrap().len(), 3);

        assert!(output.weights.is_some());
        assert!(output.biases.is_some());
        assert_eq!(output.weights.unwrap().len(), 3);
        assert_eq!(output.biases.unwrap().len(), 1);
    }

    #[test]
    fn test_fast_layer_stacking() {
        let network_output = FullyConnected::from(vec![2, 3, 1]);
        let network_hidden = network_output.input.unwrap();
        let network_input = network_hidden.input.unwrap();

        let input = FullyConnected::new(2);
        let hidden = FullyConnected::stack(input, 3);
        let output = FullyConnected::stack(hidden, 1);

        // unravel inner layers
        let hidden = output.input.unwrap();
        let input = hidden.input.unwrap();

        assert_eq!(network_input.weights, input.weights);
        assert_eq!(network_input.biases, input.biases);
        assert_eq!(network_input.size, input.size);

        assert_eq!(
            network_hidden.weights.unwrap().len(),
            hidden.weights.unwrap().len()
        );
        assert_eq!(
            network_hidden.biases.unwrap().len(),
            hidden.biases.unwrap().len()
        );

        assert_eq!(
            network_output.weights.unwrap().len(),
            output.weights.unwrap().len()
        );
        assert_eq!(
            network_output.biases.unwrap().len(),
            output.biases.unwrap().len()
        );
    }
}
