use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[derive(Debug)]
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

    fn predict(&self, network_input: &[f32]) -> Array1<f32> {
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
}

impl From<&[usize]> for FullyConnected {
    fn from(dims: &[usize]) -> Self {
        assert!(!dims.is_empty());

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
