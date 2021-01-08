use crate::layers::Layer;
use ndarray::prelude::*;

pub struct Network<L>
where
    L: Layer,
{
    layers: Vec<L>,
}

impl<L> Network<L>
where
    L: Layer,
{
    pub fn new(layers: &[L]) -> Self {
        Network {
            layers: layers.into(),
        }
    }

    pub fn predict(&self, input: &Array1<f32>) -> Array1<f32> {
        self.layers
            .iter()
            .fold(input.clone(), |prev_layer_output, layer| {
                layer.forward(&prev_layer_output)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestLayer(usize, usize);

    impl Layer for TestLayer {
        fn forward(&self, _input: &Array1<f32>) -> Array1<f32> {
            arr1(&vec![0.; self.0])
        }

        fn input_size(&self) -> usize {
            self.1
        }

        fn output_size(&self) -> usize {
            self.0
        }
    }

    #[test]
    fn test_network_predict() {
        let l1 = TestLayer(3, 2);
        let l2 = TestLayer(1, 3);

        let network = Network::new(&[l1, l2]);

        let input = [0., 1.];
        let output = network.predict(&arr1(&input));
        assert_eq!(output.len(), 1);
    }
}
