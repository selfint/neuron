use crate::layers::FeedForwardLayer;
use ndarray::prelude::*;

#[derive(Debug, PartialEq, Clone)]
pub struct FeedForwardNetwork<L>
where
    L: FeedForwardLayer,
{
    layers: Vec<L>,
}

trait Predict {
    fn predict(&self, input: &Array1<f32>) -> Array1<f32>;
}

impl<L> FeedForwardNetwork<L>
where
    L: FeedForwardLayer,
{
    pub fn new(layers: &[L]) -> Self {
        FeedForwardNetwork {
            layers: layers.into(),
        }
    }
}

impl<L> Predict for FeedForwardNetwork<L>
where
    L: FeedForwardLayer,
{
    fn predict(&self, input: &Array1<f32>) -> Array1<f32> {
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
    use crate::layers::ReLuLayer;

    #[test]
    fn test_network_predict() {
        let l1 = ReLuLayer::new(3, 2);
        let l2 = ReLuLayer::new(1, 3);

        let network = FeedForwardNetwork::new(&[l1, l2]);

        let input = [0., 1.];
        let output = network.predict(&arr1(&input));
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_network_is_cloneable() {
        let l1 = ReLuLayer::new(3, 2);
        let l2 = ReLuLayer::new(1, 3);

        let network = FeedForwardNetwork::new(&[l1, l2]);
        let network2 = network.clone();

        let input = arr1(&[0., 1.]);
        assert_eq!(network.predict(&input), network2.predict(&input));
    }
}
