use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub struct Network {}

impl Network {
    fn generate_weights(dims: &[usize]) -> Vec<Array2<f32>> {
        dims.iter()
            .zip(dims.iter().skip(1))
            .map(|(&layer, &next)| Array2::random((next, layer), Uniform::new(-0.01, 0.01)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_weights() {
        let weights: Vec<Array2<f32>> = Network::generate_weights(&[2, 3, 1]);

        assert_eq!(weights.len(), 2);

        assert_eq!(weights[0].len(), 6);
        assert_eq!(weights[1].len(), 3);
    }
}
