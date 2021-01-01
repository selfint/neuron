pub struct Network {}

impl Network {
    pub fn new(dims: &[usize]) -> Self {
        Network
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weights_and_biases() {
        let network = Network::new(&[2, 3, 1]);

        assert_eq!(network.weights.len(), 3);
    }
}
