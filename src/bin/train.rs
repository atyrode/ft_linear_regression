use ft_linear_regression::core::train_model;
use ft_linear_regression::Error;
use ft_linear_regression::{Dataset, Weights};

fn main() -> Result<(), Box<dyn Error>> {
    let weights: Weights = Weights::from_csv()?;
    let dataset: Dataset = Dataset::from_csv()?;

    let learning_rate: f64 = 0.1;
    let iterations: u32 = 100;
    let batch: u32 = 10;

    let new_weights: Weights = train_model(&dataset, weights, learning_rate, iterations, batch)?;

    new_weights.to_csv()?;
    Ok(())
}
