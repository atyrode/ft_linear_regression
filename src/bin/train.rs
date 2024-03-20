use std::error::Error;

use ft_linear_regression::core::train_model;

fn main() -> Result<(), Box<dyn Error>> {
    let learning_rate: f64 = 0.1;
    let iterations: u32 = 100;
    let reset_weights: bool = true;

    train_model(learning_rate, iterations, reset_weights)?;
    Ok(())
}
