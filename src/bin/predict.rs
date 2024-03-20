use ft_linear_regression::core::predict_price;
use ft_linear_regression::Error;
use ft_linear_regression::{Dataset, Weights};

fn main() -> Result<(), Box<dyn Error>> {
    let weights: Weights = Weights::from_csv()?;
    let dataset: Dataset = Dataset::from_csv()?;

    predict_price(&dataset, &weights)?;
    Ok(())
}
