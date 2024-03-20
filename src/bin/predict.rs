use std::error::Error;

use ft_linear_regression::core::predict_price;

fn main() -> Result<(), Box<dyn Error>> {
    predict_price()?;
    Ok(())
}
