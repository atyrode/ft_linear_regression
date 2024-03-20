use std::error::Error;

use crate::io;
use crate::model;
use crate::parsing;

#[allow(clippy::missing_errors_doc)]
pub fn predict_price() -> Result<(), Box<dyn Error>> {
    let weights = parsing::Weights::get()?;
    let dataset = parsing::Dataset::get()?;

    let user_input: String = io::get_user_input("Enter the number of kilometers: ")?;
    let km: f64 = user_input.parse::<f64>()?;

    let std_km: f64 = dataset.get_standardized_km(km);

    let price: f64 = model::predict(std_km, weights.theta0, weights.theta1);
    println!("The estimated price is: {}", price.round());

    Ok(())
}

#[allow(clippy::missing_errors_doc)]
pub fn train_model(learning_rate: f64, iterations: u32, reset_weights: bool) -> Result<(), Box<dyn Error>> {
    
    let initial_weights = match reset_weights {
        true => parsing::Weights::default(),
        false => parsing::Weights::get()?,
    };

    let dataset = parsing::Dataset::get()?;

    let (new_theta0, new_theta1) = model::train(&dataset, &initial_weights, learning_rate, iterations);

    parsing::Weights::set(new_theta0, new_theta1)?;
    Ok(())
}
