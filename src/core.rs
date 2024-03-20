use std::error::Error;

use crate::io;
use crate::model;
use crate::{Dataset, Weights};

/// # Errors
/// Will return an error if it fails to parse the user input.
/// Or if the user input is not a valid number.
pub fn predict_price(dataset: &Dataset, weights: &Weights) -> Result<(), Box<dyn Error>> {
    let user_input: String = io::get_user_input("Enter the number of kilometers: ")?;

    let km: f64 = match user_input.parse::<f64>() {
        Ok(km) => km,
        Err(_) => {
            println!("Please enter a valid number.");
            return Ok(());
        }
    };

    let std_km: f64 = dataset.scaler.standardize(km);
    let price: f64 = model::predict(std_km, weights.theta0, weights.theta1);

    println!("The estimated price is: {}", price.round());

    Ok(())
}

/// # Errors
/// Will return an error if it fails to modify the pretty table or to print it
/// due to an I/O error.
pub fn train_model(
    dataset: &Dataset,
    mut weights: Weights,
    learning_rate: f64,
    iterations: u32,
    mut batch: u32,
) -> Result<Weights, Box<dyn Error>> {
    if batch as usize > (io::get_term_width() / 2) || batch > iterations {
        batch = 10;
        println!("> Batch size can't be more than iterations or half of the terminal height.");
        println!("> It was adjusted to 10.");
    }

    let mut table = io::create_table(&["n°", "Iter.", "Margin"]);
    let mut height: usize = io::print_dyn_table(&table)?;

    let mut mse: f64 = model::calculate_mean_squared_error(dataset, weights.theta0, weights.theta1);

    io::add_dyn_row(&mut table, &mut height, 0, 0, mse)?;

    let iter_per_batch: u32 = iterations / batch;

    for step in 0..batch {
        (weights.theta0, weights.theta1) =
            model::gradient_descent(dataset, &weights, learning_rate, iter_per_batch);

        mse = model::calculate_mean_squared_error(dataset, weights.theta0, weights.theta1);

        io::add_dyn_row(
            &mut table,
            &mut height,
            step + 1,
            iter_per_batch * (step + 1),
            mse,
        )?;
    }

    Ok(weights)
}
