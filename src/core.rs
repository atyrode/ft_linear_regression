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

use prettytable::{format, row, Table};
use term_size::dimensions;

fn add_row(table: &mut Table, height: &mut usize, step: u32, iterations: u32, mse: f64) {
    let mut terminal = term::stdout().unwrap();
    
    table.add_row(row![step, iterations, mse.round()]);

    for _ in 0..*height {
        terminal.cursor_up().unwrap();
        terminal.delete_line().unwrap();
    }
    
    *height = table.print_tty(false).unwrap();
}

#[allow(clippy::missing_errors_doc)]
pub fn train_model(
    learning_rate: f64,
    iterations: u32,
    mut batch: u32,
    reset_weights: bool,
) -> Result<(), Box<dyn Error>> {
    if reset_weights {
        parsing::Weights::set(0.0, 0.0)?;
    }

    if batch > iterations {
        batch = iterations;
        println!("> Batch size is larger than the number of iterations. It was adjusted to fit.");
        println!("> From {batch} to {iterations}");
    }

    if batch > dimensions().unwrap().1 as u32 {
        batch = 10;
        println!("> Batch size is too large for the terminal size. It was set to 10.");
    }

    let dataset = parsing::Dataset::get()?;

    let mut table: Table = Table::new();
    table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);
    table.set_titles(row!["", "It.", "Margin"]);
    let mut height: usize = table.print_tty(false).unwrap();
    

    let iter_per_batch: u32 = iterations / batch;
    let (mut new_theta0, mut new_theta1) = (0.0, 0.0);
    
    let mut mse: f64 = model::calculate_mean_squared_error(&dataset, new_theta0, new_theta1);
    add_row(&mut table, &mut height, 0, 0, mse);

    for step in 0..batch {
        (new_theta0, new_theta1) = model::gradient_descent(
            &dataset,
            &parsing::Weights::get()?,
            learning_rate,
            iter_per_batch,
        );
        parsing::Weights::set(new_theta0, new_theta1)?;

        mse = model::calculate_mean_squared_error(&dataset, new_theta0, new_theta1);

        add_row(&mut table, &mut height, step + 1, iter_per_batch * (step + 1), mse);
    }

    Ok(())
}
