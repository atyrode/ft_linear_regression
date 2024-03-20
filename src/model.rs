use crate::parsing::{Dataset, Weights};
use prettytable::{Table, format, row};

/* Standardization */
/* Hypothesis function */
#[must_use]
pub fn predict(km: f64, theta0: f64, theta1: f64) -> f64 {
    theta1.mul_add(km, theta0)
}

#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn get_mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn get_variance(values: &[f64]) -> f64 {
    let mean = get_mean(values);
    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
}

#[must_use]
pub fn get_standard_deviation(values: &[f64]) -> f64 {
    get_variance(values).sqrt()
}

/* Cost function */
// The MSE function calculates the average squared difference between actual and predicted prices.
// It iterates over each Record of the dataset
// Calculates the squared difference between the predicted and actual prices
// Sums all these squared differences
// and finally divides by the number of records to compute the MSE.
#[allow(clippy::cast_precision_loss)]
fn calculate_mean_squared_error(dataset: &Dataset, theta0: f64, theta1: f64) -> f64 {
    let m = dataset.records.len() as f64;
    let mut sum_squared_error = 0.0;

    for record in &dataset.records {
        let km: f64 = dataset.get_standardized_km(record.km);
        let price = record.price;
        let prediction = predict(km, theta0, theta1);
        sum_squared_error += (prediction - price).abs();
    }

    sum_squared_error / m
}

/* Gradient Descent */
#[allow(clippy::cast_precision_loss)]
fn gradient_descent(
    dataset: &Dataset,
    weights: &Weights,
    learning_rate: f64,
    iterations: u32,
) -> (f64, f64) {

    let mut table = Table::new();
    table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
    table.set_titles(row!["Iterations", "Error Margin"]);
    let mut height = table.print_tty(false).unwrap();
    let mut terminal = term::stdout().unwrap();

    let m = dataset.records.len() as f64;

    let mut new_theta0: f64 = weights.theta0;
    let mut new_theta1: f64 = weights.theta1;

    for iteration in 0..iterations {
        let mut sum_errors_theta_0 = 0.0;
        let mut sum_errors_theta_1 = 0.0;

        // Calculate errors for gradients
        for record in &dataset.records {
            let km: f64 = dataset.get_standardized_km(record.km);
            let price: f64 = record.price;
            let prediction: f64 = predict(km, new_theta0, new_theta1);
            sum_errors_theta_0 += prediction - price;
            sum_errors_theta_1 += (prediction - price) * km;
        }

        // Update weights
        new_theta0 -= learning_rate * (1.0 / m) * sum_errors_theta_0;
        new_theta1 -= learning_rate * (1.0 / m) * sum_errors_theta_1;

        // Optionally print MSE to monitor training progress
        if iteration % (iterations / 10) == 0 {
            let mse = calculate_mean_squared_error(dataset, new_theta0, new_theta1);
            table.add_row(row![iteration, mse.round()]);
            for _ in 0..height {
                terminal.cursor_up().unwrap();
                terminal.delete_line().unwrap();
            }
            height = table.print_tty(false).unwrap();
        }
    }
    (new_theta0, new_theta1)
}

/* Training */
#[must_use]
pub fn train(
    dataset: &Dataset,
    initial_weights: &Weights,
    learning_rate: f64,
    iterations: u32,
) -> (f64, f64) {
    gradient_descent(dataset, initial_weights, learning_rate, iterations)
}
