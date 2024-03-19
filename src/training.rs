use crate::parse::{Record, Weights};

/* Standardization */

pub fn get_mean(values: &Vec<f64>) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn get_variance(values: &Vec<f64>) -> f64 {
    let mean = get_mean(values);
    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
}

pub fn get_standard_deviation(values: &Vec<f64>) -> f64 {
    get_variance(values).sqrt()
}

/* Hypothesis function */
pub fn predict_price(km: f64, theta0: f64, theta1: f64) -> f64 {
    theta1.mul_add(km, theta0)
}

/* Cost function */
// The MSE function calculates the average squared difference between actual and predicted prices.
// It iterates over each Record of the dataset
// Calculates the squared difference between the predicted and actual prices
// Sums all these squared differences
// and finally divides by the number of records to compute the MSE.
fn calculate_mean_squared_error(dataset: &Vec<Record>, theta0: f64, theta1: f64) -> f64 {
    let m = dataset.len() as f64;
    let mut sum_squared_error = 0.0;

    for record in dataset {
        let km = record.km;
        let price = record.price;
        let prediction = predict_price(km, theta0, theta1);
        sum_squared_error += (prediction - price).abs();
    }

    sum_squared_error / m
}

/* Gradient Descent */
fn gradient_descent(
    dataset: &Vec<Record>,
    weights: Weights,
    learning_rate: f64,
    iterations: u32,
) -> (f64, f64) {
    let m = dataset.len() as f64;

    let mut new_theta0: f64 = weights.theta0;
    let mut new_theta1: f64 = weights.theta1;

    for iteration in 0..iterations {
        let mut sum_errors_theta_0 = 0.0;
        let mut sum_errors_theta_1 = 0.0;

        // Calculate errors for gradients
        for record in dataset {
            let km = record.km;
            let price = record.price;
            let prediction = predict_price(km, new_theta0, new_theta1);
            sum_errors_theta_0 += prediction - price;
            sum_errors_theta_1 += (prediction - price) * km;
        }

        // Update weights
        new_theta0 -= learning_rate * (1.0 / m) * sum_errors_theta_0;
        new_theta1 -= learning_rate * (1.0 / m) * sum_errors_theta_1;

        // Optionally print MSE to monitor training progress
        if iteration % (iterations / 10) == 0 {
            // For example, every 100 iterations
            let mse = calculate_mean_squared_error(dataset, new_theta0, new_theta1);
            println!("Iteration {iteration}: MSE = {mse}");
        }
    }

    println!("==================================================");

    (new_theta0, new_theta1)
}

/* Training */
pub fn train_model(
    dataset: Vec<Record>,
    learning_rate: f64,
    iterations: u32,
) -> Result<Weights, Box<dyn std::error::Error>> {
    let initial_weights = Weights::get()?;

    let (new_theta0, new_theta1) =
        gradient_descent(&dataset, initial_weights, learning_rate, iterations);

    let new_weights = Weights::set(new_theta0, new_theta1)?;
    Ok(new_weights)
}
