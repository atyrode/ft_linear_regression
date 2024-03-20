use crate::{Dataset, Weights};

/* Hypothesis function */
#[must_use]
pub fn predict(km: f64, theta0: f64, theta1: f64) -> f64 {
    theta1.mul_add(km, theta0)
}

/* Cost function */
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn calculate_mean_squared_error(dataset: &Dataset, theta0: f64, theta1: f64) -> f64 {
    let m = dataset.records.len() as f64;
    let mut sum_squared_error = 0.0;

    for record in &dataset.records {
        let km: f64 = dataset.scaler.standardize(record.km);
        let price = record.price;
        let prediction = predict(km, theta0, theta1);
        sum_squared_error += (prediction - price).abs();
    }

    sum_squared_error / m
}

/* Gradient Descent */
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn gradient_descent(
    dataset: &Dataset,
    weights: &Weights,
    learning_rate: f64,
    iterations: u32,
) -> (f64, f64) {
    let m = dataset.records.len() as f64;

    let mut new_theta0: f64 = weights.theta0;
    let mut new_theta1: f64 = weights.theta1;

    for _ in 0..iterations {
        let mut sum_errors_theta_0 = 0.0;
        let mut sum_errors_theta_1 = 0.0;

        // Calculate errors for gradients
        for record in &dataset.records {
            let km: f64 = dataset.scaler.standardize(record.km);
            let price: f64 = record.price;
            let prediction: f64 = predict(km, new_theta0, new_theta1);
            sum_errors_theta_0 += prediction - price;
            sum_errors_theta_1 += (prediction - price) * km;
        }

        // Update weights
        new_theta0 -= learning_rate * (1.0 / m) * sum_errors_theta_0;
        new_theta1 -= learning_rate * (1.0 / m) * sum_errors_theta_1;
    }

    (new_theta0, new_theta1)
}
