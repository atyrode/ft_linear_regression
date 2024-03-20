use crate::io::{get_term_height, get_term_width};
use crate::model::predict;
use crate::{Dataset, Weights};
use textplots::{Chart, Plot, Shape};

#[must_use]
pub fn prediction_compare(dataset: &Dataset, weights: &Weights) -> Vec<Vec<f64>> {
    dataset
        .records
        .iter()
        .map(|record| {
            let std_km = dataset.scaler.standardize(record.km);
            let prediction = predict(std_km, weights.theta0, weights.theta1);
            let precision = 100.0 * (1.0 - (record.price - prediction).abs() / record.price);
            vec![
                record.km,
                record.price,
                prediction.round(),
                (precision * 100.0).round() / 100.0,
            ]
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn calculate_precision(dataset: &Dataset, weights: &Weights) -> f64 {
    let mean_prices: f64 = dataset
        .records
        .iter()
        .map(|record| record.price)
        .sum::<f64>()
        / dataset.records.len() as f64;

    let total_sum_squares: f64 = dataset
        .records
        .iter()
        .map(|record| (record.price - mean_prices).powi(2))
        .sum::<f64>();

    let residuals_sum_squares: f64 = dataset
        .records
        .iter()
        .map(|record| {
            let std_km = dataset.scaler.standardize(record.km);
            let prediction = predict(std_km, weights.theta0, weights.theta1);
            (record.price - prediction).powi(2)
        })
        .sum::<f64>();

    let r_squared: f64 = 1.0 - residuals_sum_squares / total_sum_squares;
    
    if r_squared < 0.0 {
        0.0
    } else {
        r_squared
    }
}

#[allow(clippy::cast_possible_truncation)]
pub fn display_dataset_plot(dataset: &Dataset, weights: &Weights) {
    let term_w: u32 = get_term_width() as u32;
    let term_h: u32 = get_term_height() as u32;

    let points: Vec<(f32, f32)> = dataset
        .records
        .iter()
        .map(|r| (r.km as f32, r.price as f32))
        .collect();

    // Determine the buffer as a percentage of the range (e.g., 10%)
    let buffer_percent = 0.1;

    /* X */
    let x_min = points.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
    let x_max = points.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);

    // Calculate the range of x values
    let x_range = x_max - x_min;
    let buffer = x_range * buffer_percent;

    // Apply the buffer
    let x_min = x_min - buffer;
    let x_max = x_max + buffer;

    /* Y */
    let y_min = points.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
    let y_max = points.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);

    // Calculate the range of y values
    let y_range = y_max - y_min;
    let buffer = y_range * buffer_percent;

    // Apply the buffer
    let y_min = y_min - buffer;
    let y_max = y_max + buffer;

    /* Create the chart */
    let mut chart = Chart::new_with_y_range(term_w + 60, term_h, x_min, x_max, y_min, y_max);

    // Calculate the linear regression start and end X points
    let std_x_min = dataset.scaler.standardize(f64::from(x_min));
    let std_x_max = dataset.scaler.standardize(f64::from(x_max));

    // Calculate the linear regression start and end Y points
    let linear_regression: &[(f32, f32)] = &[
        (
            x_min,
            predict(std_x_min, weights.theta0, weights.theta1) as f32,
        ),
        (
            x_max,
            predict(std_x_max, weights.theta0, weights.theta1) as f32,
        ),
    ];

    // Plot the dataset
    println!("X -> Car Mileage | Y -> Car Price\n");
    chart
        .lineplot(&Shape::Points(&points))
        .lineplot(&Shape::Lines(linear_regression))
        .nice();
}
