use crate::model::predict;
use crate::parsing::{Dataset, Weights};
use textplots::{Chart, Plot, Shape};

pub fn prediction_compare(dataset: &Dataset, weights: &Weights) {
    println!("km\t\tprice\t\tprediction");
    println!("{}", "-".repeat(50));
    for record in &dataset.records {
        let km: f64 = record.km;
        let price: f64 = record.price;

        let std_km: f64 = dataset.get_standardized_km(km);

        let prediction: f64 = predict(std_km, weights.theta0, weights.theta1);
        println!("{}\t=>\t{}\t=>\t{}", km, price, prediction.round());
    }
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
            let std_km = dataset.get_standardized_km(record.km);
            let prediction = predict(std_km, weights.theta0, weights.theta1);
            (record.price - prediction).powi(2)
        })
        .sum::<f64>();

    let r_squared: f64 = 1.0 - residuals_sum_squares / total_sum_squares;

    r_squared
}

#[allow(clippy::cast_possible_truncation)]
pub fn display_dataset_plot(dataset: &Dataset, weights: &Weights) {
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

    let y_range = y_max - y_min;
    let buffer = y_range * buffer_percent;

    let y_min = y_min - buffer;
    let y_max = y_max + buffer;

    let mut chart = Chart::new_with_y_range(180, 80, x_min, x_max, y_min, y_max);

    let std_x_min = dataset.get_standardized_km(f64::from(x_min));
    let std_x_max = dataset.get_standardized_km(f64::from(x_max));

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
    println!("\nX -> Car Mileage | Y -> Car Price\n");
    chart
        .lineplot(&Shape::Points(&points))
        .lineplot(&Shape::Lines(linear_regression))
        .nice();
}
