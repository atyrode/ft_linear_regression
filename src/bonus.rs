use crate::parsing::{Dataset, Weights};
use crate::training::predict_price;

pub fn prediction_compare(dataset: &Dataset, weights: &Weights) {
    println!("km\t\tprice\t\tprediction");
    println!("{}", "-".repeat(50));
    for record in &dataset.records {
        let km: f64 = record.km;
        let price: f64 = record.price;

        let std_km: f64 = dataset.get_standardized_km(km);

        let prediction: f64 = predict_price(std_km, weights.theta0, weights.theta1);
        println!("{}\t=>\t{}\t=>\t{}", km, price, prediction.round());
    }
}

#[allow(clippy::cast_precision_loss)]
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
            let prediction = predict_price(std_km, weights.theta0, weights.theta1);
            (record.price - prediction).powi(2)
        })
        .sum::<f64>();

    let r_squared: f64 = 1.0 - residuals_sum_squares / total_sum_squares;

    r_squared
}
