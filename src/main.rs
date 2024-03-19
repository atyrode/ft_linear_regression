mod menu;
use menu::{get_user_input, user_want_prediction};

mod parse;
use parse::{Dataset, Weights};

mod training;
use training::{predict_price, train_model};

use std::error::Error;

fn prediction_compare() -> Result<(), Box<dyn Error>> {
    let dataset: Dataset = Dataset::get()?;
    let weights: Weights = Weights::get()?;

    for record in &dataset.records {
        let km = record.km;
        let price = record.price;

        let std_km: f64 = dataset.get_standardized_km(km);

        let prediction = predict_price(std_km, weights.theta0, weights.theta1);
        println!("km: {} => {} => {}", km, price, prediction.round());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    loop {
        match user_want_prediction()? {
            true => {
                let user_input: String = get_user_input("Enter the number of kilometers: ")?;
                let km: f64 = user_input.parse::<f64>()?;

                let weights: Weights = Weights::get()?;
                let dataset: Dataset = Dataset::get()?;
                let std_km: f64 = dataset.get_standardized_km(km);

                let price: f64 = predict_price(std_km, weights.theta0, weights.theta1);
                println!("The estimated price is: {}", price.round());
            }
            false => {
                let std_dataset: Dataset = Dataset::get_standardized()?;

                let user_input: String = get_user_input("Start with new weights? (y/n): ")?;
                let user_choice: bool = user_input == "y";
                if user_choice {
                    Weights::set(0.0, 0.0)?;
                }
                println!("==================================================");
                let new_weights: Weights = train_model(std_dataset.records, 0.1, 100)?;
                println!(
                    "New computed weights:\ntheta0: {}\ntheta1: {}",
                    new_weights.theta0, new_weights.theta1
                );
                println!("==================================================");
                prediction_compare()?;
            }
        }
    }
}
