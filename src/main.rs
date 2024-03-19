use std::error::Error;
use std::io::{stdin, stdout, Write};

mod parse;
use parse::{Dataset, Weights};

mod training;
use training::{predict_price, train_model};

#[allow(clippy::missing_errors_doc)]
pub fn get_user_input(prompt: &str) -> Result<String, Box<dyn Error>> {
    print!("{prompt}");
    stdout().flush()?;

    let mut input: String = String::new();
    stdin().read_line(&mut input)?;

    Ok(input.trim().to_string())
}

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
        println!("==================================================");
        let option_list = ["1. Predict price", "2. Train model"];
        let options: String = option_list.join("\n") + "\n> ";
        let user_choice: String = get_user_input(&options)?;

        match user_choice.as_str() {
            "1" => {
                let user_input: String = get_user_input("Enter the number of kilometers: ")?;
                let km: f64 = user_input.parse::<f64>()?;

                let weights: Weights = Weights::get()?;
                let dataset: Dataset = Dataset::get()?;
                let std_km: f64 = dataset.get_standardized_km(km);

                let price: f64 = predict_price(std_km, weights.theta0, weights.theta1);
                println!("The estimated price is: {}", price.round());
            }
            "2" => {
                let std_dataset: Dataset = Dataset::get_standardized()?;

                let user_input: String = get_user_input("Start with new weights? (y/n): ")?;
                let user_choice: bool = user_input == "y";
                if user_choice {
                    Weights::set(0.0, 0.0)?;
                }
                println!("==================================================");
                let new_weights: Weights = train_model(&std_dataset.records, 0.1, 100)?;
                println!(
                    "New computed weights:\ntheta0: {}\ntheta1: {}",
                    new_weights.theta0, new_weights.theta1
                );
                println!("==================================================");
                prediction_compare()?;
            }
            _ => {
                println!("Invalid option. Please try again.");
            }
        }
    }
}
