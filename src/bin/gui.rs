use std::error::Error;

use ft_linear_regression::core::{predict_price, train_model};
use ft_linear_regression::io;

fn main() -> Result<(), Box<dyn Error>> {
    loop {
        let option_list = ["1. Predict price", "2. Train model"];
        let options: String = option_list.join("\n") + "\n> ";
        let user_choice: String = io::get_user_input(&options)?;

        match user_choice.as_str() {
            "1" => {
                predict_price()?;
            }
            "2" => {
                train_model(0.2, 100, 10, true)?;
            }
            _ => {
                println!("Invalid option. Please try again.");
            }
        }
    }
}

// let user_input: String = get_user_input("Start with new weights? (y/n): ")?;
// let user_choice: bool = user_input == "y";
// if user_choice {
//     Weights::set(0.0, 0.0)?;
// }
// println!("==================================================");
// let new_weights: Weights = model::train_model(&dataset, 0.1, 100)?;
// println!(
//     "New computed weights:\ntheta0: {}\ntheta1: {}",
//     new_weights.theta0, new_weights.theta1
// );
// println!("==================================================");
// bonus::prediction_compare(&dataset, &new_weights);
// println!("==================================================");
// let r_squared: f64 = bonus::calculate_precision(&dataset, &new_weights);
// let rounded_r_squared: f64 = (r_squared * 100.0).round() / 100.0;
// println!("Precision (R²) closer to 1 == better: {rounded_r_squared}");
// println!("==================================================");
// bonus::display_dataset_plot(&dataset, &new_weights);
