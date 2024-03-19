mod parse;

use std::error::Error;
use std::io::{stdout, stdin, Write};

fn user_menu() -> Result<f64, Box<dyn Error>> {
    // Print the menu
    let prompt: String = "Pick an option:\n1. Predict price\n2. Train model\n> ".to_string();
    print!("{prompt}");
    stdout().flush()?;

    // Get user input
    let mut input = String::new();
    stdin().read_line(&mut input)?;
    
    // Match the input to the corresponding action
    match input.trim().parse() {
        // Predict price
        Ok(1) => {
            print!("Enter mileage (km): ");
            stdout().flush()?;
            let mut input = String::new();
            stdin().read_line(&mut input)?;
            
            let km: f64 = input.trim().parse()?;
            Ok(km)
        }
        // Train model
        Ok(2) => {
            println!("Training model...");
            Ok(0.0)
        }
        // Invalid option, re-prompt the user
        _ => {
            println!("! Invalid option !");
            Ok(user_menu()?)
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    user_menu()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::Array;

    #[test]
    fn test_ndarray_working() {
        let a = Array::from_vec(vec![1, 2, 3, 4]);
        assert_eq!(a.sum(), 10);
    }
}
