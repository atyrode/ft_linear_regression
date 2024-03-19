use std::error::Error;
use std::io::{stdin, stdout, Write};

pub fn get_user_input(prompt: &str) -> Result<String, Box<dyn Error>> {
    print!("{prompt}");
    stdout().flush()?;

    let mut input: String = String::new();
    stdin().read_line(&mut input)?;

    Ok(input.trim().to_string())
}

pub fn user_want_prediction() -> Result<bool, Box<dyn Error>> {
    let user_choice: String = get_user_input("==================================================\nPick an option:\n1. Predict price\n2. Train model\n> ")?;

    match user_choice.as_str() {
        "1" => Ok(true),
        "2" => Ok(false),
        _ => {
            println!("Invalid choice!\n");
            user_want_prediction()
        }
    }
}
