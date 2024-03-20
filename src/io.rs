use std::error::Error;
use std::io::{stdin, stdout, Write};

#[allow(clippy::missing_errors_doc)]
pub fn get_user_input(prompt: &str) -> Result<String, Box<dyn Error>> {
    print!("{prompt}");
    stdout().flush()?;

    let mut input: String = String::new();
    stdin().read_line(&mut input)?;

    Ok(input.trim().to_string())
}

fn get_term_width() -> usize {
    term_size::dimensions().map_or(80, |(w, _)| w)
}

pub fn print_separator(style: &str) {
    let width: usize = get_term_width();
    println!("{}", style.repeat(width));
}
