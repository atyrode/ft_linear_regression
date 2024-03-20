use std::fs::File;
use std::io::{stdin, stdout, Write};
use std::path::Path;

use csv::{Reader, StringRecord, Writer};

use crate::Error;

/* CSV */

#[must_use]
pub fn csv_exists(file_path: &str) -> bool {
    Path::new(file_path).exists()
}

/// # Errors
/// Return an error if path does not already exist.
/// Other errors may also be returned according to `OpenOptions::open`.
pub fn read_csv(file_path: &str) -> Result<Vec<csv::StringRecord>, Box<dyn Error>> {
    let file: File = File::open(file_path)?;
    let mut reader: Reader<File> = Reader::from_reader(file);
    let mut data: Vec<StringRecord> = Vec::new();

    for result in reader.records() {
        let row: StringRecord = result?;
        data.push(row);
    }
    Ok(data)
}

/// # Errors
/// Return an error if it couldn't open the path.
pub fn write_csv<T: serde::Serialize>(file_path: &str, data: T) -> Result<(), Box<dyn Error>> {
    let mut wtr: Writer<File> = Writer::from_path(file_path)?;

    wtr.serialize(data)?;
    wtr.flush()?;

    Ok(())
}

/* User input/output */

/// # Errors
/// Return an error if not all bytes could be written due to I/O errors or EOF being reached.
pub fn get_user_input(prompt: &str) -> Result<String, Box<dyn Error>> {
    print!("{prompt}");
    stdout().flush()?;

    let mut input: String = String::new();
    let _ = stdin().read_line(&mut input);

    Ok(input.trim().to_string())
}

#[must_use]
pub fn get_term_width() -> usize {
    term_size::dimensions().map_or(80, |(w, _)| w)
}

#[must_use]
pub fn get_term_height() -> usize {
    term_size::dimensions().map_or(24, |(_, h)| h)
}

#[must_use]
pub fn separator(style: &str) -> String {
    let width: usize = get_term_width();
    style.repeat(width)
}

#[must_use]
pub fn center(text: &str) -> String {
    let width: usize = get_term_width();
    let padding: usize = (width - text.len()) / 2;
    format!("{}{}{}", " ".repeat(padding), text, " ".repeat(padding))
}

#[must_use]
pub fn underline(text: &str, style: &str) -> String {
    format!("{}\n{}", text, style.repeat(text.len()))
}

/* Pretty Tables */

use prettytable::{format, row, Cell, Row, Table};

#[must_use]
pub fn create_table(columns_title: &[&str]) -> Table {
    let mut table: Table = Table::new();
    table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);

    let columns_title: Vec<Cell> = columns_title.iter().map(|c| Cell::new(c)).collect();
    table.set_titles(Row::new(columns_title));

    table
}

/// # Errors
/// May error if it can't get the terminal stdout.
pub fn print_dyn_table(table: &Table) -> Result<usize, Box<dyn Error>> {
    let height: usize = table.print_tty(true)?;
    Ok(height)
}

/// # Errors
/// May error if it can't get the terminal stdout.
/// May error if the terminal cursor can't move up or delete a line.
/// May error if the table can't be printed.
pub fn add_dyn_row(
    table: &mut Table,
    height: &mut usize,
    step: u32,
    iterations: u32,
    mse: f64,
) -> Result<(), Box<dyn Error>> {
    let Some(mut terminal) = term::stdout() else {
        return Err("Failed to get terminal".into());
    };

    table.add_row(row![step, iterations, mse.round()]);

    for _ in 0..*height {
        terminal.cursor_up()?;
        terminal.delete_line()?;
    }

    *height = table.print_tty(false)?;
    Ok(())
}
