use ft_linear_regression::core::{predict_price, train_model};
use ft_linear_regression::{bonus, io, Dataset, Error, Weights};

use prettytable::{format, row, Cell, Row, Table};

fn create_menu() -> Table {
    let mut menu = Table::new();
    menu.set_format(*format::consts::FORMAT_BORDERS_ONLY);
    menu.set_titles(Row::new(vec![Cell::new("Menu").style_spec("Fcbc")]));

    let options = [
        "Predict price",
        "Train model",
        "Reset weights",
        "Show weights",
        "Show comparison",
        "Show graph",
        "Show precision",
        "Show training parameters",
        "Edit training parameters",
    ];

    for (i, option) in options.iter().enumerate() {
        menu.add_row(row![format!("{}) {}", i + 1, option)]);
    }

    menu
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut learning_rate: f64 = 0.1;
    let mut iterations: u32 = 100;
    let mut batch: u32 = 10;

    let menu = create_menu();

    loop {
        // Clear the terminal
        print!("{esc}[2J{esc}[1;1H", esc = 27 as char);

        io::print_dyn_table(&menu)?;
        println!("{}", io::underline("Pick an option", "-"));

        let user_choice: String = io::get_user_input("> ")?;

        println!();

        let dataset: Dataset = Dataset::from_csv()?;
        let weights: Weights = Weights::from_csv()?;

        match user_choice.as_str() {
            "1" => {
                predict_price(&dataset, &weights)?;
            }
            "2" => {
                let new_weights: Weights =
                    train_model(&dataset, weights, learning_rate, iterations, batch)?;
                new_weights.to_csv()?;
            }
            "3" => {
                Weights::initialize()?;
                let mut table: Table = io::create_table(&["theta0", "theta1"]);
                table.add_row(row![0.0, 0.0]);
                io::print_dyn_table(&table)?;
            }
            "4" => {
                let mut table: Table = io::create_table(&["theta0", "theta1"]);
                table.add_row(row![weights.theta0, weights.theta1]);
                io::print_dyn_table(&table)?;
            }
            "5" => {
                let mut table: Table =
                    io::create_table(&["km", "price", "predicted", "accuracy (%)"]);
                let predictions = bonus::prediction_compare(&dataset, &weights);
                for prediction in predictions {
                    table.add_row(row![
                        &prediction[0],
                        &prediction[1],
                        &prediction[2],
                        &prediction[3]
                    ]);
                }
                io::print_dyn_table(&table)?;
            }
            "6" => {
                bonus::display_dataset_plot(&dataset, &weights);
            }
            "7" => {
                let r_squared: f64 = bonus::calculate_precision(&dataset, &weights);
                let rounded_r_squared: f64 = (r_squared * 100.0).round();
                println!("The model's precision (R²) is: {rounded_r_squared}%");
            }
            "8" => {
                println!("Training parameters:");
                let mut table: Table = io::create_table(&["Learning rate", "Iterations", "Batch"]);
                table.add_row(row![learning_rate, iterations, batch]);
                io::print_dyn_table(&table)?;
            }
            "9" => {
                println!("Current learning rate: {learning_rate}");
                let new_learning_rate = io::get_user_input("New => ")?;
                learning_rate = new_learning_rate.parse::<f64>()?;

                println!("\nCurrent iterations: {iterations}");
                let new_iterations = io::get_user_input("New => ")?;
                iterations = new_iterations.parse::<u32>()?;

                println!("\nCurrent batch: {batch}");
                let new_batch = io::get_user_input("New => ")?;
                batch = new_batch.parse::<u32>()?;
            }
            _ => {
                println!("Invalid option. Please try again.");
            }
        }

        let _ = io::get_user_input("\n-> Press any key to continue <-");
        println!();
    }
}
