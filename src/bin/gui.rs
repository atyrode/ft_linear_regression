use ft_linear_regression::core::{predict_price, train_model};
use ft_linear_regression::{bonus, io, Dataset, Error, Weights};

use prettytable::row;
use tabled::{
    settings::{
        object::Rows, peaker::PriorityMax, style::Style, themes::Colorization, Alignment, Color,
        Settings, Theme, Width,
    },
    Table, Tabled,
};

static MENU_OPTIONS: [&str; 10] = [
    "Predict price",
    "Train model",
    "Reset weights",
    "Show weights",
    "Show comparison",
    "Show graph",
    "Show precision",
    "Show training parameters",
    "Edit training parameters",
    "Reset training parameters",
];

#[derive(Tabled)]
struct MenuItem {
    #[tabled(rename = "Menu")]
    option: String,
}

fn create_menu() {
    // Clear the terminal
    print!("{esc}[2J{esc}[1;1H", esc = 27 as char);

    let term_width: usize = io::get_term_width();

    let menu_items: Vec<MenuItem> = MENU_OPTIONS
        .iter()
        .enumerate()
        .map(|(id, &option)| MenuItem {
            option: format!("{}. {}", id + 1, option),
        })
        .collect();

    let mut table = Table::new(menu_items);
    table.modify(Rows::new(..1), Alignment::center());
    table.with(Colorization::exact([Color::FG_CYAN], Rows::first()));

    let style = Theme::from_style(Style::rounded());

    table.with(style);

    let settings = Settings::default()
        .with(Width::wrap(term_width).priority::<PriorityMax>())
        .with(Width::increase(term_width));
    table.with(settings);

    println!("{table}");
}

fn print_training_parameters(
    learning_rate: f64,
    iterations: u32,
    batch: u32,
) -> Result<(), Box<dyn Error>> {
    let mut table = io::create_table(&["Learning rate", "Iterations", "Batch"]);
    table.add_row(row![learning_rate, iterations, batch]);
    io::print_dyn_table(&table)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut learning_rate: f64 = 0.1;
    let mut iterations: u32 = 100;
    let mut batch: u32 = 10;

    loop {
        create_menu();

        println!("\n{}", io::underline("Pick an option", "-"));
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
                let mut table = io::create_table(&["theta0", "theta1"]);
                table.add_row(row![0.0, 0.0]);
                io::print_dyn_table(&table)?;
            }
            "4" => {
                let mut table = io::create_table(&["theta0", "theta1"]);
                table.add_row(row![weights.theta0, weights.theta1]);
                io::print_dyn_table(&table)?;
            }
            "5" => {
                let mut table = io::create_table(&["km", "price", "predicted", "accuracy (%)"]);
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
                print_training_parameters(learning_rate, iterations, batch)?;
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

                println!();
                print_training_parameters(learning_rate, iterations, batch)?;
            }
            "10" => {
                learning_rate = 0.1;
                iterations = 100;
                batch = 10;
                println!("Training parameters were reset to default values.");
                println!();
                print_training_parameters(learning_rate, iterations, batch)?;
            }
            _ => {
                println!("Invalid option. Please try again.");
            }
        }

        let gray_color = "\x1b[90m";
        let reset = "\x1b[0m";
        let _ = io::get_user_input(&format!(
            "{}{}{}",
            gray_color, "\nPress Enter to continue...", reset
        ));
        println!();
    }
}
