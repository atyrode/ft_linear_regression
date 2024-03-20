pub mod bonus;
pub mod core;
pub mod io;
pub mod model;
pub use std::error::Error;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct Dataset {
    records: Vec<Record>,
    scaler: Scaler,
}

#[derive(Debug, Deserialize)]
pub struct Record {
    km: f64,
    price: f64,
}

#[derive(Debug, Deserialize)]
struct Scaler {
    mean: f64,
    std_dev: f64,
}

impl Dataset {
    const DATASET_PATH: &'static str = "data.csv";

    fn new(records: Vec<Record>) -> Self {
        let scaler = Scaler::from_records(&records);
        Self { records, scaler }
    }

    /// # Errors
    /// Return an error if it couldn't read the file.
    pub fn from_csv() -> Result<Self, Box<dyn Error>> {
        let mut records: Vec<Record> = Vec::new();

        for data in io::read_csv(Self::DATASET_PATH)? {
            let record = Record {
                km: data[0].parse::<f64>()?,
                price: data[1].parse::<f64>()?,
            };
            records.push(record);
        }

        Ok(Self::new(records))
    }
}

impl Scaler {
    fn standardize(&self, value: f64) -> f64 {
        (value - self.mean) / self.std_dev
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_records(records: &[Record]) -> Self {
        let kms: Vec<f64> = records.iter().map(|record| record.km).collect();

        let mean = kms.iter().sum::<f64>() / kms.len() as f64;
        let variance: f64 = kms.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / kms.len() as f64;
        let std_dev = variance.sqrt();

        Self { mean, std_dev }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Weights {
    pub theta0: f64,
    pub theta1: f64,
}

impl Weights {
    const WEIGHTS_PATH: &'static str = "weights.csv";

    const fn new(theta0: f64, theta1: f64) -> Self {
        Self { theta0, theta1 }
    }

    /// # Errors
    /// Return an error if it couldn't create the file.
    pub fn initialize() -> Result<Self, Box<dyn Error>> {
        let weights = Self::new(0.0, 0.0);
        weights.to_csv()?;
        Ok(weights)
    }

    /// # Errors
    /// Return an error if it couldn't read the file.
    pub fn from_csv() -> Result<Self, Box<dyn Error>> {
        if !io::csv_exists(Self::WEIGHTS_PATH) {
            return Self::initialize();
        }

        let data = io::read_csv(Self::WEIGHTS_PATH)?;
        let theta0 = data[0][0].parse::<f64>()?;
        let theta1 = data[0][1].parse::<f64>()?;

        Ok(Self::new(theta0, theta1))
    }

    /// # Errors
    /// Return an error if it couldn't write to the file.
    pub fn to_csv(&self) -> Result<(), Box<dyn Error>> {
        io::write_csv(Self::WEIGHTS_PATH, self)?;
        Ok(())
    }
}
