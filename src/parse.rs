use crate::training::{get_mean, get_standard_deviation};
use csv::{Reader, Writer};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::path::Path;

/* Dataset */

#[derive(Debug, Deserialize)]
pub struct Dataset {
    pub records: Vec<Record>,
}

#[derive(Debug, Deserialize)]
pub struct Record {
    pub km: f64,
    pub price: f64,
}

impl Dataset {
    const DATASET_PATH: &'static str = "data.csv";

    pub fn new(records: Vec<Record>) -> Self {
        Self { records }
    }

    pub fn get() -> Result<Self, Box<dyn Error>> {
        let mut rdr: Reader<File> = Reader::from_path(Self::DATASET_PATH)?;
        let mut records: Vec<Record> = Vec::new();

        for result in rdr.deserialize() {
            records.push(result?);
        }

        Ok(Self::new(records))
    }

    fn get_kms(&self) -> Vec<f64> {
        self.records.iter().map(|record| record.km).collect()
    }

    pub fn get_standardized_km(&self, km: f64) -> f64 {
        let kms = self.get_kms();

        let mean = get_mean(&kms);
        let std_dev = get_standard_deviation(&kms);

        (km - mean) / std_dev
    }

    pub fn get_standardized() -> Result<Self, Box<dyn Error>> {
        let dataset = Self::get()?;

        let standardized_records: Vec<Record> = dataset
            .records
            .iter()
            .map(|record| {
                let km = record.km;
                let price = record.price;

                let std_km = dataset.get_standardized_km(km);

                Record { km: std_km, price }
            })
            .collect();

        Ok(Self::new(standardized_records))
    }
}

/* Weights */

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

    fn write(&self) -> Result<(), Box<dyn Error>> {
        let mut wtr = Writer::from_path(Self::WEIGHTS_PATH)?;
        wtr.serialize(self)?;
        wtr.flush()?;

        Ok(())
    }

    fn read() -> Result<Self, Box<dyn Error>> {
        let mut rdr: Reader<File> = Reader::from_path(Self::WEIGHTS_PATH)?;
        let mut result = rdr.deserialize();

        let weights: Self = match result.next() {
            Some(weights) => weights?,
            None => Self::new(0.0, 0.0),
        };

        Ok(weights)
    }

    fn file_exists() -> bool {
        Path::new(Self::WEIGHTS_PATH).exists()
    }

    pub fn get() -> Result<Self, Box<dyn Error>> {
        if Self::file_exists() {
            Self::read()
        } else {
            let default_weights = Self::new(0.0, 0.0);
            default_weights.write()?; // Write default weights to file if it doesn't exist
            Ok(default_weights)
        }
    }

    pub fn set(theta0: f64, theta1: f64) -> Result<Self, Box<dyn Error>> {
        let weights = Self::new(theta0, theta1);
        weights.write()?;
        Ok(weights)
    }
}
