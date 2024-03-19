use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Record {
    km: u32,
    price: u32,
}

use csv::Reader;
use std::error::Error;

fn load_csv(file_path: &str) -> Result<Vec<Record>, Box<dyn Error>> {
    let mut rdr = Reader::from_path(file_path)?;
    let mut records = Vec::new();

    for result in rdr.deserialize() {
        let record: Record = result?;
        records.push(record);
    }

    Ok(records)
}

fn main() -> Result<(), Box<dyn Error>> {
    let records = load_csv("data.csv")?;
    println!("Loaded {} records", records.len());
    for record in records {
        println!("km: {}, price: {}", record.km, record.price);
    }
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

    use super::load_csv;
    use super::Record;

    #[test]
    #[allow(clippy::expect_used)]
    fn test_valid_csv() {
        let records: Vec<Record> = load_csv("data.csv").expect("Failed to load 'data.csv'");
        assert_eq!(records.len(), 24); // Ensures there are 24 rows of data
        for record in records {
            // This asserts that each record has a 'km' and 'price' field.
            // It will panic if any of the fields are missing in any row, effectively testing the presence of these columns.
            println!("km: {}, price: {}", record.km, record.price);
        }
    }
}
