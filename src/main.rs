// testing speed of Floyd-Warshall
// Version with 825x825 Vec<u64> took 80s
// Version with 825x825 array of u64 took 12s

use std::fs;

#[macro_use]
extern crate serde_derive;
use serde_json::{self, Result};

const ARRAY_SIZE: usize = 825;
const ARRAY_DATA_PATH: &str = "/var/tmp/nd.json";

#[derive(Deserialize, Debug)]
#[serde(transparent)]
pub struct Row {
    pub cells: Vec<u64>,
}

#[derive(Deserialize, Debug)]
#[serde(transparent)]
pub struct Table {
    pub rows: Vec<Row>,
}

impl Table {
    pub fn new(data: &str) -> Result<Table> {
        let table = serde_json::from_str(data);
        table
    }
}

fn floyd_warshall(dist: &mut [[u64; ARRAY_SIZE]; ARRAY_SIZE]) {
    for i in 0..dist.len() {
        dist[i][i] = 0;
    }
    for k in 0..dist.len() {
        for i in 0..dist.len() {
            for j in 0..dist.len() {
                if dist[i][j] > dist[i][k] + dist[k][j] {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
}

fn main() {
    let json_str: &str = &fs::read_to_string(ARRAY_DATA_PATH).expect("failed to read file");
    let table: Table = Table::new(&json_str).expect("failed to parse json");

    let mut matrix: [[u64; ARRAY_SIZE]; ARRAY_SIZE] = [[0; ARRAY_SIZE]; ARRAY_SIZE];
    for (i, row) in table.rows.iter().enumerate() {
        for (j, cell) in row.cells.iter().enumerate() {
            matrix[i][j] = *cell;
        }
    }
    println!("matrix before {:?}\n", matrix);

    floyd_warshall(&mut matrix);
    println!("matrix after {:?}\n", matrix);
}
