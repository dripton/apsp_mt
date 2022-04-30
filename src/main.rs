use std::env;
use std::fs;
use std::process;

#[macro_use]
extern crate serde_derive;
use serde_json::{self, Result};

#[derive(Serialize, Deserialize, Debug)]
#[serde(transparent)]
pub struct Row {
    pub cells: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
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

const NO_PRED_NODE: f64 = -9999.0;
const INFINITY: f64 = f64::MAX;

fn floyd_warshall(dist: &mut Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let size = dist.len();
    let mut pred = vec![vec![NO_PRED_NODE; size]; size];

    // Set all zero vertexes to infinity
    for i in 0..size {
        for j in 0..size {
            if dist[i][j] == 0.0 {
                dist[i][j] = INFINITY;
            }
        }
    }

    // Set each vertex at zero distance to itself
    for i in 0..size {
        dist[i][i] = 0.0;
    }
    // Assume bidirectional movement
    for i in 0..size {
        for j in 0..size {
            if dist[i][j] > dist[j][i] {
                dist[i][j] = dist[j][i];
            }
        }
    }
    for i in 0..size {
        for j in 0..size {
            if dist[i][j] > 0.0 && dist[i][j] < INFINITY {
                pred[i][j] = i as f64;
            }
        }
    }
    for k in 0..size {
        for i in 0..size {
            for j in 0..size {
                if dist[i][k] != INFINITY
                    && dist[k][j] != INFINITY
                    && dist[i][j] > dist[i][k] + dist[k][j]
                {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    pred[i][j] = pred[k][j];
                }
            }
        }
    }
    return pred;
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        println!("Usage: {:?} json_input dist_output pred_output", args[0]);
        process::exit(1);
    }
    let json_input_path = &args[1];
    let dist_output_path = &args[2];
    let pred_output_path = &args[3];
    let json_str: &str = &fs::read_to_string(json_input_path).expect("failed to read file");
    let table: Table = Table::new(&json_str).expect("failed to parse json");
    let size = table.rows.len();

    let mut matrix = vec![vec![0.0; size]; size];
    for (i, row) in table.rows.iter().enumerate() {
        for (j, cell) in row.cells.iter().enumerate() {
            if *cell != 0.0 {
                matrix[i][j] = *cell;
            }
        }
    }

    let pred = floyd_warshall(&mut matrix);

    let dist_output_str = serde_json::to_string(&matrix).unwrap();
    fs::write(dist_output_path, dist_output_str)?;

    let pred_output_str = serde_json::to_string(&pred).unwrap();
    fs::write(pred_output_path, pred_output_str)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floyd_warshall_scipy() {
        // https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html
        let mut matrix = vec![vec![INFINITY; 4]; 4];
        matrix[0][1] = 1.0;
        matrix[0][2] = 2.0;
        matrix[1][3] = 1.0;
        matrix[2][0] = 2.0;
        matrix[2][3] = 3.0;
        println!("matrix before {:?}\n", matrix);

        let pred = floyd_warshall(&mut matrix);
        println!("matrix after {:?}\n", matrix);
        println!("pred after {:?}\n", pred);

        assert_eq!(matrix[0][0], 0.0);
        assert_eq!(matrix[0][1], 1.0);
        assert_eq!(matrix[0][2], 2.0);
        assert_eq!(matrix[0][3], 2.0);

        assert_eq!(matrix[1][0], 1.0);
        assert_eq!(matrix[1][1], 0.0);
        assert_eq!(matrix[1][2], 3.0);
        assert_eq!(matrix[1][3], 1.0);

        assert_eq!(matrix[2][0], 2.0);
        assert_eq!(matrix[2][1], 3.0);
        assert_eq!(matrix[2][2], 0.0);
        assert_eq!(matrix[2][3], 3.0);

        assert_eq!(matrix[3][0], 2.0);
        assert_eq!(matrix[3][1], 1.0);
        assert_eq!(matrix[3][2], 3.0);
        assert_eq!(matrix[3][3], 0.0);

        assert_eq!(pred[0][0], NO_PRED_NODE);
        assert_eq!(pred[0][1], 0.0);
        assert_eq!(pred[0][2], 0.0);
        assert_eq!(pred[0][3], 1.0);

        assert_eq!(pred[1][0], 1.0);
        assert_eq!(pred[1][1], NO_PRED_NODE);
        assert_eq!(pred[1][2], 0.0);
        assert_eq!(pred[1][3], 1.0);

        assert_eq!(pred[2][0], 2.0);
        assert_eq!(pred[2][1], 0.0);
        assert_eq!(pred[2][2], NO_PRED_NODE);
        assert_eq!(pred[2][3], 2.0);

        assert_eq!(pred[3][0], 1.0);
        assert_eq!(pred[3][1], 3.0);
        assert_eq!(pred[3][2], 3.0);
        assert_eq!(pred[3][3], NO_PRED_NODE);
    }
}
