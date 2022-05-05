use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::path;

use clap::{ArgEnum, Parser};
use rayon::prelude::*;

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

const NO_PRED_NODE: i64 = -9999;
const INFINITY: f64 = f64::MAX;

fn floyd_warshall(dist: &mut Vec<Vec<f64>>) -> Vec<Vec<i64>> {
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

    // Initialize predecessors where we have paths
    for i in 0..size {
        for j in 0..size {
            if dist[i][j] > 0.0 && dist[i][j] < INFINITY {
                pred[i][j] = i as i64;
            }
        }
    }

    // Do the Floyd Warshall triple nested loop
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

fn dijkstra_one_row(
    start: u64,
    size: usize,
    neighbors_map: &HashMap<u64, HashSet<u64>>,
    weights: &HashMap<(u64, u64), u64>,
) -> (Vec<f64>, Vec<i64>) {
    let mut dist_row = vec![INFINITY; size];
    let mut pred_row = vec![NO_PRED_NODE; size];

    let mut heap = BinaryHeap::new();
    let mut set = HashSet::new();

    dist_row[start as usize] = 0.0;
    heap.push(Reverse((0, start)));
    set.insert(start);

    while !heap.is_empty() {
        if let Some(Reverse((priority, u))) = heap.pop() {
            set.remove(&u);
            if priority == dist_row[u as usize] as u64 {
                if let Some(neighbors) = neighbors_map.get(&u) {
                    for v in neighbors {
                        if let Some(weight) = weights.get(&(u, *v)) {
                            let alt = dist_row[u as usize] as u64 + weight;
                            if alt < (dist_row[*v as usize]) as u64 {
                                dist_row[*v as usize] = alt as f64;
                                pred_row[*v as usize] = u as i64;
                                let tup = (alt, *v);
                                heap.push(Reverse(tup));
                                set.insert(*v);
                            }
                        } else {
                            panic!("bug: neighbor not in weights");
                        }
                    }
                }
            }
        }
    }

    return (dist_row, pred_row);
}

fn dijkstra(dist: &mut Vec<Vec<f64>>) -> Vec<Vec<i64>> {
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

    // Populate neighbors_map
    let mut neighbors_map: HashMap<u64, HashSet<u64>> = HashMap::new();
    for i in 0..size {
        let set = HashSet::new();
        neighbors_map.insert(i as u64, set);
    }
    for i in 0..size {
        for j in 0..size {
            if dist[i][j] > 0.0 && dist[i][j] < INFINITY {
                if let Some(set) = neighbors_map.get_mut(&(i as u64)) {
                    set.insert(j as u64);
                } else {
                    panic!("problem with map");
                }
            }
        }
    }

    // Populate weights
    let mut weights: HashMap<(u64, u64), u64> = HashMap::new();
    for i in 0..size {
        for j in 0..size {
            if dist[i][j] > 0.0 && dist[i][j] != INFINITY {
                weights.insert((i as u64, j as u64), dist[i][j] as u64);
            }
        }
    }

    // Initialize predecessors where we have paths
    for i in 0..size {
        for j in 0..size {
            if dist[i][j] > 0.0 && dist[i][j] < INFINITY {
                pred[i][j] = i as i64;
            }
        }
    }

    // Do the Dijkstra algorithm for each row, in parallel using Rayon
    let tuples: Vec<(Vec<f64>, Vec<i64>)> = (0..size)
        .into_par_iter()
        .map(|i| dijkstra_one_row(i as u64, size, &neighbors_map, &weights))
        .collect();
    for (i, (dist_row, pred_row)) in tuples.iter().enumerate() {
        dist[i] = dist_row.to_vec();
        pred[i] = pred_row.to_vec();
    }

    return pred;
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, arg_enum)]
    algorithm: Algorithm,

    /// Path to the input matrix in json format
    #[clap(short, long)]
    json_input_path: path::PathBuf,

    /// Path to the output distance matrix in json format
    #[clap(short, long)]
    dist_output_path: path::PathBuf,

    /// Path to the output predecessor matrix in json format
    #[clap(short, long)]
    pred_output_path: path::PathBuf,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
enum Algorithm {
    D,
    FW,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let json_str: &str = &fs::read_to_string(&args.json_input_path).expect("failed to read file");
    let table: Table = Table::new(&json_str).expect("failed to parse json");
    let size = table.rows.len();

    let mut dist = vec![vec![0.0; size]; size];
    for (i, row) in table.rows.iter().enumerate() {
        for (j, cell) in row.cells.iter().enumerate() {
            if *cell != 0.0 {
                dist[i][j] = *cell;
            }
        }
    }

    let pred;
    if args.algorithm == Algorithm::D {
        pred = dijkstra(&mut dist);
    } else {
        pred = floyd_warshall(&mut dist);
    }

    let dist_output_str = serde_json::to_string(&dist).unwrap();
    fs::write(&args.dist_output_path, dist_output_str)?;

    let pred_output_str = serde_json::to_string(&pred).unwrap();
    fs::write(&args.pred_output_path, pred_output_str)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floyd_warshall_scipy() {
        // https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html
        let mut dist = vec![vec![INFINITY; 4]; 4];
        dist[0][1] = 1.0;
        dist[0][2] = 2.0;
        dist[1][3] = 1.0;
        dist[2][0] = 2.0;
        dist[2][3] = 3.0;
        println!("dist before {:?}\n", dist);

        let pred = floyd_warshall(&mut dist);
        println!("dist after {:?}\n", dist);
        println!("pred after {:?}\n", pred);

        assert_eq!(dist[0][0], 0.0);
        assert_eq!(dist[0][1], 1.0);
        assert_eq!(dist[0][2], 2.0);
        assert_eq!(dist[0][3], 2.0);

        assert_eq!(dist[1][0], 1.0);
        assert_eq!(dist[1][1], 0.0);
        assert_eq!(dist[1][2], 3.0);
        assert_eq!(dist[1][3], 1.0);

        assert_eq!(dist[2][0], 2.0);
        assert_eq!(dist[2][1], 3.0);
        assert_eq!(dist[2][2], 0.0);
        assert_eq!(dist[2][3], 3.0);

        assert_eq!(dist[3][0], 2.0);
        assert_eq!(dist[3][1], 1.0);
        assert_eq!(dist[3][2], 3.0);
        assert_eq!(dist[3][3], 0.0);

        assert_eq!(pred[0][0], NO_PRED_NODE);
        assert_eq!(pred[0][1], 0);
        assert_eq!(pred[0][2], 0);
        assert_eq!(pred[0][3], 1);

        assert_eq!(pred[1][0], 1);
        assert_eq!(pred[1][1], NO_PRED_NODE);
        assert_eq!(pred[1][2], 0);
        assert_eq!(pred[1][3], 1);

        assert_eq!(pred[2][0], 2);
        assert_eq!(pred[2][1], 0);
        assert_eq!(pred[2][2], NO_PRED_NODE);
        assert_eq!(pred[2][3], 2);

        assert_eq!(pred[3][0], 1);
        assert_eq!(pred[3][1], 3);
        assert_eq!(pred[3][2], 3);
        assert_eq!(pred[3][3], NO_PRED_NODE);
    }

    #[test]
    fn test_dijkstra_scipy() {
        // https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html
        let mut dist = vec![vec![INFINITY; 4]; 4];
        dist[0][1] = 1.0;
        dist[0][2] = 2.0;
        dist[1][3] = 1.0;
        dist[2][0] = 2.0;
        dist[2][3] = 3.0;
        println!("dist before {:?}\n", dist);

        let pred = dijkstra(&mut dist);
        println!("dist after {:?}\n", dist);
        println!("pred after {:?}\n", pred);

        assert_eq!(dist[0][0], 0.0);
        assert_eq!(dist[0][1], 1.0);
        assert_eq!(dist[0][2], 2.0);
        assert_eq!(dist[0][3], 2.0);

        assert_eq!(dist[1][0], 1.0);
        assert_eq!(dist[1][1], 0.0);
        assert_eq!(dist[1][2], 3.0);
        assert_eq!(dist[1][3], 1.0);

        assert_eq!(dist[2][0], 2.0);
        assert_eq!(dist[2][1], 3.0);
        assert_eq!(dist[2][2], 0.0);
        assert_eq!(dist[2][3], 3.0);

        assert_eq!(dist[3][0], 2.0);
        assert_eq!(dist[3][1], 1.0);
        assert_eq!(dist[3][2], 3.0);
        assert_eq!(dist[3][3], 0.0);

        assert_eq!(pred[0][0], NO_PRED_NODE);
        assert_eq!(pred[0][1], 0);
        assert_eq!(pred[0][2], 0);
        assert_eq!(pred[0][3], 1);

        assert_eq!(pred[1][0], 1);
        assert_eq!(pred[1][1], NO_PRED_NODE);
        assert_eq!(pred[1][2], 0);
        assert_eq!(pred[1][3], 1);

        assert_eq!(pred[2][0], 2);
        assert_eq!(pred[2][1], 0);
        assert_eq!(pred[2][2], NO_PRED_NODE);
        assert_eq!(pred[2][3], 2);

        assert_eq!(pred[3][0], 1);
        assert_eq!(pred[3][1], 3);
        assert_eq!(pred[3][2], 3);
        assert_eq!(pred[3][3], NO_PRED_NODE);
    }
}
