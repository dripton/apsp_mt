use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path;

use anyhow::Result;
use clap::{ArgEnum, Parser};
use rayon::prelude::*;

extern crate bucket_queue;
use bucket_queue::*;

extern crate ndarray;
use ndarray::prelude::*;
use ndarray_npy::{read_npy, NpzWriter};

extern crate rand;
use rand::prelude::*;

const NO_PRED_NODE: i64 = -9999;
const INFINITY: f64 = f64::MAX;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
enum Algorithm {
    D,
    FW,
    Dial,
}

// TODO multi-thread
fn floyd_warshall(dist: &mut Array2<f64>) -> Array2<i64> {
    let size = dist.nrows();
    let mut pred = Array2::<i64>::from_elem((size, size), NO_PRED_NODE);

    // Set all zero vertexes to infinity
    for i in 0..size {
        for j in 0..size {
            if dist[[i, j]] == 0.0 {
                dist[[i, j]] = INFINITY;
            }
        }
    }

    // Set each vertex at zero distance to itself
    for i in 0..size {
        dist[[i, i]] = 0.0;
    }

    // Assume bidirectional movement
    for i in 0..size {
        for j in 0..size {
            if dist[[i, j]] > dist[[j, i]] {
                dist[[i, j]] = dist[[j, i]];
            }
        }
    }

    // Initialize predecessors where we have paths
    for i in 0..size {
        for j in 0..size {
            if dist[[i, j]] > 0.0 && dist[[i, j]] < INFINITY {
                pred[[i, j]] = i as i64;
            }
        }
    }

    // Do the Floyd Warshall triple nested loop
    for k in 0..size {
        for i in 0..size {
            for j in 0..size {
                if dist[[i, k]] != INFINITY
                    && dist[[k, j]] != INFINITY
                    && dist[[i, j]] > dist[[i, k]] + dist[[k, j]]
                {
                    dist[[i, j]] = dist[[i, k]] + dist[[k, j]];
                    pred[[i, j]] = pred[[k, j]];
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

    // TODO Try a Fibonacci heap instead
    let mut heap = BinaryHeap::new();

    dist_row[start as usize] = 0.0;
    heap.push(Reverse((0, start)));

    while !heap.is_empty() {
        if let Some(Reverse((priority, u))) = heap.pop() {
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

fn dial_one_row(
    start: u64,
    size: usize,
    neighbors_map: &HashMap<u64, HashSet<u64>>,
    weights: &HashMap<(u64, u64), u64>,
) -> (Vec<f64>, Vec<i64>) {
    let mut dist_row = vec![INFINITY; size];
    let mut pred_row = vec![NO_PRED_NODE; size];

    let mut queue = BucketQueue::<VecDeque<u64>>::new();

    dist_row[start as usize] = 0.0;
    queue.enqueue(start, 0);

    while !queue.is_empty() {
        if let Some(priority) = queue.min_priority() {
            if let Some(u) = queue.dequeue_min() {
                if priority == dist_row[u as usize] as usize {
                    if let Some(neighbors) = neighbors_map.get(&u) {
                        for v in neighbors {
                            if let Some(weight) = weights.get(&(u, *v)) {
                                let alt = dist_row[u as usize] as u64 + weight;
                                if alt < (dist_row[*v as usize]) as u64 {
                                    dist_row[*v as usize] = alt as f64;
                                    pred_row[*v as usize] = u as i64;
                                    queue.enqueue(*v as u64, alt as usize);
                                }
                            } else {
                                panic!("bug: neighbor not in weights");
                            }
                        }
                    }
                }
            }
        }
    }

    return (dist_row, pred_row);
}

fn dijkstra_dial_inner(dist: &mut Array2<f64>, alg: Algorithm) -> Array2<i64> {
    let size = dist.nrows();
    let mut pred = Array2::<i64>::from_elem((size, size), NO_PRED_NODE);

    // Set all zero vertexes to infinity
    for i in 0..size {
        for j in 0..size {
            if dist[[i, j]] == 0.0 {
                dist[[i, j]] = INFINITY;
            }
        }
    }

    // Set each vertex at zero distance to itself
    for i in 0..size {
        dist[[i, i]] = 0.0;
    }

    // Assume bidirectional movement
    for i in 0..size {
        for j in 0..size {
            if dist[[i, j]] > dist[[j, i]] {
                dist[[i, j]] = dist[[j, i]];
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
            if dist[[i, j]] > 0.0 && dist[[i, j]] < INFINITY {
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
            if dist[[i, j]] > 0.0 && dist[[i, j]] != INFINITY {
                weights.insert((i as u64, j as u64), dist[[i, j]] as u64);
            }
        }
    }

    // Initialize predecessors where we have paths
    for i in 0..size {
        for j in 0..size {
            if dist[[i, j]] > 0.0 && dist[[i, j]] < INFINITY {
                pred[[i, j]] = i as i64;
            }
        }
    }

    let tuples: Vec<(Vec<f64>, Vec<i64>)>;
    // Do the Dijkstra or algorithm for each row, in parallel using Rayon
    if alg == Algorithm::D {
        tuples = (0..size)
            .into_par_iter()
            .map(|i| dijkstra_one_row(i as u64, size, &neighbors_map, &weights))
            .collect();
    } else if alg == Algorithm::Dial {
        tuples = (0..size)
            .into_par_iter()
            .map(|i| dial_one_row(i as u64, size, &neighbors_map, &weights))
            .collect();
    } else {
        panic!("broken Algorithm");
    }
    for (i, (dist_row, pred_row)) in tuples.iter().enumerate() {
        // TODO Find a way to copy the entire row
        for (j, dist_el) in dist_row.iter().enumerate() {
            dist[[i, j]] = *dist_el;
        }
        for (j, pred_el) in pred_row.iter().enumerate() {
            pred[[i, j]] = *pred_el;
        }
    }

    return pred;
}

fn dijkstra(dist: &mut Array2<f64>) -> Array2<i64> {
    return dijkstra_dial_inner(dist, Algorithm::D);
}

fn dial(dist: &mut Array2<f64>) -> Array2<i64> {
    return dijkstra_dial_inner(dist, Algorithm::Dial);
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, arg_enum)]
    algorithm: Algorithm,

    /// Path to input matrix in numpy ndy format
    #[clap(short, long)]
    input_path: path::PathBuf,

    /// Path to output distance and predecessor matrixes in numpy ndz format
    #[clap(short, long)]
    output_path: path::PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut dist: Array2<f64> = read_npy(args.input_path)?;

    let pred: Array2<i64>;
    if args.algorithm == Algorithm::D {
        pred = dijkstra(&mut dist);
    } else if args.algorithm == Algorithm::Dial {
        pred = dial(&mut dist);
    } else if args.algorithm == Algorithm::FW {
        pred = floyd_warshall(&mut dist);
    } else {
        panic!("bad algorithm");
    }

    let mut npz = NpzWriter::new(File::create(args.output_path)?);
    npz.add_array("dist", &dist)?;
    npz.add_array("pred", &pred)?;
    npz.finish()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_scipy_test() -> Array2<f64> {
        // https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html
        let mut dist = Array2::<f64>::from_elem((4, 4), INFINITY);
        dist[[0, 1]] = 1.0;
        dist[[0, 2]] = 2.0;
        dist[[1, 3]] = 1.0;
        dist[[2, 0]] = 2.0;
        dist[[2, 3]] = 3.0;
        println!("dist before {:?}\n", dist);
        return dist;
    }

    fn compare_scipy_test(dist: Array2<f64>, pred: Array2<i64>) {
        println!("dist after {:?}\n", dist);
        println!("pred after {:?}\n", pred);

        assert_eq!(dist[[0, 0]], 0.0);
        assert_eq!(dist[[0, 1]], 1.0);
        assert_eq!(dist[[0, 2]], 2.0);
        assert_eq!(dist[[0, 3]], 2.0);

        assert_eq!(dist[[1, 0]], 1.0);
        assert_eq!(dist[[1, 1]], 0.0);
        assert_eq!(dist[[1, 2]], 3.0);
        assert_eq!(dist[[1, 3]], 1.0);

        assert_eq!(dist[[2, 0]], 2.0);
        assert_eq!(dist[[2, 1]], 3.0);
        assert_eq!(dist[[2, 2]], 0.0);
        assert_eq!(dist[[2, 3]], 3.0);

        assert_eq!(dist[[3, 0]], 2.0);
        assert_eq!(dist[[3, 1]], 1.0);
        assert_eq!(dist[[3, 2]], 3.0);
        assert_eq!(dist[[3, 3]], 0.0);

        assert_eq!(pred[[0, 0]], NO_PRED_NODE);
        assert_eq!(pred[[0, 1]], 0);
        assert_eq!(pred[[0, 2]], 0);
        assert_eq!(pred[[0, 3]], 1);

        assert_eq!(pred[[1, 0]], 1);
        assert_eq!(pred[[1, 1]], NO_PRED_NODE);
        assert_eq!(pred[[1, 2]], 0);
        assert_eq!(pred[[1, 3]], 1);

        assert_eq!(pred[[2, 0]], 2);
        assert_eq!(pred[[2, 1]], 0);
        assert_eq!(pred[[2, 2]], NO_PRED_NODE);
        assert_eq!(pred[[2, 3]], 2);

        assert_eq!(pred[[3, 0]], 1);
        assert_eq!(pred[[3, 1]], 3);
        assert_eq!(pred[[3, 2]], 3);
        assert_eq!(pred[[3, 3]], NO_PRED_NODE);
    }

    #[test]
    fn test_floyd_warshall_scipy() {
        let mut dist = setup_scipy_test();
        let pred = floyd_warshall(&mut dist);
        compare_scipy_test(dist, pred);
    }

    #[test]
    fn test_dijkstra_scipy() {
        let mut dist = setup_scipy_test();
        let pred = dijkstra(&mut dist);
        compare_scipy_test(dist, pred);
    }

    #[test]
    fn test_dial_scipy() {
        let mut dist = setup_scipy_test();
        let pred = dial(&mut dist);
        compare_scipy_test(dist, pred);
    }

    fn setup_random_matrix() -> Array2<f64> {
        let mut rng = thread_rng();
        let vertexes = 100;
        let edges = 100;
        let max_cost = 4;
        let mut dist = Array2::<f64>::from_elem((vertexes, vertexes), INFINITY);
        for _ in 0..edges {
            let i = rng.gen_range(0..edges);
            let j = rng.gen_range(0..edges);
            let cost = rng.gen_range(1..=max_cost);
            dist[[i, j]] = cost as f64;
        }
        return dist;
    }

    #[test]
    fn multi_algorithm_random_matrix() {
        let mut dist1 = setup_random_matrix();
        let mut dist2 = dist1.clone();
        let mut dist3 = dist2.clone();

        floyd_warshall(&mut dist1);
        dijkstra(&mut dist2);
        dial(&mut dist3);

        assert_eq!(dist1, dist2);
        assert_eq!(dist1, dist3);
        // predecessors are not guaranteed to be identical
    }
}
