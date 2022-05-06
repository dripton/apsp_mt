This is a small Rust implementation of solutions for the All Pairs Shortest
Paths algorithm.  It currently contains a multi-threaded (with Rayon)
implementation of Dijkstra's Algorithm, and a single-threaded implementation of
the Floyd-Warshall Algorithm.

Input is a square matrix in numpy's npy format.  Neither Dijkstra or
Floyd-Warshall works if the input contains negative cycles.  Output is a
npz-format file containing two matrixes: dist (the distances) and pred (the
predecessors).

This was created as a dependency for my (Python) traderoutes program, which
needed fast APSP.  Both scipy and retworkx had implementations of Dijkstra and
Floyd-Warshall, but scipy's were too slow, retworkx's Floyd-Warshall didn't
have a way to return the predecessors, and retworkx's Dijkstra used too much
memory.  (Note that unlike those programs, this project is not currently a
Python extension; it's just a stand-alone command-line Rust program.)
