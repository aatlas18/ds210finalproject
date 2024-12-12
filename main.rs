extern crate ndarray;
extern crate csv;

use ndarray::{Array1, Array2};
use std::error::Error;
use csv::ReaderBuilder;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    // Read the data from CSV files
    let file_paths = vec![
        "/opt/app-root/src/FinalProject/src/al_jazeera.csv",
        "/opt/app-root/src/FinalProject/src/bbc.csv",
        "/opt/app-root/src/FinalProject/src/cnn.csv",
        "/opt/app-root/src/FinalProject/src/reuters.csv",
    ];

    let mut data: Vec<(String, u32)> = Vec::new();

    for path in file_paths {
        let file = File::open(path)?;
        let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

        for result in rdr.records() {
            let record = result?;
            let source = record.get(0).unwrap().to_string();
            let likes: u32 = record.get(1).unwrap().parse()?;

            if likes > 0 {
                data.push((source, likes));
            }
        }
    }

    // Prepare the data for clustering (only the 'likes' field)
    let likes_data: Vec<f64> = data.iter().map(|(_, likes)| *likes as f64).collect();
    let n_samples = likes_data.len();
    let n_features = 1; // We only have 'likes' as the feature

    let arr = Array2::<f64>::from_shape_vec((n_samples, n_features), likes_data)?;

    // Perform KMeans clustering (K = 3 clusters)
    let k = 3;
    let max_iters = 100;
    let tolerance = 0.0001;

    let mut centroids = initialize_centroids(&arr, k);
    let mut labels = vec![0; n_samples];

    for _ in 0..max_iters {
        // Assign each point to the closest centroid
        for (i, point) in arr.axis_iter(ndarray::Axis(0)).enumerate() {
            // Clone the point to an owned Array1<f64> before passing it to the function
            let point_owned = point.to_owned();
            labels[i] = find_closest_centroid(&point_owned, &centroids);
        }

        // Recompute centroids
        let new_centroids = recompute_centroids(&arr, &labels, k);

        // Check for convergence (if centroids don't change)
        if has_converged(&centroids, &new_centroids, tolerance) {
            break;
        }

        centroids = new_centroids;
    }

    // Map each source to its cluster
    for (i, (source, _)) in data.into_iter().enumerate() {
        let cluster = labels[i];
        println!("Source: {}, Cluster: {}", source, cluster);
    }

    Ok(())
}

// Function to initialize random centroids (k centroids)
fn initialize_centroids(arr: &Array2<f64>, k: usize) -> Vec<Array1<f64>> {
    let mut centroids = Vec::new();
    for i in 0..k {
        centroids.push(arr.row(i).to_owned());  // Clone the row into a new Array1
    }
    centroids
}

// Function to find the closest centroid for a given data point
fn find_closest_centroid(point: &Array1<f64>, centroids: &Vec<Array1<f64>>) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, centroid)| (i, distance(point, centroid)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

// Function to compute the Euclidean distance between two points
fn distance(p1: &Array1<f64>, p2: &Array1<f64>) -> f64 {
    p1.iter()
        .zip(p2.iter())
        .map(|(x1, x2)| (x1 - x2).powi(2))
        .sum::<f64>()
        .sqrt()
}

// Function to recompute centroids based on the current cluster labels
fn recompute_centroids(arr: &Array2<f64>, labels: &Vec<usize>, k: usize) -> Vec<Array1<f64>> {
    let mut new_centroids = vec![Array1::<f64>::zeros(arr.shape()[1]); k];
    let mut counts = vec![0; k];

    for (i, point) in arr.axis_iter(ndarray::Axis(0)).enumerate() {
        let cluster = labels[i];
        new_centroids[cluster] = &new_centroids[cluster] + &point;
        counts[cluster] += 1;
    }

    for i in 0..k {
        if counts[i] > 0 {
            new_centroids[i] = &new_centroids[i] / counts[i] as f64;
        }
    }

    new_centroids
}

// Function to check if centroids have converged
fn has_converged(old_centroids: &Vec<Array1<f64>>, new_centroids: &Vec<Array1<f64>>, tolerance: f64) -> bool {
    old_centroids
        .iter()
        .zip(new_centroids.iter())
        .all(|(old, new)| distance(old, new) < tolerance)
}
