extern crate ndarray;
extern crate csv;

use ndarray::{Array1, Array2};
use std::error::Error;
use csv::ReaderBuilder;
use std::fs::File;
use std::collections::HashMap;

//Initialize random centroids
fn initialize_centroids(arr: &Array2<f64>, k: usize) -> Vec<Array1<f64>> {
    let mut centroids = Vec::new();
    for i in 0..k {
        // Clone row into new Array1
        centroids.push(arr.row(i).to_owned());  
    }
    centroids
}

//Compute euclidean distance between points
fn distance(p1: &Array1<f64>, p2: &Array1<f64>) -> f64 {
    p1.iter()
        .zip(p2.iter())
        .map(|(x1, x2)| (x1 - x2).powi(2))
        .sum::<f64>()
        .sqrt()
}

//Find closest centroid
fn find_closest_centroid(point: &Array1<f64>, centroids: &Vec<Array1<f64>>) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, centroid)| (i, distance(point, centroid)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

//Recompute centroids based on current cluster
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

//Check if centroids converged
fn has_converged(old_centroids: &Vec<Array1<f64>>, new_centroids: &Vec<Array1<f64>>, tolerance: f64) -> bool {
    old_centroids
        .iter()
        .zip(new_centroids.iter())
        .all(|(old, new)| distance(old, new) < tolerance)
}

fn main() -> Result<(), Box<dyn Error>> {
    //Define mapping between file paths and custom names
    let file_name_map: HashMap<&str, &str> = [
        ("/opt/app-root/src/FinalProject/src/al_jazeera.csv", "Al Jazeera"),
        ("/opt/app-root/src/FinalProject/src/bbc.csv", "BBC"),
        ("/opt/app-root/src/FinalProject/src/cnn.csv", "CNN"),
        ("/opt/app-root/src/FinalProject/src/reuters.csv", "Reuters"),
    ]
    .iter()
    .cloned()
    .collect();

    //Read data from CSV
    let file_paths = vec![
        "/opt/app-root/src/FinalProject/src/al_jazeera.csv",
        "/opt/app-root/src/FinalProject/src/bbc.csv",
        "/opt/app-root/src/FinalProject/src/cnn.csv",
        "/opt/app-root/src/FinalProject/src/reuters.csv",
    ];

    let mut data: Vec<(String, u32)> = Vec::new(); 
    //Iterate through each CSV
    for path in file_paths {
        let file = File::open(path)?;
        let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

        //Read from CSV
        for result in rdr.records() {
            let record = result?;
            let likes: u32 = record.get(1).unwrap().parse()?;

            if likes > 0 {
                //Get custom name for file
                let news_source = file_name_map.get(path).unwrap_or(&"Unknown").to_string();
                //Store the custom file source name and likes
                data.push((news_source, likes)); 
            }
        }
    }

    //Prepare the data for clustering, only using likes
    let likes_data: Vec<f64> = data.iter().map(|(_, likes)| *likes as f64).collect();
    let n_samples = likes_data.len();
    let n_features = 1;

    //Convert data into a 2D array
    let arr = Array2::<f64>::from_shape_vec((n_samples, n_features), likes_data)?;

    //Kmeans clustering
    let k = 4;
    let max_iters = 100;
    let tolerance = 0.0001;

    //Initialize centroids starting w/random points
    let mut centroids = initialize_centroids(&arr, k);
    let mut labels = vec![0; n_samples];

    //Perform kmeans iterations
    for _ in 0..max_iters {
        // Assign each point to the closest centroid
        for (i, point) in arr.axis_iter(ndarray::Axis(0)).enumerate() {
            let point_owned = point.to_owned();
            labels[i] = find_closest_centroid(&point_owned, &centroids);
        }

        //Recompute centroids
        let new_centroids = recompute_centroids(&arr, &labels, k);

        //Check for convergence
        if has_converged(&centroids, &new_centroids, tolerance) {
            // Stop if the centroids have converged
            break;
        }

        //Update centroids
        centroids = new_centroids;
    }

    //Map each news to cluster and print results
    for (i, (news_source, likes)) in data.into_iter().enumerate() {
        let cluster = labels[i];
        println!("News Source: {}, Likes: {}, Cluster: {}", news_source, likes, cluster);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use std::fs::File;
    use csv::ReaderBuilder;
    use std::collections::HashMap;

    //Test for clustering CNN
    #[test]
    fn test_cnn_clustering() -> Result<(), Box<dyn std::error::Error>> {
        //Simulate reading CNN
        let file_name_map: HashMap<&str, &str> = [
            ("/opt/app-root/src/FinalProject/src/cnn.csv", "CNN"),
        ]
        .iter()
        .cloned()
        .collect();

        let file_paths = vec![
            "/opt/app-root/src/FinalProject/src/cnn.csv",
        ];

        let mut data: Vec<(String, u32)> = Vec::new(); 

        //Read CSV file for CNN
        for path in file_paths {
            let file = File::open(path)?;
            let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

            //Read each record for CSV 
            for result in rdr.records() {
                let record = result?;
                let likes: u32 = record.get(1).unwrap().parse()?;

                if likes > 0 {
                    let news_source = file_name_map.get(path).unwrap_or(&"Unknown").to_string();
                    //Store custom file source name and likes
                    data.push((news_source, likes)); 
                }
            }
        }

        //Prepare data for clustering, only likes
        let likes_data: Vec<f64> = data.iter().map(|(_, likes)| *likes as f64).collect();
        let n_samples = likes_data.len();
        let n_features = 1;

        //Convert into a 2D array
        let arr = Array2::<f64>::from_shape_vec((n_samples, n_features), likes_data)?;

        //Perform kmeans clustering
        let k = 4;
        let max_iters = 100;
        let tolerance = 0.0001;

        //Initialize centroids
        let mut centroids = initialize_centroids(&arr, k);
        let mut labels = vec![0; n_samples];

        //Perform kmeans iterations
        for _ in 0..max_iters {
            //Assign each point to closest centroid
            for (i, point) in arr.axis_iter(ndarray::Axis(0)).enumerate() {
                let point_owned = point.to_owned();
                labels[i] = find_closest_centroid(&point_owned, &centroids);
            }

            //Recompute centroids
            let new_centroids = recompute_centroids(&arr, &labels, k);

            //Check for convergence
            if has_converged(&centroids, &new_centroids, tolerance) {
                break;
            }

            //Update centroids
            centroids = new_centroids;
        }

        //Verify CNN clustering
        for (i, (news_source, likes)) in data.into_iter().enumerate() {
            let cluster = labels[i];
            assert_eq!(news_source, "CNN");
            //Check likes > 0
            assert!(likes > 0);
            // Check cluster validity
            assert!(cluster < k); 
        }

        Ok(())
    }

    //Test for clustering BBC
    #[test]
    fn test_bbc_clustering() -> Result<(), Box<dyn std::error::Error>> {
        //Simulate reading BBC file
        let file_name_map: HashMap<&str, &str> = [
            ("/opt/app-root/src/FinalProject/src/bbc.csv", "BBC"),
        ]
        .iter()
        .cloned()
        .collect();

        let file_paths = vec![
            "/opt/app-root/src/FinalProject/src/bbc.csv",
        ];

        let mut data: Vec<(String, u32)> = Vec::new(); 

        //Read BBC CSV
        for path in file_paths {
            let file = File::open(path)?;
            let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

            //Read each record from CSV
            for result in rdr.records() {
                let record = result?;
                let likes: u32 = record.get(1).unwrap().parse()?;

                if likes > 0 {
                    let news_source = file_name_map.get(path).unwrap_or(&"Unknown").to_string();
                    //Store custom file source name and likes
                    data.push((news_source, likes)); 
                }
            }
        }

        //Prep data
        let likes_data: Vec<f64> = data.iter().map(|(_, likes)| *likes as f64).collect();
        let n_samples = likes_data.len();
        let n_features = 1;

        //Convert into a 2D array
        let arr = Array2::<f64>::from_shape_vec((n_samples, n_features), likes_data)?;

        //Perform kmeans clustering
        let k = 4;
        let max_iters = 100;
        let tolerance = 0.0001;

        //Initialize centroids
        let mut centroids = initialize_centroids(&arr, k);
        let mut labels = vec![0; n_samples];

        //Perform kmeans iterations
        for _ in 0..max_iters {
            //Assign each point to closest centroid
            for (i, point) in arr.axis_iter(ndarray::Axis(0)).enumerate() {
                let point_owned = point.to_owned();
                labels[i] = find_closest_centroid(&point_owned, &centroids);
            }

            //Recompute centroids
            let new_centroids = recompute_centroids(&arr, &labels, k);

            //Check for convergence (if centroids don't change)
            if has_converged(&centroids, &new_centroids, tolerance) {
                break;
            }

            //Update centroids
            centroids = new_centroids;
        }

        //Verify BBC clustering
        for (i, (news_source, likes)) in data.into_iter().enumerate() {
            let cluster = labels[i];
            assert_eq!(news_source, "BBC");
            assert!(likes > 0); 
            assert!(cluster < k); 
        }

        Ok(())
    }
}