use rayon::prelude::*;

use statrs::distribution::Poisson;

use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::distributions::WeightedIndex;
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::seq::SliceRandom;
use rand::{Rng};

use ndarray::Zip;
use ndarray::{s, Array1, Array2, Axis};
use std::f64::MIN_POSITIVE;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::fs::read_to_string;
use std::path::Path;
use std::collections::HashSet;
use std::io;

use crate::distances::*;
use logsumexp::LogSumExp;

fn average(numbers: &[f64]) -> f64 {
    numbers.iter().sum::<f64>() as f64 / numbers.len() as f64
}

pub fn standard_deviation(values: &[f64]) -> (f64, f64) {
    let mean = average(values);

    let sum_of_squares: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum();

    let variance = sum_of_squares / (values.len() as f64);
    (variance.sqrt(), mean)
}


pub fn read_pangenome_matrix(matrix: &str) -> Result<(Array1<u8>, Array1<u8>, Array2<u8>), Box<dyn std::error::Error>>{
    let content = read_to_string(matrix).unwrap();

    // Parse whole matrix as Vec<Vec<u8>>
    let matrix: Vec<Vec<u8>> = content
        .lines()
        .map(|line| {
            line.split('\t')
                .map(|x| x.parse::<u8>().expect("Expected binary 0 or 1"))
                .collect()
        })
        .collect();

    let nrows = matrix.len();
    let ncols = matrix[0].len();

    // Flatten matrix and create ndarray
    let flat: Vec<u8> = matrix.into_iter().flatten().collect();
    let full_array = Array2::from_shape_vec((nrows, ncols), flat)?;

    // --- Extract vaccine types (row 0, skip column 0) ---
    let vaccine_types: Array1<u8> = full_array.slice(s![0, 1..]).to_owned();

    // --- Extract NFDS (column 0, skip row 0) ---
    let under_nfds: Array1<u8> = full_array.slice(s![1.., 0]).to_owned();

    // --- Extract presence matrix (skip row 0 and column 0) ---
    let presence_matrix: Array2<u8> = full_array.slice(s![1.., 1..]).to_owned();

    // --- Output ---
    println!("Vaccine types: {:?}", vaccine_types);
    println!("Under NFDS:    {:?}", under_nfds);
    println!("Presence matrix:\n{:?}", presence_matrix);

    Ok((vaccine_types, under_nfds, presence_matrix))
}

fn average_sum_per_row(matrix: &Array2<u8>) -> f64 {
    let total_sum: usize = matrix
        .rows()
        .into_iter()
        .map(|row| row.sum() as usize)
        .sum();

    let nrows = matrix.nrows();

    total_sum as f64 / nrows as f64
}

pub struct Population {
    presence_matrix: Array2<u8>,
    vaccine_types: Array1<u8>,
    under_nfds: Array1<u8>,
    avg_gene_freq: f64,
    equilibrium_freq: Vec<f64>, // equilibrium frequency for each gene under NFDS
    nfds_weight: f64,           // strength of NFDS effect
}

// stacks vector of arrays into 2D array
fn to_array2<T: Copy>(source: Vec<Array1<T>>) -> Result<Array2<T>, impl std::error::Error> {
    let width = source.len();
    let flattened: Array1<T> = source.into_iter().flat_map(|row| row.to_vec()).collect();
    let height = flattened.len() / width;
    flattened.into_shape((width, height))
}

impl Population {
    /// Compute per-genome fitness under NFDS: for each gene under NFDS, compute its frequency in the population.
    /// For each genome, fitness is the sum of (1 - gene frequency) for present genes under NFDS.
    /// Returns a fitness vector (Vec<f64>), one entry per genome (row).
    pub fn compute_nfds_fitness(&self) -> Vec<f64> {
        let n_genomes = self.presence_matrix.nrows();
        let n_genes = self.presence_matrix.ncols();
        // under_nfds is a column vector: 1 if gene is under NFDS, 0 otherwise
        let under_nfds = self.under_nfds.iter().collect::<Vec<&u8>>();
        // Compute frequencies for each gene (column)
        let mut gene_freq = vec![0.0; n_genes];
        for j in 0..n_genes {
            let mut count = 0.0;
            for i in 0..n_genomes {
                count += self.presence_matrix[[i, j]] as f64;
            }
            gene_freq[j] = count / n_genomes as f64;
        }
        // Compute fitness for each genome
        let mut fitness = vec![0.0; n_genomes];
        for i in 0..n_genomes {
            let mut fit = 0.0;
            for j in 0..n_genes {
                if *under_nfds[j] == 1 && self.presence_matrix[[i, j]] == 1 {
                    // Apply NFDS effect: push toward equilibrium frequency
                    fit += 1.0 - self.nfds_weight * (gene_freq[j] - self.equilibrium_freq[j]).abs();
                }
            }
            // Add a small constant to prevent zero fitness
            fitness[i] = fit + 1e-8;
        }
        fitness
    }

    pub fn pop_size(&self) -> usize {
        self.presence_matrix.nrows()
    }   

    pub fn calc_gene_freq(&mut self) -> f64 {
        // Calculate the proportion of 1s for each row
        let proportions: Vec<f64> = self
            .presence_matrix
            .axis_iter(Axis(0))
            .map(|row| {
                let sum: usize = row.iter().map(|&x| x as usize).sum();
                let count = row.len();
                sum as f64 / count as f64
            })
            .collect();

        //println!("proportions: {:?}", proportions);
        
        // Sum all the elements in the vector
        let sum: f64 = proportions.iter().sum();

        // Calculate the number of elements in the vector
        let count = proportions.len();

        // Calculate the average
        let average: f64 = sum as f64 / count as f64;

        average
    }

    pub fn new(
        presence_matrix: &Array2<u8>,
        vaccine_types: &Array1<u8>,
        under_nfds: &Array1<u8>,
        nfds_weight: f64,
    ) -> Self {
        let avg_gene_freq = average_sum_per_row(presence_matrix);
        let n_genomes = presence_matrix.nrows();
        let n_genes = presence_matrix.ncols();
        // Compute equilibrium frequency for each gene (initial frequency)
        let mut equilibrium_freq = vec![0.0; n_genes];
        for j in 0..n_genes {
            let mut count = 0.0;
            for i in 0..n_genomes {
                count += presence_matrix[[i, j]] as f64;
            }
            equilibrium_freq[j] = count / n_genomes as f64;
        }
        Self {
            presence_matrix: presence_matrix.clone(),
            vaccine_types: vaccine_types.clone(),
            under_nfds: under_nfds.clone(),
            avg_gene_freq,
            equilibrium_freq,
            nfds_weight,
        }
    }

    pub fn migration(&mut self, rng: &mut StdRng, migration: f64, original_population: &Array2<u8>) {
        let pop_size = self.presence_matrix.nrows();
        let poisson = Poisson::new(migration * pop_size as f64).unwrap();
        let n_migrants = poisson.sample(rng).round() as usize;
        let mut migrant_indices: Vec<usize> = (0..pop_size).collect();
        migrant_indices.shuffle(rng);

        // Replace genomes of migrants with random samples from the original population
        for &idx in migrant_indices.iter().take(n_migrants) {
            let orig_idx = rng.gen_range(0..original_population.nrows());
            for gene in 0..self.presence_matrix.ncols() {
                self.presence_matrix[[idx, gene]] = original_population[[orig_idx, gene]];
            }
        }
    }

    pub fn sample_indices(
        &mut self,
        rng: &mut StdRng,
        avg_gene_num: i32,
        avg_pairwise_dists: Vec<f64>,
        selection_coefficients: &Vec<f64>,
        verbose: bool,
        no_control_genome_size: bool,
        genome_size_penalty: f64,
        competition_strength: f64
    ) -> Vec<usize> {
        // Calculate the proportion of 1s for each row
        let num_genes: Vec<i32> = self
            .presence_matrix
            .axis_iter(Axis(0))
            .map(|row| {
                let sum: i32 = row.iter().map(|&x| x as i32).sum();
                sum
                //let count = row.len();
                //sum as f64 / count as f64
            })
            .collect();

        let mut selection_weights: Vec<f64> = vec![1.0; self.presence_matrix.nrows()];

        // ensure accessory genome present
        if self.presence_matrix.ncols() > 0 {
            
            // TODO generate log sum value for entire row, then do logsumexp across whole selection weights array
            selection_weights = self
                .presence_matrix
                .axis_iter(Axis(0))
                .map(|row| {
                    let log_values: Vec<f64> = row
                        .iter()
                        .enumerate()
                        .map(|(col_idx, &col_val)| (1.0 + selection_coefficients[col_idx] * col_val as f64).ln())
                        .collect();

                    let neg_inf = log_values.contains(&std::f64::NEG_INFINITY);

                    //let log_mean = log_values.into_iter().map(|x| x).ln_sum_exp() - (row.len() as f64).ln();
                    let mut log_sum = 0.0;
                    if neg_inf == false {
                        log_sum = log_values.iter().sum();
                    }

                    log_sum
                })
                .collect();

            // TODO: work out why when prop_positive = 0, genome size still increases (should favour reduction in genome size)
            let logsumexp_value = selection_weights.iter().ln_sum_exp();

            //println!("logsumexp_value: {:?}", logsumexp_value);

            //println!("raw_selection_weights: {:?}", selection_weights);

            // Exponentiate and normalize
            selection_weights = selection_weights.into_iter()
                .map(|x| (x - logsumexp_value).exp()) // exp(log(w) - logsumexp)
                .collect();

            //println!("pre_norm_selection_weights: {:?}", selection_weights);

            let sum_weights: f64 = selection_weights.iter().sum();
            //println!("sum_weights: {:?}", sum_weights);
            selection_weights = selection_weights.iter().map(|&w| if w != std::f64::NEG_INFINITY {w / sum_weights} else {0.0}).collect();
        }

        // Convert differences to weights (lower difference should have higher weight)
        //println!("raw_weights: {:?}", selection_weights);
        let mut weights : Vec<f64>;
        if no_control_genome_size == false {
            // Calculate the differences from avg_gene_freq
            let differences: Vec<i32> = num_genes
                .iter()
                .map(|&n_genes| (n_genes - avg_gene_num).abs())
                .collect();

            //println!("differences: {:?}", differences);

            weights = differences
                .iter()
                .enumerate()
                .map(|(row_idx, &diff)| genome_size_penalty.powi(diff) * selection_weights[row_idx]) // based on https://pmc.ncbi.nlm.nih.gov/articles/instance/5320679/bin/mgen-01-38-s001.pdf
                .collect();
        } else {
            weights = selection_weights.clone();
        }

        //println!("post_genome_size_weights: {:?}", weights);
        // update weights with average pairwise distance
        for i in 0..weights.len() {
            let scaled_distance = avg_pairwise_dists[i] * (1.0 / competition_strength);
            let exponent = weights[i].powf(scaled_distance);
            weights[i] = exponent;
        }

        // println!("avg_pairwise_dists: {:?}", avg_pairwise_dists);
        // println!("post_pairwise_weights: {:?}", weights);
        // let mean_avg_pairwise_dists = average(&avg_pairwise_dists);
        // println!("mean_avg_pairwise_dists: {:?}", mean_avg_pairwise_dists);

        // determine whether weights is only 0s
        let max_final_weights = weights.iter().cloned().fold(-1./0. /* -inf */, f64::max);

        // account for only zeros
        if max_final_weights == 0.0 {
            weights = vec![1.0; self.presence_matrix.nrows()];
        }

        // Create a WeightedIndex distribution based on weights
        let dist = WeightedIndex::new(&weights).unwrap();

        // Sample rows based on the distribution
        let sampled_indices: Vec<usize> = (0..self.presence_matrix.nrows()).map(|_| dist.sample(rng)).collect();

        //println!("sampled_indices: {:?}", sampled_indices);

        sampled_indices
    }

    pub fn next_generation(&mut self, sample: &Vec<usize>) 
    {
        let nrows = sample.len();
        let ncols = self.presence_matrix.ncols();
    
        // Pre-allocate the new population array
        let mut next_pop = Array2::zeros((nrows, ncols));
    
        // Fill in each row directly
        for (row_idx, &sample_idx) in sample.iter().enumerate() {
            let source_row = self.presence_matrix.slice(s![sample_idx, ..]);
            let mut target_row = next_pop.slice_mut(s![row_idx, ..]);
            target_row.assign(&source_row);
        }
    
        self.presence_matrix = next_pop;
    }

    pub fn mutate_alleles(
        &mut self,
        mutations: f64,
    ) {
        // avoid rate parameter of 0
        if mutations == 0.0 {
            return;
        }
        
        let poisson = Poisson::new(mutations).unwrap();

        self.presence_matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                // thread-specific random number generator
                let mut thread_rng = rand::thread_rng();
                //let thread_index = rayon::current_thread_index();
                //print!("{:?} ", thread_index);

                // sample from Poisson distribution for number of sites to mutate in this isolate
                let n_sites = thread_rng.sample(poisson) as usize;

                // iterate for number of mutations required to reach mutation rate
                for _ in 0..n_sites {
                    // sample new site to mutate
                    let mutant_site = thread_rng.gen_range(0..row.len());

                    // Flip the value (binary: 0 <-> 1)
                    row[mutant_site] = 1 - row[mutant_site];
                }
            });
    }

    // TODO update vector in place
    pub fn pairwise_distances(
        &mut self,
        max_distances: usize,
        range1: &Vec<usize>,
        range2: &Vec<usize>,
    ) -> Vec<f64> {
        //let (contiguous_array, matches) = get_variable_loci(self.core, &self.pop);

        //let mut idx = 0;
        let range = 0..max_distances;
        let distances: Vec<_> = range
            .into_par_iter()
            .map(|current_index| {
                let i = range1[current_index];
                let j = range2[current_index];

                let row1 = self.presence_matrix.index_axis(Axis(0), i);
                let row2 = self.presence_matrix.index_axis(Axis(0), j);
                let row1_slice = row1.as_slice().unwrap();
                let row2_slice = row2.as_slice().unwrap();
                //println!("i: {:?}", i);
                //println!("j: {:?}", j);

                //println!("rowi: {:?}", row1);
                //println!("rowj: {:?}", row2);

                let mut _final_distance: f64 = 0.0;

    
                let (intersection, union) = jaccard_distance_fast(row1_slice, row2_slice);
                // let (intersection_test, union_test) = jaccard_distance_test(row1_slice, row2_slice);
                // println!("intersection_test: {:?} intersection: {:?}", intersection_test, intersection);
                // println!("union_test: {:?} union: {:?}", union_test, union);
                _final_distance = 1.0
                    - ((intersection as f64)
                        / (union as f64));
            
                //println!("_final_distance: {:?}", _final_distance);
                _final_distance
            })
            .collect();
        distances
    }

    pub fn write(&mut self, outpref: &str) -> io::Result<()>
    {

        // Open output file
        let mut output_file = outpref.to_owned();
        let extension: &str = "_pangenome.csv";
        output_file.push_str(extension);

        let mut file = File::create(output_file)?;

        // Iterate rows and write
        for row in self.presence_matrix.outer_iter() {
            let line: Vec<String> = row.iter().map(|&x| x.to_string()).collect();
            writeln!(file, "{}", line.join(","))?;
        }
        Ok(())
    }
}