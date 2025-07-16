use rand_distr::Beta;

//use rand_distr::Distribution;
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
use std::f64::MAX;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use logsumexp::LogSumExp;
use crate::distances::*;
use safe_arch::*;

use std::fs::File;
use std::io::{self, Write};

use std::usize;

use std::error::Error;
use std::fs::read_to_string;

pub fn read_pangenome_matrix(matrix: &str) -> (Array2<u8>, Array2<u8>, &Array2<u8>){
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
    let vaccine_types: ndarray::ArrayBase<ndarray::OwnedRepr<u8>, ndarray::Dim<[usize; 2]>> = full_array.slice(s![0, 1..]).to_owned();

    // --- Extract NFDS (column 0, skip row 0) ---
    let under_nfds: ndarray::ArrayBase<ndarray::OwnedRepr<u8>, ndarray::Dim<[usize; 2]>> = full_array.slice(s![1.., 0]).to_owned();

    // --- Extract presence matrix (skip row 0 and column 0) ---
    let presence_matrix: ndarray::ArrayBase<ndarray::OwnedRepr<u8>, ndarray::Dim<[usize; _]>> = full_array.slice(s![1.., 1..]).to_owned();

    // --- Output ---
    println!("Vaccine types: {:?}", vaccine_types);
    println!("Under NFDS:    {:?}", under_nfds);
    println!("Presence matrix:\n{:?}", presence_matrix);

    (vaccine_types, under_nfds, presence_matrix)
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
    vaccine_types: Array2<u8>,
    under_nfds: Array2<u8>,
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
    pub fn new(
        presence_matrix: &Array2<u8>,
        vaccine_types: &Array2<u8>,
        under_nfds: &Array2<u8>,
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
            .pop
            .axis_iter(Axis(0))
            .map(|row| {
                let sum: i32 = row.iter().map(|&x| x as i32).sum();
                sum
                //let count = row.len();
                //sum as f64 / count as f64
            })
            .collect();

        let mut selection_weights: Vec<f64> = vec![1.0; self.pop.nrows()];

        // ensure accessory genome present
        if self.pop.ncols() > 0 {
            
            // TODO generate log sum value for entire row, then do logsumexp across whole selection weights array
            selection_weights = self
                .pop
                .axis_iter(Axis(0))
                .map(|row| {
                    let log_values: Vec<f64> = row
                        .iter()
                        .enumerate()
                        .map(|(col_idx, &col_val)| (1.0 + selection_coefficients[col_idx] * col_val as f64).ln())
                        .collect();

                    //println!("log_values: {:?}", log_values);
                    
                    //println!("log_values: {:?}", log_values);
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
            let exponent = safe_pow(weights[i], scaled_distance);
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
            weights = vec![1.0; self.pop.nrows()];
        }

        // Create a WeightedIndex distribution based on weights
        let dist = WeightedIndex::new(&weights).unwrap();

        // Sample rows based on the distribution
        let sampled_indices: Vec<usize> = (0..self.pop.nrows()).map(|_| dist.sample(rng)).collect();

        //println!("sampled_indices: {:?}", sampled_indices);

        sampled_indices
    }

    pub fn next_generation(&mut self, sample: &Vec<usize>) {
        let nrows = sample.len();
        let ncols = self.pop.ncols();
    
        // Pre-allocate the new population array
        let mut next_pop = Array2::zeros((nrows, ncols));
    
        // Fill in each row directly
        for (row_idx, &sample_idx) in sample.iter().enumerate() {
            let source_row = self.pop.slice(s![sample_idx, ..]);
            let mut target_row = next_pop.slice_mut(s![row_idx, ..]);
            target_row.assign(&source_row);
        }
    
        self.pop = next_pop;
    }

    pub fn mutate_alleles(
        &mut self,
        mutations_vec: &Vec<f64>,
        weighted_dist: &Vec<WeightedIndex<f32>>,
    ) {
        // index for random number generation
        let _index = AtomicUsize::new(0);
        let _update_rng = AtomicUsize::new(0);

        for site_idx in 0..mutations_vec.len() {
            let mutations = mutations_vec[site_idx];
            
            // avoid rate parameter of 0
            if mutations == 0.0 {
                continue;
            }
            
            let poisson = Poisson::new(mutations).unwrap();

            //println!("mutations {:?} site_idx {:?}", mutations, site_idx);
            self.pop
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
                        let mutant_site = weighted_dist[site_idx].sample(&mut thread_rng);
                        //println!("mutant_site {:?} site_idx {:?} ", mutant_site, site_idx);

                        // get possible values to mutate to, must be different from current value
                        let value = row[mutant_site];
                        let values = &self.core_vec[1 >> value];

                        // sample new allele
                        let new_allele = values.iter().choose_multiple(&mut thread_rng, 1)[0];

                        // set value in place
                        row[mutant_site] = *new_allele;
                    }
                });
        }
    }

    pub fn recombine(
        &mut self,
        recombinations_vec: &Vec<f64>,
        rng: &mut StdRng,
        locus_weights: &Vec<Vec<f32>>,
    ) {
        // index for random number generation
        let _index = AtomicUsize::new(0);
        let _update_rng = AtomicUsize::new(0);

        for site_idx in 0..recombinations_vec.len() {
            let n_recombinations = recombinations_vec[site_idx];

            // avoid rate parameter of 0
            if n_recombinations == 0.0 {
                continue;
            }

            let poisson_recomb = Poisson::new(n_recombinations).unwrap();

            // Preallocate results vector with one entry per row
            //let mut loci: Vec<Vec<usize>> = vec![Vec::new(); self.pop.nrows()];
            let loci = Arc::new(RwLock::new(vec![Vec::new(); self.pop.nrows()]));
            let values = Arc::new(RwLock::new(vec![Vec::new(); self.pop.nrows()]));
            let recipients = Arc::new(RwLock::new(vec![Vec::new(); self.pop.nrows()]));

            // let contiguous_array:ndarray::ArrayBase<ndarray::OwnedRepr<u8>, ndarray::Dim<[usize; 2]>>;
            // let matches:f64;

            // // get mutation matrix
            // match pangenome_matrix {
            //     Some(matrix) => {
            //         (contiguous_array, matches) =  get_variable_loci(false, &matrix);
            //     }
            //     None => {
            //         (contiguous_array, matches) =  get_variable_loci(false, &self.pop);
            //     }
            // }

            // recipient distribution, minus one to avoid comparison with self
            let dist: Uniform<usize> = Uniform::new(0, self.pop.nrows() - 1);

            // for each genome, determine which positions are being transferred
            self.pop
                .axis_iter(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(row_idx, row)| {
                    //use std::time::Instant;
                    //let now = Instant::now();

                    // thread-specific random number generator
                    let mut thread_rng = rand::thread_rng();

                    // sample from Poisson distribution for number of sites to mutate in this isolate
                    let n_sites = poisson_recomb.sample(&mut thread_rng) as usize;

                    // get sampling weights for each pairwise comparison
                    // TODO remove this, jsut have same distance for all individuals
                    //let binding = get_distance(row_idx, self.pop.nrows(), self.core_genes, matches, false, &contiguous_array, self.pop.ncols());
                    //let mut elapsed = now.elapsed();

                    //println!("finished distances: {}, {:.2?}", row_idx, elapsed);
                    //let binding = vec![0.1; self.pop.nrows()];
                    //let i_distances = binding
                    //.iter().map(|i| {1.0 - i});
                    //let sample_dist = WeightedIndex::new(i_distances).unwrap();

                    //elapsed = now.elapsed();
                    //println!("finished sampling dist: {}, {:.2?}", row_idx, elapsed);

                    // Sample rows based on the distribution, adjusting as self comparison not conducted
                    let sampled_recipients: Vec<usize> = (0..n_sites)
                        .map(|_| dist.sample(&mut thread_rng))
                        .map(|value| value + (value >= row_idx) as usize)
                        .collect();
                    let mut sampled_loci: Vec<usize> = Vec::with_capacity(n_sites);

                    // for _ in 0..n_sites {
                    //     let value = sample_dist.sample(&mut thread_rng);
                    //     sampled_recipients.push(value + (value >= row_idx) as usize);
                    // }

                    //elapsed = now.elapsed();
                    //println!("finished sampling total: {}, {:.2?}", row_idx, elapsed);

                    //let sampled_recipients: Vec<usize> = vec![1, 7, 20, 705, 256];

                    let mut sampled_values: Vec<u8> = vec![1; n_sites];
                    // get non-zero indices
                    if self.core == false {
                        //if accessory, set elements with no genes to 0
                        let mut non_zero_weights: Vec<f32> = locus_weights[site_idx].clone();
                        let mut total_0: usize = 0;
                        for (idx, &val) in row.indexed_iter() {
                            let mut update: bool = false;
                            
                            // check for any zeroes in either vector
                            if non_zero_weights[idx] == 0.0 {
                                update = true;
                            }

                            if val == 0 {
                                non_zero_weights[idx] = 0.0;
                                update = true;
                            }

                            if update == true 
                            {
                                total_0 += 1;
                            }
                        }

                        // // get all sites to be recombined
                        // sampled_loci = (0..n_sites)
                        // .map(|_| *non_zero_indices.choose(&mut thread_rng).unwrap()) // Sample with replacement
                        // .collect(); 
                        
                        //let sum : f32 = non_zero_weights.clone().iter().sum();
                        // if sum == 0.0 {
                        //     println!("total_0: {:?}", total_0);
                        //     println!("non_zero_weights.len(): {:?}", non_zero_weights.len());
                            
                        //     println!("non_zero_weights.sum(): {:?}", sum);
                        // }


                        // ensure some non-zero values present
                        if total_0 < non_zero_weights.len() {
                            let locus_weighted_dist: WeightedIndex<f32> =
                                WeightedIndex::new(non_zero_weights).unwrap();

                            // iterate for number of mutations required to reach mutation rate, include deletions and insertions
                            sampled_loci = thread_rng
                                .sample_iter(locus_weighted_dist)
                                .take(n_sites)
                                .collect();
                        }
                    } else {
                        // sampled_loci = (0..n_sites)
                        // .map(|_| row.indexed_iter().map(|(idx, _)| idx).choose(&mut thread_rng).unwrap()) // Sample with replacement
                        // .collect();

                        sampled_loci = thread_rng
                            .sample_iter(rand::distributions::Uniform::new(0, self.pop.ncols()))
                            .take(n_sites)
                            .collect();

                        // assign site value from row
                        for site in 0..n_sites {
                            sampled_values[site] = row[sampled_loci[site]];
                        }
                    }

                    //elapsed = now.elapsed();
                    //println!("finished getting sites total: {}, {:.2?}", row_idx, elapsed);


                    // assign values
                    {
                        let mut mutex = loci.write().unwrap(); // Lock for writing
                        let entry = &mut mutex[row_idx]; // Now you can index safely
                        *entry = sampled_loci;
                    }
                    {
                        let mut mutex = values.write().unwrap(); // Lock for writing
                        let entry = &mut mutex[row_idx]; // Now you can index safely
                        *entry = sampled_values;
                    }
                    {
                        let mut mutex = recipients.write().unwrap(); // Lock for writing
                        let entry = &mut mutex[row_idx]; // Now you can index safely
                        *entry = sampled_recipients;
                    }

                    //elapsed = now.elapsed();
                    //println!("finished entering data: {}, {:.2?}", row_idx, elapsed);
                });

            // go through entries in loci, values and recipients, mutating the rows in each case
            // randomise order in which rows are moved through
            let mut row_indices: Vec<usize> = (0..self.pop.nrows()).collect();
            row_indices.shuffle(rng);

            for pop_idx in row_indices {
                // sample for given donor
                let sampled_loci: Vec<usize> = loci.write().unwrap()[pop_idx].to_vec();
                let sampled_recipients: Vec<usize> = recipients.write().unwrap()[pop_idx].to_vec();
                let sampled_values: Vec<u8> = values.write().unwrap()[pop_idx].to_vec();

                // println!("index: {}", pop_idx);
                // println!("sampled_loci: {:?}", sampled_loci);
                // println!("sampled_recipients: {:?}", sampled_recipients);
                // println!("sampled_values: {:?}", sampled_values);

                // update recipients in place if any recombinations allowed
                if sampled_loci.len() > 0 {
                    Zip::from(&sampled_recipients)
                        .and(&sampled_loci)
                        .and(&sampled_values)
                        .for_each(|&row_idx, &col_idx, &value| {
                            self.pop[[row_idx, col_idx]] = value;
                        });
                }
            }

        }
    }

    pub fn average_distance(&mut self) -> Vec<f64> {
        //let (contiguous_array, matches) = get_variable_loci(self.core, &self.pop);

        let range = 0..self.pop.nrows();
        let distances: Vec<f64> = range
            .into_par_iter()
            .map(|i| {
                let i_distances = get_distance(
                    i,
                    self.pop.nrows(),
                    self.core_genes,
                    0.0,
                    self.core,
                    &self.pop,
                    self.pop.ncols(),
                );

                let (sum, count) = i_distances.iter().fold((0.0, 0), |(s, c), &x| (s + x, c + 1));
                let mut _final_distance = sum / count as f64;

                // ensure no zero distances that may cause no selection of isolates.
                if _final_distance == 0.0 {
                    _final_distance = MIN_POSITIVE;
                }

                _final_distance
            })
            .collect();

        //println!("new distances: {:?}", distances);
        distances
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

                let row1 = self.pop.index_axis(Axis(0), i);
                let row2 = self.pop.index_axis(Axis(0), j);
                let row1_slice = row1.as_slice().unwrap();
                let row2_slice = row2.as_slice().unwrap();
                //println!("i: {:?}", i);
                //println!("j: {:?}", j);

                //println!("rowi: {:?}", row1);
                //println!("rowj: {:?}", row2);

                let mut _final_distance: f64 = 0.0;

                if self.core == true {
                    
                    let distance = hamming_bitwise_fast(row1_slice, row2_slice) / 2;
           
                    // let test_distance = row1_slice.to_vec().iter().zip(&row2_slice.to_vec()).filter(|&(a, b)| a != b).count();
                    // println!("distance: {:?} test_distance: {:?}", distance, test_distance);

                    _final_distance = distance as f64 / (self.pop.ncols() as f64);
                } else {
                    let (intersection, union) = jaccard_distance_fast(row1_slice, row2_slice);
                    // let (intersection_test, union_test) = jaccard_distance_test(row1_slice, row2_slice);
                    // println!("intersection_test: {:?} intersection: {:?}", intersection_test, intersection);
                    // println!("union_test: {:?} union: {:?}", union_test, union);
                    _final_distance = 1.0
                        - ((intersection as f64 + self.core_genes as f64)
                            / (union as f64 + self.core_genes as f64));
                }
                //println!("_final_distance: {:?}", _final_distance);
                _final_distance
            })
            .collect();
        distances
    }

    pub fn write(&mut self, outpref: &str) -> io::Result<()>
    {
        // core genome
        if self.core == true {
            // Open output file
            let mut output_file = outpref.to_owned();
            let extension: &str = "_core_genome.csv";
            output_file.push_str(extension);

            let mut file = File::create(output_file)?;

            // Iterate rows and write
            for row in self.pop.outer_iter() {
                let line: Vec<String> = row.iter().map(|&x| int_to_base(x).to_string()).collect();
                writeln!(file, "{}", line.join(","))?;
            }
        } else {
            // Open output file
            let mut output_file = outpref.to_owned();
            let extension: &str = "_pangenome.csv";
            output_file.push_str(extension);

            let mut file = File::create(output_file)?;

            // Iterate rows and write
            for row in self.pop.outer_iter() {
                let mut line: Vec<String> = vec![1_u8.to_string(); self.core_genes];
                line.extend(row.iter().map(|&x| x.to_string()));
                writeln!(file, "{}", line.join(","))?;
            }
        }
        Ok(())
    }
}