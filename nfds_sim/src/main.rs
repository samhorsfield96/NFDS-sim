use rand::prelude::*;
use rand::rngs::StdRng;
use rand::distributions::Uniform;

use ndarray::{Array1, Array2};

use nfds_sim::populations::*;

use clap::{Arg, Command};

use std::fs::File;
use std::io::{self, Write};
use std::vec;

fn main() -> io::Result<()> {
    // Define the command-line arguments using clap
    let matches = Command::new("nfds-sim")
    .version("0.0.1")
    .author("Samuel Horsfield shorsfield@ebi.ac.uk")
    .about("Runs Wright-Fisher simulation, simulating NFDS.")
    .arg(Arg::new("matrix")
        .long("matrix")
        .help("Input csv file of gene presence/absence.")
        .required(true)
    .arg(Arg::new("n_gen")
        .long("n_gen")
        .help("Number of generations to simulate.")
        .required(false)
        .default_value("100"))
    .arg(Arg::new("rate_genes")
        .long("rate_genes")
        .help("Average number of accessory pangenome that mutates per generation in gene. Must be >= 0.0")
        .required(false)
        .default_value("1.0"))
    .arg(Arg::new("seed")
        .long("seed")
        .help("Seed for random number generation.")
        .required(false)
        .default_value("0"))
    .arg(Arg::new("outpref")
        .long("outpref")
        .help("Output prefix path.")
        .required(false)
        .default_value("distances"))
    .arg(Arg::new("threads")
        .long("threads")
        .help("Number of threads.")
        .required(false)
        .default_value("1"))
    .arg(Arg::new("verbose")
        .long("verbose")
        .help("Prints generation and time to completion")
        .required(false)
        .takes_value(false))
    .arg(Arg::new("genome_size_penalty")
        .long("genome_size_penalty")
        .help("Multiplier for each gene difference between avg_gene_freq and observed value. Default = 0.99")
        .required(false)
        .default_value("0.99"))
    .arg(Arg::new("nfds_weight")
        .long("nfds-weight")
        .help("Strength of NFDS effect (default: 1.0)")
        .required(false)
        .default_value("0.2"))
    .arg(Arg::new("migration")
        .long("migration")
        .help("Strength of migration effect (default: 0.01)")
        .required(false)
        .default_value("0.01"))
    .arg(Arg::new("vaccine_gen")
        .long("vaccine-gen")
        .help("Generation at which vaccine is introduced (default: never)")
        .required(false)
        .default_value("-1"))
    .arg(Arg::new("vaccine_strength")
        .long("vaccine-strength")
        .help("Fitness penalty for vaccine-type genomes after vaccine introduction (default: 0.0)")
        .required(false)
        .default_value("0.0"))
    .get_matches());

    // Set the argument to a variable
    let matrix: String = matches.value_of_t("matrix").unwrap();
    let n_gen: i32 = matches.value_of_t("n_gen").unwrap();
    let outpref = matches.value_of("outpref").unwrap_or("distances");
    let mut n_threads: usize = matches.value_of_t("threads").unwrap();
    let verbose = matches.is_present("verbose");
    let seed: u64 = matches.value_of_t("seed").unwrap();
    let nfds_weight: f64 = matches.value_of_t("nfds_weight").unwrap();
    let migration: f64 = matches.value_of_t("migration").unwrap();
    let rate_genes: f64 = matches.value_of_t("rate_genes").unwrap();
    let vaccine_gen: i32 = matches.value_of_t("vaccine_gen").unwrap();
    let vaccine_strength: f64 = matches.value_of_t("vaccine_strength").unwrap();

    if n_threads < 1 {
        n_threads = 1;
    }

    // enable multithreading
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .unwrap();

    let mut rng: StdRng = StdRng::seed_from_u64(seed);

    // Read matrix and extract relevant arrays
    let (vaccine_types, under_nfds, presence_matrix) = read_pangenome_matrix(&matrix).unwrap();
    let mut pan_genome = Population::new(&presence_matrix, &vaccine_types, &under_nfds, nfds_weight); // pangenome alignment
    // Save the original population for migration source
    let original_population = presence_matrix.clone();

    for j in 0..n_gen {
        // 1. mutate generations
        pan_genome.mutate_alleles(rate_genes);

        // Run for n_gen generations
        // 1. Compute NFDS fitness vector
        let mut fitness = pan_genome.compute_nfds_fitness();
        // Apply vaccine penalty after vaccine introduction
        if vaccine_gen >= 0 && j >= vaccine_gen {
            // vaccine_types is a 1-row Array2<u8>, so use [0, idx] or flatten
            let vaccine_types_vec: Vec<u8> = vaccine_types.iter().cloned().collect();
            for (idx, &is_vaccine_type) in vaccine_types_vec.iter().enumerate() {
                if is_vaccine_type == 1 {
                    fitness[idx] -= vaccine_strength;
                    if fitness[idx] < 0.0 {
                        fitness[idx] = 0.0; // Prevent negative fitness
                    }
                }
            }
        }

        // 2. Migration: replace a fraction of individuals with random genomes
        let pop_size = pan_genome.presence_matrix.nrows();
        use rand_distr::{Poisson, Distribution};
        let poisson = Poisson::new(migration * pop_size as f64).unwrap();
        let n_migrants = poisson.sample(&mut rng).round() as usize;
        let mut migrant_indices: Vec<usize> = (0..pop_size).collect();
        migrant_indices.shuffle(&mut rng);
        // For migrants, set fitness to 1.0 (neutral sampling)
        let mut fitness_with_migration = fitness.clone();
        for &idx in migrant_indices.iter().take(n_migrants) {
            fitness_with_migration[idx] = 1.0;
        }
        // Replace genomes of migrants with random samples from the original population
        for &idx in migrant_indices.iter().take(n_migrants) {
            let orig_idx = rng.gen_range(0..original_population.nrows());
            for gene in 0..pan_genome.presence_matrix.ncols() {
                pan_genome.presence_matrix[[idx, gene]] = original_population[[orig_idx, gene]];
            }
        }

        // 3. Wright-Fisher sampling: sample next generation with fitness+migration
        let weights = &fitness_with_migration;
        let dist = rand::distributions::WeightedIndex::new(weights).unwrap();
        let sampled_individuals: Vec<usize> = (0..pop_size).map(|_| dist.sample(&mut rng)).collect();

        pan_genome.next_generation(&sampled_individuals);

        // mutate core genome
        pan_genome.mutate_alleles(&n_pan_mutations, &pan_weighted_dist);
        
        // if at final generation, sample
        if j == n_gen -1 {

            // else calculate hamming and jaccard distances
            let core_distances = core_genome.pairwise_distances(max_distances, &range1, &range2);
            let acc_distances = pan_genome.pairwise_distances(max_distances, &range1, &range2);

            let mut output_file = outpref.to_owned();
            let extension: &str = ".tsv";
            output_file.push_str(extension);
            let mut file = File::create(output_file)?;

            // Iterate through the vectors and write each pair to the file
            for (core, acc) in core_distances.iter().zip(acc_distances.iter()) {
                writeln!(file, "{}\t{}", core, acc);
            }
        }

        // get average distances
        if print_dist {
            let acc_distances = pan_genome.pairwise_distances(max_distances, &range1, &range2);

            let mut std_acc = 0.0;
            let mut avg_acc = 0.0;
            (std_acc, avg_acc) = standard_deviation(&acc_distances);

            avg_acc_dist[j as usize] = avg_acc;

            std_acc_dist[j as usize] = std_acc;
        }

        //let elapsed = now_gen.elapsed();
        if verbose {
            println!("Finished gen: {}", j + 1);
            let avg_gene_freq = pan_genome.calc_gene_freq();
            println!("avg_gene_freq: {}", avg_gene_freq);
        }
        //println!("Elapsed: {:.2?}", elapsed);
    }

    // print per generation distances
    if print_dist {
        let mut output_file = outpref.to_owned();
        let extension: &str = "_per_gen.tsv";
        output_file.push_str(extension);

        let mut file = File::create(output_file)?;

        // Iterate through the vectors and write each pair to the file
        for (avg_acc, std_acc) in avg_acc_dist
            .iter()
            .zip(std_acc_dist.iter())
            .map(|(w, x)| (w, x))
        {
            writeln!(file, "{}\t{}", avg_acc, std_acc);
        }
    }

    if print_matrices {
        pan_genome.write(outpref);
    }

    //let elapsed = now.elapsed();

    // if verbose {
    //     println!("Total elapsed: {:.2?}", elapsed);
    // }

    // println!("Total elapsed: {:.2?}", elapsed);

    return Ok(());
}
