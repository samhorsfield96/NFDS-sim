use rand::prelude::*;
use rand::rngs::StdRng;
use rand::distributions::Uniform;
use statrs::distribution::Poisson;

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
        .arg(
            Arg::new("matrix")
                .long("matrix")
                .help("Input csv file of gene presence/absence.")
                .takes_value(true)
                .required(true)
        )
        .arg(
            Arg::new("n_gen")
                .long("n_gen")
                .help("Number of generations to simulate.")
                .required(false)
                .default_value("100")
        )
        .arg(
            Arg::new("rate_genes")
                .long("rate_genes")
                .help("Average number of accessory pangenome that mutates per generation in gene. Must be >= 0.0")
                .required(false)
                .default_value("1.0")
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .help("Seed for random number generation.")
                .required(false)
                .default_value("0")
        )
        .arg(
            Arg::new("outpref")
                .long("outpref")
                .help("Output prefix path.")
                .required(false)
                .default_value("distances")
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .help("Number of threads.")
                .required(false)
                .default_value("1")
        )
        .arg(
            Arg::new("verbose")
                .long("verbose")
                .help("Prints generation and time to completion")
                .required(false)
                .takes_value(false)
        )
        .arg(
            Arg::new("nfds_weight")
                .long("nfds-weight")
                .help("Strength of NFDS effect (default: 1.0)")
                .required(false)
                .default_value("0.2")
        )
        .arg(
            Arg::new("migration")
                .long("migration")
                .help("Strength of migration effect (default: 0.01)")
                .required(false)
                .default_value("0.01")
        )
        .arg(
            Arg::new("vaccine_gen")
                .long("vaccine-gen")
                .help("Generation at which vaccine is introduced (default: never)")
                .required(false)
                .default_value("-1")
        )
        .arg(
            Arg::new("vaccine_strength")
                .long("vaccine-strength")
                .help("Fitness penalty for vaccine-type genomes after vaccine introduction (default: 0.0)")
                .required(false)
                .default_value("0.0")
        )
        .arg(
            Arg::new("max_distances")
            .long("max_distances")
            .help("Maximum number of pairwise distances to calculate.")
            .required(false).default_value("100000")
        )
        .get_matches();

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
    let max_distances: usize = matches.value_of_t("max_distances").unwrap();

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
    let pop_size = pan_genome.pop_size();

    // Save the original population for migration source
    let original_population = presence_matrix.clone();

    let range1: Vec<usize> = (0..max_distances)
        .map(|_| rng.gen_range(0..pop_size))
    .collect();
    let mut range2: Vec<usize> = vec![0; max_distances];

    let mut avg_acc_dist = vec![0.0; n_gen as usize];
    let mut std_acc_dist = vec![0.0; n_gen as usize];

    for j in 0..n_gen {
        // 1. mutate generations
        println!("Mutating generation {}", j + 1);
        pan_genome.print_pop();
        pan_genome.mutate_alleles(rate_genes);
        println!("Mutated generation {}", j + 1);
        pan_genome.print_pop();

        // 2. Compute NFDS fitness vector
        let mut fitness = pan_genome.compute_nfds_fitness();
        println!("Computed NFDS fitness vector");
        println!("Fitness vector: {:?}", fitness);

        // 3. Apply vaccine penalty after vaccine introduction
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
        
        // 4. Wright-Fisher sampling: sample next generation with fitness+migration
        let weights = &fitness;
        println!("Computed Wright-Fisher sampling");    
        println!("Weights: {:?}", weights);
        let dist = rand::distributions::WeightedIndex::new(weights).unwrap();
        let sampled_individuals: Vec<usize> = (0..pop_size).map(|_| dist.sample(&mut rng)).collect();
        println!("Sampled individuals: {:?}", sampled_individuals);

        pan_genome.next_generation(&sampled_individuals);
        println!("Next generation");
        pan_genome.print_pop();

        // 5. Migration: replace a fraction of individuals with random genomes
        pan_genome.migration(&mut rng, migration, &original_population);
        println!("Migration");
        pan_genome.print_pop();

        // 6. get average distances
        let acc_distances = pan_genome.pairwise_distances(max_distances, &range1, &range2);

        let mut std_acc = 0.0;
        let mut avg_acc = 0.0;
        (std_acc, avg_acc) = standard_deviation(&acc_distances);

        avg_acc_dist[j as usize] = avg_acc;

        std_acc_dist[j as usize] = std_acc;

        //let elapsed = now_gen.elapsed();
        if verbose {
            println!("Finished gen: {}", j + 1);
            let avg_gene_freq = pan_genome.calc_gene_freq();
            println!("avg_gene_freq: {}", avg_gene_freq);
        }
        //println!("Elapsed: {:.2?}", elapsed);
    }

    // print per generation distances
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

    pan_genome.write(outpref);

    //let elapsed = now.elapsed();

    // if verbose {
    //     println!("Total elapsed: {:.2?}", elapsed);
    // }

    // println!("Total elapsed: {:.2?}", elapsed);

    return Ok(());
}
