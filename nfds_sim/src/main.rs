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
        .long("rate_genes1")
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
    .get_matches());

    // Set the argument to a variable
    let matrix: str = matches.value_of_t("matrix").unwrap();
    let n_gen: i32 = matches.value_of_t("n_gen").unwrap();
    let outpref = matches.value_of("outpref").unwrap_or("distances");
    let mut n_threads: usize = matches.value_of_t("threads").unwrap();
    let verbose = matches.is_present("verbose");
    let seed: u64 = matches.value_of_t("seed").unwrap();
    
    // probability that existing 
    let migration: u64 = matches.value_of_t("seed").unwrap();

    // add vaccine strength, time at point when vaccine introduced

    if n_threads < 1 {
        n_threads = 1;
    }

    // enable multithreading
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .unwrap();

    let mut pan_weights: Vec<Vec<f32>> = vec![];
    let mut n_pan_mutations: Vec<f64> = vec![];
    let mut selection_weights: Vec<f64> = vec![0.0; pan_size];

    let mut rng: StdRng = StdRng::seed_from_u64(seed);

    let mut pan_genome = Population::new(presence_matrix); // pangenome alignment

    for j in 0..n_gen {
        // Run for n_gen generations
        //let now_gen = Instant::now();

        // sample new individuals
        //let sampled_individuals: Vec<usize> = (0..pop_size).map(|_| rng.gen_range(0..pop_size)).collect();
        let mut avg_pairwise_dists = vec![1.0; pop_size];

        // include competition
        if competition == true {
            avg_pairwise_dists = pan_genome.average_distance();
        }

        let sampled_individuals =
            pan_genome.sample_indices(&mut rng, avg_gene_num, avg_pairwise_dists, &selection_weights, verbose, no_control_genome_size, genome_size_penalty, competition_strength);
        
        core_genome.next_generation(&sampled_individuals);
        //println!("finished copying core genome {}", j);
        pan_genome.next_generation(&sampled_individuals);
        //println!("finished copying pangenome {}", j);

        // mutate core genome
        //println!("started {}", j);
        core_genome.mutate_alleles(&n_core_mutations, &core_weighted_dist);

        //println!("finished mutating core genome {}", j);
        pan_genome.mutate_alleles(&n_pan_mutations, &pan_weighted_dist);
        //println!("finished mutating pangenome {}", j);

        // recombine populations
        if HR_rate > 0.0 {
            core_genome.recombine(&n_recombinations_core, &mut rng, &core_weights);
        }
        if HGT_rate > 0.0 {
            pan_genome.recombine(&n_recombinations_pan, &mut rng, &pan_weights);
        }
        
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
            let core_distances = core_genome.pairwise_distances(max_distances, &range1, &range2);
            let acc_distances = pan_genome.pairwise_distances(max_distances, &range1, &range2);

            let mut std_core = 0.0;
            let mut avg_core = 0.0;
            (std_core, avg_core) = standard_deviation(&core_distances);

            let mut std_acc = 0.0;
            let mut avg_acc = 0.0;
            (std_acc, avg_acc) = standard_deviation(&acc_distances);

            avg_core_dist[j as usize] = avg_core;
            avg_acc_dist[j as usize] = avg_acc;

            std_core_dist[j as usize] = std_core;
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
        for (avg_core, avg_acc, std_core, std_acc) in avg_core_dist
            .iter()
            .zip(avg_acc_dist.iter())
            .zip(std_core_dist.iter())
            .zip(std_acc_dist.iter())
            .map(|(((w, x), y), z)| (w, x, y, z))
        {
            writeln!(file, "{}\t{}\t{}\t{}", avg_core, std_core, avg_acc, std_acc);
        }
    }

    if print_matrices {
        core_genome.write(outpref);
        pan_genome.write(outpref);
    }

    //let elapsed = now.elapsed();

    // if verbose {
    //     println!("Total elapsed: {:.2?}", elapsed);
    // }

    // println!("Total elapsed: {:.2?}", elapsed);

    return Ok(());
}
