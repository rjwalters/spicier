//! spicier-validate CLI tool.
//!
//! This is the command-line interface for the spicier-validate crate.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use spicier_validate::{
    ComparisonConfig, DcTolerances, NgspiceConfig, is_ngspice_available, load_golden_directory,
    ngspice_version, validate_against_golden,
};

#[derive(Parser)]
#[command(name = "spicier-validate")]
#[command(about = "Cross-simulator validation tool comparing spicier with ngspice")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compare a netlist through both ngspice and spicier
    Compare {
        /// Path to the netlist file
        netlist: PathBuf,

        /// Voltage absolute tolerance (V)
        #[arg(long, default_value = "1e-6")]
        voltage_tol: f64,

        /// Current absolute tolerance (A)
        #[arg(long, default_value = "1e-9")]
        current_tol: f64,

        /// Output results as JSON
        #[arg(long)]
        json: bool,
    },

    /// Run validation suite against golden data
    Suite {
        /// Directory containing golden data files
        #[arg(long, default_value = "tests/golden_data")]
        golden_dir: PathBuf,

        /// Only run tests matching this pattern
        #[arg(long)]
        filter: Option<String>,

        /// Output results as JSON
        #[arg(long)]
        json: bool,
    },

    /// Generate golden data from ngspice
    Generate {
        /// Path to the netlist file
        netlist: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Check ngspice availability
    Check,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compare {
            netlist,
            voltage_tol,
            current_tol,
            json,
        } => cmd_compare(netlist, voltage_tol, current_tol, json),
        Commands::Suite {
            golden_dir,
            filter,
            json,
        } => cmd_suite(golden_dir, filter, json),
        Commands::Generate { netlist, output } => cmd_generate(netlist, output),
        Commands::Check => cmd_check(),
    }
}

fn cmd_compare(netlist_path: PathBuf, voltage_tol: f64, current_tol: f64, json: bool) -> ExitCode {
    // Check ngspice availability
    let ng_config = NgspiceConfig::default();
    if !is_ngspice_available(&ng_config) {
        eprintln!("Error: ngspice not found in PATH");
        return ExitCode::FAILURE;
    }

    // Read netlist
    let netlist = match std::fs::read_to_string(&netlist_path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading netlist: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Configure tolerances
    let config = ComparisonConfig::default().with_dc_tolerances(DcTolerances {
        voltage_abs: voltage_tol,
        voltage_rel: 1e-4,
        current_abs: current_tol,
        current_rel: 1e-4,
    });

    // Run comparison
    match spicier_validate::compare_simulators(&netlist, &config) {
        Ok(report) => {
            if json {
                match serde_json::to_string_pretty(&report) {
                    Ok(s) => println!("{}", s),
                    Err(e) => eprintln!("Error serializing report: {}", e),
                }
            } else {
                println!("{}", report.to_text());
            }

            if report.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn cmd_suite(golden_dir: PathBuf, filter: Option<String>, json: bool) -> ExitCode {
    // Load golden data files
    let files = match load_golden_directory(&golden_dir) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error loading golden data: {}", e);
            return ExitCode::FAILURE;
        }
    };

    if files.is_empty() {
        eprintln!("No golden data files found in {}", golden_dir.display());
        return ExitCode::FAILURE;
    }

    let mut total_tests = 0;
    let mut passed_tests = 0;
    let mut failed_tests = Vec::new();

    for file in &files {
        for circuit in &file.circuits {
            // Apply filter if specified
            if let Some(ref pattern) = filter {
                if !circuit.name.contains(pattern) {
                    continue;
                }
            }

            total_tests += 1;

            match validate_against_golden(circuit) {
                Ok(report) => {
                    if report.passed {
                        passed_tests += 1;
                        if !json {
                            println!("  PASS: {}", circuit.name);
                        }
                    } else {
                        failed_tests.push((circuit.name.clone(), report.to_text()));
                        if !json {
                            println!("  FAIL: {}", circuit.name);
                        }
                    }
                }
                Err(e) => {
                    failed_tests.push((circuit.name.clone(), format!("Error: {}", e)));
                    if !json {
                        println!("  ERROR: {} - {}", circuit.name, e);
                    }
                }
            }
        }
    }

    if json {
        let result = serde_json::json!({
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests.len(),
            "failures": failed_tests.iter().map(|(name, msg)| {
                serde_json::json!({"name": name, "message": msg})
            }).collect::<Vec<_>>()
        });
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    } else {
        println!("\nSummary: {}/{} tests passed", passed_tests, total_tests);

        if !failed_tests.is_empty() {
            println!("\nFailed tests:");
            for (name, msg) in &failed_tests {
                println!("\n--- {} ---\n{}", name, msg);
            }
        }
    }

    if failed_tests.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn cmd_generate(netlist_path: PathBuf, output_path: PathBuf) -> ExitCode {
    // Check ngspice availability
    let ng_config = NgspiceConfig::default();
    if !is_ngspice_available(&ng_config) {
        eprintln!("Error: ngspice not found in PATH");
        return ExitCode::FAILURE;
    }

    // Read netlist
    let netlist = match std::fs::read_to_string(&netlist_path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading netlist: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Run ngspice and generate golden data
    match spicier_validate::run_ngspice(&netlist, &ng_config) {
        Ok(rawfile) => {
            let result = spicier_validate::NgspiceResult::from_rawfile(&rawfile);

            // Serialize to JSON
            let golden = serde_json::json!({
                "generator": format!("ngspice (via spicier-validate)"),
                "generated_at": chrono_lite::Utc::now().to_string(),
                "description": format!("Generated from {}", netlist_path.display()),
                "circuits": [{
                    "name": netlist_path.file_stem().map(|s| s.to_string_lossy()).unwrap_or_default(),
                    "description": "Auto-generated golden data",
                    "netlist": netlist,
                    "analysis": match result {
                        spicier_validate::NgspiceResult::DcOp(dc) => serde_json::json!({
                            "type": "dc_op",
                            "results": dc.values,
                            "tolerances": {
                                "voltage": 1e-9,
                                "current": 1e-12
                            }
                        }),
                        _ => serde_json::json!({"type": "unsupported"})
                    }
                }]
            });

            match std::fs::write(&output_path, serde_json::to_string_pretty(&golden).unwrap()) {
                Ok(_) => {
                    println!("Golden data written to {}", output_path.display());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("Error writing output: {}", e);
                    ExitCode::FAILURE
                }
            }
        }
        Err(e) => {
            eprintln!("Error running ngspice: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn cmd_check() -> ExitCode {
    let config = NgspiceConfig::default();

    if is_ngspice_available(&config) {
        match ngspice_version(&config) {
            Ok(version) => {
                println!("ngspice is available: {}", version);
                ExitCode::SUCCESS
            }
            Err(e) => {
                println!("ngspice found but version check failed: {}", e);
                ExitCode::SUCCESS
            }
        }
    } else {
        println!("ngspice not found in PATH");
        ExitCode::FAILURE
    }
}

mod chrono_lite {
    pub struct Utc;
    impl Utc {
        pub fn now() -> impl std::fmt::Display {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| format!("{}", d.as_secs()))
                .unwrap_or_else(|_| "unknown".to_string())
        }
    }
}
