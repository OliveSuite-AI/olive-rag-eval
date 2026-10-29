use crate::{clients::LocalClient, config::Config, data, pipeline, report};
use anyhow::{Result, ensure};
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Local Odia-first multilingual RAG evaluation")]
struct Cli {
    #[arg(short, long, default_value = "configs/local.toml", global = true)]
    config: PathBuf,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Validate {
        #[arg(long)]
        online: bool,
    },
    Index,
    Retrieve {
        query: String,
        #[arg(long, conflicts_with = "embedding")]
        pipeline: Option<String>,
        #[arg(long)]
        embedding: Option<String>,
        #[arg(long, value_delimiter = ',')]
        languages: Vec<String>,
    },
    Compare {
        #[arg(long)]
        limit: Option<usize>,
    },
    Evaluate {
        #[arg(long)]
        limit: Option<usize>,
    },
    Report {
        run: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}
pub async fn run(args: Vec<String>) -> Result<i32> {
    let parsed = if args.first().is_some_and(|name| name == "olive-rag-eval") {
        Cli::command()
            .name("olive-rag-eval")
            .bin_name("olive-rag-eval")
            .try_get_matches_from(args)
            .and_then(|matches| Cli::from_arg_matches(&matches))
    } else {
        Cli::try_parse_from(args)
    };
    let cli = match parsed {
        Ok(cli) => cli,
        Err(e) => {
            let code = if e.use_stderr() { 2 } else { 0 };
            e.print()?;
            return Ok(code);
        }
    };
    if let Command::Report { run, output } = &cli.command {
        let result: pipeline::Run = serde_json::from_slice(&std::fs::read(run)?)?;
        ensure!(
            matches!(result.version, 1 | 2),
            "unsupported run artifact version"
        );
        let dir = output
            .clone()
            .unwrap_or_else(|| run.parent().unwrap().join("reports"));
        report::write(&result, &dir)?;
        println!("{}", dir.display());
        return Ok(0);
    }
    let c = Config::load(&cli.config)?;
    let (chunks, mut cases) = data::load(&c.corpus, &c.dataset)?;
    let client = LocalClient::new(&c)?;
    match cli.command {
        Command::Validate { online } => {
            if online {
                for m in c
                    .embeddings
                    .iter()
                    .chain(&c.rerankers)
                    .chain(c.answer.iter())
                    .chain(c.judge.iter())
                    .chain(c.llm_reranker.iter())
                {
                    println!("{}: {}", m.name, client.health(m).await?);
                }
            }
            println!(
                "Validated {} chunks, {} cases, {} semantic cases; {} draft queries. Fixture: {}",
                chunks.len(),
                cases.len(),
                cases
                    .iter()
                    .map(|c| &c.semantic_case_id)
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                cases.iter().filter(|c| c.review_status == "draft").count(),
                c.fixture
            );
            for m in &c.embeddings {
                println!("Embedding profile {}: {}", m.name, crate::fingerprint(m));
            }
        }
        Command::Index => {
            for p in pipeline::index(&c, &chunks, &client).await? {
                println!("{}", p.display());
            }
        }
        Command::Retrieve {
            query,
            pipeline: selected,
            embedding,
            languages,
        } => {
            if let Some(name) = selected {
                let p = c
                    .resolved_pipelines()
                    .into_iter()
                    .find(|p| p.name == name)
                    .ok_or_else(|| anyhow::anyhow!("unknown pipeline {name}"))?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &pipeline::retrieve_named(&c, &chunks, &client, &p, &query, &languages)
                            .await?
                    )?
                );
                return Ok(0);
            }
            let m = if let Some(name) = embedding {
                c.embeddings
                    .iter()
                    .find(|m| m.name == name)
                    .ok_or_else(|| anyhow::anyhow!("unknown embedding profile {name}"))?
            } else {
                c.embeddings.first().ok_or_else(|| {
                    anyhow::anyhow!("no embedding profile; use --pipeline for BM25-only retrieval")
                })?
            };
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &pipeline::retrieve(&c, &chunks, &client, m, &query, &languages).await?
                )?
            );
        }
        Command::Compare { limit } | Command::Evaluate { limit } => {
            let full = matches!(cli.command, Command::Evaluate { .. });
            if let Some(limit) = limit {
                ensure!(limit > 0, "limit must be positive");
                cases.truncate(limit);
            }
            let (run, path) = pipeline::evaluate(&c, &chunks, &cases, full, &client).await?;
            report::write(&run, &path.parent().unwrap().join("reports"))?;
            println!("{}", path.display());
            let failures = run
                .outcomes
                .iter()
                .filter(|o| o.status != "completed")
                .count()
                + run.calibration.iter().filter(|o| o.error.is_some()).count();
            ensure!(
                failures == 0,
                "{failures} comparisons failed; successful stages retained for resume"
            );
        }
        Command::Report { .. } => unreachable!(),
    }
    Ok(0)
}
