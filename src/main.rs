#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let code = rag_eval::cli::run(std::env::args().collect()).await?;
    if code != 0 {
        std::process::exit(code);
    }
    Ok(())
}
