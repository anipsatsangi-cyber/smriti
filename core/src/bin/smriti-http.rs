use std::env;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use smriti::http::{serve, HttpState};
use smriti::Smriti;

fn parse_args() -> Result<(String, u16)> {
    let mut db = String::from(".smriti/memory.db");
    let mut port = 4000u16;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--db" => {
                db = args.next().context("missing value for --db")?;
            }
            "--port" => {
                let value = args.next().context("missing value for --port")?;
                port = value.parse().context("invalid value for --port")?;
            }
            "--help" | "-h" => {
                println!("Usage: smriti-http [--db PATH] [--port PORT]");
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    Ok((db, port))
}

fn open_store(db: &str) -> Result<Smriti> {
    if db != ":memory:" {
        if let Some(parent) = std::path::Path::new(db).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
    }

    Smriti::open(db).context("failed to open Smriti store")
}

#[tokio::main]
async fn main() -> Result<()> {
    let (db, port) = parse_args()?;
    let smriti = open_store(&db)?;

    eprintln!("📖 Smriti memory engine");
    eprintln!("   Database: {}", db);
    eprintln!("   Listening: http://localhost:{}", port);
    eprintln!();
    eprintln!("   POST /api/remember");
    eprintln!("   POST /api/recall");
    eprintln!("   POST /api/forget");
    eprintln!("   POST /api/supersede");
    eprintln!("   POST /api/link");
    eprintln!("   POST /api/consolidate");
    eprintln!("   GET  /api/stats");
    eprintln!("   POST /smrp");

    let state = Arc::new(HttpState {
        smriti: Mutex::new(smriti),
    });
    serve(state, port).await
}
