pub mod cli;
pub mod clients;
pub mod config;
pub mod data;
pub mod fusion;
pub mod metrics;
pub mod persistence;
pub mod pipeline;
pub mod report;
pub mod sparse;
pub mod store;

use serde::Serialize;
use sha2::{Digest, Sha256};

pub fn fingerprint(value: &impl Serialize) -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(value).expect("serializable fingerprint"))
    )
}
