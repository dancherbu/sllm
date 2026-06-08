//! sLLM Inference Runner
//!
//! Read-only, multi-model HTTP API server. Loads brain.sllm files via
//! memory-mapped I/O and serves predictions over a REST API.
//!
//! Usage:
//!   sllm-run --models-dir ./models/ --port 11435
//!   sllm-run --model ./models/code.sllm

use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, Sse},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tokio::net::TcpListener;
use tracing::{info, error};

use sllm_core::brain::{NgramBrain, Sampler};
use sllm_core::format::{BrainReader, BrainHeader};
use sllm_core::tokenizer::BpeTokenizer;

/// sLLM Inference Runner — multi-model HTTP API server
#[derive(Parser, Debug)]
#[command(name = "sllm-run", about = "Serve sLLM models via HTTP API")]
struct Args {
    /// Directory containing .sllm model files
    #[arg(long, default_value = "./models")]
    models_dir: PathBuf,

    /// Load a specific model file on startup
    #[arg(long)]
    model: Option<PathBuf>,

    /// Port to listen on
    #[arg(short, long, default_value_t = sllm_core::DEFAULT_PORT)]
    port: u16,

    /// Bind address
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

/// A loaded model in the registry.
struct LoadedModel {
    header: BrainHeader,
    tokenizer: BpeTokenizer,
    brain: NgramBrain,
}

/// Shared application state.
struct AppState {
    models: RwLock<HashMap<String, LoadedModel>>,
    models_dir: PathBuf,
}

// === API Request/Response types ===

#[derive(Deserialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_top_p")]
    top_p: f64,
    #[serde(default = "default_repetition_penalty")]
    repetition_penalty: f64,
    #[serde(default)]
    stream: bool,
}

fn default_max_tokens() -> usize { 128 }
fn default_temperature() -> f64 { 0.8 }
fn default_top_k() -> usize { 40 }
fn default_top_p() -> f64 { 0.95 }
fn default_repetition_penalty() -> f64 { 1.1 }

#[derive(Serialize)]
struct GenerateResponse {
    model: String,
    response: String,
    tokens_generated: usize,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    vocab_size: u32,
    context_window: u32,
    total_associations: u64,
    training_tokens_seen: u64,
}

#[derive(Serialize)]
struct ModelsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    models_loaded: usize,
}

#[derive(Deserialize)]
struct LoadModelRequest {
    path: String,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// === Handlers ===

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let models = state.models.read().unwrap();
    Json(HealthResponse {
        status: "ok".to_string(),
        models_loaded: models.len(),
    })
}

async fn list_models_handler(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let models = state.models.read().unwrap();
    let model_list = models
        .iter()
        .map(|(name, model)| ModelInfo {
            name: name.clone(),
            vocab_size: model.header.vocab_size,
            context_window: model.header.context_window,
            total_associations: model.header.total_associations,
            training_tokens_seen: model.header.training_tokens_seen,
        })
        .collect();

    Json(ModelsResponse { models: model_list })
}

async fn generate_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, Json<ErrorResponse>)> {
    let models = state.models.read().unwrap();
    let model = models.get(&req.model).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Model '{}' not loaded", req.model),
            }),
        )
    })?;

    let sampler = Sampler {
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        repetition_penalty: req.repetition_penalty,
    };

    // Encode prompt
    let mut tokens = model.tokenizer.encode(&req.prompt);
    let mut rng = rand::rngs::StdRng::from_os_rng();
    let mut generated_count = 0;

    // Generate tokens
    for _ in 0..req.max_tokens {
        let context = if tokens.len() > 4 {
            &tokens[tokens.len() - 4..]
        } else {
            &tokens
        };

        let predictions = model.brain.predict_next(context);
        if predictions.is_empty() {
            break;
        }

        let recent = if tokens.len() > 20 {
            &tokens[tokens.len() - 20..]
        } else {
            &tokens
        };

        match sampler.sample(&predictions, recent, &mut rng) {
            Some(token) => {
                // Stop on EOS
                if token == sllm_core::tokenizer::SpecialToken::Eos.id() {
                    break;
                }
                tokens.push(token);
                generated_count += 1;
            }
            None => break,
        }
    }

    // Decode the generated portion
    let generated_tokens = &tokens[tokens.len() - generated_count..];
    let response_text = model.tokenizer.decode(generated_tokens);

    Ok(Json(GenerateResponse {
        model: req.model,
        response: response_text,
        tokens_generated: generated_count,
    }))
}

async fn load_model_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> Result<Json<ModelInfo>, (StatusCode, Json<ErrorResponse>)> {
    let path = PathBuf::from(&req.path);
    let loaded = BrainReader::read_mmap(&path).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Failed to load model: {}", e),
            }),
        )
    })?;

    let name = req.name.unwrap_or_else(|| loaded.header.model_name.clone());
    let info = ModelInfo {
        name: name.clone(),
        vocab_size: loaded.header.vocab_size,
        context_window: loaded.header.context_window,
        total_associations: loaded.header.total_associations,
        training_tokens_seen: loaded.header.training_tokens_seen,
    };

    let mut models = state.models.write().unwrap();
    models.insert(
        name.clone(),
        LoadedModel {
            header: loaded.header,
            tokenizer: loaded.tokenizer,
            brain: loaded.brain,
        },
    );

    info!("Loaded model: {}", name);
    Ok(Json(info))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sllm_run=info,tower_http=info".into()),
        )
        .init();

    let args = Args::parse();

    // Create shared state
    let state = Arc::new(AppState {
        models: RwLock::new(HashMap::new()),
        models_dir: args.models_dir.clone(),
    });

    // Auto-load model if specified
    if let Some(ref model_path) = args.model {
        info!("Loading model from {}...", model_path.display());
        match BrainReader::read_mmap(model_path) {
            Ok(loaded) => {
                let name = loaded.header.model_name.clone();
                info!(
                    "Loaded '{}': {} associations, vocab={}",
                    name, loaded.header.total_associations, loaded.header.vocab_size
                );
                state.models.write().unwrap().insert(
                    name,
                    LoadedModel {
                        header: loaded.header,
                        tokenizer: loaded.tokenizer,
                        brain: loaded.brain,
                    },
                );
            }
            Err(e) => {
                error!("Failed to load model: {}", e);
            }
        }
    }

    // Scan models directory
    if args.models_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&args.models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("sllm") {
                    info!("Found model: {}", path.display());
                    match BrainReader::read_mmap(&path) {
                        Ok(loaded) => {
                            let name = loaded.header.model_name.clone();
                            info!("  Loaded '{}'", name);
                            state.models.write().unwrap().insert(
                                name,
                                LoadedModel {
                                    header: loaded.header,
                                    tokenizer: loaded.tokenizer,
                                    brain: loaded.brain,
                                },
                            );
                        }
                        Err(e) => {
                            error!("  Failed to load {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }
    }

    // Build router
    let app = Router::new()
        .route("/v1/health", get(health_handler))
        .route("/v1/models", get(list_models_handler))
        .route("/v1/models/load", post(load_model_handler))
        .route("/v1/generate", post(generate_handler))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    info!("sLLM Runner starting on http://{}", addr);
    info!("Endpoints:");
    info!("  GET  /v1/health         — Health check");
    info!("  GET  /v1/models         — List models");
    info!("  POST /v1/models/load    — Load a model");
    info!("  POST /v1/generate       — Generate text");

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
