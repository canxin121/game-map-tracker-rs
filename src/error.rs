use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("failed to resolve project root from manifest directory: {0}")]
    ProjectRootResolution(String),
}
