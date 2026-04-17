pub mod ai;
pub mod burn_support;
pub mod capture;
pub mod debug;
pub mod precompute;
pub mod presence;
pub mod runtime;
pub mod template;
pub mod vision;

#[cfg(test)]
pub(crate) mod test_support;

pub use runtime::{TrackerSession, TrackingEvent, TrackingStatus, spawn_tracker_session};
