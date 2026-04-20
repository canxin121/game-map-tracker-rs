use std::{
    sync::Arc,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, unbounded};
use derive_more::Display;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info};

use crate::{
    domain::tracker::{PositionEstimate, TrackerEngineKind, TrackerLifecycle, TrackingSource},
    resources::WorkspaceSnapshot,
    tracking::{
        ai::BurnTrackerWorker, debug::TrackingDebugSnapshot, template::TemplateTrackerWorker,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingStatus {
    pub engine: TrackerEngineKind,
    pub frame_index: u64,
    pub message: String,
    pub lifecycle: TrackerLifecycle,
    pub source: Option<TrackingSource>,
    pub match_score: Option<f32>,
    pub probe_summary: String,
    pub locate_summary: String,
}

impl TrackingStatus {
    #[must_use]
    pub fn new(engine: TrackerEngineKind, message: impl Into<String>) -> Self {
        Self {
            engine,
            frame_index: 0,
            message: message.into(),
            lifecycle: TrackerLifecycle::Idle,
            source: None,
            match_score: None,
            probe_summary: "等待探针".to_owned(),
            locate_summary: "等待首帧".to_owned(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrackingTick {
    pub status: TrackingStatus,
    pub estimate: Option<PositionEstimate>,
    pub debug: Option<TrackingDebugSnapshot>,
}

#[derive(Debug, Clone, Display)]
pub enum TrackingEvent {
    #[display("Lifecycle")]
    LifecycleChanged(TrackerLifecycle),
    #[display("Status")]
    Status(TrackingStatus),
    #[display("Position")]
    Position(PositionEstimate),
    #[display("Debug")]
    Debug(TrackingDebugSnapshot),
    #[display("Error")]
    Error(String),
}

pub trait TrackingWorker: Send {
    fn refresh_interval(&self) -> Duration;
    fn tick(&mut self) -> Result<TrackingTick>;
    fn set_debug_enabled(&mut self, _enabled: bool) {}
    fn initial_status(&self) -> TrackingStatus;
    fn engine_kind(&self) -> TrackerEngineKind;
}

#[derive(Debug)]
enum TrackerCommand {
    Stop,
    SetDebugEnabled(bool),
}

#[derive(Debug)]
pub struct TrackerSession {
    command_tx: Sender<TrackerCommand>,
    event_rx: Receiver<TrackingEvent>,
    thread: Option<JoinHandle<()>>,
}

impl TrackerSession {
    #[must_use]
    pub fn event_rx(&self) -> &Receiver<TrackingEvent> {
        &self.event_rx
    }

    pub fn stop(&mut self) {
        let _ = self.command_tx.send(TrackerCommand::Stop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }

    pub fn set_debug_enabled(&self, enabled: bool) {
        let _ = self
            .command_tx
            .send(TrackerCommand::SetDebugEnabled(enabled));
    }
}

impl Drop for TrackerSession {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn spawn_tracker_session(
    workspace: Arc<WorkspaceSnapshot>,
    engine: TrackerEngineKind,
    debug_enabled: bool,
) -> Result<TrackerSession> {
    info!(%engine, debug_enabled, "spawning tracker session");
    let (command_tx, command_rx) = unbounded();
    let (event_tx, event_rx) = unbounded();

    let thread = thread::Builder::new()
        .name(format!("tracker-{}", engine))
        .spawn(move || {
            run_tracker_session(workspace, engine, debug_enabled, command_rx, event_tx)
        })?;

    Ok(TrackerSession {
        command_tx,
        event_rx,
        thread: Some(thread),
    })
}

fn run_tracker_session(
    workspace: Arc<WorkspaceSnapshot>,
    engine: TrackerEngineKind,
    mut debug_enabled: bool,
    command_rx: Receiver<TrackerCommand>,
    event_tx: Sender<TrackingEvent>,
) {
    info!(%engine, debug_enabled, "tracker thread started");
    let _ = event_tx.send(TrackingEvent::Status(TrackingStatus {
        engine,
        frame_index: 0,
        message: format!("正在初始化 {engine} 追踪器。"),
        lifecycle: TrackerLifecycle::Idle,
        source: None,
        match_score: None,
        probe_summary: "等待探针".to_owned(),
        locate_summary: "等待初始化".to_owned(),
    }));

    if handle_tracker_commands_without_worker(&command_rx, &mut debug_enabled) {
        info!(%engine, "tracker thread stopped before worker initialization");
        let _ = event_tx.send(TrackingEvent::LifecycleChanged(TrackerLifecycle::Idle));
        return;
    }

    let worker = match build_worker(workspace, engine) {
        Ok(worker) => {
            info!(%engine, "tracker worker initialized");
            worker
        }
        Err(error) => {
            error!(%engine, error = %error, "failed to initialize tracker worker");
            let _ = event_tx.send(TrackingEvent::Error(error.to_string()));
            let _ = event_tx.send(TrackingEvent::LifecycleChanged(TrackerLifecycle::Failed));
            return;
        }
    };

    if handle_tracker_commands_without_worker(&command_rx, &mut debug_enabled) {
        info!(%engine, "tracker thread stopped after worker initialization");
        let _ = event_tx.send(TrackingEvent::LifecycleChanged(TrackerLifecycle::Idle));
        return;
    }

    let mut worker = worker;
    worker.set_debug_enabled(debug_enabled);

    run_worker_loop(worker, command_rx, event_tx);
}

fn build_worker(
    workspace: Arc<WorkspaceSnapshot>,
    engine: TrackerEngineKind,
) -> Result<Box<dyn TrackingWorker>> {
    debug!(
        %engine,
        group_count = workspace.groups.len(),
        "building tracker worker"
    );
    Ok(match engine {
        TrackerEngineKind::MultiScaleTemplateMatch => {
            Box::new(TemplateTrackerWorker::new(workspace)?)
        }
        TrackerEngineKind::ConvolutionFeatureMatch => Box::new(BurnTrackerWorker::new(workspace)?),
    })
}

fn run_worker_loop(
    mut worker: Box<dyn TrackingWorker>,
    command_rx: Receiver<TrackerCommand>,
    event_tx: Sender<TrackingEvent>,
) {
    let engine = worker.engine_kind();
    info!(%engine, "tracker worker loop entered");
    let _ = event_tx.send(TrackingEvent::LifecycleChanged(TrackerLifecycle::Running));
    let mut initial_status = worker.initial_status();
    initial_status.lifecycle = TrackerLifecycle::Running;
    let _ = event_tx.send(TrackingEvent::Status(initial_status));

    loop {
        if handle_tracker_commands(worker.as_mut(), &command_rx) {
            let _ = event_tx.send(TrackingEvent::LifecycleChanged(TrackerLifecycle::Idle));
            return;
        }

        let loop_started = Instant::now();
        match worker.tick() {
            Ok(tick) => {
                if should_log_tick(tick.status.frame_index) {
                    debug!(
                        %engine,
                        frame_index = tick.status.frame_index,
                        lifecycle = ?tick.status.lifecycle,
                        source = ?tick.status.source,
                        match_score = ?tick.status.match_score,
                        has_estimate = tick.estimate.is_some(),
                        has_debug = tick.debug.is_some(),
                        elapsed_ms = loop_started.elapsed().as_millis(),
                        "tracker tick completed"
                    );
                }
                let _ = event_tx.send(TrackingEvent::Status(tick.status));
                if let Some(estimate) = tick.estimate {
                    let _ = event_tx.send(TrackingEvent::Position(estimate));
                }
                if let Some(debug) = tick.debug {
                    let _ = event_tx.send(TrackingEvent::Debug(debug));
                }
            }
            Err(error) => {
                error!(%engine, error = %error, "tracker tick failed");
                let _ = event_tx.send(TrackingEvent::Error(error.to_string()));
                let _ = event_tx.send(TrackingEvent::LifecycleChanged(TrackerLifecycle::Failed));
                return;
            }
        }

        let refresh_interval = worker.refresh_interval();
        let elapsed = loop_started.elapsed();
        let wait_for = refresh_interval.saturating_sub(elapsed);
        if wait_for.is_zero() {
            continue;
        }

        match command_rx.recv_timeout(wait_for) {
            Ok(TrackerCommand::Stop) => {
                info!(%engine, "tracker worker loop received stop command");
                let _ = event_tx.send(TrackingEvent::LifecycleChanged(TrackerLifecycle::Idle));
                return;
            }
            Ok(TrackerCommand::SetDebugEnabled(enabled)) => {
                info!(%engine, enabled, "tracker debug mode changed");
                worker.set_debug_enabled(enabled);
            }
            Err(_) => {}
        }
    }
}

fn handle_tracker_commands(
    worker: &mut dyn TrackingWorker,
    command_rx: &Receiver<TrackerCommand>,
) -> bool {
    for command in command_rx.try_iter() {
        match command {
            TrackerCommand::Stop => {
                info!(engine = %worker.engine_kind(), "tracker command requested stop");
                return true;
            }
            TrackerCommand::SetDebugEnabled(enabled) => {
                info!(engine = %worker.engine_kind(), enabled, "tracker command changed debug mode");
                worker.set_debug_enabled(enabled);
            }
        }
    }
    false
}

fn handle_tracker_commands_without_worker(
    command_rx: &Receiver<TrackerCommand>,
    debug_enabled: &mut bool,
) -> bool {
    for command in command_rx.try_iter() {
        match command {
            TrackerCommand::Stop => {
                info!("tracker command requested stop before worker was ready");
                return true;
            }
            TrackerCommand::SetDebugEnabled(enabled) => {
                info!(enabled, "tracker command changed pending debug mode");
                *debug_enabled = enabled;
            }
        }
    }
    false
}

fn should_log_tick(frame_index: u64) -> bool {
    frame_index <= 3 || frame_index % 30 == 0
}
