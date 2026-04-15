use std::{
    sync::Arc,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, unbounded};
use derive_more::Display;
use serde::{Deserialize, Serialize};

use crate::{
    domain::tracker::{PositionEstimate, TrackerEngineKind, TrackerLifecycle, TrackingSource},
    resources::WorkspaceSnapshot,
    tracking::{
        ai::CandleTrackerWorker, debug::TrackingDebugSnapshot, template::TemplateTrackerWorker,
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
    fn initial_status(&self) -> TrackingStatus;
    fn engine_kind(&self) -> TrackerEngineKind;
}

#[derive(Debug)]
enum TrackerCommand {
    Stop,
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
}

impl Drop for TrackerSession {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn spawn_tracker_session(
    workspace: Arc<WorkspaceSnapshot>,
    engine: TrackerEngineKind,
) -> Result<TrackerSession> {
    let (command_tx, command_rx) = unbounded();
    let (event_tx, event_rx) = unbounded();

    let thread = thread::Builder::new()
        .name(format!("tracker-{}", engine))
        .spawn(move || run_tracker_session(workspace, engine, command_rx, event_tx))?;

    Ok(TrackerSession {
        command_tx,
        event_rx,
        thread: Some(thread),
    })
}

fn run_tracker_session(
    workspace: Arc<WorkspaceSnapshot>,
    engine: TrackerEngineKind,
    command_rx: Receiver<TrackerCommand>,
    event_tx: Sender<TrackingEvent>,
) {
    let _ = event_tx.send(TrackingEvent::Status(TrackingStatus {
        engine,
        frame_index: 0,
        message: format!("正在初始化 {engine} 追踪器。"),
        lifecycle: TrackerLifecycle::Idle,
        source: None,
        match_score: None,
    }));

    if matches!(command_rx.try_recv(), Ok(TrackerCommand::Stop)) {
        let _ = event_tx.send(TrackingEvent::LifecycleChanged(TrackerLifecycle::Idle));
        return;
    }

    let worker = match build_worker(workspace, engine) {
        Ok(worker) => worker,
        Err(error) => {
            let _ = event_tx.send(TrackingEvent::Error(error.to_string()));
            let _ = event_tx.send(TrackingEvent::LifecycleChanged(TrackerLifecycle::Failed));
            return;
        }
    };

    if matches!(command_rx.try_recv(), Ok(TrackerCommand::Stop)) {
        let _ = event_tx.send(TrackingEvent::LifecycleChanged(TrackerLifecycle::Idle));
        return;
    }

    run_worker_loop(worker, command_rx, event_tx);
}

fn build_worker(
    workspace: Arc<WorkspaceSnapshot>,
    engine: TrackerEngineKind,
) -> Result<Box<dyn TrackingWorker>> {
    Ok(match engine {
        TrackerEngineKind::MultiScaleTemplateMatch => {
            Box::new(TemplateTrackerWorker::new(workspace)?)
        }
        TrackerEngineKind::ConvolutionFeatureMatch => {
            Box::new(CandleTrackerWorker::new(workspace)?)
        }
    })
}

fn run_worker_loop(
    mut worker: Box<dyn TrackingWorker>,
    command_rx: Receiver<TrackerCommand>,
    event_tx: Sender<TrackingEvent>,
) {
    let _ = event_tx.send(TrackingEvent::LifecycleChanged(TrackerLifecycle::Running));
    let mut initial_status = worker.initial_status();
    initial_status.lifecycle = TrackerLifecycle::Running;
    let _ = event_tx.send(TrackingEvent::Status(initial_status));

    loop {
        if matches!(command_rx.try_recv(), Ok(TrackerCommand::Stop)) {
            let _ = event_tx.send(TrackingEvent::LifecycleChanged(TrackerLifecycle::Idle));
            return;
        }

        let loop_started = Instant::now();
        match worker.tick() {
            Ok(tick) => {
                let _ = event_tx.send(TrackingEvent::Status(tick.status));
                if let Some(estimate) = tick.estimate {
                    let _ = event_tx.send(TrackingEvent::Position(estimate));
                }
                if let Some(debug) = tick.debug {
                    let _ = event_tx.send(TrackingEvent::Debug(debug));
                }
            }
            Err(error) => {
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

        if let Ok(TrackerCommand::Stop) = command_rx.recv_timeout(wait_for) {
            let _ = event_tx.send(TrackingEvent::LifecycleChanged(TrackerLifecycle::Idle));
            return;
        }
    }
}
