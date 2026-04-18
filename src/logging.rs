use std::{
    collections::VecDeque,
    fmt,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::Instant,
};

use tracing::{
    Event, Subscriber,
    field::{Field, Visit},
};
use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

const DEFAULT_LOG_LIMIT: usize = 2_000;

static DEBUG_LOG_STORE: OnceLock<Arc<DebugLogStore>> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct DebugLogEntry {
    pub sequence: u64,
    pub elapsed_millis: u128,
    pub level: String,
    pub target: String,
    pub message: String,
    pub fields: Vec<String>,
    pub span_path: Option<String>,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub thread_name: String,
}

#[derive(Debug)]
pub struct DebugLogStore {
    entries: Mutex<VecDeque<DebugLogEntry>>,
    max_entries: usize,
    started_at: Instant,
    next_sequence: AtomicU64,
    revision: AtomicU64,
}

impl DebugLogStore {
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::with_capacity(max_entries.min(256))),
            max_entries: max_entries.max(1),
            started_at: Instant::now(),
            next_sequence: AtomicU64::new(1),
            revision: AtomicU64::new(0),
        }
    }

    pub fn push(&self, mut entry: DebugLogEntry) {
        entry.sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        entry.elapsed_millis = self.started_at.elapsed().as_millis();

        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while entries.len() >= self.max_entries {
            entries.pop_front();
        }
        entries.push_back(entry);
        self.revision.fetch_add(1, Ordering::Release);
    }

    #[must_use]
    pub fn snapshot(&self, limit: usize) -> Vec<DebugLogEntry> {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let limit = limit.max(1);
        let skip = entries.len().saturating_sub(limit);
        entries.iter().skip(skip).cloned().collect()
    }

    pub fn clear(&self) {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        entries.clear();
        self.revision.fetch_add(1, Ordering::Release);
    }

    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }
}

pub fn install_debug_log_store(max_entries: usize) -> Arc<DebugLogStore> {
    let store = Arc::new(DebugLogStore::new(max_entries));
    let _ = DEBUG_LOG_STORE.set(store.clone());
    DEBUG_LOG_STORE.get().cloned().unwrap_or(store)
}

#[must_use]
pub fn debug_log_store() -> Arc<DebugLogStore> {
    DEBUG_LOG_STORE
        .get_or_init(|| Arc::new(DebugLogStore::new(DEFAULT_LOG_LIMIT)))
        .clone()
}

#[must_use]
pub fn debug_log_snapshot(limit: usize) -> Vec<DebugLogEntry> {
    debug_log_store().snapshot(limit)
}

pub fn clear_debug_logs() {
    debug_log_store().clear();
}

#[must_use]
pub fn debug_log_revision() -> u64 {
    debug_log_store().revision()
}

#[derive(Debug, Clone)]
pub struct DebugLogLayer {
    store: Arc<DebugLogStore>,
}

impl DebugLogLayer {
    #[must_use]
    pub fn new(store: Arc<DebugLogStore>) -> Self {
        Self { store }
    }
}

impl<S> Layer<S> for DebugLogLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let metadata = event.metadata();
        let mut visitor = DebugLogVisitor::default();
        event.record(&mut visitor);

        let thread_name = thread::current()
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| "unnamed".to_owned());
        let span_path = ctx.event_scope(event).map(|scope| {
            scope
                .from_root()
                .map(|span| span.name())
                .collect::<Vec<_>>()
                .join("::")
        });

        self.store.push(DebugLogEntry {
            sequence: 0,
            elapsed_millis: 0,
            level: metadata.level().to_string(),
            target: metadata.target().to_owned(),
            message: visitor.message.unwrap_or_else(|| {
                if visitor.fields.is_empty() {
                    metadata.name().to_owned()
                } else {
                    visitor.fields.join(" ")
                }
            }),
            fields: visitor.fields,
            span_path,
            file: metadata.file().map(str::to_owned),
            line: metadata.line(),
            thread_name,
        });
    }
}

#[derive(Default)]
struct DebugLogVisitor {
    message: Option<String>,
    fields: Vec<String>,
}

impl DebugLogVisitor {
    fn record_value(&mut self, field: &Field, value: impl Into<String>) {
        if field.name() == "message" {
            self.message = Some(value.into());
        } else {
            self.fields
                .push(format!("{}={}", field.name(), value.into()));
        }
    }
}

impl Visit for DebugLogVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_value(field, value.to_owned());
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.record_value(field, value.to_string());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record_value(field, value.to_string());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.record_value(field, value.to_string());
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.record_value(field, value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.record_value(field, format!("{value:?}"));
    }
}
