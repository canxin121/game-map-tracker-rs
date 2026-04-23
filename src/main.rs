#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

#[cfg(debug_assertions)]
use std::ffi::OsStr;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL_ALLOCATOR: MiMalloc = MiMalloc;

fn main() -> rocom_compass::error::Result<()> {
    #[cfg(debug_assertions)]
    {
        let mut args = std::env::args_os();
        let _ = args.next();
        match args.next().as_deref() {
            Some(command)
                if command == OsStr::new("train-conv")
                    || command == OsStr::new("train-encoder") =>
            {
                rocom_compass::app::init_tracing();
                return rocom_compass::tracking::ai::run_encoder_training_cli(args.collect());
            }
            _ => {}
        }
    }

    rocom_compass::app::launch()
}
