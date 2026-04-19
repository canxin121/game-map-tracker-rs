use std::ffi::OsStr;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args_os();
    let _ = args.next();
    match args.next().as_deref() {
        Some(command)
            if command == OsStr::new("train-conv") || command == OsStr::new("train-encoder") =>
        {
            game_map_tracker_rs::app::init_tracing();
            game_map_tracker_rs::tracking::ai::run_encoder_training_cli(args.collect())
        }
        _ => game_map_tracker_rs::app::launch(),
    }
}
