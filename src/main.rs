mod app;
mod config;
mod models;
mod pixels;

use config::Config;
use models::{Classify, Models};
use std::path::Path;
use std::process::exit;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::watch;

#[tokio::main]
async fn main() {
    let config = Config::from_env().unwrap_or_else(|message| {
        eprintln!("{message}");
        exit(1)
    });

    // Listen right away so the platform sees the port open; /health answers 503
    // until the models are ready, which is what Railway's healthcheck waits for.
    let port = config.port;
    let listener = match TcpListener::bind(("::", port)).await {
        Ok(listener) => listener,
        // No IPv6 on this host.
        Err(_) => TcpListener::bind(("0.0.0.0", port))
            .await
            .unwrap_or_else(|err| {
                eprintln!("[server] cannot listen on :{port}: {err}");
                exit(1)
            }),
    };
    println!("[server] listening on :{port}");

    let (ready, models) = watch::channel(None::<Arc<dyn Classify>>);
    let categories = config.clip_categories.clone();
    let open = config.api_key.is_empty();
    tokio::task::spawn_blocking(move || {
        let started = Instant::now();
        let loaded = Models::load(Path::new("models"), &categories).unwrap_or_else(|err| {
            eprintln!("[models] failed to load: {err}");
            exit(1)
        });
        let clip = loaded
            .clip_variant()
            .map(|variant| format!(" + CLIP {variant} ({})", categories.join(", ")));
        println!(
            "[models] {}{} ready in {} ms",
            loaded.name(),
            clip.unwrap_or_default(),
            started.elapsed().as_millis()
        );
        if open {
            eprintln!("[auth] API_KEY is not set: /classify is open to anyone");
        }
        release_memory();
        ready.send_replace(Some(Arc::new(loaded)));
        // Never closed: /classify waits on this channel.
        std::mem::forget(ready);
    });

    tokio::spawn(release_memory_when_idle());
    let server = axum::serve(
        listener,
        app::router(config, models, Duration::from_secs(60)),
    );
    if let Err(err) = server.with_graceful_shutdown(shutdown()).await {
        eprintln!("[server] {err}");
        exit(1);
    }
}

/// Resolves on SIGTERM or SIGINT; in-flight requests then get 10 seconds to finish.
async fn shutdown() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut terminate = signal(SignalKind::terminate()).expect("SIGTERM handler");
    let name = tokio::select! {
        _ = terminate.recv() => "SIGTERM",
        _ = tokio::signal::ctrl_c() => "SIGINT",
    };
    println!("[server] {name} received, closing");
    tokio::spawn(async {
        tokio::time::sleep(Duration::from_secs(10)).await;
        exit(0)
    });
}

/// Hands the heap's free pages back to the OS. glibc keeps them otherwise:
/// ~100 MB of scratch space after optimising the models, ~75 MB of tensors after a burst of requests.
fn release_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        // SAFETY: plain glibc call, no pointers involved; safe to call from any thread.
        unsafe { malloc_trim(0) };
    }
}

/// Releases memory once the API has been idle for a few seconds, never between
/// two requests of a burst (getting the pages back costs ~30% of the throughput).
async fn release_memory_when_idle() {
    let mut seen = 0;
    loop {
        tokio::time::sleep(Duration::from_secs(5)).await;
        let now = app::CLASSIFIED.load(Ordering::Relaxed);
        let previous = std::mem::replace(&mut seen, now);
        // One quiet tick right after an active one: the burst is over.
        if now == previous && RELEASED.swap(now, Ordering::Relaxed) != now {
            release_memory();
        }
    }
}

/// Value of `CLASSIFIED` at the last idle release.
static RELEASED: AtomicU64 = AtomicU64::new(0);
