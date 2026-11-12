//! The HTTP API: healthcheck, playground and POST /classify.
use crate::config::{CLASSES, Config};
use crate::models::{Classify, ClassifyError, Verdict};
use axum::body::{Body, Bytes, to_bytes};
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Map, Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

/// Images classified since the server started.
pub static CLASSIFIED: AtomicU64 = AtomicU64::new(0);

/// The loaded models; None while they are still loading.
pub type ModelsRx = watch::Receiver<Option<Arc<dyn Classify>>>;

struct App {
    config: Config,
    models: ModelsRx,
    /// How long /classify waits for the models instead of failing, so a request
    /// that wakes a sleeping service still gets an answer.
    ready_timeout: Duration,
    /// Caps how many inferences run at once...
    slots: Arc<Semaphore>,
    /// ...and how many may wait behind them.
    queued: AtomicUsize,
}

pub fn router(config: Config, models: ModelsRx, ready_timeout: Duration) -> Router {
    let mut router = Router::new()
        // "/" is the healthcheck; "/health" stays as an explicit alias (railway.json uses it).
        .route("/", get(health))
        .route("/health", get(health))
        .route("/classify", post(classify))
        .fallback(|| async { error(StatusCode::NOT_FOUND, "Not found") })
        // /classify enforces MAX_FILE_SIZE_MB itself.
        .layer(DefaultBodyLimit::disable());
    if config.enable_playground {
        router = router.route(
            "/playground",
            get(|| async { Html(include_str!("playground.html")) }),
        );
    }
    if !config.cors_origin.is_empty() {
        let origins: Vec<&str> = config.cors_origin.split(',').map(str::trim).collect();
        let allow = if origins.contains(&"*") {
            AllowOrigin::any()
        } else {
            AllowOrigin::list(origins.iter().filter_map(|o| HeaderValue::from_str(o).ok()))
        };
        router = router.layer(
            CorsLayer::new()
                .allow_origin(allow)
                .allow_methods(Any)
                .allow_headers(Any),
        );
    }
    router.with_state(Arc::new(App {
        slots: Arc::new(Semaphore::new(config.max_concurrency)),
        queued: AtomicUsize::new(0),
        config,
        models,
        ready_timeout,
    }))
}

fn error(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({ "error": message }))).into_response()
}

async fn health(State(app): State<Arc<App>>) -> Response {
    let Some(models) = app.models.borrow().clone() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "loading" })),
        )
            .into_response();
    };
    let config = &app.config;
    Json(json!({
        "status": "ok",
        "model": models.name(),
        "authRequired": !config.api_key.is_empty(),
        "threshold": config.threshold,
        "nsfwClasses": config.nsfw_classes,
        "clip": models.has_clip().then(|| json!({ "categories": config.clip_categories, "threshold": config.clip_threshold })),
        "maxFileSizeMb": megabytes(config.max_file_size),
        "inFlight": config.max_concurrency - app.slots.available_permits(),
        "queued": app.queued.load(Ordering::Relaxed),
    }))
    .into_response()
}

fn megabytes(bytes: usize) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

fn round(n: f64) -> f64 {
    (n * 10_000.0).round() / 10_000.0
}

/// Compares without stopping at the first difference, so timing doesn't leak the key.
fn safe_equal(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |diff, (x, y)| diff | (x ^ y)) == 0
}

struct Queued<'a>(&'a AtomicUsize);
impl Drop for Queued<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl App {
    /// A slot to run one inference, waiting in the queue if needed; None when the queue is full.
    async fn slot(&self) -> Option<OwnedSemaphorePermit> {
        if let Ok(permit) = self.slots.clone().try_acquire_owned() {
            return Some(permit);
        }
        // The guard also gives the place back if the client goes away while waiting.
        let _queued = Queued(&self.queued);
        if self.queued.fetch_add(1, Ordering::Relaxed) >= self.config.max_queue {
            return None;
        }
        self.slots.clone().acquire_owned().await.ok()
    }
}

async fn classify(State(app): State<Arc<App>>, request: Request) -> Response {
    let config = &app.config;
    let (parts, body) = request.into_parts();
    let header = |name| {
        parts
            .headers
            .get(name)
            .and_then(|v: &HeaderValue| v.to_str().ok())
            .unwrap_or("")
    };

    if !config.api_key.is_empty() {
        let token = header("authorization")
            .strip_prefix("Bearer ")
            .unwrap_or(header("x-api-key"));
        if !safe_equal(token.as_bytes(), config.api_key.as_bytes()) {
            return error(StatusCode::UNAUTHORIZED, "Invalid or missing API key");
        }
    }

    let too_large = || {
        error(
            StatusCode::PAYLOAD_TOO_LARGE,
            &format!("Image larger than {} MB", megabytes(config.max_file_size)),
        )
    };
    if header("content-length")
        .parse::<usize>()
        .is_ok_and(|length| length > config.max_file_size)
    {
        return too_large();
    }
    let Ok(body) = to_bytes(body, config.max_file_size).await else {
        return too_large();
    };

    let mut models = app.models.clone();
    let ready = tokio::time::timeout(app.ready_timeout, models.wait_for(Option::is_some)).await;
    let Some(models) = ready
        .ok()
        .and_then(Result::ok)
        .and_then(|models| models.clone())
    else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Model is still loading, retry in a few seconds",
        );
    };

    let mut threshold = config.threshold;
    let query = parts.uri.query().unwrap_or("");
    if let Some(raw) = query
        .split('&')
        .find_map(|pair| pair.strip_prefix("threshold="))
    {
        match raw.parse::<f64>() {
            Ok(t) if (0.0..=1.0).contains(&t) => threshold = t,
            _ => {
                return error(
                    StatusCode::BAD_REQUEST,
                    "threshold must be a number between 0 and 1",
                );
            }
        }
    }

    let Some(image) = read_image(parts.headers.clone(), body).await else {
        return error(
            StatusCode::BAD_REQUEST,
            "Send an image as multipart/form-data (field \"image\") or as a raw image/* body",
        );
    };

    let started = Instant::now();
    let Some(slot) = app.slot().await else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, "1")],
            Json(json!({ "error": "Server busy, retry later" })),
        )
            .into_response();
    };
    let classifier = models.clone();
    let verdict = tokio::task::spawn_blocking(move || {
        let _slot = slot; // held until the inference is over, even if the client is gone
        classifier.classify(&image)
    })
    .await;
    CLASSIFIED.fetch_add(1, Ordering::Relaxed);
    let Verdict {
        predictions,
        categories,
    } = match verdict {
        Ok(Ok(verdict)) => verdict,
        Ok(Err(ClassifyError::Decode)) => {
            return error(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "Unsupported or corrupt image",
            );
        }
        Ok(Err(ClassifyError::Internal(detail))) => return internal(&detail),
        Err(panic) => return internal(&panic.to_string()),
    };

    let probability = |name: &str| {
        CLASSES
            .iter()
            .position(|c| *c == name)
            .map_or(0.0, |i| f64::from(predictions[i]))
    };
    let score: f64 = config
        .nsfw_classes
        .iter()
        .map(|name| probability(name))
        .sum();
    let top = CLASSES.iter().fold(CLASSES[0], |top, name| {
        if probability(name) > probability(top) {
            name
        } else {
            top
        }
    });
    let nsfw = score > threshold;
    let mut flags = Vec::from_iter(nsfw.then_some("nsfw".to_string()));

    let mut response = Map::new();
    let mut flagged = Map::new();
    for category in categories.iter().flatten() {
        let over = category.score > config.clip_threshold;
        flagged.insert(
            category.name.clone(),
            json!({ "score": round(category.score), "match": category.matched, "flagged": over }),
        );
        if over {
            flags.push(category.name.clone());
        }
    }
    response.insert("safe".into(), flags.is_empty().into());
    response.insert("flags".into(), flags.into());
    response.insert("nsfw".into(), nsfw.into());
    response.insert("score".into(), round(score).into());
    response.insert("threshold".into(), threshold.into());
    response.insert(
        "top".into(),
        json!({ "className": top, "probability": round(probability(top)) }),
    );
    response.insert(
        "predictions".into(),
        Value::Object(
            CLASSES
                .iter()
                .map(|name| (name.to_string(), round(probability(name)).into()))
                .collect(),
        ),
    );
    if categories.is_some() {
        response.insert("categories".into(), flagged.into());
    }
    response.insert("model".into(), models.name().into());
    response.insert(
        "timeMs".into(),
        (started.elapsed().as_millis() as u64).into(),
    );
    Json(response).into_response()
}

fn internal(detail: &str) -> Response {
    eprintln!("[classify] {detail}");
    error(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error")
}

/// The uploaded image: the multipart field "image" (or failing that, any file
/// field), or the raw request body.
async fn read_image(headers: HeaderMap, body: Bytes) -> Option<Bytes> {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !content_type.starts_with("multipart/form-data") {
        return (!body.is_empty()).then_some(body);
    }
    let mut request = Request::new(Body::from(body));
    *request.headers_mut() = headers;
    let mut form = Multipart::from_request(request, &()).await.ok()?;
    let mut any_file = None;
    while let Ok(Some(field)) = form.next_field().await {
        let named_image = field.name() == Some("image");
        let is_file = field.file_name().is_some();
        if named_image && !is_file {
            return None; // "image" sent as plain text
        }
        if named_image || (is_file && any_file.is_none()) {
            let bytes = field.bytes().await.ok()?;
            if named_image {
                return Some(bytes);
            }
            any_file = Some(bytes);
        }
    }
    any_file
}

#[cfg(test)]
mod tests {
    //! HTTP behaviour with a stub classifier: fast, no model involved.
    use super::*;
    use crate::models::CategoryScore;
    use tower::ServiceExt;

    type Outcome = Result<Verdict, ClassifyError>;

    struct Stub<F>(F, bool);
    impl<F: Fn() -> Outcome + Send + Sync> Classify for Stub<F> {
        fn name(&self) -> &str {
            "Stub"
        }
        fn has_clip(&self) -> bool {
            self.1
        }
        fn classify(&self, _: &[u8]) -> Outcome {
            (self.0)()
        }
    }

    // Drawing, Hentai, Neutral, Porn, Sexy
    const SCORES: [f32; 5] = [0.05, 0.05, 0.6, 0.2, 0.1];
    const NEUTRAL: [f32; 5] = [0.0, 0.0, 1.0, 0.0, 0.0];

    fn scores() -> Outcome {
        Ok(Verdict {
            predictions: SCORES,
            categories: None,
        })
    }

    fn with_clip(
        predictions: [f32; 5],
        categories: &'static [(&'static str, f64)],
    ) -> impl Fn() -> Outcome + Send + Sync {
        move || {
            let categories = categories
                .iter()
                .map(|(name, score)| CategoryScore {
                    name: name.to_string(),
                    score: *score,
                    matched: Some(format!("a photo of {name}")),
                })
                .collect();
            Ok(Verdict {
                predictions,
                categories: Some(categories),
            })
        }
    }

    fn config(edit: impl FnOnce(&mut Config)) -> Config {
        let mut config = Config::load(&|_| None).unwrap();
        edit(&mut config);
        config
    }

    fn app_with(
        config: Config,
        stub: impl Fn() -> Outcome + Send + Sync + 'static,
        clip: bool,
    ) -> Router {
        let (tx, rx) = watch::channel(Some(Arc::new(Stub(stub, clip)) as Arc<dyn Classify>));
        std::mem::forget(tx); // keep the channel open
        router(config, rx, Duration::from_secs(60))
    }

    fn app() -> Router {
        app_with(config(|_| {}), scores, false)
    }

    /// An app whose models never finish loading.
    fn loading(ready_timeout: Duration) -> (Router, watch::Sender<Option<Arc<dyn Classify>>>) {
        let (tx, rx) = watch::channel(None);
        (router(config(|_| {}), rx, ready_timeout), tx)
    }

    async fn send(app: &Router, request: Request) -> (StatusCode, HeaderMap, String) {
        let response = app.clone().oneshot(request).await.unwrap();
        let (parts, body) = response.into_parts();
        let body = to_bytes(body, usize::MAX).await.unwrap();
        (
            parts.status,
            parts.headers,
            String::from_utf8(body.to_vec()).unwrap(),
        )
    }

    async fn get_json(app: &Router, path: &str) -> (StatusCode, Value) {
        let (status, _, body) = send(app, Request::get(path).body(Body::empty()).unwrap()).await;
        (status, serde_json::from_str(&body).unwrap_or(Value::Null))
    }

    /// A multipart POST; pair it with `file_part` for the body.
    fn upload(path: &str) -> axum::http::request::Builder {
        Request::post(path).header("content-type", "multipart/form-data; boundary=XX")
    }

    fn file_part(field: &str) -> Body {
        Body::from(format!(
            "--XX\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"photo.jpg\"\r\nContent-Type: image/jpeg\r\n\r\npixels\r\n--XX--\r\n"
        ))
    }

    async fn classify_with(
        app: &Router,
        path: &str,
        headers: &[(&str, &str)],
    ) -> (StatusCode, HeaderMap, Value) {
        let mut request = upload(path);
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        let (status, headers, body) = send(app, request.body(file_part("image")).unwrap()).await;
        (status, headers, serde_json::from_str(&body).unwrap())
    }

    async fn classify(app: &Router) -> (StatusCode, Value) {
        let (status, _, body) = classify_with(app, "/classify", &[]).await;
        (status, body)
    }

    #[tokio::test]
    async fn health_is_503_while_the_models_load() {
        let (app, _tx) = loading(Duration::from_millis(50));
        for path in ["/health", "/"] {
            assert_eq!(
                get_json(&app, path).await,
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    json!({ "status": "loading" })
                )
            );
        }
    }

    #[tokio::test]
    async fn health_describes_the_service_once_ready() {
        let (status, body) = get_json(&app(), "/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            json!({
                "status": "ok", "model": "Stub", "authRequired": false, "threshold": 0.7,
                "nsfwClasses": ["Porn", "Hentai", "Sexy"], "clip": null, "maxFileSizeMb": 10.0, "inFlight": 0, "queued": 0,
            })
        );
        assert_eq!(get_json(&app(), "/").await.1, body);
    }

    #[tokio::test]
    async fn playground_is_served_unless_disabled() {
        let (status, headers, body) = send(
            &app(),
            Request::get("/playground").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            headers[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .contains("text/html")
        );
        assert!(body.contains("NSFW Detection API"));

        let disabled = app_with(config(|c| c.enable_playground = false), scores, false);
        assert_eq!(
            get_json(&disabled, "/playground").await,
            (StatusCode::NOT_FOUND, json!({ "error": "Not found" }))
        );
    }

    #[tokio::test]
    async fn classify_accepts_multipart_and_computes_the_verdict() {
        let (status, body) = classify(&app()).await;
        assert_eq!(status, StatusCode::OK);
        // default NSFW classes are Porn + Hentai + Sexy = 0.35, below 0.7
        assert_eq!(
            (
                &body["nsfw"],
                &body["score"],
                &body["threshold"],
                &body["model"]
            ),
            (&json!(false), &json!(0.35), &json!(0.7), &json!("Stub"))
        );
        assert_eq!(
            body["top"],
            json!({ "className": "Neutral", "probability": 0.6 })
        );
        assert_eq!(
            body["predictions"],
            json!({ "Drawing": 0.05, "Hentai": 0.05, "Neutral": 0.6, "Porn": 0.2, "Sexy": 0.1 })
        );
        assert_eq!((&body["safe"], &body["flags"]), (&json!(true), &json!([])));
        assert!(body.get("categories").is_none());
    }

    #[tokio::test]
    async fn classify_accepts_a_raw_body_and_any_file_field() {
        let raw = Request::post("/classify")
            .header("content-type", "image/png")
            .body(Body::from("pixels"))
            .unwrap();
        assert_eq!(send(&app(), raw).await.0, StatusCode::OK);
        let other_field = upload("/classify").body(file_part("file")).unwrap();
        assert_eq!(send(&app(), other_field).await.0, StatusCode::OK);
    }

    #[tokio::test]
    async fn classify_honours_threshold_and_nsfw_classes() {
        let app = app_with(config(|c| c.nsfw_classes = vec!["Porn"]), scores, false);
        let (_, _, body) = classify_with(&app, "/classify?threshold=0.1", &[]).await;
        assert_eq!(
            (&body["nsfw"], &body["score"], &body["threshold"]),
            (&json!(true), &json!(0.2), &json!(0.1))
        );
        assert_eq!(body["flags"], json!(["nsfw"]));
        for bad in ["-1", "2", "abc"] {
            assert_eq!(
                classify_with(&app, &format!("/classify?threshold={bad}"), &[])
                    .await
                    .0,
                StatusCode::BAD_REQUEST
            );
        }
    }

    #[tokio::test]
    async fn classify_is_400_without_an_image() {
        let empty_form = upload("/classify").body(Body::from("--XX--\r\n")).unwrap();
        assert_eq!(send(&app(), empty_form).await.0, StatusCode::BAD_REQUEST);
        let empty_body = Request::post("/classify").body(Body::empty()).unwrap();
        assert_eq!(send(&app(), empty_body).await.0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn classify_is_413_above_max_file_size() {
        let app = app_with(config(|c| c.max_file_size = 1024), scores, false);
        let big = || Request::post("/classify").header("content-type", "image/jpeg");
        // with and without a Content-Length header
        let declared = big()
            .header("content-length", "4096")
            .body(Body::from(vec![0; 4096]))
            .unwrap();
        assert_eq!(send(&app, declared).await.0, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            send(&app, big().body(Body::from(vec![0; 4096])).unwrap())
                .await
                .0,
            StatusCode::PAYLOAD_TOO_LARGE
        );
    }

    #[tokio::test]
    async fn classify_is_415_when_the_image_cannot_be_decoded() {
        let app = app_with(config(|_| {}), || Err(ClassifyError::Decode), false);
        assert_eq!(classify(&app).await.0, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }

    #[tokio::test]
    async fn classify_hides_internal_errors() {
        let app = app_with(
            config(|_| {}),
            || Err(ClassifyError::Internal("secret detail".into())),
            false,
        );
        let (status, _, body) =
            send(&app, upload("/classify").body(file_part("image")).unwrap()).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!body.contains("secret"));
    }

    #[tokio::test]
    async fn classify_gives_up_with_503_when_the_models_take_too_long() {
        let (app, _tx) = loading(Duration::from_millis(50));
        assert_eq!(classify(&app).await.0, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn classify_waits_for_models_that_are_about_to_be_ready() {
        let (app, tx) = loading(Duration::from_secs(60));
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            tx.send_replace(Some(Arc::new(Stub(scores, false))));
            tokio::time::sleep(Duration::from_secs(5)).await; // keep the channel open
        });
        assert_eq!(classify(&app).await.0, StatusCode::OK);
    }

    #[tokio::test]
    async fn classify_is_429_when_the_queue_is_full() {
        let (release, gate) = std::sync::mpsc::channel::<()>();
        let gate = std::sync::Mutex::new(gate);
        let blocked = move || {
            gate.lock().unwrap().recv().unwrap();
            scores()
        };
        let app = app_with(
            config(|c| (c.max_concurrency, c.max_queue) = (1, 0)),
            blocked,
            false,
        );
        let first = tokio::spawn({
            let app = app.clone();
            async move { classify(&app).await.0 }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(get_json(&app, "/health").await.1["inFlight"], 1);
        let (status, headers, _) = classify_with(&app, "/classify", &[]).await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(headers[header::RETRY_AFTER], "1");
        release.send(()).unwrap();
        assert_eq!(first.await.unwrap(), StatusCode::OK);
        assert_eq!(get_json(&app, "/health").await.1["queued"], 0);
    }

    #[tokio::test]
    async fn api_key_protects_classify_only() {
        let app = app_with(config(|c| c.api_key = "s3cret".into()), scores, false);
        let status =
            async |headers: &[(&str, &str)]| classify_with(&app, "/classify", headers).await.0;
        assert_eq!(status(&[]).await, StatusCode::UNAUTHORIZED);
        assert_eq!(
            status(&[("authorization", "Bearer nope")]).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status(&[("authorization", "Bearer s3cret")]).await,
            StatusCode::OK
        );
        assert_eq!(status(&[("x-api-key", "s3cret")]).await, StatusCode::OK);
        assert_eq!(get_json(&app, "/health").await.0, StatusCode::OK);
    }

    #[tokio::test]
    async fn clip_adds_categories_flags_and_safe() {
        let app = app_with(
            config(|_| {}),
            with_clip(NEUTRAL, &[("weapon", 0.91), ("gore", 0.02)]),
            true,
        );
        let (_, body) = classify(&app).await;
        assert_eq!(
            (&body["safe"], &body["flags"], &body["nsfw"]),
            (&json!(false), &json!(["weapon"]), &json!(false))
        );
        assert_eq!(
            body["categories"],
            json!({
                "weapon": { "score": 0.91, "match": "a photo of weapon", "flagged": true },
                "gore": { "score": 0.02, "match": "a photo of gore", "flagged": false },
            })
        );

        let calm = app_with(config(|_| {}), with_clip(NEUTRAL, &[("weapon", 0.1)]), true);
        let (_, body) = classify(&calm).await;
        assert_eq!((&body["safe"], &body["flags"]), (&json!(true), &json!([])));
    }

    #[tokio::test]
    async fn clip_lists_nsfw_first_when_both_trigger() {
        let app = app_with(
            config(|_| {}),
            with_clip([0.0, 0.0, 0.0, 1.0, 0.0], &[("gore", 0.9)]),
            true,
        );
        assert_eq!(classify(&app).await.1["flags"], json!(["nsfw", "gore"]));
    }

    #[tokio::test]
    async fn clip_is_described_in_health() {
        let app = app_with(
            config(|c| (c.clip_categories, c.clip_threshold) = (vec!["weapon".into()], 0.6)),
            with_clip(NEUTRAL, &[]),
            true,
        );
        assert_eq!(
            get_json(&app, "/health").await.1["clip"],
            json!({ "categories": ["weapon"], "threshold": 0.6 })
        );
    }

    #[tokio::test]
    async fn cors_headers_follow_cors_origin() {
        let app = app_with(
            config(|c| c.cors_origin = "https://a.example, https://b.example".into()),
            scores,
            false,
        );
        let ask = async |origin: &str| {
            let request = Request::get("/health")
                .header("origin", origin)
                .body(Body::empty())
                .unwrap();
            send(&app, request)
                .await
                .1
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .cloned()
        };
        assert_eq!(ask("https://b.example").await.unwrap(), "https://b.example");
        assert!(ask("https://evil.example").await.is_none());
    }
}
