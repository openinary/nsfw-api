pub const CLASSES: [&str; 5] = ["Drawing", "Hentai", "Neutral", "Porn", "Sexy"];

#[derive(Clone, Debug)]
pub struct Config {
    pub port: u16,
    pub api_key: String,
    pub threshold: f64,
    pub nsfw_classes: Vec<&'static str>,
    /// Bytes.
    pub max_file_size: usize,
    pub max_concurrency: usize,
    pub max_queue: usize,
    pub clip_categories: Vec<String>,
    pub clip_threshold: f64,
    pub enable_playground: bool,
    pub cors_origin: String,
}

fn number(
    env: &dyn Fn(&str) -> Option<String>,
    name: &str,
    fallback: f64,
    min: f64,
    max: f64,
) -> Result<f64, String> {
    let Some(raw) = env(name).filter(|raw| !raw.is_empty()) else {
        return Ok(fallback);
    };
    match raw.trim().parse::<f64>() {
        Ok(value) if value.is_finite() && value >= min && value <= max => Ok(value),
        _ => Err(format!(
            "{name} must be a number between {min} and {max} (got \"{raw}\")"
        )),
    }
}

impl Config {
    pub fn from_env() -> Result<Config, String> {
        Config::load(&|name| std::env::var(name).ok())
    }

    pub fn load(env: &dyn Fn(&str) -> Option<String>) -> Result<Config, String> {
        let list = |raw: String| {
            raw.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        };

        let nsfw_classes = list(
            env("NSFW_CLASSES")
                .filter(|s| !s.is_empty())
                .unwrap_or("Porn,Hentai,Sexy".into()),
        )
        .into_iter()
        .map(|name| {
            CLASSES
                .into_iter()
                .find(|class| *class == name)
                .ok_or(format!(
                    "NSFW_CLASSES contains unknown class \"{name}\" (allowed: {})",
                    CLASSES.join(", ")
                ))
        })
        .collect::<Result<_, _>>()?;

        let mut clip_categories =
            list(env("CLIP_CATEGORIES").unwrap_or("weapon,gore,drugs".into()));
        clip_categories.retain(|c| c != "none");

        Ok(Config {
            port: number(env, "PORT", 8080.0, 1.0, 65535.0)? as u16,
            api_key: env("API_KEY").unwrap_or_default(),
            threshold: number(env, "NSFW_THRESHOLD", 0.7, 0.0, 1.0)?,
            nsfw_classes,
            max_file_size: (number(env, "MAX_FILE_SIZE_MB", 10.0, 0.1, 100.0)? * 1024.0 * 1024.0)
                as usize,
            max_concurrency: number(env, "MAX_CONCURRENCY", 2.0, 1.0, 64.0)? as usize,
            max_queue: number(env, "MAX_QUEUE", 32.0, 0.0, 10_000.0)? as usize,
            clip_categories,
            clip_threshold: number(env, "CLIP_THRESHOLD", 0.5, 0.0, 1.0)?,
            enable_playground: env("ENABLE_PLAYGROUND").as_deref() != Some("false"),
            cors_origin: env("CORS_ORIGIN").unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(vars: &[(&str, &str)]) -> Result<Config, String> {
        Config::load(&|name| {
            vars.iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.to_string())
        })
    }

    #[test]
    fn has_sensible_defaults() {
        let c = load(&[]).unwrap();
        assert_eq!((c.port, c.api_key.as_str(), c.threshold), (8080, "", 0.7));
        assert_eq!(c.nsfw_classes, ["Porn", "Hentai", "Sexy"]);
        assert_eq!(c.max_file_size, 10 * 1024 * 1024);
        assert!(c.enable_playground);
        assert_eq!(c.clip_categories, ["weapon", "gore", "drugs"]);
        assert_eq!(c.clip_threshold, 0.5);
    }

    #[test]
    fn reads_railway_style_env_vars() {
        let c = load(&[
            ("PORT", "3000"),
            ("API_KEY", "k"),
            ("NSFW_THRESHOLD", "0.5"),
            ("NSFW_CLASSES", "Porn, Hentai"),
            ("MAX_FILE_SIZE_MB", "2"),
            ("ENABLE_PLAYGROUND", "false"),
        ])
        .unwrap();
        assert_eq!((c.port, c.api_key.as_str(), c.threshold), (3000, "k", 0.5));
        assert_eq!(c.nsfw_classes, ["Porn", "Hentai"]);
        assert_eq!(c.max_file_size, 2 * 1024 * 1024);
        assert!(!c.enable_playground);
    }

    #[test]
    fn clip_categories_can_be_narrowed_or_disabled() {
        assert_eq!(
            load(&[("CLIP_CATEGORIES", "weapon, gore")])
                .unwrap()
                .clip_categories,
            ["weapon", "gore"]
        );
        assert!(
            load(&[("CLIP_CATEGORIES", "none")])
                .unwrap()
                .clip_categories
                .is_empty()
        );
        assert!(
            load(&[("CLIP_CATEGORIES", "")])
                .unwrap()
                .clip_categories
                .is_empty()
        );
    }

    #[test]
    fn fails_fast_on_invalid_values() {
        for (var, value, needle) in [
            ("CLIP_THRESHOLD", "2", "CLIP_THRESHOLD"),
            ("NSFW_THRESHOLD", "1.5", "NSFW_THRESHOLD"),
            ("NSFW_CLASSES", "Porn,Gore", "Gore"),
            ("PORT", "abc", "PORT"),
        ] {
            assert!(load(&[(var, value)]).unwrap_err().contains(needle));
        }
    }
}
