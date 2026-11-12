//! The two models, run by tract (pure Rust ONNX inference):
//!  - nudity: nsfwjs's MobileNetV2Mid, converted to ONNX by scripts/prepare-nsfw.ts;
//!  - optional zero-shot categories: MobileCLIP-S0's vision model against the
//!    prompt embeddings computed by scripts/prepare-clip.ts.
use crate::config::CLASSES;
use crate::pixels::{DecodeError, Fit, Rgb};
use fast_image_resize::FilterType;
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;

pub const MODEL_NAME: &str = "MobileNetV2Mid";
const NSFW_SIZE: usize = 224;
/// MobileCLIP-S0 preprocessing: shortest edge to 256 (bilinear), centre crop,
/// pixels scaled to 0..1 with no mean/std normalisation.
const CLIP_SIZE: usize = 256;
const CLIP_MODEL: &str = "Xenova/mobileclip_s0"; // must match scripts/prepare-clip.ts
const LOGIT_SCALE: f32 = 100.0; // MobileCLIP's learned temperature

/// One probability per entry of `CLASSES`.
pub type Predictions = [f32; CLASSES.len()];

pub struct CategoryScore {
    pub name: String,
    pub score: f64,
    /// The category's prompt CLIP matched best.
    pub matched: Option<String>,
}

pub struct Verdict {
    pub predictions: Predictions,
    /// None when CLIP is disabled.
    pub categories: Option<Vec<CategoryScore>>,
}

#[derive(Debug)]
pub enum ClassifyError {
    /// Unsupported or corrupt image.
    Decode,
    Internal(String),
}

impl From<DecodeError> for ClassifyError {
    fn from(_: DecodeError) -> Self {
        ClassifyError::Decode
    }
}

impl From<TractError> for ClassifyError {
    fn from(err: TractError) -> Self {
        ClassifyError::Internal(format!("{err:#}"))
    }
}

/// What the HTTP layer needs from the models (tests swap in a stub).
pub trait Classify: Send + Sync {
    fn name(&self) -> &str;
    fn has_clip(&self) -> bool;
    /// CPU-bound: call it off the async runtime.
    fn classify(&self, image: &[u8]) -> Result<Verdict, ClassifyError>;
}

type Plan = Arc<TypedRunnableModel>;

fn load(path: &Path, shape: [usize; 4]) -> TractResult<Plan> {
    tract_onnx::onnx()
        // The fp16 CLIP export annotates its tensors with a symbolic batch size,
        // which clashes with the fixed input shape set below.
        .with_ignore_value_info(true)
        .model_for_path(path)
        .map_err(|err| err.context(format!("loading {}", path.display())))?
        .with_input_fact(0, f32::fact(shape).into())?
        .into_optimized()?
        .into_runnable()
}

fn run(plan: &Plan, shape: [usize; 4], data: Vec<f32>) -> TractResult<Vec<f32>> {
    let input: Tensor = tract_ndarray::Array4::from_shape_vec(shape, data)?.into();
    let output = plan.run(tvec!(input.into()))?;
    Ok(output[0]
        .to_plain_array_view::<f32>()?
        .iter()
        .copied()
        .collect())
}

/// models/clip/prompts.json, written by scripts/prepare-clip.ts.
#[derive(Deserialize)]
struct PromptsFile {
    model: String,
    #[serde(default)]
    variant: String,
    dim: usize,
    prompts: Vec<Prompt>,
}

#[derive(Deserialize)]
struct Prompt {
    category: String,
    text: String,
    embedding: Vec<f32>,
}

struct Clip {
    plan: Plan,
    categories: Vec<String>,
    /// The enabled categories' prompts plus the "other" ones: only those compete in the softmax.
    prompts: Vec<Prompt>,
    variant: String,
}

impl Clip {
    fn load(dir: &Path, categories: &[String]) -> Result<Clip, String> {
        let path = dir.join("prompts.json");
        let raw = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file: PromptsFile =
            serde_json::from_slice(&raw).map_err(|e| format!("{}: {e}", path.display()))?;
        if file.model != CLIP_MODEL {
            return Err(format!(
                "{} was made for {}, not {CLIP_MODEL}: run \"npm run prepare-models\" again",
                path.display(),
                file.model
            ));
        }
        for c in categories {
            if c == "other" || !file.prompts.iter().any(|p| &p.category == c) {
                return Err(format!("Unknown CLIP category \"{c}\""));
            }
        }
        let prompts: Vec<Prompt> = file
            .prompts
            .into_iter()
            .filter(|p| p.category == "other" || categories.contains(&p.category))
            .collect();
        if prompts.iter().any(|p| p.embedding.len() != file.dim) {
            return Err(format!(
                "{}: embeddings don't match dim {}",
                path.display(),
                file.dim
            ));
        }
        let plan = load(&dir.join("vision.onnx"), [1, 3, CLIP_SIZE, CLIP_SIZE])
            .map_err(|e| format!("{e:#}"))?;
        Ok(Clip {
            plan,
            categories: categories.to_vec(),
            prompts,
            variant: file.variant,
        })
    }

    fn classify(&self, image: &Rgb) -> TractResult<Vec<CategoryScore>> {
        let hwc = image.resize(CLIP_SIZE as u32, Fit::Cover, FilterType::Bilinear);
        let plane = CLIP_SIZE * CLIP_SIZE;
        let mut chw = vec![0f32; 3 * plane];
        for (i, px) in hwc.chunks_exact(3).enumerate() {
            for c in 0..3 {
                chw[c * plane + i] = f32::from(px[c]) / 255.0;
            }
        }
        let embedding = run(&self.plan, [1, 3, CLIP_SIZE, CLIP_SIZE], chw)?;
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        let logits: Vec<f32> = self
            .prompts
            .iter()
            .map(|p| {
                LOGIT_SCALE
                    * p.embedding
                        .iter()
                        .zip(&embedding)
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                    / norm
            })
            .collect();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = logits.iter().map(|l| f64::from(l - max).exp()).collect();
        let total: f64 = weights.iter().sum();

        Ok(self
            .categories
            .iter()
            .map(|name| {
                let mut score = 0.0;
                let mut best: Option<(f64, &str)> = None;
                for (prompt, weight) in self
                    .prompts
                    .iter()
                    .zip(&weights)
                    .filter(|(p, _)| &p.category == name)
                {
                    let p = weight / total;
                    score += p;
                    if best.is_none_or(|(top, _)| p > top) {
                        best = Some((p, &prompt.text));
                    }
                }
                CategoryScore {
                    name: name.clone(),
                    score,
                    matched: best.map(|(_, text)| text.to_string()),
                }
            })
            .collect())
    }
}

pub struct Models {
    nsfw: Plan,
    clip: Option<Clip>,
}

impl Models {
    /// Loads the models from `dir` (models/) and runs one inference on a blank
    /// image so the first real request isn't slow.
    pub fn load(dir: &Path, clip_categories: &[String]) -> Result<Models, String> {
        // Both loads are CPU-bound: do them side by side.
        let (nsfw, clip) = std::thread::scope(|scope| {
            let clip = scope.spawn(|| {
                (!clip_categories.is_empty())
                    .then(|| Clip::load(&dir.join("clip"), clip_categories))
                    .transpose()
            });
            let nsfw = load(
                &dir.join("nsfw/mobilenet_v2_mid.onnx"),
                [1, NSFW_SIZE, NSFW_SIZE, 3],
            )
            .map_err(|e| format!("{e:#}"));
            (nsfw, clip.join().expect("CLIP loading panicked"))
        });
        let (nsfw, clip) = (nsfw?, clip?);
        let models = Models { nsfw, clip };
        models
            .run(&Rgb::blank(NSFW_SIZE as u32))
            .map_err(|e| format!("{e:#}"))?;
        Ok(models)
    }

    pub fn clip_variant(&self) -> Option<&str> {
        self.clip.as_ref().map(|c| c.variant.as_str())
    }

    fn nsfw(&self, image: &Rgb) -> TractResult<Predictions> {
        // NHWC, 0..1, as nsfwjs; stretched to the input size, as nsfwjs would.
        let pixels = image.resize(NSFW_SIZE as u32, Fit::Fill, FilterType::Lanczos3);
        let input = pixels.iter().map(|&v| f32::from(v) / 255.0).collect();
        let scores = run(&self.nsfw, [1, NSFW_SIZE, NSFW_SIZE, 3], input)?;
        scores
            .try_into()
            .map_err(|_| TractError::msg("the nudity model must output one score per class"))
    }

    fn run(&self, image: &Rgb) -> TractResult<Verdict> {
        let Some(clip) = &self.clip else {
            return Ok(Verdict {
                predictions: self.nsfw(image)?,
                categories: None,
            });
        };
        // Both models are single-threaded: run them side by side.
        let (predictions, categories) = std::thread::scope(|scope| {
            let clip = scope.spawn(|| clip.classify(image));
            (
                self.nsfw(image),
                clip.join().expect("CLIP inference panicked"),
            )
        });
        Ok(Verdict {
            predictions: predictions?,
            categories: Some(categories?),
        })
    }
}

impl Classify for Models {
    fn name(&self) -> &str {
        MODEL_NAME
    }

    fn has_clip(&self) -> bool {
        self.clip.is_some()
    }

    fn classify(&self, image: &[u8]) -> Result<Verdict, ClassifyError> {
        Ok(self.run(&Rgb::decode(image)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_prompts_embedded_with_another_clip_model() {
        let dir = std::env::temp_dir().join(format!("nsfw-api-clip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("prompts.json"),
            r#"{"model":"Xenova/clip-vit-base-patch32","prompts":[],"dim":512}"#,
        )
        .unwrap();
        let err = Clip::load(&dir, &["weapon".into()]).err().unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(err.contains("prepare-models"), "{err}");
    }
}
