<p align="center">
    <img src="assets/icon.svg" width="120" alt="Openinary NSFW API" />
</p>

<h1 align="center">Openinary NSFW API</h1>

<p align="center">
    <a href="https://github.com/openinary/nsfw-api/blob/main/LICENSE" target="_blank" rel="noopener"><img src="https://img.shields.io/github/license/openinary/nsfw-api.svg" alt="license" /></a>
</p>

[Openinary NSFW API](https://github.com/openinary/nsfw-api) is a self-hosted image moderation API that:

- detects nudity with [nsfwjs](https://github.com/infinitered/nsfwjs)'s model
- flags weapons, gore and drugs with Apple's [MobileCLIP](https://github.com/apple/ml-mobileclip)
- accepts any image format: JPEG, PNG, WebP, AVIF, GIF, TIFF, SVG
- runs on a small instance: ~75 ms per image on 1 vCPU, ~140 MB of RAM, no GPU
- ships as a single Docker image, with no external service

**For the full API, every setting and the benchmarks, see [docs/details.md](docs/details.md).**

> [!WARNING]
> Weapon, gore and drug detection is a prototype: its scores are not calibrated, so test it on your own images.
> Read the [MobileCLIP license note](docs/details.md#mobileclip-license) before any commercial use.

## Quick Start

<p align="left">
    <a href="https://railway.com/deploy/nsfw-api?referralCode=imHr5N&utm_medium=integration&utm_source=template&utm_campaign=generic" target="_blank" rel="noopener" style="display: inline-block;"><img src="https://www.openinary.dev/deploy/railway.svg" alt="Deploy with Railway" style="display: block;" /></a>
</p>

Or run it with Docker:

```bash
docker build -t nsfw-api .
docker run -p 8080:8080 -e API_KEY=your-key nsfw-api
```

Then open http://localhost:8080/playground to try it in your browser.

## Usage

```bash
curl -X POST https://your-app.up.railway.app/classify \
  -H "Authorization: Bearer $API_KEY" \
  -F image=@photo.jpg
```

```json
{
  "safe": false,
  "flags": ["weapon"],
  "nsfw": false,
  "score": 0.0036,
  "categories": {
    "weapon": { "score": 0.91, "match": "a person holding a gun", "flagged": true },
    "gore": { "score": 0.02, "match": "a photo of a bloody wound", "flagged": false },
    "drugs": { "score": 0.01, "match": "a photo of illegal drugs", "flagged": false }
  },
  "timeMs": 76
}
```

`safe` is `false` as soon as one flag is raised. The response also includes each nudity class's score: see the [full response](docs/details.md#post-classify).

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `API_KEY` | _(empty)_ | Key required on `/classify`. Leave empty for an open API. |
| `NSFW_THRESHOLD` | `0.7` | Nudity score above which an image is flagged. |
| `CLIP_CATEGORIES` | `weapon,gore,drugs` | Extra categories to check, or `none`. |
| `CLIP_THRESHOLD` | `0.5` | Score above which a category is flagged. |

[All variables](docs/details.md#configuration)

## Development

Requires Node 22.18+.

```bash
npm install
npm run prepare-models  # downloads and converts the models
npm run dev             # http://localhost:8080
npm test
```

## Resources

[Details](docs/details.md) | [Railway template](TEMPLATE.md) | [Issues](https://github.com/openinary/nsfw-api/issues) | [Contact](https://x.com/initflorian)

## License

The code is MIT, see [LICENSE](LICENSE). The nsfwjs model is © [Infinite Red](https://github.com/infinitered/nsfwjs) (MIT). MobileCLIP-S0 is © Apple Inc., under the Apple Sample Code License copied into the image at build time.
