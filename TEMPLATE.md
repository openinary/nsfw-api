# Publishing the Railway template

Railway templates are created in the dashboard from a public GitHub repo. This repo already contains everything the service needs (`Dockerfile` + `railway.json`), so the template itself is just one service with a few variables.

## 1. Push the repo

```bash
gh repo create openinary/nsfw-api --public --source . --push
```

## 2. Create the template

Railway dashboard → **Workspace settings → Templates → New Template**.

**Service**: *Add service → GitHub repo* → `openinary/nsfw-api`, branch `main`.

**Settings → Networking**: enable *Public networking* (HTTP). Railway generates the domain at deploy time.

**Variables** (in the service's *Variables* tab of the template editor):

| Name | Value | Description shown to users |
| --- | --- | --- |
| `API_KEY` | `${{secret(32)}}` | Key required on `/classify`. Generated for you; clear it for an open API. |
| `NSFW_THRESHOLD` | `0.7` | Score above which an image is flagged. |
| `NSFW_CLASSES` | `Porn,Hentai,Sexy` | Classes counted as NSFW. |
| `MAX_FILE_SIZE_MB` | `10` | Max upload size. |
| `CLIP_CATEGORIES` | `weapon,gore,drugs` | Extra categories detected with CLIP. `none` to turn them off. |
| `CLIP_THRESHOLD` | `0.5` | Score above which a CLIP category is flagged. |

Leave `PORT` unset; Railway injects it. Everything else has defaults in the code; see the README for the full list.

The healthcheck (`/health`, 120 s timeout) and restart policy come from `railway.json`, so there is nothing to configure for them in the template editor.

App Sleeping is left off so the first request never waits. For an API called only now and then, it is worth mentioning in the overview: the service then costs almost nothing while idle, and a request that wakes it waits for the models (~0.75 s) instead of failing.

## 3. Template listing

**Name**: NSFW API (nudity, weapons, gore, drugs)

**Short description**: Self-hosted image moderation API: nudity, weapons, gore, drugs. No GPU, ~75 ms per image.

**Category**: AI/ML

**Overview** (markdown shown on the template page):

```markdown
Deploy a private image moderation API in one click.

- Nudity (Drawing, Hentai, Neutral, Porn, Sexy) with [nsfwjs](https://github.com/infinitered/nsfwjs)'s model
- Weapons, gore and drugs with Apple MobileCLIP-S0 (zero-shot, prototype)
- `POST /classify` with a file and get `{ safe, flags, predictions, categories }` back
- JPEG, PNG, WebP, AVIF, GIF and more
- ~75 ms per image on 1 vCPU, ~125 MB RAM idle (~140 MB under load), no GPU
- Turn on App Sleeping to pay almost nothing while it is idle
- Send ~512 px thumbnails rather than originals: 2 to 6× less CPU on large photos
- API key generated for you; playground at `/playground` to try it in the browser
- No data stored; images stay in memory for the length of the request

After deploy, open `https://<your-domain>/playground` to try it, or:

    curl -H "Authorization: Bearer $API_KEY" -F image=@photo.jpg https://<your-domain>/classify

The models only look at a 224–256 px version of each image, so send a thumbnail (~512 px, at least 256 px on its shortest side) when your app has one: a 4000×3000 WebP takes ~414 ms, its 512 px thumbnail ~69 ms.
```

## 4. Before publishing

**License first**: read the README's *MobileCLIP license* note. Apple moved MobileCLIP to a research-only license in August 2025; this template uses a copy published under the earlier 2024 license. Decide whether that is acceptable before making the template public, or switch back to CLIP ViT-B/32 (MIT).

1. Deploy the template once from its URL in a fresh project.
2. Run the smoke test against it:
   ```bash
   API_KEY=<value from the Variables tab> npm run smoke -- https://<generated-domain>
   ```
3. Put the template code in the README's *Deploy on Railway* button.

