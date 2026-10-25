# Assets

**This folder holds the banner at the top of the main README, in four versions, and the benchmark figure, in two.** Each is an SVG file drawn with code, with no outside fonts, images or scripts, and each is under 50 KB.

<br>

## The four files

| File | Theme | Motion |
| --- | --- | --- |
| [vade-banner-dark.svg](vade-banner-dark.svg) | Dark | The faint field drifts, a soft light passes over the letters, a few letter cells flicker |
| [vade-banner-light.svg](vade-banner-light.svg) | Light | The same, on a pale field |
| [vade-banner-dark-static.svg](vade-banner-dark-static.svg) | Dark | None |
| [vade-banner-light-static.svg](vade-banner-light-static.svg) | Light | None |

<br>

## What it shows

- **VADE**, drawn large in monospace characters, the same letter shapes as on [vade.lol](https://vade.lol).
- **One line beneath it:** an LLM inference engine and agent harness, written in Rust, built for the DGX Spark.
- **The project sites** in the corner: vade.lol and getvade.com.

<br>

## How the right one is chosen

**The README uses a `<picture>` element, and the browser picks.**

- Dark mode shows a dark banner; light mode shows a light one.
- If your system asks for reduced motion, you get a still version.
- The moving files also stop moving on their own when reduced motion is on.

<br>

## Why it looks this way

- It copies the look of [vade.lol](https://vade.lol) as it renders: a deep indigo, nearly black field with a few tiny dots and dashes, violet letters lit from the upper left, a soft veil at the edges.
- The colors are the site's indigo and violet. The light version keeps the violet letters on a near-white lavender field.
- The type follows the site: monospace for the letters and the corner, a plain sans-serif for the tagline. The files name system fonts only, so nothing is downloaded.
- The motion is slow and low in contrast: a drift of a few pixels a second, one soft light every twelve seconds, no flashing.

<br>

## The benchmark figure

**[benchmarks-2026-10-24-light.svg](benchmarks-2026-10-24-light.svg) and [benchmarks-2026-10-24-dark.svg](benchmarks-2026-10-24-dark.svg) compare one DGX Spark with two.** Both use drafting ON, the default, on our own Sparks: writing speed for code, prose and eight chats, and the wait for the first reply after an 8K-token prompt. They appear in the main README and in [Benchmarks](../benchmarks.md#current-2026-10-24), chosen by the same `<picture>` light/dark rule; they do not move.

- Do not edit them by hand. `python3 -I tools/benchmarks-chart.py` writes both from the committed receipts, so every number is a value a receipt prints.
- The headline says in words what the bars show. The script checks it against every same-day home card with the same settings, six single Sparks and three pairs, and stops if any of them disagrees. Under the eight-chat bars, a note gives the range those cards show.
- 1 Spark is blue (`#2a78d6` light, `#3987e5` dark) and 2 Sparks is orange (`#eb6834` light, `#d95926` dark), a pair chosen to stay apart for colorblind readers. Every bar also carries its name, so color is never the only clue. Text uses neutral ink, never the bar colors.
- Drafting OFF and the cloud VMs stay in the tables on the [Benchmarks](../benchmarks.md#current-2026-10-24) page.

<br>

## Rules for changes

- Keep each file under 50 KB.
- No `<script>`, no `<foreignObject>`, no links to other files or sites.
- Keep a still version for every moving one.

The engine and harness banners are the same design: [`engine/docs/assets/`](../../engine/docs/assets/README.md) and [`harness/docs/assets/`](../../harness/docs/assets/README.md). The engine's older logo drawings live in the engine folder too.

<br>

**Back:** [Docs index](../README.md) · **Next:** [README](../../README.md) →
