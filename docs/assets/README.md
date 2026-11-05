# Assets

**This folder holds the banner at the top of the main README, moving and still, and the benchmark figure, in two themes.**

<br>

## The banner

| File | What it is |
| --- | --- |
| [vade-banner.webp](vade-banner.webp) | The moving banner: 12 seconds, 30 frames per second, a loop, 1280 x 400 pixels, lossless, about 3.8 MB. |
| [vade-banner-static.png](vade-banner-static.png) | Its first frame, for readers who ask for reduced motion. |

The README uses a `<picture>` element: the browser shows the still frame when your system asks for reduced motion, and the moving banner otherwise. The same dark banner shows in light and dark mode.

<br>

## Where it comes from

**The banner is a recording of [getvade.com](https://getvade.com), the project site, rendered by the site's own code.** Nothing is redrawn by hand, so the two match.

- The site draws VADE on a canvas as a grid of monospace characters over a slow current of dots and dashes. A soft light drifts across the letters, which are lit and beveled; cells under the light change quickly through random characters and settle; small dots far back blink softly. Under the letters is one line of the site's own text, "Pure-Rust inference engine for the DGX Spark", in the site's monospace style. The banner leaves out the site's GitHub link and footer.
- To make the banner, the page is opened in headless Chrome at 1280 x 400 pixels, with the site's nebula palette, the letters sized and placed for a wide banner and the intro skipped. The animation clock is stepped by hand, one thirtieth of a second per frame, and every frame is captured.
- The loop is made by the renderer, not by cross-fading pictures. Over the last four seconds every continuous quantity (the current, the light's path, the highlight sweep, the blinking dots) is blended toward its value twelve seconds earlier, and the cells that were changing at the start are brought back, so the last frame runs into the first. Measured on this file: decode it to raw frames, then compare each frame with the next (mean squared difference over the RGBA values, 0 to 255). The loop step is 32.9, the median ordinary step is 16.7, the largest ordinary step is 70.6, and 77 of the 359 ordinary steps are larger than the loop step.
- The frames are encoded as a lossless animated WebP, so every frame is exact and the loop point gets no codec seam (a lossy file softens ordinary frames and restarts sharp at the first frame, which reads as a jump), and the first frame as a PNG.

To make it again: take the page as the site serves it, fix the palette and the letter size, step the clock, blend the last four seconds as above, capture the frames and encode them. The small Chrome DevTools Protocol script that does this is kept with the lab scripts, outside this repository, so the repository stays Rust and docs. The site's renderer is the only source of the look; when the site changes, record the banner again.

<br>

## Rules for changes

- Do not edit the moving banner by hand. Record it again from the site.
- Keep the loop made by the renderer, never by cross-fading pictures. Measure the loop step against the ordinary frame steps on the committed file and write the numbers here; the loop step must not be larger than the largest ordinary step.
- Keep 30 frames per second and continuous motion: no frame swaps, steps or blinking.
- Keep a still frame next to the moving file.
- Keep the moving file lossless and under 5 MB: the release scan blocks any file over 5 MB, so a larger banner never reaches the public repository. Record at the page's own pixel size (1280 x 400, no 2x capture); a 2x capture scaled down is about twice as large once compressed.

The engine and harness banners use the earlier, quieter drawn design: [`engine/docs/assets/`](../../engine/docs/assets/README.md) and [`harness/docs/assets/`](../../harness/docs/assets/README.md).

<br>

## The benchmark figure

**[benchmarks-2026-10-24-light.svg](benchmarks-2026-10-24-light.svg) and [benchmarks-2026-10-24-dark.svg](benchmarks-2026-10-24-dark.svg) compare one DGX Spark with two.** Both use drafting ON, the default, on our own Sparks: writing speed for code, prose and eight chats, and the wait for the first reply after an 8K-token prompt. They appear in [Benchmarks](../history/benchmarks-2026-10.md#current-2026-10-24), chosen by the same `<picture>` light/dark rule; they do not move.

- Do not edit them by hand. `cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- benchmarks-chart` writes both from the committed receipts, so every number is a value a receipt prints. The Rust SVG writer is under 300 lines.
- The headline says in words what the bars show. The script checks it against every same-day home card with the same settings, six single Sparks and three pairs, and stops if any of them disagrees. Under the eight-chat bars, a note gives the range those cards show.
- 1 Spark is blue (`#2a78d6` light, `#3987e5` dark) and 2 Sparks is orange (`#eb6834` light, `#d95926` dark), a pair chosen to stay apart for colorblind readers. Every bar also carries its name, so color is never the only clue. Text uses neutral ink, never the bar colors.
- Drafting OFF and the cloud VMs stay in the tables on the [Benchmarks](../history/benchmarks-2026-10.md#current-2026-10-24) page.

<br>

## Rules for changes

- Keep each file under 50 KB.
- No `<script>`, no `<foreignObject>`, no links to other files or sites.
- Keep a still version for every moving one.
- Keep the motion continuous: animate only translate and opacity, with linear or ease-in-out timing, seamless loops and no `steps()`, blinking or frame swaps.

The engine and harness banners still use the earlier, quieter version of this design: [`engine/docs/assets/`](../../engine/docs/assets/README.md) and [`harness/docs/assets/`](../../harness/docs/assets/README.md). The engine's older logo drawings live in the engine folder too.

<br>

**Back:** [Docs index](../README.md) · **Next:** [README](../../README.md) →
