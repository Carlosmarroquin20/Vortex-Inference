# Diagrams (diagram-as-code)

The architecture diagram is authored as a small, validated **JSON specification** and
rendered to a self-contained, interactive HTML page with
[Archify](https://github.com/tt-a1i/archify). The spec is the committed source of truth;
the rendered `.html` is a build artifact and is **git-ignored** (it embeds a font and a
viewer runtime, about 850 KB, which does not belong in the repository). This keeps the
diagram reviewable in pull requests and stops it drifting into a stale screenshot.

| Spec | Shows |
|---|---|
| [`vortex-architecture.json`](vortex-architecture.json) | Runtime topology: synchronous inference, asynchronous jobs, autoscaling and observability, with guided views for each |

The top-level [`README.md`](../../README.md#architecture) shows a Mermaid overview of the
same topology inline (GitHub renders it natively). The rendered page is published at
**[carlosmarroquin20.github.io/Vortex-Inference](https://carlosmarroquin20.github.io/Vortex-Inference/)**
(GitHub Pages, served from the `gh-pages` branch so `main` stays free of build artifacts).
When the code changes what the diagram shows, update the spec and the Mermaid block
together, then re-render, re-run `add_csp.py` and replace the page on `gh-pages`.

## Render

```bash
npx skills add tt-a1i/archify -g            # or clone https://github.com/tt-a1i/archify
cd docs/diagrams
node <archify>/archify/bin/archify.mjs validate architecture vortex-architecture.json --quality showcase --json
node <archify>/archify/bin/archify.mjs deliver  architecture vortex-architecture.json vortex-architecture.html --quality showcase --json
```

`validate` must report `"ok": true`: a showcase render has to pass Archify's nine
composition checks (single SVG, orthogonal arrows, label/route clearance, no crossings,
corridor spacing, container border runs, route rhythm, legend clearance). `deliver` also
writes a `vortex-architecture.delivery.json` receipt with local paths; it is git-ignored
too.

Open the resulting `.html` in a browser to explore it (pan/zoom, guided views, theme
toggle) or use the viewer's **Export → PNG** for a static image.

## Before publishing

Give each render a Content-Security-Policy before it is published (for example to
GitHub Pages):

```bash
python docs/diagrams/add_csp.py docs/diagrams/vortex-architecture.html
```

It allows only the page's own inline scripts, by SHA-256, plus embedded fonts and images,
and no network at all, so a future Archify change cannot quietly start loading or sending
anything. Re-running it after a new render replaces the tag. GitHub Pages cannot send
headers, which is why the policy is a `<meta>` tag. `tests/test_diagram_csp.py` covers
the script.
