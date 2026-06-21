# `ui/vendor/` — locally vendored frontend assets (air-gap requirement)

Every third-party asset the console needs is committed here so the UI loads with
**zero network access** at runtime. No CDNs, no Google Fonts, no remote imports.

| File | Library | Version | License | SHA-256 |
|---|---|---|---|---|
| `cytoscape.min.js` | [Cytoscape.js](https://js.cytoscape.org/) | 3.30.2 | MIT | `83e8c54a6bec655bfd81df07df605649c268af69aeca67a5ea2da54ea42dac81` |

**Notes**
- The **risk timeline chart is hand-rolled** in `../app.js` on an HTML5 `<canvas>`
  (no charting library) to keep the vendored footprint to a single file.
- **Fonts:** the console uses the OS **system font stack** only
  (`-apple-system, Segoe UI, Roboto, …`) — no web-font download.
- Re-vendoring (on a connected build host only):
  ```bash
  curl -fsSL https://cdn.jsdelivr.net/npm/cytoscape@3.30.2/dist/cytoscape.min.js \
    -o cytoscape.min.js
  sha256sum cytoscape.min.js   # must match the table above
  ```
- Verify no remote refs remain anywhere in `ui/` (the test in
  `tests/test_api.py::test_ui_is_air_gapped` enforces this):
  ```bash
  ! grep -rEn "https?://" ui --include='*.html' --include='*.js' --include='*.css' \
      | grep -vE "w3\.org|localhost|127\.0\.0\.1"
  ```
