# Changelog

## 0.3.0 (first release on PyPI)

The import package is `asciimagic`; install with `pipx install ascii-magic-tools`
(extras: `[web]`, `[video]`, `[translate]`, `[subject]`). Python 3.11+.

### Pictures
- Image to ASCII by glyph matching or braille dots, with color from the picture,
  themes, gradient overlays, and captions.
- **Line art**: trace outlines instead of shading, which makes logos, drawings
  and faces much clearer (`--line-art`, `--line-detail`).
- **Subject focus**: remove, blur, fade, or replace the background, zoom to the
  subject, or convert just a region (on-device model, downloaded on first use).
- Matrix rain, still or animated.
- `tune` renders the same picture many ways side by side; presets save a look.

### Text
- Crisp block, small, shadow, solid, and figlet styles, plus frames (box,
  rounded, double, heavy, banner) that scale with the letters.
- Japanese, Chinese, and other non-Latin text.
- Free offline translation (English to Japanese and ~100 languages).
- 24 animated effects (wave, spin, flips, bounce, typewriter, rainbow...) as
  terminal playback, GIF, HTML, or a login greeting.

### Video
- Video and webcam to ASCII: terminal playback, GIF, HTML, or MP4 with the
  original sound; a live ASCII mirror from a camera.

### Compose
- Layer images and text anywhere on one canvas; save the scene as JSON.

### Output
- ANSI (truecolor, 256, or 16 colors for SSH), HTML, plain text, **PNG**
  (looks the same everywhere, transparent background optional), GIF, MP4,
  and `.frames` login animations.

### Web GUI
- `ascii-magic web`: every feature in the browser with a live preview, resize
  handles, and a download menu. There is no login: run it on your own machine
  or a network you trust (see "Server limits" in the README).

### Security
- Dependency floors are releases with no known vulnerabilities, and CI scans
  the lockfile weekly.
- Uploads are size-checked before decoding, and uploaded video goes through a
  restricted ffmpeg. Model downloads are capped and run one at a time, and
  pages can't be framed by other sites.
