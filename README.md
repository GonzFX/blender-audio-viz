# Audio Viz

3D visualizers that react to audio, without leaving Blender.

![Bars, landscape, plexus and a particle swarm all reacting to the same track](docs/demo.gif)

*Four presets running at once on the same track. Rendered with the plugin; the
GIF has no sound.*

Load a song, the plugin analyzes it right there and hands you presets ready to
animate: bars, LEDs, a plexus, a scrolling landscape, an orbital swarm of particles
and vibrating strings. You can have all of them at once, several times over, each listening
to a different audio file with its own settings.

Needs Blender 5.0 or newer. Nothing else: it uses the audio decoder and the
ffmpeg that Blender already ships.

The interface is in English, with Spanish registered as a translation — if your
Blender is set to Spanish, the panel comes up in Spanish on its own.

---

## Install

1. Download `audio_viz-1.0.0.zip` from [Releases](../../releases).
2. In Blender: **Edit → Preferences → Add-ons**, the arrow at the top right →
   **Install from Disk…**, pick the zip.
3. Tick the "Audio Viz" checkbox.

The panel lives in the 3D viewport: press **N**, tab **Audio Viz**.

To build the zip yourself, from `extension/audio_viz`:

```
blender --factory-startup --command extension build --output-dir ../../dist
```

---

## How it works

You load an audio file and the plugin analyzes it into logarithmic bands (8 by
default, configurable), stores the **raw** values inside an Empty in the scene
and builds animation curves from them.

Keeping the raw values separately is what lets you change the smoothing — the
attack and the decay — at any time without analyzing anything again. Move a
slider and the curves are rebuilt instantly.

The tempo is detected on load. Next to the BPM you get what it rests on: *"5 of
6 chunks of the track agree"*, which is literally what was checked. If the track
has no clear pulse, no tempo is invented.

### The presets

| Preset | What it is |
|---|---|
| **Bars** | The classic equalizer. Runs on drivers, so it needs no plugin to play back. |
| **LEDs** | Columns of separate cubes lighting up from the bottom, green to red. |
| **Beat cubes** | One cube per beat of the bar, growing and lighting on its own beat. |
| **Plexus** | A cloud of points joined by lines. It can be generated from the surface or the volume of any scene object, and the faces can go to a separate object. |
| **Landscape** | A grid with frequencies on one axis and time on the other: the relief travels towards the horizon. |
| **Orbital swarm** | Thousands of particles orbiting a centre, pushed and lit by their band. |
| **Strings** | One string per band, held at both ends and vibrating like a guitar's — plucked by the beat, ringing and fading. |

### What they have in common

- **Several at once.** The settings live on each object, not on the scene, so you
  can have three different plexus listening to three different songs.
- **You can scrub the timeline.** No preset depends on the previous frame: frame
  4000 is worked out without ever passing through 3999.
- **Optional stereo**, if the file has two channels.
- **A Bake button**, which leaves a copy with the animation baked into shape
  keys. That copy works without the plugin installed: for sending the .blend to a
  render farm or to someone who does not have it.
- **Attributes for your own material**: `av_level` (where it sits in the
  spectrum), `av_intensity` (how loud it is right now) and, on the swarm,
  `av_beat`. The panel carries a cheat sheet on how to read them.

### Two ways to react (the swarm)

**Direct push** places the particle according to what sounds *now*: if the audio
jumps, it jumps. Easiest to read, but with busy music it jitters.

**Force with inertia** treats every hit as a shove on something with weight: it
eases out and returns by itself. Measured against the direct push on the same
audio: it shakes 4.5× less and arrives 3 frames late. That lag is exactly what
makes it feel like it has mass.

There is no simulation behind it. A damped spring's response to a force is a
convolution, so the plugin looks *backwards* over the recent audio instead of
carrying state forward. That is what keeps scrubbing and baking working.

### Seeing the analysis

A button draws the whole track as a spectrogram inside Blender: time across,
bands from low to high, the beats along the bottom and a line on the current
frame. It is for tuning the decay by eye instead of blind.

---

## Rendering

For long animations, do it from a console rather than from Blender's window:

```
blender --factory-startup --addons bl_ext.user_default.audio_viz -b "my_scene.blend" -o "out/####" -F PNG -a
```

That way the 3D viewport is never drawn, and with it goes a whole family of
crashes that have nothing to do with the final image. `--factory-startup
--addons …` enables only this plugin, so no other add-on can interfere: measured
pixel by pixel, the image comes out identical.

Numbered PNGs rather than a video on purpose: if it stops halfway, you keep what
was rendered.

---

## What is in the repository

```
extension/audio_viz/
  blender_audio_viz.py     all the code
  traducciones.py          the panel in Spanish
  blender_manifest.toml    name, version, licence
  __init__.py              the hook into Blender

MANUAL-es.txt              long manual, in Spanish
README.md                  this
LICENSE                    GPL-3.0-or-later
```

The plugin is a single Python file. It needs nothing from outside: it uses the
audio decoder and the ffmpeg Blender already ships.

### The standalone analyzer (optional)

[Releases](../../releases) also carries `extra-audio-analyzer-app.exe`: a small
windowed program, independent of Blender, that analyzes an audio file and leaves
the result in a `.json` the plugin can import.

**It is not needed for normal work** — Blender analyzes the audio by itself — but
it is useful for batch-processing many tracks, for saving an analysis to reuse
without repeating it, or for handing someone the analysis without the audio.

It ships alongside `extra-audio-analyzer-source.zip` with its source, because the
licence is GPL and the source has to travel with the binary. And because an
unsigned `.exe` downloaded from the internet deserves to be inspectable.

---

## Status

It works and it is in use. What I know is missing or shaky:

- Tempo detection gets it right on music with a clear pulse, but goes wrong when
  the tempo breathes (live recordings, things played by hand). In those cases you
  can type the BPM by hand and everything else works the same.
- The plexus and the landscape rebuild their geometry every frame, so with many
  points viewport playback suffers. The Bake button solves this for the final
  render.
- Comments and internal function names in the source are in Spanish. The user
  interface is not, so this only affects anyone reading the code.

---

## Licence

GPL-3.0-or-later, as befits a Blender add-on. Full text in [LICENSE](LICENSE).
