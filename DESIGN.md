---
name: footfall viewer
colors:
  ink: "#12151A"
  muted: "#545B66"
  background: "#FFFFFF"
  surface: "#EEF0F3"
  line: "#D5D9DF"
  floor: "#F7F8FA"
  accent: "#087A50"        # one meaning only: occupancy (the ramp and focus)
  ramp: ["#DDEFE6", "#7CC4A0", "#087A50", "#03311F"]
  ink-dark: "#EEF0F3"
  background-dark: "#0D0F13"
  ramp-dark: ["#12382A", "#1F7A52", "#3FB884", "#C9F2DD"]
typography:
  family: "system-ui"
  weights: [400, 600]
  sizes: [12, 13, 16, 22]
  numerals: tabular-nums
rounded: { control: 12, sheet: 20 }
spacing: { base: 8 }
---

# footfall viewer

The page `footfall serve` hosts (`src/footfall_heatmaps/web/static/index.html`):
one static file, no build step, nothing fetched from outside the server.

## Overview (brand and style)

A measuring instrument for one camera's floor, or for a site's floor with
every camera fused (`serve --site`). The floor grid in metres is the
page; everything else is the control and the reading of what it shows.

## Colors

Monochrome ink on white, and one hue, green, for occupancy only: a four-stop
sequential ramp from light to dark (dark mode inverts it so higher reads
brighter). Empty cells are the floor colour, never a ramp colour. Controls use
ink: the selected segment is ink on background.

## Typography

System UI. 22 the title, 16 body, 13 labels, values and notes, 12 the axis ticks
on the canvas. Numbers are tabular.

## Layout

Below 960 px one column: the map (its height follows the grid's aspect), then the
panel: controls and the slot's occupancy, then the zone ranking and the summary.
From 960 px two columns: the map takes the full viewport height and the width
its grid needs at that height, with no empty side gutters; the panel takes the
rest (at least 340 px) in two columns of at least 300 px when they fit: controls,
occupancy and legend in the first, the zone ranking and summary beside them.

## Elevation & Depth

None: no sheets, no dialogs, no shadows.

## Shapes

12 px radius on controls; 4 px on the legend bar; square cells on the map.

## Components

- Map: canvas, x to the right and y up, grid lines on the grid's 2 m blocks
  labelled in metres. A stored map (slot, fused scene) is cropped to its
  observed extent (cells above zero in either layer) plus 2 m on each side; the
  live map keeps the whole grid. The ranked zones are outlined in ink and
  numbered as in the zone table.
- Source and layer switches: segmented radio groups. Layer names are exactly
  "Dwell (person-seconds)" and "Passage (crossings)".
- Slot picker: a UTC day and the slots that start on it, each read as
  "on average N people in view" ("on average about N people in view" on a
  single camera); hidden in live mode.
- Occupancy (slot only): people in view on average (22 px), the peak 1-minute
  occupancy with its minute, and the observed time. Identity-free: dwell over
  observed seconds, never a count of ids. On a single camera (not a site) the
  mean reads "about N" (13 px before the number), the peak "about N people",
  and a second note says people hidden from it are not counted, with the error
  range measured on held-out scenes. Never "at least N": a single camera reads
  above the truth for some views.
- Zone table: the top 10 of the grid's 2 m zones by dwell, with their x and y
  ranges in metres, share of the dwell shown and person-seconds.
- Legend: the ramp from 0 to the highest cell shown, with the unit.
- Summary: sum over cells, cells above zero, highest cell, cell size, data time,
  map extent;
  for a site also the cameras fused, the overlap weighting and what passage counts.
- Site mode (`serve --site`): the title names the site; Live and 15-min slot are
  the fused layers. A camera table spans the panel's columns: camera, state,
  processed fps, frames skipped, source drops, refreshed every 2 s. Counts only.
- Empty states: a map with no cell at k says so on the map; a process with no
  live layer opens on its newest slot and never polls for one; choosing Live
  there, or a day without a slot, says why on the map.

## Do's and Don'ts

| Don't | Do |
|---|---|
| A narrow centred column on desktop | Map and panel side by side |
| A second hue for anything but occupancy | Ink for controls and text |
| Tinted cells for empty or suppressed cells | The floor colour |
| Informal or tracking wording | Neutral occupancy and foot-traffic terms |
| "people" on a number counted from ids | "people" only on numbers derived from dwell (occupancy); passage in crossings; the k rule stated as track ids, which are not people |

## Testable rules

| Rule | Value | How it is checked |
|---|---|---|
| Colors off palette | 0 | verify-design |
| Text contrast | >=4.5:1 (3:1 large) | verify-design |
| Touch targets | >=24x24 px (controls are 40 px high) | verify-design (`--min-target`) |
| Type | 1 family, the weights and sizes above | verify-design |
| Horizontal overflow at 360, 390 and 1440 px | none | Playwright `scrollWidth` |
