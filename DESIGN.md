---
name: space-usage viewer
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

# space-usage viewer

The page `space-usage serve` hosts (`src/space_usage_heatmaps/web/static/index.html`):
one static file, no build step, nothing fetched from outside the server.

## Overview (brand and style)

A measuring instrument for one camera's floor. The floor grid in metres is the
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
panel. From 960 px two columns: the map takes the remaining width and the full
viewport height, the panel is 340 px on the right.

## Elevation & Depth

None: no sheets, no dialogs, no shadows.

## Shapes

12 px radius on controls; 4 px on the legend bar; square cells on the map.

## Components

- Map: canvas, x to the right and y up, grid lines every 2 m labelled in metres.
- Source and layer switches: segmented radio groups. Layer names are exactly
  "Dwell (person-seconds)" and "Passage (people)".
- Slot picker: a UTC day and the slots that start on it.
- Legend: the ramp from 0 to the highest cell shown, with the unit.
- Summary: sum over cells, cells above zero, highest cell, cell size, data time.

## Do's and Don'ts

| Don't | Do |
|---|---|
| A narrow centred column on desktop | Map and panel side by side |
| A second hue for anything but occupancy | Ink for controls and text |
| Tinted cells for empty or suppressed cells | The floor colour |
| Informal or tracking wording | Neutral occupancy and foot-traffic terms |

## Testable rules

| Rule | Value | How it is checked |
|---|---|---|
| Colors off palette | 0 | verify-design |
| Text contrast | >=4.5:1 (3:1 large) | verify-design |
| Touch targets | >=24x24 px (controls are 40 px high) | verify-design (`--min-target`) |
| Type | 1 family, the weights and sizes above | verify-design |
| Horizontal overflow at 360, 390 and 1440 px | none | Playwright `scrollWidth` |
