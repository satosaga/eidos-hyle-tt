# Analyzer: How Activity, Strategy, and Rebuild Are Compared

The Analyzer's Altitude/Power/Velocity/W'/Δt panels overlay three
different kinds of line — the **Strategy** (the original plan), any
number of **Rebuilds** (a re-simulation, manual or Auto Fit), and the
**Activity** (what a FIT recording says actually happened) — and none
of them share a single, simple distance axis by construction. This page
explains how they're actually lined up, so the numbers in those panels
mean what you think they mean.

## Finding the right lap in the recording

Before any comparison can happen, the Analyzer has to find which part
of a loaded FIT file corresponds to the loaded course: it scans for a
standing-start departure followed by a pass through the course's own
goal coordinate, then checks that the recorded path's shape actually
resembles the course (not just its start/end points) and that its
recorded distance roughly matches the course's own length. If none of
this finds a match, see `docs/RUNBOOK.md`'s "Analyzer fails to match a
recorded FIT file to its course" entry — usually a recording that
doesn't extend far enough before the start or after the finish.

## Why comparisons use "percent of the way through," not raw distance

A rider's actual line through a corner is never identical to the
course's own reference line — cutting the apex shortens it, swinging
wide lengthens it — so "500m" on the Activity and "500m" on the course
don't quite describe the same physical spot. Comparing at the same
**percentage of the way through** each one's own total distance, rather
than at the same raw meter mark, keeps both sides honestly anchored to
"start" and "finish" regardless of that difference.

This is why the Δt panel's x-axis reads as a fraction of the course,
not a distance in meters, and why it's the right way to read "how far
ahead/behind was I at this point" even though the two tracks are never
pixel-perfect copies of each other.

## The Activity's own total distance is pinned to the course's

The Activity's recorded GPS track is fitted to a smooth curve (see
below) before anything is measured from it, and that curve's own total
length is deliberately set to **exactly** match the course's known
distance — rather than left to disagree with it by whatever small
amount (typically well under 1%) the rider's real line choice through
corners produced.

This correction is spread evenly across the whole ride (the curve is
uniformly stretched or compressed about its own middle), not
concentrated at any one point — the same reasoning the Analyzer already
applies when replaying an Activity's recorded power against the
course. Practically: every point on the Activity's track shifts by at
most a small fraction of the total mismatch, never by the whole thing
at once, and the comparison never has to choose between "trust the
rider's odometer" and "trust the course's."

## Smoothed GPS, not the raw recording

The Activity's displayed position, distance, and speed all come from
one smooth curve fitted through its raw GPS points — not from the FIT
file's own recorded speed/odometer fields directly. A device's own
speed reading reacts late coming out of a standing start and is more
heavily smoothed elsewhere in the ride; a curve fitted through position
directly tracks real acceleration more faithfully, particularly in the
opening seconds that matter most for pacing analysis.

## Rebuilds use the same convention

A Rebuild's own finish distance (from its own simulation, not the
course's) stands in for "100%" the same way the course distance does
for the Strategy — so a Rebuild, the Strategy, and the Activity are all
read at the same percentage mark, never at the same raw distance.
