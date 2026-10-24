//! Film dust and hair detection on a scanned negative or slide, after its conversion: dust and
//! hair on the film block light, so in the positive they show as **sharp, bright** specks and thin
//! bright lines (sensor dust, [`crate::dust`], shows as soft dark round spots instead). Classical:
//! a grey opening (a local minimum, then maximum, over a few pixels) removes every bright feature
//! narrower than that, so the original minus its opening (a white top-hat) holds exactly the
//! small and thin bright features. Those standing well above the local texture are grouped into
//! blobs; a small roundish blob is a speck (one heal spot), a long thin one a hair or scratch (a
//! heal stroke along its centre line).
//!
//! Sources: morphological top-hat transform (J. Serra, "Image Analysis and Mathematical
//! Morphology", 1982); blob shape from second moments; own design otherwise.

use lcc_raster::{Plane, Rgba8};

/// A found defect: a speck (one point) or a hair or scratch (a path of points), in 0..1 of the
/// image's width and height, with its radius (fraction of the long edge) and how much brighter
/// than its surroundings it is (0..1). A line also suggests where to heal it from: `source`, an
/// offset (0..1 of the width and height) to the side, onto the smoother of the two strips that
/// run beside it.
#[derive(Clone, Debug, PartialEq)]
pub struct FilmDefect {
    pub points: Vec<(f64, f64)>,
    pub radius: f64,
    pub strength: f32,
    pub source: Option<(f64, f64)>,
}

impl FilmDefect {
    pub fn is_line(&self) -> bool {
        self.points.len() > 1
    }
}

/// Most defects reported: the strongest are kept.
const MAX_DEFECTS: usize = 150;
/// Most points on a hair's path.
const MAX_PATH_POINTS: usize = 48;

/// Separable running minimum (`max == false`) or maximum over a square of side `2r + 1`, edges
/// repeated.
fn rank(src: &Plane, r: usize, max: bool) -> Plane {
    let (w, h) = (src.width, src.height);
    let pick = |a: f32, b: f32| if max { a.max(b) } else { a.min(b) };
    let start = if max { f32::NEG_INFINITY } else { f32::INFINITY };
    let mut rows = Plane::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (lo, hi) = (x.saturating_sub(r), (x + r).min(w - 1));
            let mut v = start;
            for xi in lo..=hi {
                v = pick(v, src.data.get(y * w + xi).copied().unwrap_or(start));
            }
            if let Some(o) = rows.data.get_mut(y * w + x) {
                *o = v;
            }
        }
    }
    let mut out = Plane::new(w, h);
    for y in 0..h {
        let (lo, hi) = (y.saturating_sub(r), (y + r).min(h - 1));
        for x in 0..w {
            let mut v = start;
            for yi in lo..=hi {
                v = pick(v, rows.data.get(yi * w + x).copied().unwrap_or(start));
            }
            if let Some(o) = out.data.get_mut(y * w + x) {
                *o = v;
            }
        }
    }
    out
}

/// Film dust, hair and scratches in a rendered positive; `sensitivity` 0..100 (higher finds
/// fainter ones).
pub fn detect(img: &Rgba8, sensitivity: f32) -> Vec<FilmDefect> {
    let (w, h) = (img.width, img.height);
    if w < 64 || h < 64 || img.data.len() < w * h {
        return Vec::new();
    }
    let long = w.max(h) as f32;
    let l = Plane::from_fn(w, h, |x, y| {
        let p = img.data.get(y * w + x).copied().unwrap_or_default();
        (0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])) / 255.0
    });
    // the widest defect: about 0.15 % of the long edge (4 px at 2400)
    let r = ((0.0015 * long).round() as usize).max(2);
    let opened = rank(&rank(&l, r, false), r, true);
    let tophat: Vec<f32> = l.data.iter().zip(&opened.data).map(|(a, b)| (a - b).max(0.0)).collect();
    // local texture, in small cells: the mean top-hat (grain and foliage make it busy
    // everywhere) or the spread of the opened image (the photo with dust removed: edges,
    // railings, window frames), whichever is larger; then the busiest of the 3 x 3 cells around
    // each pixel, so a bright pixel at a busy area's edge doesn't stand out either. A speck on a
    // smooth area (sky, skin, a wall) stands alone.
    let cell = ((0.008 * long).round() as usize).max(4);
    let (cw, ch) = (w.div_ceil(cell), h.div_ceil(cell));
    let mut sums = vec![(0.0f32, 0.0f32, 0.0f32, 0u32); cw * ch];
    for (i, (t, o)) in tophat.iter().zip(&opened.data).enumerate() {
        if let Some(c) = sums.get_mut((i / w) / cell * cw + (i % w) / cell) {
            c.0 += t;
            c.1 += o;
            c.2 += o * o;
            c.3 += 1;
        }
    }
    let busy: Vec<f32> = sums
        .iter()
        .map(|c| {
            let n = c.3.max(1) as f32;
            let mean = c.1 / n;
            (c.0 / n).max((c.2 / n - mean * mean).max(0.0).sqrt())
        })
        .collect();
    let busiest: Vec<f32> = (0..cw * ch)
        .map(|c| {
            let (cx, cy) = (c % cw, c / cw);
            let mut m = 0.0f32;
            for y in cy.saturating_sub(1)..=(cy + 1).min(ch - 1) {
                for x in cx.saturating_sub(1)..=(cx + 1).min(cw - 1) {
                    m = m.max(busy.get(y * cw + x).copied().unwrap_or(0.0));
                }
            }
            m
        })
        .collect();
    let texture = |i: usize| busiest.get((i / w) / cell * cw + (i % w) / cell).copied().unwrap_or(1.0);
    let s = (sensitivity / 100.0).clamp(0.0, 1.0);
    let min_bright = 0.35 - 0.27 * s;
    let contrast = 4.0 - 1.5 * s;
    // the outermost pixels read as a thin bright line wherever the photo is brighter at its
    // very edge: left out
    let margin = r + 1;
    let candidate = |i: usize| {
        let (x, y) = (i % w, i / w);
        if x < margin || y < margin || x + margin >= w || y + margin >= h {
            return false;
        }
        let t = tophat.get(i).copied().unwrap_or(0.0);
        t > min_bright && t > contrast * texture(i)
    };
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    let max_blob = ((0.06 * long) * (0.004 * long)).max(64.0) as usize;
    let mut stack = Vec::new();
    let mut px: Vec<(usize, usize, f32)> = Vec::new();
    for start in 0..w * h {
        if seen.get(start).copied().unwrap_or(true) || !candidate(start) {
            continue;
        }
        // one blob of candidate pixels (8-connected)
        px.clear();
        stack.clear();
        stack.push(start);
        if let Some(s) = seen.get_mut(start) {
            *s = true;
        }
        let mut too_big = false;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            px.push((x, y, tophat.get(i).copied().unwrap_or(0.0)));
            if px.len() > max_blob {
                too_big = true;
            }
            for (dx, dy) in [(-1i64, -1i64), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if !seen.get(j).copied().unwrap_or(true) && candidate(j) {
                    if let Some(s) = seen.get_mut(j) {
                        *s = true;
                    }
                    if !too_big {
                        stack.push(j);
                    }
                }
            }
        }
        if too_big || px.len() < 2 {
            continue; // a bright structure, not dust (or a single noisy pixel)
        }
        if let Some(d) = classify(&px, &l, long) {
            out.push(d);
        }
    }
    out.sort_by(|a, b| b.strength.total_cmp(&a.strength));
    out.truncate(MAX_DEFECTS);
    out
}

/// A blob's shape: a speck when small and roundish, a hair or scratch when long and thin,
/// `None` otherwise (a bright edge or texture).
fn classify(px: &[(usize, usize, f32)], luma: &Plane, long: f32) -> Option<FilmDefect> {
    let (w, h) = (luma.width, luma.height);
    let n = px.len() as f64;
    let (mx, my) = px.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0 as f64, a.1 + p.1 as f64));
    let (mx, my) = (mx / n, my / n);
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for p in px {
        let (dx, dy) = (p.0 as f64 - mx, p.1 as f64 - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let (sxx, syy, sxy) = (sxx / n, syy / n, sxy / n);
    // principal axes of the blob
    let tr = sxx + syy;
    let det = sxx * syy - sxy * sxy;
    let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
    let (l1, l2) = (tr / 2.0 + disc, (tr / 2.0 - disc).max(0.0));
    let length = 4.0 * l1.sqrt() + 1.0;
    let width = (n / length).max(1.0);
    let strength = px.iter().map(|p| p.2).fold(0.0, f32::max);
    let long = f64::from(long);
    let to_norm = |x: f64, y: f64| ((x + 0.5) / w as f64, (y + 0.5) / h as f64);
    let elongation = (l1 / l2.max(0.05)).sqrt();
    if elongation < 2.5 && length <= 0.008 * long {
        // a speck: heal a circle a little larger than it
        return Some(FilmDefect { points: vec![to_norm(mx, my)], radius: (length / 2.0 * 1.6 + 1.0) / long, strength, source: None });
    }
    if elongation >= 3.0 && width <= 0.003 * long && length >= 0.008 * long {
        // a hair or scratch: its centre line, as the mean position along the main axis in steps
        let (ax, ay) = if sxy.abs() < 1e-12 && sxx >= syy {
            (1.0, 0.0)
        } else if sxy.abs() < 1e-12 {
            (0.0, 1.0)
        } else {
            let (vx, vy) = (l1 - syy, sxy);
            let m = (vx * vx + vy * vy).sqrt();
            (vx / m, vy / m)
        };
        let t: Vec<f64> = px.iter().map(|p| (p.0 as f64 - mx) * ax + (p.1 as f64 - my) * ay).collect();
        let (t0, t1) = t.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, v| (a.0.min(*v), a.1.max(*v)));
        let steps = (((t1 - t0) / (2.0 * width).max(3.0)).ceil() as usize).clamp(2, MAX_PATH_POINTS);
        let mut sums = vec![(0.0f64, 0.0f64, 0usize); steps];
        for (p, tv) in px.iter().zip(&t) {
            let k = (((tv - t0) / (t1 - t0).max(1e-9)) * (steps as f64 - 1.0)).round() as usize;
            if let Some(s) = sums.get_mut(k.min(steps - 1)) {
                s.0 += p.0 as f64;
                s.1 += p.1 as f64;
                s.2 += 1;
            }
        }
        let path: Vec<(f64, f64)> = sums.iter().filter(|s| s.2 > 0).map(|s| (s.0 / s.2 as f64, s.1 / s.2 as f64)).collect();
        if path.len() >= 2 && same_both_sides(&path, luma, width + 2.0, 0.004 * long, strength) {
            let points = path.iter().map(|p| to_norm(p.0, p.1)).collect();
            let radius = width * 1.5 + 1.5;
            let source = side_source(&path, luma, (ay, -ax), 2.5 * radius + 2.0).map(|d| (d.0 / w as f64, d.1 / h as f64));
            return Some(FilmDefect { points, radius: radius / long, strength, source });
        }
    }
    None
}

/// The heal source for a line along `path`: `normal * reach` or its opposite (in pixels),
/// whichever lands on the smoother strip of `luma` (the one most like plain background), or
/// `None` when both fall off the image.
fn side_source(path: &[(f64, f64)], luma: &Plane, normal: (f64, f64), reach: f64) -> Option<(f64, f64)> {
    let (w, h) = (luma.width, luma.height);
    let spread = |sign: f64| {
        let vals: Vec<f32> = path
            .iter()
            .filter_map(|p| {
                let (x, y) = ((p.0 + sign * normal.0 * reach).round(), (p.1 + sign * normal.1 * reach).round());
                if x < 0.0 || y < 0.0 || x >= w as f64 || y >= h as f64 {
                    return None;
                }
                luma.data.get(y as usize * w + x as usize).copied()
            })
            .collect();
        if vals.len() < path.len() {
            return None;
        }
        let n = vals.len() as f32;
        let mean = vals.iter().sum::<f32>() / n;
        Some(vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n)
    };
    let sign = match (spread(1.0), spread(-1.0)) {
        (Some(a), Some(b)) => {
            if a <= b {
                1.0
            } else {
                -1.0
            }
        }
        (Some(_), None) => 1.0,
        (None, Some(_)) => -1.0,
        (None, None) => return None,
    };
    Some((sign * normal.0 * reach, sign * normal.1 * reach))
}

/// Whether a bright line lies across one background (a hair or scratch: the photo looks the same
/// on both sides of it) rather than along the border of two areas or as part of a structure (the
/// lit rim of a roof, a railing's top rail, a window frame): the mean `luma` over a strip from
/// `near` to `near + depth` pixels to either side of the path is compared, and the typical
/// difference must be small next to the line's own brightness `strength`.
fn same_both_sides(path: &[(f64, f64)], luma: &Plane, near: f64, depth: f64, strength: f32) -> bool {
    let (w, h) = (luma.width, luma.height);
    let at = |x: f64, y: f64| {
        let (x, y) = (x.round(), y.round());
        if x < 0.0 || y < 0.0 || x >= w as f64 || y >= h as f64 {
            return None;
        }
        luma.data.get(y as usize * w + x as usize).copied()
    };
    let steps = (depth.ceil() as usize).max(2);
    let side = |p: (f64, f64), nx: f64, ny: f64| {
        let mut sum = 0.0;
        for k in 0..steps {
            let d = near + depth * k as f64 / (steps - 1) as f64;
            sum += at(p.0 + nx * d, p.1 + ny * d)?;
        }
        Some(sum / steps as f32)
    };
    let mut diffs: Vec<f32> = (0..path.len())
        .filter_map(|i| {
            let (a, b) = (path.get(i.saturating_sub(1))?, path.get((i + 1).min(path.len() - 1))?);
            let p = *path.get(i)?;
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let m = (dx * dx + dy * dy).sqrt();
            if m < 1e-9 {
                return None;
            }
            let (nx, ny) = (-dy / m, dx / m);
            Some((side(p, nx, ny)? - side(p, -nx, -ny)?).abs())
        })
        .collect();
    if diffs.len() * 2 < path.len() {
        return false; // mostly off the image: can't tell
    }
    diffs.sort_by(f32::total_cmp);
    let typical = diffs.get(diffs.len() * 3 / 4).copied().unwrap_or(f32::INFINITY);
    typical <= (0.25 * strength).max(0.03)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth grey gradient (a sky) with a busy textured strip along the bottom.
    fn scene(w: usize, h: usize) -> Rgba8 {
        let mut img = Rgba8::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let base = 120.0 + 60.0 * x as f32 / w as f32;
                let v = if y > h * 3 / 4 {
                    // texture: a fine high-contrast pattern, as foliage
                    let n = ((x * 7919 + y * 104_729) % 97) as f32 / 97.0;
                    base - 40.0 + 80.0 * n
                } else {
                    base
                };
                let v = v.clamp(0.0, 255.0) as u8;
                img.data[y * w + x] = [v, v, v, 255];
            }
        }
        img
    }

    fn dab(img: &mut Rgba8, cx: usize, cy: usize, r: i64) {
        let w = img.width;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    let (x, y) = ((cx as i64 + dx) as usize, (cy as i64 + dy) as usize);
                    img.data[y * w + x] = [250, 250, 250, 255];
                }
            }
        }
    }

    #[test]
    fn bright_specks_and_a_hair_on_a_smooth_area_are_found_and_texture_is_left() {
        let (w, h) = (800, 600);
        let mut img = scene(w, h);
        for (x, y) in [(100, 100), (300, 80), (500, 200), (650, 120)] {
            dab(&mut img, x, y, 1);
        }
        // a hair: a thin bright arc across the sky
        for i in 0..1200 {
            let t = i as f32 / 1200.0;
            let (x, y) = (150.0 + 400.0 * t, 300.0 + 60.0 * (t * 3.0).sin());
            img.data[y as usize * w + x as usize] = [245, 245, 245, 255];
            img.data[(y as usize + 1) * w + x as usize] = [245, 245, 245, 255];
        }
        let found = detect(&img, 50.0);
        let specks: Vec<_> = found.iter().filter(|d| !d.is_line()).collect();
        let hairs: Vec<_> = found.iter().filter(|d| d.is_line()).collect();
        for (x, y) in [(100, 100), (300, 80), (500, 200), (650, 120)] {
            let (nx, ny) = (x as f64 / w as f64, y as f64 / h as f64);
            assert!(
                specks.iter().any(|d| (d.points[0].0 - nx).abs() < 0.01 && (d.points[0].1 - ny).abs() < 0.01),
                "speck at {x},{y} missed: {found:?}"
            );
        }
        assert!(!hairs.is_empty(), "the hair was missed: {found:?}");
        // the hair's path runs along it, end to end
        let longest = hairs.iter().map(|d| d.points.len()).max().unwrap_or(0);
        assert!(longest >= 5, "a hair should be a path: {hairs:?}");
        // the textured strip gives (almost) nothing
        let in_texture = found.iter().filter(|d| d.points.iter().any(|p| p.1 > 0.76)).count();
        assert!(in_texture <= 2, "texture taken for dust: {in_texture}");
    }

    #[test]
    fn a_clean_photo_and_a_bright_object_give_nothing() {
        let (w, h) = (640, 480);
        let mut img = scene(w, h);
        // a large bright object (a white wall, the sun): not dust
        for y in 50..200 {
            for x in 300..500 {
                img.data[y * w + x] = [250, 250, 250, 255];
            }
        }
        assert!(detect(&img, 50.0).is_empty());
        assert!(detect(&scene(w, h), 100.0).len() <= 2);
    }

    #[test]
    fn hostile_input_never_panics() {
        for (w, h) in [(0, 0), (1, 1), (63, 63), (64, 64), (1000, 64)] {
            let img = Rgba8::new(w, h);
            let _ = detect(&img, 50.0);
        }
        let short = Rgba8 { width: 100, height: 100, data: vec![[0; 4]; 10] };
        assert!(detect(&short, 50.0).is_empty());
        let _ = detect(&scene(200, 200), f32::NAN);
    }
}
