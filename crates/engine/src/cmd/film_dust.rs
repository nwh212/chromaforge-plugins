//! Find Film Dust and Hair (`spot.findFilmDust`): on a scanned negative or slide, the dust
//! specks, hairs and scratches on the film, healed with Remove spots. A scanner's infrared pass
//! (a VueScan or SilverFast RGBI TIFF, [`lcc_codecs::Decoded::infrared`]) says exactly where they
//! are; without one the converted picture is searched for the sharp bright marks they leave
//! ([`lcc_pipeline::film_dust`]).

use lcc_develop::{Spot, SpotMode};
use lcc_geom::{Affine, Point};
use lcc_pipeline::RenderRequest;
use serde_json::{Value, json};

use super::masks::{SPOT_SIZE, pick_source};
use super::{CommandSpec, bad, cmd, f64_or, has_active, str_param};
use crate::media::SourceLevel;
use crate::{Result, Session};

/// The long edge the photo is searched at: film dust and hairs are only a few pixels wide.
const SEARCH_EDGE: usize = 2400;

/// Where the defects come from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    /// The infrared pass when the file has one, else the picture.
    Auto,
    /// The infrared pass only: an error without one.
    Infrared,
    /// The picture only.
    Picture,
}

fn find_film_dust(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "spot.findFilmDust";
    let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
    let source = match str_param(p, "infrared").unwrap_or("auto") {
        "auto" => Source::Auto,
        "only" | "true" => Source::Infrared,
        "never" | "false" => Source::Picture,
        other => return Err(bad(c, format!("`infrared` must be auto, only or never, not {other:?}"))),
    };
    let sensitivity = f64_or(p, "sensitivity", 50.0) as f32;
    // the uncropped photo, so positions are the spots' own coordinates
    let job = s.render_job(id, SEARCH_EDGE, SEARCH_EDGE, false, false).ok_or_else(|| bad(c, "no photo"))?;
    let img = job.run().rendered.map_err(|e| bad(c, e))?.image;
    let mut dd = (*s.develop_of(id).unwrap_or_default()).clone();
    // the infrared pass through the same geometry as the picture (orientation, lens, perspective;
    // not the crop), at the picture's size
    let infrared = if source == Source::Picture {
        None
    } else {
        let src = s.source_now(id, SourceLevel::Preview).map_err(|e| bad(c, e))?;
        let info = s.source_info(id);
        info.infrared.as_deref().and_then(|ir| {
            let mut uncropped = dd.clone();
            uncropped.crop.geometry = lcc_geom::CropGeometry::default();
            let req = RenderRequest { apply_crop: false, ..RenderRequest::fit(SEARCH_EDGE, SEARCH_EDGE) };
            let plane = lcc_pipeline::render_plane(&src, ir, &info, &uncropped, &req)?;
            Some(if (plane.width, plane.height) == (img.width, img.height) {
                plane
            } else {
                lcc_raster::resample::resize(&plane, img.width, img.height, lcc_raster::resample::Filter::Mitchell)
            })
        })
    };
    if source == Source::Infrared && infrared.is_none() {
        return Err(bad(c, "this photo has no infrared pass: a film scanner's RGBI file (VueScan 64-bit RGBI, SilverFast HDRi) carries one"));
    }
    let found_all = match &infrared {
        Some(ir) => {
            let luma = lcc_raster::Plane::from_fn(img.width, img.height, |x, y| {
                let q = img.get(x, y);
                (0.2126 * f32::from(q[0]) + 0.7152 * f32::from(q[1]) + 0.0722 * f32::from(q[2])) / 255.0
            });
            lcc_pipeline::film_dust::detect_infrared(ir, &luma, sensitivity)
        }
        None => lcc_pipeline::film_dust::detect(&img, sensitivity),
    };
    // only what's inside the crop: a scan's rebate (edge print, sprocket holes) is usually
    // cropped away, and its lettering isn't dust
    let (w, h) = (img.width as f64, img.height as f64);
    let geometry = dd.crop.geometry;
    let (crop, to_crop) = (geometry.rect_px(w, h), Affine::rotate_about(geometry.angle.to_radians(), Point::new(w / 2.0, h / 2.0)));
    let inside = |q: &(f64, f64)| crop.contains(to_crop.apply(Point::new(q.0 * w, q.1 * h)));
    let found: Vec<_> = found_all.into_iter().filter(|d| d.points.iter().filter(|q| inside(q)).count() * 2 > d.points.len()).collect();
    let out: Vec<Value> =
        found.iter().map(|d| json!({"points": d.points.iter().map(|q| [q.0, q.1]).collect::<Vec<_>>(), "size": d.radius})).collect();
    let used_infrared = infrared.is_some();
    if !p.get("add").and_then(Value::as_bool).unwrap_or(true) || found.is_empty() {
        return Ok(json!({"spots": out, "added": 0, "infrared": used_infrared}));
    }
    let base = Spot::default();
    for d in &found {
        let mut spot = Spot {
            mode: SpotMode::Heal,
            points: d.points.iter().map(|q| Point::new(q.0, q.1)).collect(),
            size: d.radius.clamp(SPOT_SIZE.0, SPOT_SIZE.1),
            feather: base.feather,
            opacity: 100.0,
            source_offset: d.source.map(|o| Point::new(o.0, o.1)),
            ..Spot::default()
        };
        if spot.source_offset.is_none() {
            spot.source_offset = pick_source(s, id, &dd, &spot, None);
        }
        dd.spots.push(spot);
    }
    s.set_develop(id, dd, "Find Film Dust and Hair")?;
    Ok(json!({"spots": out, "added": found.len(), "infrared": used_infrared}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![cmd!(
        "spot.findFilmDust",
        "Find Film Dust and Hair",
        ["Photo", "Develop Settings"],
        None,
        "{sensitivity?: 0..100 (50), add?: bool (true), infrared?: auto|only|never (auto)} — on a scanned negative or slide, find dust specks, hairs and scratches on the film inside the crop and add a heal spot on each speck or patch and a heal stroke along each line (one undo step). A film scanner's infrared pass (a VueScan or SilverFast RGBI TIFF) says where they are, whatever the picture shows; without one the converted picture is searched for the sharp bright marks they leave on smooth areas. `only`: an error without the pass; `never`: the picture alone → {spots: [{points: [[x, y], …], size}], added, infrared: bool}",
        has_active,
        find_film_dust
    )]
}
