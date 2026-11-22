//! llevant: a fast geometry engine for MCNP constructive solid geometry.
//!
//! Load a model with [`Model::load`], then classify batches of points with
//! [`Model::classify`], trace batches of lines with [`Model::trace`], find
//! where rays first meet visible cells with [`Model::first_hits`], and
//! fingerprint cells and universes with [`Model::fingerprints`]. Views build
//! on these: [`Model::slice`] traces a [`Slice`] of a plane row by row, and
//! [`Model::render`] casts a [`Camera`]'s rays, and [`Model::volumes`]
//! measures the volume of every label (and finds gaps and overlaps) on a
//! [`VolumeGrid`] of lines.
//!
//! Results refer to cells by index into [`Model::cells`] and to surfaces by
//! compiled-surface index; [`Model::cell_id`], [`Model::surface_id`] and
//! [`Model::describe`] give the MCNP numbers. [`export`] gives a flat copy of
//! the compiled model for other evaluators (the GPU renderer).
//! [`Model::instances`] numbers the places each cell occupies (a cell of a
//! universe that fills several cells, a lattice cell in each element), and
//! [`Model::instance`] gives the instance holding a point. [`Model::save`]
//! and [`Model::read`] keep a compiled model as bytes (for caches).
//! [`draft`] moves a surface live while it is dragged (for interactive
//! frontends; it may change in any 0.x release).
//!
//! The engine answers geometry questions only: colouring hits and slices
//! into images, and what materials look like, is the `llevant-render` crate.
//!
//! Queries check their arguments and return [`Error`] rather than panic.

mod bbox;
mod bvh;
mod classify;
pub mod draft;
mod error;
pub mod export;
mod expr;
mod fingerprint;
mod grid;
mod hit;
pub mod instance;
mod intern;
mod lattice;
mod lp;
mod model;
mod progress;
mod recompile;
pub mod reference;
mod saved;
mod skip;
mod submodel;
mod surface;
mod surfaces;
mod trace;
mod transform;
pub mod vec3;
mod view;
mod volume;
// Counting is always compiled in (it is an `Option` checked on the way);
// only the calls that return it need the `bench` feature.
#[cfg_attr(not(feature = "bench"), allow(dead_code))]
mod work;

/// Engine internals the GPU renderer uses to recentre quadrics (and, with
/// the `bench` feature, the compiled surfaces for the benchmarks). Not part
/// of the supported API: anything here can change in any release.
#[doc(hidden)]
pub mod internals {
    pub use crate::bbox::{analyse, QuadInfo};
    pub use crate::surface::Quadric;
    #[cfg(feature = "bench")]
    pub use crate::surface::{Shape, Side};
}

/// The MCNP parser llevant is built with, at the exact version
/// [`Model::from_migjorn`] and [`Model::recompile_parsed`] take.
pub use migjorn;

pub use bbox::Aabb;
pub use classify::{Classification, Depth, Label, Status, Step};
pub use error::{Error, Result};
pub use fingerprint::{
    Change, Changes, Digest, Fingerprint, FingerprintDiff, Fingerprints, DEFAULT_PRECISION,
    FINGERPRINT_VERSION,
};
pub use hit::{Ghost, Hits, Veils, Visibility, NO_CELL, NO_HIT};
pub use instance::{ElementRange, Instances, NO_BIN};
pub use lattice::{Element, Lattice, Target};
pub use model::{
    read_input_bytes, translate_input_bytes, Cell, Diagnostic, Fill, GridMode, Model, Options,
    Scope, Stats, Subject, SurfaceId, Unit,
};
pub use progress::Progress;
pub use recompile::Changed;
pub use saved::{ENGINE_HASH, FORMAT_VERSION};
pub use skip::Skip;
pub use submodel::Cut;
pub use surfaces::NearSurface;
pub use trace::{Line, Segment, Tracing, NO_SURFACE};
pub use transform::Affine;
pub use vec3::Vec3;
pub use view::{Axis, BoxCut, Camera, Clip, Cutaway, Kept, Slice};
pub use volume::{
    spacing_within, BinTarget, BinVolumes, Converged, Group, GroupBy, PlacedBox, Volume,
    VolumeGrid, VolumeTarget, Volumes, DEFAULT_PROBES, MAX_LISTED_INSTANCES, MAX_PROBES,
};
#[cfg(feature = "bench")]
pub use work::{LevelWork, Work};
