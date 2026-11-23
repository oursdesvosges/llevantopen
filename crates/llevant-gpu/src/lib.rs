//! GPU renderer for llevant's compiled MCNP models: first hits of camera rays
//! (the 3D viewer), in a wgpu compute shader.
//!
//! Approximate by design (docs/internals/rendering.md, "GPU renderer"): f32 throughout, and
//! steps of `eps` past each crossing, so a pixel may be wrong at an edge or
//! in a sliver thinner than `eps`. Exact answers (which cell is this pixel?)
//! come from the CPU engine. The renderer reads [`llevant::export::Export`],
//! never MCNP input.
//!
//! Gaps and problems (unsupported cells, nesting deeper than [`MAX_LEVELS`])
//! always stop a ray; overlaps are not detected (the first candidate cell
//! that holds a point wins).
//!
//! Hits become images through `llevant-render`: [`Frame::probes`] for
//! `render_image`.

pub mod check;
mod device;
mod hits;
mod job;
mod pack;
mod patch;
mod visibility;

pub use device::*;
pub use hits::*;
pub use job::*;
pub use pack::recentred_shapes;
pub use visibility::GpuVisibility;

use std::time::Instant;

pub use llevant::export::NONE;
use llevant::export::{Bulk, Export, ExportCell, ExportGridNode, NO_INSTANCE};
use llevant::{Aabb, Camera, Error, Line, Model, Result, Skip, Visibility};
use llevant_render::shading::{Found, Probe};
use wgpu::util::DeviceExt;

// The export format this renderer reads; update the packing (and the
// shaders) to a new version before changing this number.
const _: () = assert!(
    llevant::export::EXPORT_VERSION == 3,
    "llevant-gpu reads export format 3"
);

/// Deepest nesting of universes the shader follows. A model's shader keeps
/// per-level state for its own levels only (`Export::levels`, at most this):
/// every level held costs every ray private memory, so a model with fewer
/// levels traces faster (a model of 4 levels: 7–31 % per frame against 8).
pub const MAX_LEVELS: u32 = 8;

/// Planes and regions of a camera's cut the renderer takes at most
/// ([`llevant::Cutaway`]).
pub const MAX_CUT_PLANES: usize = 16;
pub const MAX_CUT_REGIONS: usize = 8;

/// A model on the GPU, ready to render.
pub struct GpuModel {
    n_cells: usize,
    n_units: usize,
    n_grid_nodes: usize,
    offsets: [u32; 9],
    /// Where the parts of the visibility buffer start (`VisLayout`).
    vis_offsets: [u32; 4],
    /// Where shapes, node boxes, programs and the octree start in the pool.
    sections: [u32; 4],
    /// Words per pool buffer: 2^shift.
    shift: u32,
    /// Where the units' lists of cells checked everywhere start in the
    /// pool, and the slack a patch writes its own lists to.
    always_list: u32,
    slack: u32,
    /// See `Params::grid_progs`.
    grid_progs: u32,
    pool: Vec<wgpu::Buffer>,
    /// A word bound to each pool binding the model leaves unused.
    spare: Vec<wgpu::Buffer>,
    /// The current visibility: a new one per [`GpuModel::set_visibility`],
    /// so a job keeps the one it was made with.
    vis: std::sync::Mutex<std::sync::Arc<visibility::Vis>>,
    /// Bound as the rays when rendering (the camera makes them).
    no_rays: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    trace_rays: wgpu::ComputePipeline,
    render: wgpu::ComputePipeline,
    /// The diagnostic variants ([`GpuModel::debug_steps`],
    /// [`GpuModel::render_stats`]): compiled when first used, not at upload
    /// (each takes as long as `render` to compile, ~1.7 s on lavapipe).
    debug: std::sync::OnceLock<wgpu::ComputePipeline>,
    stats: std::sync::OnceLock<wgpu::ComputePipeline>,
    /// Sizes each slice of a [`Job`] (and refills it).
    plan: wgpu::ComputePipeline,
    /// Kept to rebuild the pipelines ([`GpuModel::set_constants`]).
    module: wgpu::ShaderModule,
    /// A hash of the shader's source ([`GpuModel::shader_id`]).
    shader_id: u64,
    pipeline_layout: wgpu::PipelineLayout,
    pipeline_cache: Option<(wgpu::PipelineCache, std::path::PathBuf)>,
    /// Bytes uploaded.
    pub bytes: u64,
    /// Steps past each crossing: `eps_abs + eps_rel · t` (cm).
    pub eps_abs: f32,
    pub eps_rel: f32,
    /// Locate steps before a ray gives up.
    pub max_steps: u32,
    /// Program words a ray may walk (locating and finding exits) before it
    /// gives up ([`Status::Steps`]). Bounds the time of one ray whatever the
    /// model.
    pub max_work: u32,
    /// Program words each ray walks in one slice of a [`Job`] before it
    /// pauses: bounds the GPU time of every dispatch, whatever the rays.
    /// Slices with fewer than 32 768 rays in flight let each walk more (up
    /// to [`GpuModel::tail_work`]): the GPU then waits on single rays rather
    /// than on their number.
    pub slice_work: u32,
    /// Most words each ray walks in one slice when few rays are in flight:
    /// fewer keep every slice short (a window drawn meanwhile), more take
    /// fewer slices for the slow rays at the end of a job.
    pub tail_work: u32,
    /// Rays in flight at most in a [`Job`] (rounded up to a multiple of
    /// 64): enough to fill the GPU, each with a state of
    /// [`STATE_WORDS`] words.
    pub capacity: u32,
}

/// Default [`GpuModel::max_work`]: well under a second for one ray on a
/// laptop GPU (measured 2026-10-17), and above the
/// work of 99 % of the reference model's rays.
pub const MAX_WORK: u32 = 1 << 16;

/// Default [`GpuModel::slice_work`]: slices of ~3–25 ms on the reference
/// model on an RTX 3060 Laptop, up to ~50 ms at the tail of a job, and
/// frames within a few per cent of one dispatch, or faster when hidden
/// cells make rays long (`docs/decisions/0005-sliced-gpu-dispatches.md`).
pub const SLICE_WORK: u32 = 256;

/// Default [`GpuModel::tail_work`]: whole frames take about as long as in
/// one dispatch, with slices of a single slow ray up to ~50 ms.
pub const TAIL_WORK: u32 = 64 * SLICE_WORK;

/// Default [`GpuModel::capacity`].
pub const CAPACITY: u32 = 1 << 17;

/// Words of a paused ray's state, as in the shader.
pub const STATE_WORDS: usize = 20;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::round_out;

    #[test]
    fn outward_rounding() {
        let x = 0.1f64;
        assert!((round_out(x, true) as f64) <= x);
        assert!((round_out(x, false) as f64) >= x);
        assert_eq!(round_out(f64::INFINITY, false), 1e30);
        assert_eq!(round_out(f64::NEG_INFINITY, true), -1e30);
        assert_eq!(round_out(1.0, true), 1.0);
    }

    #[test]
    fn normals_pack_and_unpack() {
        for n in [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.6, -0.8, 0.0],
            [0.48, 0.6, -0.64],
            [-0.36, -0.48, -0.8],
        ] {
            let back = decode_normal(encode_normal(n));
            let cos: f32 = (0..3).map(|i| n[i] * back[i]).sum();
            assert!(cos > 0.9999, "{n:?} -> {back:?}");
        }
    }
}
