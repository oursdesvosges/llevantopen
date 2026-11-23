//! Uploading a model: the export packed into pool buffers (records,
//! shapes recentred for f32, programs, octrees), streamed in pieces.

use super::*;
use crate::visibility::{Vis, VisLayout};

/// The renderer's WGSL, its parts joined in order: each part uses what
/// those before it define.
const SHADER: &str = concat!(
    include_str!("shader/prelude.wgsl"),
    "
",
    include_str!("shader/visibility.wgsl"),
    "
",
    include_str!("shader/surfaces.wgsl"),
    "
",
    include_str!("shader/cells.wgsl"),
    "
",
    include_str!("shader/lattices.wgsl"),
    "
",
    include_str!("shader/tracking.wgsl"),
    "
",
    include_str!("shader/slices.wgsl"),
    "
",
    include_str!("shader/camera.wgsl"),
);

// Record sizes in words, as in the shader.
pub(crate) const CELL_WORDS: usize = 13;
pub(crate) const UNIT_WORDS: usize = 20;
pub(crate) const BVH_NODE_WORDS: usize = 8;
pub(crate) const BVH_ITEM_WORDS: usize = 8;
const TARGET_WORDS: usize = 16;
const LATTICE_WORDS: usize = 42;
const PLANE_WORDS: usize = 5;
/// Words reserved at the end of the pool for what [`GpuModel::patch`]
/// writes there (units' lists of cells checked everywhere).
pub(crate) const SLACK_WORDS: usize = 1 << 16;

/// The f32 next to `x` on the side of `-∞` (`down`) or `+∞`, finite.
pub(crate) fn round_out(x: f64, down: bool) -> f32 {
    const BIG: f32 = 1e30;
    let f = x as f32;
    let f = if down && f as f64 > x {
        f.next_down()
    } else if !down && (f as f64) < x {
        f.next_up()
    } else {
        f
    };
    f.clamp(-BIG, BIG)
}

pub(crate) fn push_box(out: &mut Vec<u32>, b: &Aabb) {
    for i in 0..3 {
        out.push(round_out(b.lo[i], true).to_bits());
    }
    for i in 0..3 {
        out.push(round_out(b.hi[i], false).to_bits());
    }
}

pub(crate) fn push_f32(out: &mut Vec<u32>, v: &[f64]) {
    out.extend(v.iter().map(|&x| (x as f32).to_bits()));
}

/// Words written in order into the pool's buffers (chunks of 2^shift
/// words) through the queue, a piece at a time, waiting for the queue now
/// and then: the staging memory alive at once stays small, whatever the
/// model's size.
struct PoolWriter<'a> {
    gpu: &'a Gpu,
    pool: &'a [wgpu::Buffer],
    shift: u32,
    /// Words written so far (pool index of the next one).
    at: usize,
    piece: Vec<u32>,
    /// Bytes handed to the queue since it was last waited for.
    queued: u64,
}

/// Words per write to the queue.
const PIECE_WORDS: usize = 1 << 22;
/// Bytes queued before waiting for the GPU to take them.
const QUEUED_BYTES: u64 = 1 << 28;

impl<'a> PoolWriter<'a> {
    fn new(gpu: &'a Gpu, pool: &'a [wgpu::Buffer], shift: u32) -> PoolWriter<'a> {
        PoolWriter {
            gpu,
            pool,
            shift,
            at: 0,
            piece: Vec::with_capacity(PIECE_WORDS),
            queued: 0,
        }
    }

    fn extend(&mut self, words: &[u32]) {
        let mut rest = words;
        while !rest.is_empty() {
            let n = (PIECE_WORDS - self.piece.len()).min(rest.len());
            self.piece.extend_from_slice(&rest[..n]);
            rest = &rest[n..];
            if self.piece.len() == PIECE_WORDS {
                self.flush();
            }
        }
    }

    /// Write the piece, split where it crosses into the next chunk.
    fn flush(&mut self) {
        let mut done = 0;
        while done < self.piece.len() {
            let i = self.at + done;
            let (chunk, off) = (i >> self.shift, i & ((1 << self.shift) - 1));
            let n = ((1 << self.shift) - off).min(self.piece.len() - done);
            self.gpu.queue.write_buffer(
                &self.pool[chunk],
                4 * off as u64,
                bytemuck::cast_slice(&self.piece[done..done + n]),
            );
            done += n;
        }
        self.at += self.piece.len();
        self.queued += 4 * self.piece.len() as u64;
        self.piece.clear();
        if self.queued >= QUEUED_BYTES {
            self.wait();
        }
    }

    /// Submit what was written and wait for it, freeing its staging memory.
    fn wait(&mut self) {
        self.gpu.queue.submit([]);
        let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
        self.queued = 0;
    }

    fn finish(&mut self) -> Result<()> {
        self.flush();
        self.gpu.queue.submit([]);
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("GPU upload: {e}")))?;
        Ok(())
    }
}

impl GpuModel {
    /// Upload `e` (a full [`Model::export`]).
    pub fn new(gpu: &Gpu, e: &Export) -> Result<GpuModel> {
        GpuModel::build(gpu, e, None)
    }

    /// Upload `m`, its largest arrays streamed from the model itself
    /// ([`Model::export_lean`]): host memory peaks at about the export
    /// without them, not the whole model a second time.
    pub fn upload(gpu: &Gpu, m: &Model) -> Result<GpuModel> {
        let (e, bulk) = m.export_lean();
        GpuModel::build(gpu, &e, Some(&bulk))
    }

    fn build(gpu: &Gpu, e: &Export, bulk: Option<&Bulk>) -> Result<GpuModel> {
        let n_grid_nodes = bulk.map_or(e.grid_nodes.len(), Bulk::grid_node_count);
        let n_grid_items = bulk.map_or(e.grid_items.len(), Bulk::grid_item_count);
        // Small records first, as u32 words; the shader reads them at
        // pool offset 0.
        let mut m: Vec<u32> = Vec::new();
        let mut offsets = [0u32; 9];

        offsets[0] = m.len() as u32;
        for (k, &s) in e.shape_kind.iter().zip(&e.shape_start) {
            if s >= 1 << 30 {
                return Err(Error::Gpu(
                    "too much surface data for 30-bit offsets".into(),
                ));
            }
            m.push(k << 30 | s);
        }

        // Cells and units refer to the lists, so the lists go first.
        let lists = m.len() as u32;
        m.extend_from_slice(&e.unit_cells);
        let always = m.len() as u32;
        m.extend_from_slice(&e.always);

        offsets[1] = m.len() as u32;
        // Instance ids in 32 bits, or none at all.
        let ids = e
            .cells
            .iter()
            .all(|c| c.instance_base == NO_INSTANCE || c.instance_base < u32::MAX as u64)
            && e.instances < u32::MAX as u64;
        let id32 = |x: u64| {
            if ids && x != NO_INSTANCE {
                x as u32
            } else {
                NONE
            }
        };
        for c in &e.cells {
            push_box(&mut m, &c.bbox);
            m.extend_from_slice(&[
                c.code_start,
                c.code_len,
                c.fill,
                c.flags,
                c.unit,
                id32(c.instance_base),
                c.node_start,
            ]);
        }

        offsets[2] = m.len() as u32;
        for u in &e.units {
            push_box(&mut m, &u.roi);
            push_box(&mut m, &u.grid_box);
            m.extend_from_slice(&[
                lists + u.cells.start,
                u.cells.len,
                always + u.always.start,
                u.always.len,
                u.bvh_root,
                u.grid_root,
                0,
                0,
            ]);
        }

        offsets[3] = m.len() as u32;
        for n in &e.bvh_nodes {
            push_box(&mut m, &n.bbox);
            m.extend_from_slice(&[n.start, n.count]);
        }

        offsets[4] = m.len() as u32;
        for (c, b) in e.bvh_items.iter().zip(&e.bvh_boxes) {
            m.push(*c);
            push_box(&mut m, b);
            m.push(0);
        }

        offsets[5] = m.len() as u32;
        for t in &e.targets {
            m.push(t.unit);
            for row in &t.transform.rot {
                push_f32(&mut m, row);
            }
            push_f32(&mut m, &t.transform.disp);
            m.extend_from_slice(&[
                id32(t.instance_offset),
                t.instance_stride.min(u32::MAX as u64) as u32,
                0,
            ]);
        }

        offsets[6] = m.len() as u32;
        for l in &e.lattices {
            m.push(l.hexagonal as u32);
            m.push((0..3).map(|i| (l.bounded[i] as u32) << i).sum());
            push_f32(&mut m, &l.center);
            for v in &l.basis {
                push_f32(&mut m, v);
            }
            for v in &l.dual {
                push_f32(&mut m, v);
            }
            m.extend_from_slice(&[l.sides.start, l.sides.len, l.faces.start, l.faces.len]);
            m.extend(l.lo.iter().map(|&x| x as u32));
            m.extend(l.size.iter().map(|&x| x as u32));
            m.extend_from_slice(&[l.elements, 0, 0]);
            m.extend(l.numbered.0.iter().map(|&x| x as u32));
            m.extend(l.numbered.1.iter().map(|&x| x as u32));
        }

        offsets[7] = m.len() as u32;
        for ((n, d), s) in e.lattice_planes.iter().zip(&e.lattice_plane_shapes) {
            push_f32(&mut m, n);
            push_f32(&mut m, &[*d]);
            m.push(*s);
        }

        offsets[8] = m.len() as u32;
        m.extend_from_slice(&e.lattice_elements);
        let sizes = [
            (1, CELL_WORDS * e.cells.len()),
            (2, UNIT_WORDS * e.units.len()),
            (3, BVH_NODE_WORDS * e.bvh_nodes.len()),
            (4, BVH_ITEM_WORDS * e.bvh_items.len()),
            (5, TARGET_WORDS * e.targets.len()),
            (6, LATTICE_WORDS * e.lattices.len()),
            (7, PLANE_WORDS * e.lattice_planes.len()),
        ];
        for (k, n) in sizes {
            assert_eq!((offsets[k + 1] - offsets[k]) as usize, n, "section {k}");
        }

        // The pool: records, shapes, node boxes, programs, then the octree
        // (nodes as (first, len) pairs, the candidate lists, each
        // candidate's program in its leaf). Sizes first: the sections are
        // streamed into the GPU buffers without a copy of the whole pool.
        let shapes: Vec<u32> = match bulk {
            Some(b) => recentred_shapes_by(e, |c| b.cell_code(c)),
            None => recentred_shapes(e),
        }
        .iter()
        .map(|&x| (x as f32).to_bits())
        .collect();
        let n_nodes = n_grid_nodes as u32;
        let lens = [
            shapes.len(),
            12 * e.node_boxes.len(),
            bulk.map_or(e.code.len(), Bulk::code_len),
            2 * n_grid_nodes + 2 * n_grid_items,
        ];
        let mut sections = [0u32; 4];
        let mut total = m.len();
        for (k, n) in lens.iter().enumerate() {
            sections[k] = u32::try_from(total)
                .map_err(|_| Error::Gpu("model too large for 32-bit indices".into()))?;
            total += n;
        }
        // Room for the lists a draft gives units ([`GpuModel::patch`]).
        let slack = u32::try_from(total)
            .map_err(|_| Error::Gpu("model too large for 32-bit indices".into()))?;
        total += SLACK_WORDS;
        u32::try_from(total)
            .map_err(|_| Error::Gpu("model too large for 32-bit indices".into()))?;

        // Bindings of 2^shift words each, as large as the device allows.
        let limits = gpu.device.limits();
        let device = &gpu.device;
        let max_bytes = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            .min(1 << 32);
        let shift = (max_bytes / 4).ilog2().min(30);
        let n_chunks = total.div_ceil(1 << shift).max(1);
        let max_chunks = limits.max_storage_buffers_per_shader_stage as usize - 3;
        if n_chunks > max_chunks {
            return Err(Error::Gpu(format!(
                "the model needs {} MB in {} storage buffers of {} MB; the device allows {}",
                total >> 18,
                n_chunks,
                1u64 << (shift - 18),
                max_chunks
            )));
        }
        // Out of memory is an error to return (the caller can trace on the
        // CPU), not a panic. The pool is checked before it is written: a
        // write to a buffer the device could not make is a validation error,
        // which wgpu's default handler turns into a panic (seen with a large
        // model on a laptop GPU, D3D12).
        let invalid = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let oom = gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let mut bytes = 0u64;
        let pool: Vec<wgpu::Buffer> = (0..n_chunks)
            .map(|k| {
                // Bind groups need non-empty buffers.
                let words = (total - (k << shift)).min(1 << shift).max(1);
                bytes += 4 * words as u64;
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pool"),
                    size: 4 * words as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let errors = [oom.pop(), invalid.pop()].map(pollster::block_on);
        if let Some(err) = errors.into_iter().flatten().next() {
            return Err(Error::Gpu(format!(
                "the GPU could not hold the model ({} MB in {} buffers of up to {} MB): {err}",
                bytes >> 20,
                pool.len(),
                1u64 << (shift - 18)
            )));
        }
        let oom = gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let mut w = PoolWriter::new(gpu, &pool, shift);
        w.extend(&m);
        drop(m);
        w.extend(&shapes);
        drop(shapes);
        for [t, f] in &e.node_boxes {
            let mut b = Vec::with_capacity(12);
            push_box(&mut b, t);
            push_box(&mut b, f);
            w.extend(&b);
        }
        let node = |w: &mut PoolWriter, nodes: &[ExportGridNode]| {
            for n in nodes {
                let leaf = n.len != llevant::export::GRID_INNER;
                w.extend(&[if leaf { n.first + 2 * n_nodes } else { n.first }, n.len]);
            }
        };
        match bulk {
            Some(b) => {
                b.code(|c| w.extend(c));
                b.grid_nodes(|n| node(&mut w, n));
                b.grid_items(|c| w.extend(c));
                b.grid_progs(|c| w.extend(c));
            }
            None => {
                w.extend(&e.code);
                node(&mut w, &e.grid_nodes);
                w.extend(&e.grid_items);
                w.extend(&e.grid_progs);
            }
        }
        w.extend(&vec![0; SLACK_WORDS]);
        w.finish()?;
        debug_assert_eq!(w.at, total);
        let mut storage = |name: &str, data: &[u32]| {
            // Bind groups need non-empty buffers.
            let data = if data.is_empty() { &[0u32][..] } else { data };
            bytes += 4 * data.len() as u64;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let vis_layout = VisLayout::new(e.cells.len(), e.units.len(), n_grid_nodes);
        let vis = storage("visibility", &vis_layout.everything());
        let vis_offsets = [
            vis_layout.boxes,
            vis_layout.nodes,
            vis_layout.ghosts,
            vis_layout.flags,
        ]
        .map(|o| o as u32);
        let no_rays = storage("no rays", &[0; 8]);
        // Pool bindings in a power of two (unused ones bound to a word of
        // their own), so that a model one chunk larger than another, or the
        // same model with its octrees, keeps the shader.
        let slots = pool_slots(pool.len(), max_chunks);
        let spare: Vec<wgpu::Buffer> = (pool.len()..slots)
            .map(|_| storage("pool (unused)", &[0]))
            .collect();
        if let Some(err) = pollster::block_on(oom.pop()) {
            return Err(Error::Gpu(format!(
                "not enough GPU memory for the model ({} MB): {err}",
                bytes >> 20
            )));
        }

        let bindings: String = (0..slots)
            .map(|k| {
                format!(
                    "@group(0) @binding({}) var<storage, read> pool{k}: array<u32>;\n",
                    6 + k
                )
            })
            .collect();
        let body = if slots == 1 {
            "    return pool0[i];".to_string()
        } else {
            let mut b = format!(
                "    let j = i & {}u;\n    switch i >> {shift}u {{\n",
                (1u32 << shift) - 1
            );
            for k in 0..slots - 1 {
                b +=
                    &format!("        case {k}u: {{\n            return pool{k}[j];\n        }}\n");
            }
            b += &format!(
                "        default: {{\n            return pool{}[j];\n        }}\n    }}",
                slots - 1
            );
            b
        };
        let source = SHADER
            // Per-level state for the model's own levels: a ray never goes
            // deeper (its cells there have no fill), so `locate` never
            // writes past them. Deeper models stop at `MAX_LEVELS`.
            .replace("{{MAXL}}", &e.levels.clamp(1, MAX_LEVELS).to_string())
            .replace("{{CODE_STACK}}", &(e.max_code_depth + 1).to_string())
            // Also `exit_bound`'s surfaces, one per open program node.
            .replace(
                "{{BVH_STACK}}",
                &((e.max_bvh_depth + 2).max(e.max_code_depth + 1)).to_string(),
            )
            .replace("{{POOL_BINDINGS}}", &bindings)
            .replace("{{POOL_BODY}}", &body);
        let shader_id = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            source.hash(&mut h);
            h.finish()
        };
        // A shader the driver cannot compile is an error, not a panic (wgpu's
        // default for uncaptured errors): D3D12's old compiler (FXC) gives up
        // on the renderer's loops, for one.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("llevant-gpu"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let mut entries = vec![
            entry(0, wgpu::BufferBindingType::Uniform),
            entry(1, ro),
            entry(2, ro),
            entry(3, rw),
            entry(4, rw),
            entry(5, rw),
        ];
        entries.extend((0..slots as u32).map(|k| entry(6 + k, ro)));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("llevant"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("llevant"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let cache_entry = load_pipeline_cache(gpu, shader_id);
        let pipeline = |entry: &str, constants: &[(&str, f64)]| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    ..Default::default()
                },
                cache: cache_entry.as_ref().map(|(c, _)| c),
            })
        };
        let trace_rays = pipeline("trace_rays", &[]);
        let render = pipeline("render", &[]);
        let plan = pipeline("plan", &[]);
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(Error::Gpu(format!(
                "the GPU cannot compile the renderer: {e}"
            )));
        }
        if let Some((cache, path)) = &cache_entry {
            save_pipeline_cache(cache, path);
        }
        Ok(GpuModel {
            n_cells: e.cells.len(),
            n_units: e.units.len(),
            n_grid_nodes,
            offsets,
            vis_offsets,
            sections,
            shift,
            always_list: always,
            slack,
            grid_progs: sections[3] + n_grid_items as u32,
            pool,
            spare,
            vis: std::sync::Mutex::new(std::sync::Arc::new(Vis {
                buffer: vis,
                ghosts: false,
            })),
            no_rays,
            layout,
            trace_rays,
            render,
            debug: Default::default(),
            stats: Default::default(),
            plan,
            module,
            shader_id,
            pipeline_layout,
            pipeline_cache: cache_entry,
            bytes,
            eps_abs: 1e-4,
            eps_rel: 1e-6,
            max_steps: 4096,
            max_work: MAX_WORK,
            slice_work: SLICE_WORK,
            tail_work: TAIL_WORK,
            capacity: CAPACITY,
        })
    }

    /// Which shader the model was uploaded with: models with the same id
    /// have the same shader source, which the driver compiles once (the
    /// first upload of a new one takes seconds; then it is cached).
    pub fn shader_id(&self) -> u64 {
        self.shader_id
    }

    /// Rebuild the tracing pipelines with these values of the shader's
    /// override constants (`USE_RESIDUALS`, `MAX_LEAVES`), for A/B runs and
    /// tuning in one process (`gpu_ab`).
    #[doc(hidden)]
    pub fn set_constants(&mut self, gpu: &Gpu, constants: &[(&str, f64)]) {
        let cache = self.pipeline_cache.as_ref().map(|(c, _)| c);
        let pipeline = |entry: &str, extra: &[(&str, f64)]| {
            let all: Vec<(&str, f64)> = constants.iter().chain(extra).copied().collect();
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&self.pipeline_layout),
                    module: &self.module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &all,
                        ..Default::default()
                    },
                    cache,
                })
        };
        self.trace_rays = pipeline("trace_rays", &[]);
        self.render = pipeline("render", &[]);
        self.stats = pipeline("render", &[("STATS", 1.0)]).into();
        if let Some((c, path)) = &self.pipeline_cache {
            save_pipeline_cache(c, path);
        }
    }

    /// A diagnostic variant of entry point `entry` with `constants`,
    /// compiled now.
    pub(crate) fn variant(
        &self,
        gpu: &Gpu,
        entry: &str,
        constants: &[(&str, f64)],
    ) -> wgpu::ComputePipeline {
        let cache = self.pipeline_cache.as_ref().map(|(c, _)| c);
        let pipe = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&self.pipeline_layout),
                module: &self.module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    ..Default::default()
                },
                cache,
            });
        if let Some((c, path)) = &self.pipeline_cache {
            save_pipeline_cache(c, path);
        }
        pipe
    }
}

/// The pool bindings of a model in `chunks` chunks: the next power of two
/// (as many as the device allows), so that the shader, which names each
/// binding, changes with the model's size only at powers of two.
fn pool_slots(chunks: usize, max: usize) -> usize {
    chunks.next_power_of_two().min(max).max(chunks)
}

/// Shape data with every quadric (and cone) written about its centre, so
/// that f32 values near it do not cancel (a GQ cylinder's stored origin can
/// be metres away): along an axis (a soft direction) the point of it nearest
/// the stored origin clamped into the region of the cells that use it, then
/// clamped into that region itself (a sphere far bigger than the region).
/// Cells with boxes over 100 m are left out of the region (the void around
/// everything would pull the origin far away). Exact in f64:
/// `f = d'ᵀAd' + (b + 2Aδ)·d' + f(o + δ)` with `d' = p - o - δ`.
#[doc(hidden)]
pub fn recentred_shapes(e: &Export) -> Vec<f64> {
    recentred_shapes_by(e, |c| {
        &e.code[c.code_start as usize..(c.code_start + c.code_len) as usize]
    })
}

/// [`recentred_shapes`] with the cells' programs from `code` (for a lean
/// export, [`llevant::export::Bulk::cell_code`]).
fn recentred_shapes_by<'a>(e: &Export, code: impl Fn(&ExportCell) -> &'a [u32]) -> Vec<f64> {
    use llevant::export::{SHAPE_CONE, SHAPE_QUADRIC};
    let mut usage = vec![Aabb::EMPTY; e.shape_kind.len()];
    for c in &e.cells {
        if !c.bbox.is_finite() || (0..3).any(|i| c.bbox.hi[i] - c.bbox.lo[i] > 1e4) {
            continue;
        }
        for (s, _) in llevant::export::leaves(code(c)) {
            usage[s as usize] = usage[s as usize].union(&c.bbox);
        }
    }
    let mut data = e.shape_data.clone();
    for (s, &kind) in e.shape_kind.iter().enumerate() {
        if (kind != SHAPE_QUADRIC && kind != SHAPE_CONE) || usage[s].is_empty() {
            continue;
        }
        recentre(&mut data[e.shape_start[s] as usize..][..13], &usage[s]);
    }
    data
}

/// Write the quadric `q` (13 values) about its centre clamped into `u`, the
/// box of the cells that use it (see [`recentred_shapes`]).
pub(crate) fn recentre(q: &mut [f64], u: &Aabb) {
    use llevant::internals::{analyse, Quadric};
    let near = [0, 1, 2].map(|i| q[i].clamp(u.lo[i], u.hi[i]));
    let quadric = Quadric {
        origin: [q[0], q[1], q[2]],
        a: [q[3], q[4], q[5], q[6], q[7], q[8]],
        b: [q[9], q[10], q[11]],
        c: q[12],
    };
    let mut centre = near;
    if let Some(info) = analyse(&quadric) {
        centre = info.center;
        for k in (0..3).filter(|&k| !info.stiff[k]) {
            let s = (0..3)
                .map(|i| info.u[k][i] * (near[i] - centre[i]))
                .sum::<f64>();
            for i in 0..3 {
                centre[i] += s * info.u[k][i];
            }
        }
    }
    let target = [0, 1, 2].map(|i| centre[i].clamp(u.lo[i], u.hi[i]));
    let delta = [0, 1, 2].map(|i| target[i] - q[i]);
    let (a, b) = ([q[3], q[4], q[5], q[6], q[7], q[8]], [q[9], q[10], q[11]]);
    let ad = [
        a[0] * delta[0] + a[3] * delta[1] + a[5] * delta[2],
        a[3] * delta[0] + a[1] * delta[1] + a[4] * delta[2],
        a[5] * delta[0] + a[4] * delta[1] + a[2] * delta[2],
    ];
    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    q[12] += dot(delta, ad) + dot(b, delta);
    for i in 0..3 {
        q[i] = target[i];
        q[9 + i] = b[i] + 2.0 * ad[i];
    }
}
