//! The device: opening one (or sharing one opened elsewhere), choosing an
//! adapter, and reading buffers back.

use super::*;

/// A wgpu device and queue.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
}

impl Gpu {
    /// A device opened elsewhere (by a windowing library, say), to share
    /// with it. Large models need large storage buffers: open the device
    /// with the adapter's own limits, as [`Gpu::new`] does.
    pub fn from_parts(device: wgpu::Device, queue: wgpu::Queue, info: wgpu::AdapterInfo) -> Gpu {
        Gpu {
            device,
            queue,
            info,
        }
    }

    /// The highest-performance adapter, with its own limits (large storage
    /// buffers). `WGPU_BACKEND` and `WGPU_ADAPTER_NAME` select one. Not
    /// WARP unless named ([`is_warp`]).
    pub fn new() -> Result<Gpu> {
        // Vulkan or Metal unless `WGPU_BACKEND` says otherwise: left to
        // choose, wgpu now and then takes the D3D12 adapter (when several
        // threads look at once). Every backend if neither has an adapter:
        // D3D12 then compiles with DXC, linked in (`static-dxc` in
        // Cargo.toml), as the system's FXC cannot compile the renderer's loops.
        let request = |backends: Option<wgpu::Backends>| {
            let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
            if let Some(b) = backends {
                desc.backends = b;
            }
            let instance = wgpu::Instance::new(desc);
            pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
                &instance, None,
            ))
        };
        let preferred = (std::env::var_os("WGPU_BACKEND").is_none())
            .then(|| request(Some(wgpu::Backends::VULKAN | wgpu::Backends::METAL)).ok())
            .flatten();
        let adapter = match preferred {
            Some(a) => a,
            None => request(None).map_err(|e| Error::Gpu(format!("no GPU adapter: {e}")))?,
        };
        Gpu::from_adapter(&adapter, wgpu::Features::empty())
    }

    /// A device of its own on `adapter`, with the adapter's limits and
    /// those of `features` it has. A device of its own has a queue of its
    /// own: long renders there do not hold up another device's work (a
    /// window's frames), as the driver shares the GPU between them.
    /// Not WARP (D3D12's software adapter) unless `WGPU_ADAPTER_NAME` is
    /// set ([`is_warp`]).
    pub fn from_adapter(adapter: &wgpu::Adapter, features: wgpu::Features) -> Result<Gpu> {
        let info = adapter.get_info();
        if is_warp(&info) && std::env::var_os("WGPU_ADAPTER_NAME").is_none() {
            return Err(Error::Gpu(format!(
                "{} (D3D12's software adapter) is not supported: renders on it do not finish",
                info.name
            )));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("llevant"),
            required_limits: adapter.limits(),
            required_features: adapter.features() & (features | wgpu::Features::PIPELINE_CACHE),
            ..Default::default()
        }))
        .map_err(|e| Error::Gpu(format!("cannot open {}: {e}", info.name)))?;
        Ok(Gpu {
            device,
            queue,
            info,
        })
    }
}

/// WARP, the adapter Windows offers without a GPU driver (virtual machines,
/// remote desktops, CI). The renderer's shader compiles there (DXC) but its
/// renders never finish (seen on GitHub's Windows runners; the cause is not
/// known), so [`Gpu::from_adapter`] refuses it unless `WGPU_ADAPTER_NAME`
/// asks for an adapter by name; the CI job `gpu-windows` does, to check
/// that the shader compiles. Software Vulkan (lavapipe) is fine.
pub fn is_warp(info: &wgpu::AdapterInfo) -> bool {
    info.backend == wgpu::Backend::Dx12 && info.device_type == wgpu::DeviceType::Cpu
}

/// How much to prefer an adapter (lower first): its kind, then its
/// backend.
pub fn adapter_rank(info: &wgpu::AdapterInfo) -> (u8, u8) {
    let kind = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => 2,
        wgpu::DeviceType::Cpu => 3,
    };
    let backend = match info.backend {
        wgpu::Backend::Vulkan | wgpu::Backend::Metal => 0,
        wgpu::Backend::Dx12 => 1,
        _ => 2,
    };
    (kind, backend)
}

/// A storage buffer of `words` for a shader to write and be read back.
pub(crate) fn output_buffer(gpu: &Gpu, words: u64) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hits"),
        size: 4 * words,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// The words of `buffer` (made with `COPY_SRC`), once the work queued
/// before is done.
pub(crate) fn read_buffer(gpu: &Gpu, buffer: &wgpu::Buffer) -> Result<Vec<u32>> {
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buffer, 0, &readback, 0, buffer.size());
    gpu.queue.submit([enc.finish()]);
    read_mapped(gpu, &readback)
}

/// Map `buffer` (`MAP_READ`) for reading once the work queued before is
/// done, and wait until it is: then `get_mapped_range` may be called.
///
/// Waiting for the device alone is not enough on a device other threads
/// poll too (a window's renderer, other tracers): the poll that finishes
/// the copy may be another thread's, still running the map's callback when
/// this thread's poll returns. So this waits for the callback itself,
/// polling until it has run.
pub fn map_for_reading(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<()> {
    use std::sync::mpsc::RecvTimeoutError;
    let failed = |e: &dyn std::fmt::Display| Error::Gpu(format!("GPU read-back: {e}"));
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| failed(&e))?;
    loop {
        match rx.recv_timeout(std::time::Duration::from_millis(5)) {
            Ok(Ok(())) => return Ok(()),
            Ok(Err(e)) => return Err(failed(&e)),
            Err(RecvTimeoutError::Timeout) => {
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|e| failed(&e))?;
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(failed(&"the buffer was never mapped"))
            }
        }
    }
}

/// The words of `readback` (a `MAP_READ` buffer) once the work queued
/// before is done.
pub(crate) fn read_mapped(gpu: &Gpu, readback: &wgpu::Buffer) -> Result<Vec<u32>> {
    let failed = |e: &dyn std::fmt::Display| Error::Gpu(format!("GPU trace: {e}"));
    map_for_reading(&gpu.device, readback)?;
    let words = {
        let view = readback.get_mapped_range(..).map_err(|e| failed(&e))?;
        bytemuck::cast_slice(&view).to_vec()
    };
    readback.unmap();
    Ok(words)
}

fn pipeline_cache_dir_from(
    llevant_cache_dir: Option<&str>,
    xdg_cache_home: Option<&str>,
    home: Option<&str>,
    local_app_data: Option<&str>,
) -> Option<std::path::PathBuf> {
    if let Some(dir) = llevant_cache_dir {
        let trimmed = dir.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("off") || trimmed == "0" {
            return None;
        }
        return Some(std::path::PathBuf::from(trimmed).join("pipelines"));
    }
    if let Some(xdg) = xdg_cache_home.filter(|s| !s.trim().is_empty()) {
        return Some(
            std::path::PathBuf::from(xdg)
                .join("llevant")
                .join("pipelines"),
        );
    }
    if let Some(h) = home.filter(|s| !s.trim().is_empty()) {
        return Some(
            std::path::PathBuf::from(h)
                .join(".cache")
                .join("llevant")
                .join("pipelines"),
        );
    }
    local_app_data.filter(|s| !s.trim().is_empty()).map(|p| {
        std::path::PathBuf::from(p)
            .join("llevant")
            .join("pipelines")
    })
}

pub(crate) fn pipeline_cache_path(
    info: &wgpu::AdapterInfo,
    shader_id: u64,
) -> Option<std::path::PathBuf> {
    let key = wgpu::util::pipeline_cache_key(info)?;
    let dir = pipeline_cache_dir_from(
        std::env::var("LLEVANT_CACHE_DIR").ok().as_deref(),
        std::env::var("XDG_CACHE_HOME").ok().as_deref(),
        std::env::var("HOME").ok().as_deref(),
        std::env::var("LOCALAPPDATA").ok().as_deref(),
    )?;
    let safe_key: String = key
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    Some(dir.join(format!("{safe_key}_{shader_id:016x}.bin")))
}

pub(crate) fn load_pipeline_cache(
    gpu: &Gpu,
    shader_id: u64,
) -> Option<(wgpu::PipelineCache, std::path::PathBuf)> {
    if !gpu
        .device
        .features()
        .contains(wgpu::Features::PIPELINE_CACHE)
    {
        return None;
    }
    let path = pipeline_cache_path(&gpu.info, shader_id)?;
    let data = std::fs::read(&path).ok();
    // SAFETY: `fallback: true` allows wgpu to ignore stale or mismatched bytes cleanly.
    let cache = unsafe {
        gpu.device
            .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("llevant-pipeline-cache"),
                data: data.as_deref(),
                fallback: true,
            })
    };
    Some((cache, path))
}

pub(crate) fn save_pipeline_cache(cache: &wgpu::PipelineCache, path: &std::path::Path) {
    let Some(bytes) = cache.get_data() else {
        return;
    };
    if bytes.is_empty() {
        return;
    }
    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return;
        }
    }
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    if std::fs::write(&tmp, &bytes).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    } else {
        let _ = std::fs::remove_file(&tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_pipeline_cache_dir_precedence() {
        assert_eq!(
            pipeline_cache_dir_from(Some("off"), Some("/xdg"), Some("/h"), None),
            None
        );
        assert_eq!(
            pipeline_cache_dir_from(Some("0"), Some("/xdg"), Some("/h"), None),
            None
        );
        assert_eq!(
            pipeline_cache_dir_from(Some("/custom/cache"), Some("/xdg"), Some("/h"), None),
            Some(std::path::PathBuf::from("/custom/cache/pipelines"))
        );
        assert_eq!(
            pipeline_cache_dir_from(None, Some("/xdg"), Some("/h"), None),
            Some(std::path::PathBuf::from("/xdg/llevant/pipelines"))
        );
        assert_eq!(
            pipeline_cache_dir_from(None, None, Some("/home/u"), None),
            Some(std::path::PathBuf::from("/home/u/.cache/llevant/pipelines"))
        );
    }
}
