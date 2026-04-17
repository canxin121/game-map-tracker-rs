use crate::config::AiDevicePreference;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BurnDeviceDescriptor {
    pub ordinal: usize,
    pub name: String,
}

pub(crate) fn available_burn_backends() -> &'static str {
    #[cfg(all(burn_cuda_backend, burn_vulkan_backend, burn_metal_backend))]
    {
        return "CPU / CUDA / Vulkan / Metal";
    }

    #[cfg(all(burn_cuda_backend, burn_vulkan_backend, not(burn_metal_backend)))]
    {
        return "CPU / CUDA / Vulkan";
    }

    #[cfg(all(burn_cuda_backend, not(burn_vulkan_backend), burn_metal_backend))]
    {
        return "CPU / CUDA / Metal";
    }

    #[cfg(all(not(burn_cuda_backend), burn_vulkan_backend, burn_metal_backend))]
    {
        return "CPU / Vulkan / Metal";
    }

    #[cfg(all(burn_cuda_backend, not(burn_vulkan_backend), not(burn_metal_backend)))]
    {
        return "CPU / CUDA";
    }

    #[cfg(all(not(burn_cuda_backend), burn_vulkan_backend, not(burn_metal_backend)))]
    {
        return "CPU / Vulkan";
    }

    #[cfg(all(not(burn_cuda_backend), not(burn_vulkan_backend), burn_metal_backend))]
    {
        return "CPU / Metal";
    }

    #[cfg(all(
        not(burn_cuda_backend),
        not(burn_vulkan_backend),
        not(burn_metal_backend)
    ))]
    {
        "CPU"
    }
}

pub(crate) fn available_burn_backend_preferences() -> Vec<AiDevicePreference> {
    [
        Some(AiDevicePreference::Cpu),
        #[cfg(burn_cuda_backend)]
        Some(AiDevicePreference::Cuda),
        #[cfg(not(burn_cuda_backend))]
        None,
        #[cfg(burn_vulkan_backend)]
        Some(AiDevicePreference::Vulkan),
        #[cfg(not(burn_vulkan_backend))]
        None,
        #[cfg(burn_metal_backend)]
        Some(AiDevicePreference::Metal),
        #[cfg(not(burn_metal_backend))]
        None,
    ]
    .into_iter()
    .flatten()
    .collect()
}

pub(crate) fn available_burn_device_descriptors(
    preference: AiDevicePreference,
) -> Vec<BurnDeviceDescriptor> {
    match preference {
        AiDevicePreference::Cpu => vec![BurnDeviceDescriptor {
            ordinal: 0,
            name: "主机处理器".to_owned(),
        }],
        AiDevicePreference::Cuda => available_cuda_device_descriptors(),
        AiDevicePreference::Vulkan => available_vulkan_device_descriptors(),
        AiDevicePreference::Metal => available_metal_device_descriptors(),
    }
}

#[cfg(feature = "ai-burn")]
use anyhow::Result;
#[cfg(feature = "ai-burn")]
use burn::tensor::{Tensor, backend::Backend};

#[cfg(feature = "ai-burn")]
use crate::config::{AiTrackingConfig, MinimapPresenceProbeConfig, TemplateTrackingConfig};

#[cfg(all(feature = "ai-burn", burn_cuda_backend))]
use burn::backend::cuda::CudaDevice;
#[cfg(all(feature = "ai-burn", any(burn_vulkan_backend, burn_metal_backend)))]
use burn::backend::wgpu::WgpuDevice;

#[cfg(feature = "ai-burn")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BurnDeviceSelection {
    Cpu,
    #[cfg(burn_cuda_backend)]
    Cuda(CudaDevice),
    #[cfg(burn_vulkan_backend)]
    Vulkan(WgpuDevice),
    #[cfg(burn_metal_backend)]
    Metal(WgpuDevice),
}

#[cfg(feature = "ai-burn")]
pub(crate) trait BurnDeviceConfig {
    fn device_preference(&self) -> AiDevicePreference;
    fn device_index(&self) -> usize;
}

#[cfg(feature = "ai-burn")]
impl BurnDeviceConfig for AiTrackingConfig {
    fn device_preference(&self) -> AiDevicePreference {
        self.device
    }

    fn device_index(&self) -> usize {
        self.device_index
    }
}

#[cfg(feature = "ai-burn")]
impl BurnDeviceConfig for TemplateTrackingConfig {
    fn device_preference(&self) -> AiDevicePreference {
        self.device
    }

    fn device_index(&self) -> usize {
        self.device_index
    }
}

#[cfg(feature = "ai-burn")]
impl BurnDeviceConfig for MinimapPresenceProbeConfig {
    fn device_preference(&self) -> AiDevicePreference {
        self.device
    }

    fn device_index(&self) -> usize {
        self.device_index
    }
}

#[cfg(feature = "ai-burn")]
pub(crate) fn select_burn_device(config: &impl BurnDeviceConfig) -> Result<BurnDeviceSelection> {
    match config.device_preference() {
        AiDevicePreference::Cpu => Ok(BurnDeviceSelection::Cpu),
        AiDevicePreference::Cuda => build_cuda_device(config.device_index()),
        AiDevicePreference::Vulkan => build_vulkan_device(config.device_index()),
        AiDevicePreference::Metal => build_metal_device(config.device_index()),
    }
}

#[cfg(feature = "ai-burn")]
pub(crate) fn burn_device_label(device: &BurnDeviceSelection) -> String {
    match device {
        BurnDeviceSelection::Cpu => "CPU".to_owned(),
        #[cfg(burn_cuda_backend)]
        BurnDeviceSelection::Cuda(device) => format!("CUDA:{}", device.index),
        #[cfg(burn_vulkan_backend)]
        BurnDeviceSelection::Vulkan(device) => format!("Vulkan:{}", wgpu_device_label(device)),
        #[cfg(burn_metal_backend)]
        BurnDeviceSelection::Metal(device) => format!("Metal:{}", wgpu_device_label(device)),
    }
}

#[cfg(all(feature = "ai-burn", burn_cuda_backend))]
fn build_cuda_device(ordinal: usize) -> Result<BurnDeviceSelection> {
    let device = CudaDevice { index: ordinal };
    probe_device::<burn::backend::Cuda>(&device)?;
    Ok(BurnDeviceSelection::Cuda(device))
}

#[cfg(all(feature = "ai-burn", not(burn_cuda_backend)))]
fn build_cuda_device(_ordinal: usize) -> Result<BurnDeviceSelection> {
    anyhow::bail!("配置选择了 CUDA 设备，但当前二进制未包含 CUDA 后端；Windows 默认构建会包含 CUDA")
}

#[cfg(all(feature = "ai-burn", burn_vulkan_backend))]
fn build_vulkan_device(ordinal: usize) -> Result<BurnDeviceSelection> {
    let device = wgpu_device_from_ordinal(ordinal)
        .ok_or_else(|| anyhow::anyhow!("Vulkan 设备序号 {ordinal} 不存在"))?;
    probe_device::<burn::backend::Vulkan>(&device)?;
    Ok(BurnDeviceSelection::Vulkan(device))
}

#[cfg(all(feature = "ai-burn", not(burn_vulkan_backend)))]
fn build_vulkan_device(_ordinal: usize) -> Result<BurnDeviceSelection> {
    anyhow::bail!(
        "配置选择了 Vulkan 设备，但当前二进制未包含 Vulkan 后端；Windows / Linux 默认构建会包含 Vulkan"
    )
}

#[cfg(all(feature = "ai-burn", burn_metal_backend))]
fn build_metal_device(ordinal: usize) -> Result<BurnDeviceSelection> {
    let device = wgpu_device_from_ordinal(ordinal)
        .ok_or_else(|| anyhow::anyhow!("Metal 设备序号 {ordinal} 不存在"))?;
    probe_device::<burn::backend::Metal>(&device)?;
    Ok(BurnDeviceSelection::Metal(device))
}

#[cfg(all(feature = "ai-burn", not(burn_metal_backend)))]
fn build_metal_device(_ordinal: usize) -> Result<BurnDeviceSelection> {
    anyhow::bail!(
        "配置选择了 Metal 设备，但当前二进制未包含 Metal 后端；macOS 默认构建会包含 Metal"
    )
}

#[cfg(all(feature = "ai-burn", burn_cuda_backend))]
fn available_cuda_device_descriptors() -> Vec<BurnDeviceDescriptor> {
    const MAX_PROBED_CUDA_DEVICES: usize = 16;

    let mut descriptors = Vec::new();
    for ordinal in 0..MAX_PROBED_CUDA_DEVICES {
        let device = CudaDevice { index: ordinal };
        if probe_device::<burn::backend::Cuda>(&device).is_ok() {
            descriptors.push(BurnDeviceDescriptor {
                ordinal,
                name: format!("CUDA 设备 {ordinal}"),
            });
        } else if !descriptors.is_empty() {
            break;
        }
    }
    descriptors
}

#[cfg(not(all(feature = "ai-burn", burn_cuda_backend)))]
fn available_cuda_device_descriptors() -> Vec<BurnDeviceDescriptor> {
    Vec::new()
}

#[cfg(all(feature = "ai-burn", burn_vulkan_backend))]
fn available_vulkan_device_descriptors() -> Vec<BurnDeviceDescriptor> {
    enumerate_wgpu_device_descriptors::<burn::backend::Vulkan>()
}

#[cfg(not(all(feature = "ai-burn", burn_vulkan_backend)))]
fn available_vulkan_device_descriptors() -> Vec<BurnDeviceDescriptor> {
    Vec::new()
}

#[cfg(all(feature = "ai-burn", burn_metal_backend))]
fn available_metal_device_descriptors() -> Vec<BurnDeviceDescriptor> {
    enumerate_wgpu_device_descriptors::<burn::backend::Metal>()
}

#[cfg(not(all(feature = "ai-burn", burn_metal_backend)))]
fn available_metal_device_descriptors() -> Vec<BurnDeviceDescriptor> {
    Vec::new()
}

#[cfg(all(feature = "ai-burn", any(burn_vulkan_backend, burn_metal_backend)))]
fn enumerate_wgpu_device_descriptors<B>() -> Vec<BurnDeviceDescriptor>
where
    B: Backend<Device = WgpuDevice>,
{
    let mut descriptors = Vec::new();

    for ordinal in 0..MAX_WGPU_DEVICE_ORDINALS {
        let Some(device) = wgpu_device_from_ordinal(ordinal) else {
            continue;
        };
        if probe_device::<B>(&device).is_ok() {
            descriptors.push(BurnDeviceDescriptor {
                ordinal,
                name: wgpu_device_name(&device),
            });
        }
    }

    descriptors
}

#[cfg(all(feature = "ai-burn", any(burn_vulkan_backend, burn_metal_backend)))]
const MAX_WGPU_PROBE_PER_CLASS: usize = 8;
#[cfg(all(feature = "ai-burn", any(burn_vulkan_backend, burn_metal_backend)))]
const MAX_WGPU_DEVICE_ORDINALS: usize = 1 + MAX_WGPU_PROBE_PER_CLASS * 3;

#[cfg(all(feature = "ai-burn", any(burn_vulkan_backend, burn_metal_backend)))]
fn wgpu_device_from_ordinal(ordinal: usize) -> Option<WgpuDevice> {
    if ordinal == 0 {
        return Some(WgpuDevice::DefaultDevice);
    }

    let discrete_end = 1 + MAX_WGPU_PROBE_PER_CLASS;
    if ordinal < discrete_end {
        return Some(WgpuDevice::DiscreteGpu(ordinal - 1));
    }

    let integrated_end = discrete_end + MAX_WGPU_PROBE_PER_CLASS;
    if ordinal < integrated_end {
        return Some(WgpuDevice::IntegratedGpu(ordinal - discrete_end));
    }

    let virtual_end = integrated_end + MAX_WGPU_PROBE_PER_CLASS;
    if ordinal < virtual_end {
        return Some(WgpuDevice::VirtualGpu(ordinal - integrated_end));
    }

    None
}

#[cfg(all(feature = "ai-burn", any(burn_vulkan_backend, burn_metal_backend)))]
fn wgpu_device_name(device: &WgpuDevice) -> String {
    match device {
        WgpuDevice::DefaultDevice => "默认设备".to_owned(),
        WgpuDevice::DiscreteGpu(index) => format!("独显 GPU {index}"),
        WgpuDevice::IntegratedGpu(index) => format!("核显 GPU {index}"),
        WgpuDevice::VirtualGpu(index) => format!("虚拟 GPU {index}"),
        WgpuDevice::Cpu => "WGPU CPU".to_owned(),
        #[allow(deprecated)]
        WgpuDevice::BestAvailable => "默认设备".to_owned(),
        WgpuDevice::Existing(id) => format!("现有设备 {id}"),
    }
}

#[cfg(all(feature = "ai-burn", any(burn_vulkan_backend, burn_metal_backend)))]
fn wgpu_device_label(device: &WgpuDevice) -> String {
    match device {
        WgpuDevice::DefaultDevice => "default".to_owned(),
        WgpuDevice::DiscreteGpu(index) => format!("discrete-{index}"),
        WgpuDevice::IntegratedGpu(index) => format!("integrated-{index}"),
        WgpuDevice::VirtualGpu(index) => format!("virtual-{index}"),
        WgpuDevice::Cpu => "cpu".to_owned(),
        #[allow(deprecated)]
        WgpuDevice::BestAvailable => "default".to_owned(),
        WgpuDevice::Existing(id) => format!("existing-{id}"),
    }
}

#[cfg(feature = "ai-burn")]
fn probe_device<B>(device: &B::Device) -> Result<()>
where
    B: Backend,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe({
        let device = device.clone();
        move || {
            let _ = Tensor::<B, 1>::from_data([0.0f32], &device).into_data();
        }
    }))
    .map_err(|_| anyhow::anyhow!("device probe failed"))?;
    Ok(())
}
