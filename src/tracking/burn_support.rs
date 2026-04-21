use std::{env, sync::OnceLock};

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

pub(crate) fn burn_score_map_capture_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();

    *ENABLED.get_or_init(|| {
        env::var("GAME_MAP_TRACKER_DEBUG_HEATMAP")
            .ok()
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                !matches!(value.as_str(), "" | "0" | "false" | "off" | "no")
            })
            .unwrap_or(false)
    })
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

#[cfg(any(burn_cuda_backend, burn_vulkan_backend, burn_metal_backend))]
use burn::tensor::{Tensor, backend::Backend};

#[cfg(any(burn_cuda_backend, burn_vulkan_backend, burn_metal_backend))]
use cubecl_runtime::runtime::Runtime as CubeRuntime;

use crate::{
    config::{AiTrackingConfig, MinimapPresenceProbeConfig, TemplateTrackingConfig},
    error::Result,
};

#[cfg(burn_cuda_backend)]
use burn::backend::cuda::CudaDevice;
#[cfg(any(burn_vulkan_backend, burn_metal_backend))]
use burn::backend::wgpu::WgpuDevice;
#[cfg(burn_cuda_backend)]
use cubecl_cuda::CudaRuntime;
#[cfg(any(burn_vulkan_backend, burn_metal_backend))]
use cubecl_wgpu::WgpuRuntime;

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

#[cfg(any(burn_vulkan_backend, burn_metal_backend))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct BurnWgpuDeviceChoice {
    descriptor: BurnDeviceDescriptor,
    device: WgpuDevice,
}

pub(crate) trait BurnDeviceConfig {
    fn device_preference(&self) -> AiDevicePreference;
    fn device_index(&self) -> usize;
}

impl BurnDeviceConfig for AiTrackingConfig {
    fn device_preference(&self) -> AiDevicePreference {
        self.device
    }

    fn device_index(&self) -> usize {
        self.device_index
    }
}

impl BurnDeviceConfig for TemplateTrackingConfig {
    fn device_preference(&self) -> AiDevicePreference {
        self.device
    }

    fn device_index(&self) -> usize {
        self.device_index
    }
}

impl BurnDeviceConfig for MinimapPresenceProbeConfig {
    fn device_preference(&self) -> AiDevicePreference {
        self.device
    }

    fn device_index(&self) -> usize {
        self.device_index
    }
}

pub(crate) fn select_burn_device(config: &impl BurnDeviceConfig) -> Result<BurnDeviceSelection> {
    match config.device_preference() {
        AiDevicePreference::Cpu => Ok(BurnDeviceSelection::Cpu),
        AiDevicePreference::Cuda => build_cuda_device(config.device_index()),
        AiDevicePreference::Vulkan => build_vulkan_device(config.device_index()),
        AiDevicePreference::Metal => build_metal_device(config.device_index()),
    }
}

#[cfg(any(burn_cuda_backend, burn_vulkan_backend, burn_metal_backend))]
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

#[cfg(burn_cuda_backend)]
fn build_cuda_device(ordinal: usize) -> Result<BurnDeviceSelection> {
    if !available_cuda_device_descriptors()
        .iter()
        .any(|descriptor| descriptor.ordinal == ordinal)
    {
        crate::bail!("CUDA 设备序号 {ordinal} 不存在");
    }

    let device = CudaDevice { index: ordinal };
    probe_device::<burn::backend::Cuda>(&device)?;
    Ok(BurnDeviceSelection::Cuda(device))
}

#[cfg(not(burn_cuda_backend))]
fn build_cuda_device(_ordinal: usize) -> Result<BurnDeviceSelection> {
    crate::bail!("配置选择了 CUDA 设备，但当前二进制未包含 CUDA 后端；Windows 默认构建会包含 CUDA")
}

#[cfg(burn_vulkan_backend)]
fn build_vulkan_device(ordinal: usize) -> Result<BurnDeviceSelection> {
    let device = available_vulkan_device_choices()
        .iter()
        .find(|choice| choice.descriptor.ordinal == ordinal)
        .map(|choice| choice.device.clone())
        .ok_or_else(|| crate::app_error!("Vulkan 设备序号 {ordinal} 不存在"))?;
    probe_device::<burn::backend::Vulkan>(&device)?;
    Ok(BurnDeviceSelection::Vulkan(device))
}

#[cfg(not(burn_vulkan_backend))]
fn build_vulkan_device(_ordinal: usize) -> Result<BurnDeviceSelection> {
    crate::bail!(
        "配置选择了 Vulkan 设备，但当前二进制未包含 Vulkan 后端；Windows / Linux 默认构建会包含 Vulkan"
    )
}

#[cfg(burn_metal_backend)]
fn build_metal_device(ordinal: usize) -> Result<BurnDeviceSelection> {
    let device = available_metal_device_choices()
        .iter()
        .find(|choice| choice.descriptor.ordinal == ordinal)
        .map(|choice| choice.device.clone())
        .ok_or_else(|| crate::app_error!("Metal 设备序号 {ordinal} 不存在"))?;
    probe_device::<burn::backend::Metal>(&device)?;
    Ok(BurnDeviceSelection::Metal(device))
}

#[cfg(not(burn_metal_backend))]
fn build_metal_device(_ordinal: usize) -> Result<BurnDeviceSelection> {
    crate::bail!("配置选择了 Metal 设备，但当前二进制未包含 Metal 后端；macOS 默认构建会包含 Metal")
}

#[cfg(burn_cuda_backend)]
fn available_cuda_device_descriptors() -> Vec<BurnDeviceDescriptor> {
    static DESCRIPTORS: OnceLock<Vec<BurnDeviceDescriptor>> = OnceLock::new();

    DESCRIPTORS
        .get_or_init(|| {
            <CudaRuntime as CubeRuntime>::enumerate_devices(0, &())
                .into_iter()
                .map(|device_id| {
                    let ordinal = device_id.index_id as usize;
                    BurnDeviceDescriptor {
                        ordinal,
                        name: format!("CUDA 设备 {ordinal}"),
                    }
                })
                .collect()
        })
        .clone()
}

#[cfg(not(burn_cuda_backend))]
fn available_cuda_device_descriptors() -> Vec<BurnDeviceDescriptor> {
    Vec::new()
}

#[cfg(burn_vulkan_backend)]
fn available_vulkan_device_descriptors() -> Vec<BurnDeviceDescriptor> {
    available_vulkan_device_choices()
        .iter()
        .map(|choice| choice.descriptor.clone())
        .collect()
}

#[cfg(not(burn_vulkan_backend))]
fn available_vulkan_device_descriptors() -> Vec<BurnDeviceDescriptor> {
    Vec::new()
}

#[cfg(burn_metal_backend)]
fn available_metal_device_descriptors() -> Vec<BurnDeviceDescriptor> {
    available_metal_device_choices()
        .iter()
        .map(|choice| choice.descriptor.clone())
        .collect()
}

#[cfg(not(burn_metal_backend))]
fn available_metal_device_descriptors() -> Vec<BurnDeviceDescriptor> {
    Vec::new()
}

#[cfg(any(burn_vulkan_backend, burn_metal_backend))]
const WGPU_DEVICE_CLASS_ORDINAL_STRIDE: usize = 8;

#[cfg(burn_vulkan_backend)]
fn available_vulkan_device_choices() -> &'static [BurnWgpuDeviceChoice] {
    static CHOICES: OnceLock<Vec<BurnWgpuDeviceChoice>> = OnceLock::new();
    CHOICES.get_or_init(enumerate_vulkan_device_choices)
}

#[cfg(burn_metal_backend)]
fn available_metal_device_choices() -> &'static [BurnWgpuDeviceChoice] {
    static CHOICES: OnceLock<Vec<BurnWgpuDeviceChoice>> = OnceLock::new();
    CHOICES.get_or_init(enumerate_metal_device_choices)
}

#[cfg(burn_vulkan_backend)]
fn enumerate_vulkan_device_choices() -> Vec<BurnWgpuDeviceChoice> {
    enumerate_wgpu_device_choices::<cubecl_wgpu::Vulkan>()
}

#[cfg(burn_metal_backend)]
fn enumerate_metal_device_choices() -> Vec<BurnWgpuDeviceChoice> {
    enumerate_wgpu_device_choices::<cubecl_wgpu::Metal>()
}

#[cfg(any(burn_vulkan_backend, burn_metal_backend))]
fn enumerate_wgpu_device_choices<G>() -> Vec<BurnWgpuDeviceChoice>
where
    G: cubecl_wgpu::GraphicsApi,
{
    let backend = G::backend();
    let mut choices = Vec::new();
    let discrete = <WgpuRuntime as CubeRuntime>::enumerate_devices(0, &backend).into_iter();
    let integrated = <WgpuRuntime as CubeRuntime>::enumerate_devices(1, &backend).into_iter();
    let virtual_gpus = <WgpuRuntime as CubeRuntime>::enumerate_devices(2, &backend).into_iter();

    let devices = discrete
        .chain(integrated)
        .chain(virtual_gpus)
        .collect::<Vec<_>>();

    if !devices.is_empty() {
        choices.push(BurnWgpuDeviceChoice {
            descriptor: BurnDeviceDescriptor {
                ordinal: 0,
                name: "默认设备".to_owned(),
            },
            device: WgpuDevice::DefaultDevice,
        });
    }

    choices.extend(devices.into_iter().filter_map(|device_id| {
        let index = device_id.index_id as usize;
        match device_id.type_id {
            0 => Some(BurnWgpuDeviceChoice {
                descriptor: BurnDeviceDescriptor {
                    ordinal: 1 + index,
                    name: format!("独显 GPU {index}"),
                },
                device: WgpuDevice::DiscreteGpu(index),
            }),
            1 => Some(BurnWgpuDeviceChoice {
                descriptor: BurnDeviceDescriptor {
                    ordinal: 1 + WGPU_DEVICE_CLASS_ORDINAL_STRIDE + index,
                    name: format!("核显 GPU {index}"),
                },
                device: WgpuDevice::IntegratedGpu(index),
            }),
            2 => Some(BurnWgpuDeviceChoice {
                descriptor: BurnDeviceDescriptor {
                    ordinal: 1 + WGPU_DEVICE_CLASS_ORDINAL_STRIDE * 2 + index,
                    name: format!("虚拟 GPU {index}"),
                },
                device: WgpuDevice::VirtualGpu(index),
            }),
            _ => None,
        }
    }));

    choices
}

#[cfg(any(burn_vulkan_backend, burn_metal_backend))]
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

#[cfg(any(burn_cuda_backend, burn_vulkan_backend, burn_metal_backend))]
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
    .map_err(|_| crate::app_error!("device probe failed"))?;
    Ok(())
}
