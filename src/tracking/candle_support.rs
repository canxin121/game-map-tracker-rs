#[cfg(feature = "ai-candle")]
pub(crate) fn available_candle_backends() -> &'static str {
    #[cfg(all(candle_cuda_backend, candle_metal_backend))]
    {
        return "CPU / CUDA / Metal";
    }

    #[cfg(all(candle_cuda_backend, not(candle_metal_backend)))]
    {
        return "CPU / CUDA";
    }

    #[cfg(all(not(candle_cuda_backend), candle_metal_backend))]
    {
        return "CPU / Metal";
    }

    #[cfg(all(not(candle_cuda_backend), not(candle_metal_backend)))]
    {
        "CPU"
    }
}

use crate::config::AiDevicePreference;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CandleDeviceDescriptor {
    pub ordinal: usize,
    pub name: String,
}

pub(crate) fn available_candle_backend_preferences() -> Vec<AiDevicePreference> {
    [
        Some(AiDevicePreference::Cpu),
        #[cfg(candle_cuda_backend)]
        Some(AiDevicePreference::Cuda),
        #[cfg(not(candle_cuda_backend))]
        None,
        #[cfg(candle_metal_backend)]
        Some(AiDevicePreference::Metal),
        #[cfg(not(candle_metal_backend))]
        None,
    ]
    .into_iter()
    .flatten()
    .collect()
}

pub(crate) fn available_candle_device_descriptors(
    preference: AiDevicePreference,
) -> Vec<CandleDeviceDescriptor> {
    match preference {
        AiDevicePreference::Cpu => vec![CandleDeviceDescriptor {
            ordinal: 0,
            name: "主机处理器".to_owned(),
        }],
        AiDevicePreference::Cuda => available_cuda_device_descriptors(),
        AiDevicePreference::Metal => available_metal_device_descriptors(),
    }
}

#[cfg(feature = "ai-candle")]
#[cfg(any(candle_cuda_backend, candle_metal_backend))]
use anyhow::Context as _;
#[cfg(feature = "ai-candle")]
use anyhow::Result;
#[cfg(feature = "ai-candle")]
use candle_core::{Device, DeviceLocation};

#[cfg(feature = "ai-candle")]
use crate::config::{AiTrackingConfig, TemplateTrackingConfig};

#[cfg(feature = "ai-candle")]
pub(crate) trait CandleDeviceConfig {
    fn device_preference(&self) -> AiDevicePreference;
    fn device_index(&self) -> usize;
}

#[cfg(feature = "ai-candle")]
impl CandleDeviceConfig for AiTrackingConfig {
    fn device_preference(&self) -> AiDevicePreference {
        self.device
    }

    fn device_index(&self) -> usize {
        self.device_index
    }
}

#[cfg(feature = "ai-candle")]
impl CandleDeviceConfig for TemplateTrackingConfig {
    fn device_preference(&self) -> AiDevicePreference {
        self.device
    }

    fn device_index(&self) -> usize {
        self.device_index
    }
}

#[cfg(feature = "ai-candle")]
pub(crate) fn select_candle_device(config: &impl CandleDeviceConfig) -> Result<Device> {
    match config.device_preference() {
        AiDevicePreference::Cpu => Ok(Device::Cpu),
        AiDevicePreference::Cuda => build_cuda_device(config.device_index()),
        AiDevicePreference::Metal => build_metal_device(config.device_index()),
    }
}

#[cfg(feature = "ai-candle")]
pub(crate) fn candle_device_label(device: &Device) -> String {
    match device.location() {
        DeviceLocation::Cpu => "CPU".to_owned(),
        DeviceLocation::Cuda { gpu_id } => format!("CUDA:{gpu_id}"),
        DeviceLocation::Metal { gpu_id } => format!("Metal:{gpu_id}"),
    }
}

#[cfg(candle_cuda_backend)]
fn available_cuda_device_descriptors() -> Vec<CandleDeviceDescriptor> {
    use candle_core::cuda::cudarc::driver::safe::CudaContext;

    const MAX_PROBED_CUDA_DEVICES: usize = 16;

    let reported_count = CudaContext::device_count()
        .ok()
        .and_then(|count| usize::try_from(count).ok());
    let probe_limit = reported_count
        .map(|count| count.saturating_add(4).max(1))
        .unwrap_or(MAX_PROBED_CUDA_DEVICES)
        .min(MAX_PROBED_CUDA_DEVICES);

    let mut descriptors = Vec::new();
    for ordinal in 0..probe_limit {
        let Ok(context) = CudaContext::new(ordinal) else {
            continue;
        };
        let name = context
            .name()
            .unwrap_or_else(|_| format!("CUDA 设备 {ordinal}"));
        descriptors.push(CandleDeviceDescriptor { ordinal, name });
    }

    descriptors
}

#[cfg(not(candle_cuda_backend))]
fn available_cuda_device_descriptors() -> Vec<CandleDeviceDescriptor> {
    Vec::new()
}

#[cfg(candle_metal_backend)]
fn available_metal_device_descriptors() -> Vec<CandleDeviceDescriptor> {
    const MAX_PROBED_DEVICES: usize = 8;

    let mut descriptors = Vec::new();
    for ordinal in 0..MAX_PROBED_DEVICES {
        if candle_core::Device::new_metal(ordinal).is_ok() {
            descriptors.push(CandleDeviceDescriptor {
                ordinal,
                name: format!("Metal 设备 {ordinal}"),
            });
        } else {
            break;
        }
    }

    descriptors
}

#[cfg(not(candle_metal_backend))]
fn available_metal_device_descriptors() -> Vec<CandleDeviceDescriptor> {
    Vec::new()
}

#[cfg(all(feature = "ai-candle", candle_cuda_backend))]
fn build_cuda_device(ordinal: usize) -> Result<Device> {
    Device::new_cuda(ordinal)
        .with_context(|| format!("无法初始化 CUDA 设备 {ordinal}，请检查驱动、运行库和显卡状态"))
}

#[cfg(all(feature = "ai-candle", not(candle_cuda_backend)))]
fn build_cuda_device(_ordinal: usize) -> Result<Device> {
    anyhow::bail!(
        "配置选择了 CUDA 设备，但当前二进制未包含 CUDA 后端；Windows 默认构建会包含 CUDA，其他平台请使用 `cargo run --features ai-candle-cuda` 重新构建"
    )
}

#[cfg(all(feature = "ai-candle", candle_metal_backend))]
fn build_metal_device(ordinal: usize) -> Result<Device> {
    Device::new_metal(ordinal)
        .with_context(|| format!("无法初始化 Metal 设备 {ordinal}，请检查系统和 GPU 支持状态"))
}

#[cfg(all(feature = "ai-candle", not(candle_metal_backend)))]
fn build_metal_device(_ordinal: usize) -> Result<Device> {
    anyhow::bail!(
        "配置选择了 Metal 设备，但当前二进制未包含 Metal 后端；macOS 默认构建会包含 Metal，其他平台请使用 `cargo run --features ai-candle-metal` 重新构建"
    )
}
