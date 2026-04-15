#![cfg(feature = "ai-candle")]

#[cfg(any(feature = "ai-candle-cuda", feature = "ai-candle-metal"))]
use anyhow::Context as _;
use anyhow::Result;

use crate::config::{AiDevicePreference, AiTrackingConfig, TemplateTrackingConfig};
use candle_core::{Device, DeviceLocation};

pub(crate) trait CandleDeviceConfig {
    fn device_preference(&self) -> AiDevicePreference;
    fn device_index(&self) -> usize;
}

impl CandleDeviceConfig for AiTrackingConfig {
    fn device_preference(&self) -> AiDevicePreference {
        self.device
    }

    fn device_index(&self) -> usize {
        self.device_index
    }
}

impl CandleDeviceConfig for TemplateTrackingConfig {
    fn device_preference(&self) -> AiDevicePreference {
        self.device
    }

    fn device_index(&self) -> usize {
        self.device_index
    }
}

pub(crate) fn select_candle_device(config: &impl CandleDeviceConfig) -> Result<Device> {
    match config.device_preference() {
        AiDevicePreference::Cpu => Ok(Device::Cpu),
        AiDevicePreference::Cuda => build_cuda_device(config.device_index()),
        AiDevicePreference::Metal => build_metal_device(config.device_index()),
    }
}

pub(crate) fn candle_device_label(device: &Device) -> String {
    match device.location() {
        DeviceLocation::Cpu => "CPU".to_owned(),
        DeviceLocation::Cuda { gpu_id } => format!("CUDA:{gpu_id}"),
        DeviceLocation::Metal { gpu_id } => format!("Metal:{gpu_id}"),
    }
}

pub(crate) fn available_candle_backends() -> &'static str {
    #[cfg(all(feature = "ai-candle-cuda", feature = "ai-candle-metal"))]
    {
        return "CPU / CUDA / Metal";
    }

    #[cfg(all(feature = "ai-candle-cuda", not(feature = "ai-candle-metal")))]
    {
        return "CPU / CUDA";
    }

    #[cfg(all(not(feature = "ai-candle-cuda"), feature = "ai-candle-metal"))]
    {
        return "CPU / Metal";
    }

    #[cfg(all(not(feature = "ai-candle-cuda"), not(feature = "ai-candle-metal")))]
    {
        "CPU"
    }
}

#[cfg(feature = "ai-candle-cuda")]
fn build_cuda_device(ordinal: usize) -> Result<Device> {
    Device::new_cuda(ordinal)
        .with_context(|| format!("无法初始化 CUDA 设备 {ordinal}，请检查驱动、运行库和显卡状态"))
}

#[cfg(not(feature = "ai-candle-cuda"))]
fn build_cuda_device(_ordinal: usize) -> Result<Device> {
    anyhow::bail!(
        "配置选择了 CUDA 设备，但当前二进制未启用 `ai-candle-cuda` 特性；请使用 `cargo run --features ai-candle-cuda` 重新构建"
    )
}

#[cfg(feature = "ai-candle-metal")]
fn build_metal_device(ordinal: usize) -> Result<Device> {
    Device::new_metal(ordinal)
        .with_context(|| format!("无法初始化 Metal 设备 {ordinal}，请检查系统和 GPU 支持状态"))
}

#[cfg(not(feature = "ai-candle-metal"))]
fn build_metal_device(_ordinal: usize) -> Result<Device> {
    anyhow::bail!(
        "配置选择了 Metal 设备，但当前二进制未启用 `ai-candle-metal` 特性；请使用 `cargo run --features ai-candle-metal` 重新构建"
    )
}
