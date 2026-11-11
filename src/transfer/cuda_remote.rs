// src/core/transfer/cuda_remote.rs
use super::CudaIpcMemHandle;
use candle_core::cuda_backend::cudarc::driver::sys::{lib, CUdeviceptr, CUipcMemHandle};
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{Device, Result, Tensor, WithDType};
use std::mem::{ManuallyDrop, MaybeUninit};
/// (Server) Gets a serializable IPC handle for a GPU tensor's memory.
pub(super) fn get_ipc_handle<T: WithDType + candle_core::cuda_backend::CudaDType>(
    tensor: &Tensor,
) -> Result<CudaIpcMemHandle> {
    use candle_core::Storage;
    let (storage, _) = tensor.storage_and_layout();
    let Storage::Cuda(src_storage) = &*storage else {
        candle_core::bail!("Invalid source kvcache storage!")
    };
    let ptr = src_storage.as_cuda_slice::<T>()?.device_ptr();

    let mut handle = MaybeUninit::<CUipcMemHandle>::uninit();
    let handle = unsafe {
        lib()
            .cuIpcGetMemHandle(handle.as_mut_ptr(), *ptr)
            .result()
            .map_err(|e| candle_core::Error::Msg(format!("cuIpcGetMemHandle failed: {e:?}")))?;
        handle.assume_init()
    };

    Ok(CudaIpcMemHandle(
        handle.reserved.to_vec(),
        tensor.shape().dims().to_vec(),
        tensor.dtype().into(),
    ))
}

/// Dtype-dispatching version of get_ipc_handle — picks the right T at runtime.
pub(super) fn get_ipc_handle_any(tensor: &Tensor) -> Result<CudaIpcMemHandle> {
    match tensor.dtype() {
        candle_core::DType::U8 => get_ipc_handle::<u8>(tensor),
        candle_core::DType::F32 => get_ipc_handle::<f32>(tensor),
        candle_core::DType::F16 => get_ipc_handle::<half::f16>(tensor),
        candle_core::DType::BF16 => get_ipc_handle::<half::bf16>(tensor),
        dt => candle_core::bail!("Unsupported dtype for IPC handle: {:?}", dt),
    }
}

/// Dtype-dispatching version of open_ipc_handle — picks the right T at runtime.
pub(super) fn open_ipc_handle_any(
    handle: &CudaIpcMemHandle,
    device: &Device,
) -> Result<ManuallyDrop<Tensor>> {
    let dt: candle_core::DType = handle.2.clone().into();
    match dt {
        candle_core::DType::U8 => open_ipc_handle::<u8>(handle, device),
        candle_core::DType::F32 => open_ipc_handle::<f32>(handle, device),
        candle_core::DType::F16 => open_ipc_handle::<half::f16>(handle, device),
        candle_core::DType::BF16 => open_ipc_handle::<half::bf16>(handle, device),
        dt => candle_core::bail!("Unsupported dtype for IPC open: {:?}", dt),
    }
}

/// (Client) Opens an IPC handle to get a local tensor pointing to remote GPU memory.
pub(super) fn open_ipc_handle<T: WithDType + candle_core::cuda_backend::CudaDType>(
    handle: &CudaIpcMemHandle,
    device: &Device,
) -> Result<ManuallyDrop<Tensor>> {
    use candle_core::cuda_backend::cudarc::driver::CudaSlice;

    let mut ptr: CUdeviceptr = 0;
    use core::ffi::c_char;
    if handle.0.len() != 64 {
        candle_core::bail!("Invalid CUipcMemHandle handle!");
    }
    // `CUipcMemHandle::reserved` is typed as `[c_char; 64]`, and `c_char` is
    // `i8` on x86_64 but `u8` on aarch64/powerpc64/s390x/..., so the element
    // type has to follow the target instead of being hardcoded.
    let raw_array: [c_char; 64] = handle.0.clone().try_into().expect("length checked");
    let handle_raw = CUipcMemHandle {
        reserved: raw_array,
    };
    unsafe {
        lib()
            .cuIpcOpenMemHandle_v2(&mut ptr, handle_raw, 1)
            .result()
            .map_err(|e| candle_core::Error::Msg(format!("cuIpcOpenMemHandle_v2 failed: {e:?}")))?;
    }
    let dev = device.as_cuda_device()?;
    let src_slice = unsafe {
        let slice: CudaSlice<T> = dev.upgrade_device_ptr(ptr, handle.1.iter().sum());
        // std::mem::ManuallyDrop::new(slice)
        slice
    };

    let slice = candle_core::CudaStorage::wrap_cuda_slice(src_slice, dev.clone());
    // We created a virtual Tensor, it stored the remote mem handle, so we should not release it
    Ok(ManuallyDrop::new(Tensor::from_storage(
        candle_core::Storage::Cuda(slice),
        handle.1.clone(),
    )?))
}
