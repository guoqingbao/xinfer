//! Offloaded experts stored as plain tensors (`FusedMoe` bf16/f16, `FusedMoeFp8`,
//! `FusedMoeMxfp4`, `FusedMoeWNA16`): every `[E, ...]` expert tensor of a layer (weights
//! and scales, any dtype) lives in host RAM; the GPU keeps `[num_slots, ...]` slot tensors.

use super::budget::{register, SlotPool};
use super::cache::{copy_htod, dev_ptr, profile_begin, profile_end, Plan, SlotCache};
use super::host_banks::HostBank;
use super::prefill;
use candle_core::{CpuStorage, CudaDevice, DType, Device, Result, Storage, Tensor};
use std::sync::{Arc, RwLock};

/// Slot tensors and their LRU; replaced as a whole by `resize`.
struct Slots<const N: usize> {
    cache: SlotCache,
    weights: [Tensor; N],
    dst: [u64; N],
}

struct Inner<const N: usize> {
    dev: CudaDevice,
    num_experts: usize,
    banks: [HostBank; N],
    // (dims, dtype) of each expert tensor.
    shapes: [(Vec<usize>, DType); N],
    staging: [Tensor; N],
    src: [u64; N],
    bytes: [u64; N],
    slots: RwLock<Option<Slots<N>>>,
}

/// `N` expert tensors of one MoE layer (e.g. gate_up, gate_up scales, down, down scales).
pub struct ExpertOffload<const N: usize>(Arc<Inner<N>>);

fn arr<T, const N: usize>(v: Vec<T>) -> [T; N] {
    v.try_into()
        .unwrap_or_else(|_| unreachable!("one entry per expert tensor"))
}

/// Little-endian bytes of a tensor, as the GPU stores them (fp8/e8m0 are u8-backed).
pub(crate) fn host_bytes(t: &Tensor) -> Result<Vec<u8>> {
    fn le<T: Copy, const B: usize>(v: &[T], f: fn(T) -> [u8; B]) -> Vec<u8> {
        v.iter().flat_map(|&x| f(x)).collect()
    }
    let t = t.to_device(&Device::Cpu)?.contiguous()?;
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cpu(cpu) = &*storage else {
        candle_core::bail!("expected a host tensor");
    };
    let r = layout.start_offset()..layout.start_offset() + t.elem_count();
    Ok(match cpu {
        CpuStorage::U8(v) => v[r].to_vec(),
        CpuStorage::U32(v) => le(&v[r], u32::to_le_bytes),
        CpuStorage::I64(v) => le(&v[r], i64::to_le_bytes),
        CpuStorage::BF16(v) => le(&v[r], |x| x.to_bits().to_le_bytes()),
        CpuStorage::F16(v) => le(&v[r], |x| x.to_bits().to_le_bytes()),
        CpuStorage::F32(v) => le(&v[r], f32::to_le_bytes),
        CpuStorage::F64(v) => le(&v[r], f64::to_le_bytes),
    })
}

fn make_slots<const N: usize>(
    dev: &CudaDevice,
    num_experts: usize,
    shapes: &[(Vec<usize>, DType); N],
    num_slots: usize,
) -> Result<Slots<N>> {
    let gpu = Device::Cuda(dev.clone());
    let mut weights = Vec::with_capacity(N);
    let mut dst = [0u64; N];
    for ((dims, dtype), d) in shapes.iter().zip(dst.iter_mut()) {
        let mut dims = dims.clone();
        dims[0] = num_slots;
        let w = Tensor::zeros(dims, *dtype, &gpu)?;
        *d = dev_ptr(&w)?;
        weights.push(w);
    }
    Ok(Slots {
        cache: SlotCache::new(dev, num_experts, num_slots)?,
        weights: arr(weights),
        dst,
    })
}

impl<const N: usize> SlotPool for Inner<N> {
    fn device(&self) -> &CudaDevice {
        &self.dev
    }
    fn num_experts(&self) -> usize {
        self.num_experts
    }
    fn num_slots(&self) -> usize {
        self.slots
            .read()
            .unwrap()
            .as_ref()
            .expect("MoE offload slots unavailable")
            .cache
            .num_slots()
    }
    fn slot_bytes(&self) -> usize {
        self.bytes.iter().sum::<u64>() as usize
    }
    fn resize(&self, num_slots: usize) -> Result<()> {
        // Drop the old CUDA allocations before making the replacement.  Auto sizing is
        // performed before inference, so no forward can observe the temporary `None`.
        self.slots.write().unwrap().take();
        let slots = make_slots(&self.dev, self.num_experts, &self.shapes, num_slots)?;
        *self.slots.write().unwrap() = Some(slots);
        Ok(())
    }
}

impl<const N: usize> ExpertOffload<N> {
    /// Moves `weights` (`[E, ...]` each, typically host tensors) into pinned host banks.
    /// Returns the offload and empty GPU stand-ins for the layer's expert fields: unused
    /// while offloaded (forward reads `prepare`), so a later slot resize frees the old slots.
    pub fn new(dev: &CudaDevice, weights: [Tensor; N], topk: usize) -> Result<(Self, [Tensor; N])> {
        let device = Device::Cuda(dev.clone());
        let rate = crate::utils::env::moe_offload_rate().expect("offload enabled");
        let num_experts = weights[0].dim(0)?;
        let (mut banks, mut shapes, mut staging) = (Vec::new(), Vec::new(), Vec::new());
        for (i, t) in weights.iter().enumerate() {
            if t.dim(0)? != num_experts {
                candle_core::bail!("XINFER_MOE_OFFLOAD: expert tensor {i} is not [E, ...]");
            }
            banks.push(HostBank::new(dev, &host_bytes(t)?, num_experts)?);
            // Keyed by position: tensors of one layer never share a staging buffer.
            staging.push(prefill::staging(
                dev,
                &format!("expert{i}"),
                t.dims(),
                t.dtype(),
            )?);
            shapes.push((t.dims().to_vec(), t.dtype()));
        }
        let banks: [HostBank; N] = arr(banks);
        let shapes: [(Vec<usize>, DType); N] = arr(shapes);
        let slots = make_slots(
            dev,
            num_experts,
            &shapes,
            SlotCache::slots_for(rate, num_experts, topk),
        )?;
        let empty = arr(shapes
            .iter()
            .map(|(_, dtype)| Tensor::zeros(0, *dtype, &device))
            .collect::<Result<Vec<_>>>()?);
        let inner = Arc::new(Inner {
            dev: dev.clone(),
            num_experts,
            src: banks.each_ref().map(|b| b.dev_ptr),
            bytes: banks.each_ref().map(|b| b.expert_bytes as u64),
            banks,
            shapes,
            staging: arr(staging),
            slots: RwLock::new(Some(slots)),
        });
        register(inner.clone());
        Ok((Self(inner), empty))
    }

    /// Makes the routed experts resident; returns slot ids (or the original ids with
    /// full-layer staging weights) and the expert tensors the kernels must read.
    pub fn prepare(&self, topk_ids: Tensor, is_prefill: bool) -> Result<(Tensor, [Tensor; N])> {
        let o = &self.0;
        let slots = o.slots.read().unwrap();
        let s = slots.as_ref().expect("MoE offload slots unavailable");
        let t = profile_begin(&o.dev)?;
        let out = match s.cache.plan(&topk_ids, is_prefill)? {
            Plan::Staged => {
                for (bank, dst) in o.banks.iter().zip(&o.staging) {
                    copy_htod(&o.dev, dev_ptr(dst)?, bank.all())?;
                }
                (topk_ids, o.staging.clone())
            }
            Plan::Slots(ids) => {
                s.cache.gather(&o.src, &s.dst, &o.bytes)?;
                (ids, s.weights.clone())
            }
        };
        profile_end(&o.dev, t)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{host_bytes, ExpertOffload};
    use crate::models::layers::moe::sort_expert_assignments;
    use attention_rs::moe::moe_gemm;
    use candle_core::{DType, Device, Result, Tensor};

    #[test]
    fn offloaded_moe_gemm_matches_resident_weights() -> Result<()> {
        let Ok(Device::Cuda(dev)) = Device::new_cuda(0) else {
            return Ok(());
        };
        let gpu = Device::Cuda(dev.clone());
        // 8 experts, 4 slots: decode steps hit, miss and evict; the prefill call stages.
        std::env::set_var(crate::utils::env::MOE_OFFLOAD_ENV, "0.5");
        let (e, n, k, topk) = (8, 64, 64, 2);
        let gate_up =
            Tensor::randn(0f32, 1.0, (e, 2 * n, k), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let down = Tensor::randn(0f32, 1.0, (e, k, n), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let full = [gate_up.to_device(&gpu)?, down.to_device(&gpu)?];
        let (o, _) = ExpertOffload::new(&dev, [gate_up, down], topk)?;

        // As in `FusedMoe::forward_with_routing`: gate_up without, down with routing weights.
        let run = |w: &Tensor, x: &Tensor, ids: &Tensor, prefill: bool| -> Result<Vec<Vec<f32>>> {
            let (expert_ids, sorted) = sort_expert_assignments(ids, prefill)?;
            let rw = (x.dim(0)? != ids.dim(0)?)
                .then(|| ids.ones_like()?.to_dtype(DType::F32))
                .transpose()?;
            moe_gemm(x, w, &rw, &sorted, &expert_ids, topk, prefill)?
                .to_dtype(DType::F32)?
                .to_vec2()
        };
        let steps: [&[u32]; 5] = [&[0, 1], &[2, 3], &[4, 5], &[0, 7], &[1, 3, 6, 2, 5, 0]];
        for ids in steps {
            let tokens = ids.len() / topk;
            let prefill = tokens > 1;
            let ids = Tensor::from_slice(ids, (tokens, topk), &gpu)?;
            let x = Tensor::randn(0f32, 1.0, (tokens, k), &gpu)?.to_dtype(DType::BF16)?;
            let h = Tensor::randn(0f32, 1.0, (tokens * topk, n), &gpu)?.to_dtype(DType::BF16)?;
            let (slot_ids, [w_gate_up, w_down]) = o.prepare(ids.clone(), prefill)?;
            assert_eq!(
                run(&w_gate_up, &x, &slot_ids, prefill)?,
                run(&full[0], &x, &ids, prefill)?
            );
            assert_eq!(
                run(&w_down, &h, &slot_ids, prefill)?,
                run(&full[1], &h, &ids, prefill)?
            );
        }
        Ok(())
    }

    #[test]
    fn host_bytes_keep_bit_patterns() -> Result<()> {
        let t = Tensor::new(&[1.0f32, -2.0], &Device::Cpu)?.to_dtype(DType::BF16)?;
        // bf16 1.0 = 0x3F80, -2.0 = 0xC000, little endian.
        assert_eq!(host_bytes(&t)?, vec![0x80, 0x3F, 0x00, 0xC0]);
        let t = Tensor::new(&[0x0403_0201u32], &Device::Cpu)?;
        assert_eq!(host_bytes(&t)?, vec![1, 2, 3, 4]);
        // A strided view is copied in logical order.
        let t = Tensor::new(&[[1u8, 2], [3, 4]], &Device::Cpu)?.t()?;
        assert_eq!(host_bytes(&t)?, vec![1, 3, 2, 4]);
        Ok(())
    }
}
