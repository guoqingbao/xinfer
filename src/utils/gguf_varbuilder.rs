use candle::quantized::QTensor;
use candle::{Device, Result, Shape};
use candle_core as candle;
use std::fs::File;
use std::sync::Arc;
use std::sync::Mutex;
// light-cached qvarbuilder

#[derive(Clone)]
pub struct VarBuilder {
    content: Arc<candle_core::quantized::gguf_file::Content>,
    file: Arc<std::sync::Mutex<File>>,
    cache: Arc<Mutex<Option<(String, Arc<QTensor>)>>>,
    path: Vec<String>,
    device: Device,
    file_path: Arc<String>,
}

impl VarBuilder {
    pub fn from_gguf<P: AsRef<std::path::Path>>(p: P, device: &Device) -> Result<Self> {
        let file_path = p.as_ref().to_string_lossy().to_string();
        let mut file = File::open(&p)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        Ok(Self {
            content: Arc::new(content),
            file: Arc::new(std::sync::Mutex::new(file)),
            cache: Arc::new(Mutex::new(None)),
            path: Vec::new(),
            device: device.clone(),
            file_path: Arc::new(file_path),
        })
    }

    pub fn gguf_path(&self) -> &str {
        &self.file_path
    }

    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            content: self.content.clone(),
            file: self.file.clone(),
            cache: self.cache.clone(),
            path,
            device: self.device.clone(),
            file_path: self.file_path.clone(),
        }
    }

    pub fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);

        // Check cache
        {
            let cache_guard = self.cache.lock().unwrap();
            if let Some((ref cached_name, ref cached_tensor)) = *cache_guard {
                if cached_name == &path {
                    // Return cached tensor
                    let shape = s.into();
                    if cached_tensor.shape() != &shape {
                        candle::bail!(
                            "shape mismatch for {name}, got {:?}, expected {shape:?}",
                            cached_tensor.shape()
                        );
                    }
                    return Ok(cached_tensor.clone());
                }
            }
        }

        let mut file = self.file.lock().unwrap();
        let tensor = self.content.tensor(&mut *file, &path, &self.device)?;
        let tensor = Arc::new(tensor);
        // Update cache
        *self.cache.lock().unwrap() = Some((path.clone(), tensor.clone()));

        let shape = s.into();
        if tensor.shape() != &shape {
            candle::bail!(
                "shape mismatch for {name}, got {:?}, expected {shape:?}",
                tensor.shape()
            );
        }
        Ok(tensor)
    }

    pub fn get_sharded<S: Into<Shape>>(
        &self,
        s: S,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<Option<Arc<QTensor>>> {
        if world_size <= 1 {
            return self.get(s, name).map(Some);
        }

        let path = self.path(name);
        let shape = s.into();
        if dim >= shape.dims().len() {
            candle::bail!(
                "cannot shard tensor {path} with shape {:?} on dim {dim}",
                shape
            );
        }
        if shape.dims()[dim] % world_size != 0 {
            candle::bail!(
                "cannot shard tensor {path} dim {dim} size {} into {world_size} parts",
                shape.dims()[dim]
            );
        }
        let mut shard_shape = shape.dims().to_vec();
        shard_shape[dim] /= world_size;

        let mut file = self.file.lock().unwrap();
        let Some(tensor) =
            self.content
                .tensor_shard(&mut *file, &path, dim, rank, world_size, &self.device)?
        else {
            return Ok(None);
        };
        if tensor.shape().dims() != shard_shape.as_slice() {
            candle::bail!(
                "shape mismatch for sharded {name}, got {:?}, expected {:?}",
                tensor.shape(),
                shard_shape
            );
        }
        Ok(Some(Arc::new(tensor)))
    }

    pub fn get_no_shape(&self, name: &str) -> Result<Arc<QTensor>> {
        let mut file = self.file.lock().unwrap();
        let tensor = self
            .content
            .tensor(&mut *file, &self.path(name), &self.device)?;
        Ok(Arc::new(tensor))
    }

    pub fn clear_cache(&self) {
        *self.cache.lock().unwrap() = None;
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.content.tensor_infos.contains_key(&self.path(key))
    }

    pub fn tensor_shape(&self, key: &str) -> Option<Vec<usize>> {
        self.content
            .tensor_infos
            .get(&self.path(key))
            .map(|info| info.shape.dims().to_vec())
    }
}
