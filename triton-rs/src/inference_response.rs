use crate::{DataType, Error};
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::collections::HashMap;

#[cfg(feature = "ndarray")] use ndarray::{ArrayView, IntoDimension, IxDyn};

mod detail {

  use crate::check_err;
  use super::OutputData;

  pub (crate) struct InferenceResponse {
    ptr: *mut triton_sys::TRITONSERVER_InferenceResponse,
  }

  impl InferenceResponse {
    pub (crate) fn from_ptr(ptr: *mut triton_sys::TRITONSERVER_InferenceResponse) -> Self {
        Self { ptr }
    }

    pub fn get_output_count(&self) -> Result<u32, Box<dyn std::error::Error>> {
        let mut count = 0;
        check_err(unsafe {
            triton_sys::TRITONSERVER_InferenceResponseOutputCount(self.ptr, &mut count)
        })?;
        Ok(count)
    }

    pub fn get_output_data(&self, out_idx: u32) -> Result<OutputData, Box<dyn std::error::Error>> {
        OutputData::get_output_data(self.ptr, out_idx)
    }
  }

} // detail

pub struct InferenceResponse {
    outputs: HashMap<String, OutputData>,
}

impl InferenceResponse {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONSERVER_InferenceResponse) -> Result<Self, Error> {
        let helper = detail::InferenceResponse::from_ptr(ptr);
        let count = helper.get_output_count()?;
        let mut outputs = HashMap::with_capacity(count.try_into()?);
        for index in 0..count {
            let data = helper.get_output_data(index)?;
            outputs.insert(data.name.clone(), data);
        }
        Ok(Self { outputs })
    }

    pub fn get_output_count(&self) -> usize {
        self.outputs.len()
    }

    pub fn get_output_data(&self, name: &str) -> Result<&OutputData, Error> {
        self.outputs.get(name).ok_or(format!("failed to find output {name}").into())
    }

    pub fn iter(&self) -> impl Iterator<Item = &OutputData> {
        self.outputs.values()
    }
}

pub struct OutputData {
    pub name: String,
    pub data_type: DataType,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
    pub memory_type: triton_sys::TRITONSERVER_MemoryType,
    pub memory_type_id: i64,
}

impl OutputData {
    pub fn get_output_data(
        response: *mut triton_sys::TRITONSERVER_InferenceResponse,
        out_idx: u32,
    ) -> Result<Self, Error> {
        unsafe {
            let mut name_ptr: *const c_char = std::ptr::null();
            let mut datatype = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32;
            let mut shape_ptr: *const i64 = std::ptr::null();
            let mut dim_count: u64 = 0;
            let mut base_ptr: *const c_void = std::ptr::null();
            let mut byte_size: usize = 0;
            let mut memory_type = triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
            let mut memory_type_id: i64 = 0;
            let mut user_ptr: *mut c_void = std::ptr::null_mut();

            let err = triton_sys::TRITONSERVER_InferenceResponseOutput(
                response,
                out_idx,
                &mut name_ptr,
                &mut datatype,
                &mut shape_ptr,
                &mut dim_count,
                &mut base_ptr,
                &mut byte_size,
                &mut memory_type,
                &mut memory_type_id,
                &mut user_ptr,
            );

            if !err.is_null() {
                return Err("Failed to get output".into());
            }

            let name = CStr::from_ptr(name_ptr).to_str()?.to_owned();
            let data_type = DataType::from(datatype);
            let shape = if !shape_ptr.is_null() {
                std::slice::from_raw_parts(shape_ptr, dim_count as usize).to_vec()
            } else {
                vec![]
            };

            let data = if !base_ptr.is_null() {
                std::slice::from_raw_parts(base_ptr as *const u8, byte_size).to_vec()
            } else {
                vec![]
            };

            Ok(Self { name, data_type, shape, data, memory_type, memory_type_id})
        }
    }

    #[cfg(feature="ndarray")]
    pub fn as_array<T, const N: usize>(&self)
            -> Result<ArrayView<T, IxDyn>, Error>
            where T: crate::data_type::SupportedTypes {
        assert!(<T as crate::data_type::SupportedTypes>::of() == self.data_type);
        assert!(N == self.shape.len());
        const { assert!( N < 6) };
        let shape: Vec<usize> = self.shape.iter().map(|&x| x as usize).collect();
        let shape = shape.into_dimension();
        let data: &[u8] = &self.data;
        let data: &[T] = unsafe { std::mem::transmute(data) };
        let array: ArrayView<T, IxDyn> = ArrayView::from_shape(shape, data)?;
        Ok(array)
    }
}

impl std::fmt::Debug for OutputData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f,
            "name: {:?}, datatype: {:?}, shape: {:?}, data len: {:?}, memory_type: {:?}, memory_type_id: {:?}",
            self.name, self.data_type, self.shape, self.data.len(), self.memory_type, self.memory_type_id)?;
        if self.data_type == DataType::BYTES {
            write!(f, ", data: {:?}", String::from_utf8_lossy(&self.data))?;
        }
        Ok(())
    }
}

