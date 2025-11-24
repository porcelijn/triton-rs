use crate::{check_err, DataType, Error, ModelExecutor};
use std::ffi::CString;

#[cfg(feature = "ndarray")] use ndarray::{Array, IxDyn};

pub struct InferenceRequest {
    ptr: *mut triton_sys::TRITONSERVER_InferenceRequest,
}

impl InferenceRequest {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONSERVER_InferenceRequest) -> Self {
        Self { ptr }
    }

    pub fn as_ptr(&self) -> *mut triton_sys::TRITONSERVER_InferenceRequest {
        self.ptr
    }

    pub fn new(executor: &ModelExecutor) -> Result<Self, Box<dyn std::error::Error>> {
        let mut request: *mut triton_sys::TRITONSERVER_InferenceRequest = std::ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONSERVER_InferenceRequestNew(
                &mut request,
                executor.server.as_ptr(),
                executor.model_name.as_ptr(),
                executor.model_version,
            )
        })?;
        Ok(Self { ptr: request })
    }

    pub fn add_output(&self, name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let cstr_name = CString::new(name)?;
        check_err(unsafe {
            triton_sys::TRITONSERVER_InferenceRequestAddRequestedOutput(
                self.ptr,
                cstr_name.as_ptr(),
            )
        })?;
        Ok(())
    }

    pub fn add_input(
        &self,
        name: &str,
        data_type: DataType,
        shape: &[i64],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cstr_name = CString::new(name)?;

        check_err(unsafe {
            triton_sys::TRITONSERVER_InferenceRequestAddInput(
                self.ptr,
                cstr_name.as_ptr(),
                data_type as u32,
                shape.as_ptr(),
                shape.len().try_into().expect("Shape length overflow"),
            )
        })?;
        Ok(())
    }

    pub fn set_input_data(
        &self,
        name: &str,
        data: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cstr_name = CString::new(name)?;
        let c_data_ptr = data.as_ptr() as *const std::os::raw::c_void;
        let data_memory_id = 0;
        check_err(unsafe {
            triton_sys::TRITONSERVER_InferenceRequestAppendInputData(
                self.ptr,
                cstr_name.as_ptr(),
                c_data_ptr,
                data.len(),
                triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                data_memory_id as i64,
            )
        })?;
        Ok(())
    }

    #[cfg(feature = "ndarray")]
    pub fn add_input_array<T>(&self, name: &str, array: Array<T, IxDyn>,
    ) -> Result<(), Box<dyn std::error::Error>>
    where T: Copy + crate::data_type::SupportedTypes {
        let shape: Vec<i64> = array.shape().iter().map(|x| *x as i64).collect();
        let data_type = <T as crate::data_type::SupportedTypes>::of();
        assert_eq!(data_type.byte_size() as usize, std::mem::size_of::<T>());
        self.add_input(name, data_type, &shape)?;
        let data = array.as_raw_ref();
        let data = unsafe { std::slice::from_raw_parts(data.as_ptr(), data.len()) };
        let data: &[u8] = unsafe { std::mem::transmute(data) };
        self.set_input_data(name, data)
    }

    pub fn set_request_id(&self, id: &str) -> Result<(), Error> {
        let cstr_id = CString::new(id)?;
        check_err(unsafe {
            triton_sys::TRITONSERVER_InferenceRequestSetId(self.ptr, cstr_id.as_ptr())
        })?;
        Ok(())
    }

    pub fn set_correlation_id(&self, id: u64) -> Result<(), Error>{
        check_err(unsafe {
            triton_sys::TRITONSERVER_InferenceRequestSetCorrelationId(self.ptr, id)
        })?;
        Ok(())
    }

    pub fn set_release_callback(&self) -> Result<(), Error>{
        check_err(unsafe {
            triton_sys::TRITONSERVER_InferenceRequestSetReleaseCallback(
                self.ptr,
                Some(infer_request_complete),
                std::ptr::null_mut(),
            )
        })?;
        Ok(())
    }
}

extern "C" fn infer_request_complete(
    request: *mut triton_sys::TRITONSERVER_InferenceRequest,
    _flags: u32,
    _userp: *mut ::std::os::raw::c_void,
) {
    unsafe {
        if !request.is_null() {
            let err = triton_sys::TRITONSERVER_InferenceRequestDelete(request);
            if !err.is_null() {
                let err_msg = triton_sys::TRITONSERVER_ErrorCodeString(err);
                eprintln!(
                    "Failed to delete inference request: {:?}",
                    std::ffi::CStr::from_ptr(err_msg)
                );
                triton_sys::TRITONSERVER_ErrorDelete(err);
            }
        }
    }
}
