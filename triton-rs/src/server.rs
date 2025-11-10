use crate::{InferenceRequest, ModelExecuterError};
use crate::dump_err;
use std::ptr;

pub struct Server {
    ptr: *mut triton_sys::TRITONSERVER_Server,
}

impl Server {

    pub fn from_ptr(ptr: *mut triton_sys::TRITONSERVER_Server) -> Self {
        Self { ptr }
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_sys::TRITONSERVER_Server {
        self.ptr
    }

    // Start async inference
    pub fn infer_async(&self, request: &InferenceRequest) -> Result<(), ModelExecuterError> {
        let err = unsafe {
            triton_sys::TRITONSERVER_ServerInferAsync(self.ptr, request.as_ptr(), ptr::null_mut())
        };

        if !err.is_null() {
            dump_err(err);
            return Err(ModelExecuterError::ExecutionError(
                    "Failed to start async inference".to_string(),
                    ));
        }
        Ok(())
    }
}
