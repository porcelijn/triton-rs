use crate::{Error, InferenceRequest};
use crate::check_err;
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
    pub fn infer_async(&self, request: &InferenceRequest) -> Result<(), Error> {
        check_err(unsafe {
            triton_sys::TRITONSERVER_ServerInferAsync(self.ptr, request.as_ptr(), ptr::null_mut())
        })
    }
}

impl std::fmt::Debug for Server {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        todo!()
    }
}
