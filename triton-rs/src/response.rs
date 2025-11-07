use crate::{check_err, Request, Error};
use std::ffi::CString;
use std::ptr;

pub struct Response {
    ptr: *mut triton_sys::TRITONBACKEND_Response,
}

impl Response {
    fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Response) -> Self {
        Self { ptr }
    }

    pub fn new(request: &Request) -> Result<Self, Error> {
        let mut response: *mut triton_sys::TRITONBACKEND_Response = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ResponseNew(&mut response, request.as_ptr())
        })?;
        Ok(Response::from_ptr(response))
    }

    pub fn send(self, send_flags: u32) -> Result<(), Error> {
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ResponseSend(self.ptr, send_flags, ptr::null_mut())
        })?;
        Ok(())
    }

    pub fn send_error(self, send_flags: u32, error: Error) -> Result<(), Error> {
        let error = unsafe {
            let error_code = triton_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_UNSUPPORTED;
            let error = CString::new(error.to_string()).unwrap();
            triton_sys::TRITONSERVER_ErrorNew(error_code, error.as_ptr())
        };
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ResponseSend(self.ptr, send_flags, error)
        })?;
        Ok(())
    }
}

impl Drop for Response {
    fn drop(&mut self) {
        let error = unsafe {
            triton_sys::TRITONBACKEND_ResponseDelete(self.ptr)
        };
        assert!(error == ptr::null_mut());
    }
}
