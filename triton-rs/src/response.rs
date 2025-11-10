use crate::{check_err, Request, Error};
use libc::c_void;
use std::ffi::CString;
use std::ptr;
use std::slice;

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

    fn output(&mut self, name: &str, data_type: u32, shape: &[i64]) -> Result<Output, Error> {
        let name = CString::new(name)?;
        let mut output: *mut triton_sys::TRITONBACKEND_Output = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ResponseOutput(
                self.ptr,
                &mut output,
                name.as_ptr(),
                data_type,
                shape.as_ptr(),
                shape.len().try_into().unwrap(),
            )
        })?;

        Ok(Output::from_ptr(output))
    }

    pub fn add_output(&mut self, name: &str, data_type: u32, shape: &[i64], data: &[u8]) -> Result<(), Error> {
        let mut output = self.output(name, data_type, shape)?;
        output.set_data(&data)?;
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

struct Output {
    ptr: *mut triton_sys::TRITONBACKEND_Output,
}

impl Output {
    fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Output) -> Self {
        Self { ptr }
    }

    pub fn set_data(&mut self, data: &[u8]) -> Result<(), Error> {
        let mut buffer: *mut c_void = ptr::null_mut();
        let buffer_byte_size = data.len() as u64;
        let mut memory_type: triton_sys::TRITONSERVER_MemoryType = 0;
        let mut memory_type_id = 0;
        check_err(unsafe {
            triton_sys::TRITONBACKEND_OutputBuffer(
                self.ptr,
                &mut buffer,
                buffer_byte_size,
                &mut memory_type,
                &mut memory_type_id,
            )
        })?;

        let mem: &mut [u8] = unsafe {
            slice::from_raw_parts_mut(buffer as *mut u8, buffer_byte_size as usize)
        };

        mem.copy_from_slice(&data);

        Ok(())
    }
}
