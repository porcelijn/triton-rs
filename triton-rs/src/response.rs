use crate::{check_err, Error};
use crate::{DataType, Request};
use crate::data_type::SupportedTypes;
use libc::c_void;
use std::ffi::CString;
use std::mem;
use std::ptr;
use std::slice;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum ResponseFlags {
    NONE = 0,
    FINAL = triton_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL,
}

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

    pub fn send(self, flags: ResponseFlags) -> Result<(), Error> {
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ResponseSend(self.ptr, flags as u32, ptr::null_mut())
        })?;
        mem::forget(self); // prevent Drop because, send frees Response
        Ok(())
    }

    pub fn send_error(self, flags: ResponseFlags, error: Error) -> Result<(), Error> {
        let error = unsafe {
            let error_code = triton_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_UNSUPPORTED;
            let error = CString::new(error.to_string()).unwrap();
            triton_sys::TRITONSERVER_ErrorNew(error_code, error.as_ptr())
        };
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ResponseSend(self.ptr, flags as u32, error)
        })?;
        mem::forget(self); // prevent Drop because, send frees Response
        Ok(())
    }

    fn output(&mut self, name: &str, data_type: DataType, shape: &[i64]) -> Result<Output, Error> {
        let name = CString::new(name)?;
        let mut output: *mut triton_sys::TRITONBACKEND_Output = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ResponseOutput(
                self.ptr,
                &mut output,
                name.as_ptr(),
                data_type as u32,
                shape.as_ptr(),
                shape.len().try_into().unwrap(),
            )
        })?;

        Ok(Output::from_ptr(output))
    }

    pub fn add_output<T>(&mut self, name: &str, shape: &[i64], data: &[T]) -> Result<(), Error>
    where T: Copy + SupportedTypes {
        let data_type = <T as SupportedTypes>::of();
        assert_eq!(data_type.byte_size() as usize, std::mem::size_of::<T>());
        let mut output = self.output(name, data_type, shape)?;
        output.set_data(data)?;
        Ok(())
    }
}

impl Drop for Response {
    fn drop(&mut self) {
        let error = unsafe {
            triton_sys::TRITONBACKEND_ResponseDelete(self.ptr)
        };
        assert!(error.is_null());
    }
}

struct Output {
    ptr: *mut triton_sys::TRITONBACKEND_Output,
}

impl Output {
    fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Output) -> Self {
        Self { ptr }
    }

    pub fn set_data<T: Copy>(&mut self, data: &[T]) -> Result<(), Error> {
        let mut buffer: *mut c_void = ptr::null_mut();
        let element_len = data.len();
        let buffer_byte_size = std::mem::size_of_val(data) as u64;
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

        let mem: &mut [T] = unsafe {
            slice::from_raw_parts_mut(buffer as *mut T, element_len)
        };

        mem.copy_from_slice(data);

        Ok(())
    }
}
