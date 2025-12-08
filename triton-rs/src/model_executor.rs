//! A Rust-style Triton Server model executor with FFI bindings

use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;
// use async_trait::async_trait;
use crate::{check_err, Error, InferenceRequest, InferenceResponse, Server};
#[cfg(not(test))]
use crate::to_TRITONSERVER_Error;
use tokio::sync::oneshot;

// // #[async_trait]
// pub trait ModelExecutor {
//     fn load_model(&mut self, model_path: &str) -> Result<(), Error>;
//     async fn execute(
//         &self,
//         request: *mut triton_sys::TRITONSERVER_InferenceRequest,
//     ) -> Result<*mut triton_sys::TRITONSERVER_InferenceResponse, Error>;
// }

pub struct ModelExecutor {
    pub(crate) server: Server,
    pub(crate) model_name: CString,
    pub(crate) model_version: i64,
    allocator: *mut triton_sys::TRITONSERVER_ResponseAllocator,
}

impl Drop for ModelExecutor {
    fn drop(&mut self) {
        unsafe {
            // Clean up allocator when executor is dropped
            if !self.allocator.is_null() {
                drop(Box::from_raw(self.allocator));
            }
        }
    }
}

impl std::fmt::Debug for ModelExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{:?}:{}", self.model_name, self.model_version)
    }
}

impl ModelExecutor {
    pub fn new(
        server: Server,
        model_name: &str,
        model_version: i64,
    ) -> Result<Self, Error> {
        let mut allocator: *mut triton_sys::TRITONSERVER_ResponseAllocator = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONSERVER_ResponseAllocatorNew(
                &mut allocator,
                Some(response_alloc),
                Some(response_release),
                None,
        )})?;
        let Ok(model_name) = CString::new(model_name) else {
            return Err("Failed to allcate CString".into())
        };
        Ok(Self { server, model_name, model_version, allocator })
    }
}

// #[async_trait]
impl ModelExecutor {
    pub async fn execute(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, Error> {
        let (tx, rx) = oneshot::channel();
        let tx_ptr = Box::into_raw(Box::new(tx));

        check_err(unsafe {
            // Set response callback
            triton_sys::TRITONSERVER_InferenceRequestSetResponseCallback(
                request.as_ptr(),
                self.allocator,
                ptr::null_mut(),
                Some(infer_response_complete),
                tx_ptr as *mut c_void,
        )})?;

        self.server.infer_async(request)?;

        // Wait for response
        rx.await.map(InferenceResponse::from_ptr)?
    }
}

// FFI callback functions
extern "C" fn infer_response_complete(
    response: *mut triton_sys::TRITONSERVER_InferenceResponse,
    _flags: u32,
    userp: *mut c_void,
) {
    if !response.is_null() {
        let tx = unsafe {
            Box::from_raw(
                userp as *mut oneshot::Sender<*mut triton_sys::TRITONSERVER_InferenceResponse>,
            )
        };
        let _ = tx.send(response);
    }
}

extern "C" fn response_alloc(
    _allocator: *mut triton_sys::TRITONSERVER_ResponseAllocator,
    tensor_name: *const libc::c_char,
    byte_size: libc::size_t,
    preferred_memory_type: triton_sys::TRITONSERVER_MemoryType,
    preferred_memory_type_id: i64,
    _userp: *mut c_void,
    buffer: *mut *mut c_void,
    buffer_userp: *mut *mut c_void,
    actual_memory_type: *mut triton_sys::TRITONSERVER_MemoryType,
    actual_memory_type_id: *mut i64,
) -> *mut triton_sys::TRITONSERVER_Error {
    unsafe {
        *actual_memory_type = preferred_memory_type;
        *actual_memory_type_id = preferred_memory_type_id;
    }
    if byte_size == 0 {
        unsafe {
            *buffer = ptr::null_mut();
            *buffer_userp = ptr::null_mut();
        }
        return ptr::null_mut();
    }

    let allocated_ptr = match preferred_memory_type {
        // Implement memory allocation based on type (CPU/GPU)
        triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU => unsafe {
            libc::malloc(byte_size)
        },
        triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU_PINNED =>
          unsafe {
            *actual_memory_type =
                triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
            *actual_memory_type_id = 0;
            libc::malloc(byte_size)
        },
        triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_GPU => {
            return to_TRITONSERVER_Error("GPU memory not implemented".into());
        },
        _ => {
            return to_TRITONSERVER_Error("Invalid memory type requested".into());
        },
    };

    if allocated_ptr.is_null() {
        return to_TRITONSERVER_Error("Failed to allocate memory for response".into());
    }

    // Store tensor name on Rust heap
    let tensor_name = unsafe { CString::from_vec_unchecked(
        std::ffi::CStr::from_ptr(tensor_name).to_bytes().to_vec()) };
    let tensor_name = Box::into_raw(Box::new(tensor_name));
    unsafe {
        *buffer = allocated_ptr;
        *buffer_userp = tensor_name as *mut c_void;
    }

    ptr::null_mut() // Success
}

extern "C" fn response_release(
    _allocator: *mut triton_sys::TRITONSERVER_ResponseAllocator,
    buffer: *mut c_void,
    buffer_userp: *mut c_void,
    byte_size: libc::size_t,
    memory_type: triton_sys::TRITONSERVER_MemoryType,
    _memory_type_id: i64,
) -> *mut triton_sys::TRITONSERVER_Error {
    if !buffer.is_null() {
        match memory_type {
            triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU =>
                unsafe {
                    // Detect use after free errors
                    libc::free(libc::memset(buffer, 123, byte_size));
//                  libc::free(buffer);
                },
            _ => unimplemented!("Memory type not implemented"),
        }
    }

    if !buffer_userp.is_null() {
        let tensor_name = unsafe { Box::from_raw(buffer_userp as *mut CString) };
        drop(tensor_name);
    }

    ptr::null_mut() // Success
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;
//  use tokio::test;

    #[test]
    fn test_allocation_cpu() {
        let allocator = ptr::null_mut();
        let tensor_name = c"foo";
        let byte_size: libc::size_t = 123;
        let mut memory_type =
            triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
        let mut memory_type_id: i64 = 321;
        let mut buffer = ptr::null_mut();
        let mut buffer_userp = ptr::null_mut();

        let r = response_alloc(allocator, tensor_name.as_ptr(), byte_size,
                               memory_type, memory_type_id, ptr::null_mut(),
                               &mut buffer, &mut buffer_userp,
                               &mut memory_type, &mut memory_type_id);
        assert!(r.is_null());
        assert!(!buffer.is_null());
        assert!(!buffer_userp.is_null());
        assert_eq!(memory_type, 0);
        assert_eq!(memory_type_id, 321);

        let slice: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(buffer as *mut u8, 123)
        };
        slice[122] = 0xff; // poke in buffer
        let tensor_name = unsafe { Box::from_raw(buffer_userp as *mut CString) };
        assert_eq!(tensor_name.as_c_str(), c"foo"); // peek
        Box::leak(tensor_name); // avoid double free

        let r = response_release(allocator, buffer, buffer_userp, byte_size,
                                 memory_type, memory_type_id);
        assert!(r.is_null());
    }

    #[test]
    fn test_allocation_cpu_pinned() {
        let allocator = ptr::null_mut();
        let tensor_name = c"bar";
        let byte_size: libc::size_t = 123;
        let mut memory_type =
            triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU_PINNED;
        let mut memory_type_id: i64 = 321;
        let mut buffer = ptr::null_mut();
        let mut buffer_userp = ptr::null_mut();

        let r = response_alloc(allocator, tensor_name.as_ptr(), byte_size,
                               memory_type, memory_type_id, ptr::null_mut(),
                               &mut buffer, &mut buffer_userp,
                               &mut memory_type, &mut memory_type_id);
        assert!(r.is_null());
        assert!(!buffer.is_null());
        assert!(!buffer_userp.is_null());
        assert_eq!(memory_type, 0);    // <-- changed PINNED -> Regular
        assert_eq!(memory_type_id, 0); // changed 321 -> 0

        let slice: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(buffer as *mut u8, 123)
        };
        slice[122] = 0xff; // poke in buffer
        let tensor_name = unsafe { Box::from_raw(buffer_userp as *mut CString) };
        assert_eq!(tensor_name.as_c_str(), c"bar"); // peek
        Box::leak(tensor_name); // avoid double free

        let r = response_release(allocator, buffer, buffer_userp, byte_size,
                                 memory_type, memory_type_id);
        assert!(r.is_null());
    }

    #[test]
    fn test_allocation_gpu() {
        let allocator = ptr::null_mut();
        let tensor_name = c"baz";
        let byte_size: libc::size_t = 123;
        let mut memory_type =
            triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_GPU;
        let mut memory_type_id: i64 = 321;
        let mut buffer = ptr::null_mut();
        let mut buffer_userp = ptr::null_mut();

        let r = response_alloc(allocator, tensor_name.as_ptr(), byte_size,
                               memory_type, memory_type_id, ptr::null_mut(),
                               &mut buffer, &mut buffer_userp,
                               &mut memory_type, &mut memory_type_id);
        assert!(!r.is_null());
    }

//    #[test]
//    async fn test_executor_creation() {
//        // Create with null server pointer for test
//        let server = Server::from_ptr(ptr::null_mut());
//        let executor = ModelExecutor::new(&server);
//        assert!(executor.is_ok());
//
//        let executor = executor.unwrap();
//    }
}

// TEST STUB
#[cfg(test)]
#[allow(non_snake_case)]
pub fn to_TRITONSERVER_Error(_err: Error) -> *mut triton_sys::TRITONSERVER_Error {
    std::ptr::NonNull::<triton_sys::TRITONSERVER_Error>::dangling().as_ptr()
}

