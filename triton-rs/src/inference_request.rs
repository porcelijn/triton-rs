use crate::{check_err, DataType, Error, ModelExecutor};
use std::ffi::CString;

#[cfg(feature = "ndarray")] use ndarray::{Array, IxDyn};

pub struct InferenceRequest {
    ptr: *mut triton_sys::TRITONSERVER_InferenceRequest,
    #[cfg(feature = "ndarray")]
    datas: Vec<Vec<u8>>,
}

impl InferenceRequest {
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
        Ok(Self { ptr: request, datas: Vec::with_capacity(10) })
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
        let data_memory_id: i64 = 0;
        check_err(unsafe {
            triton_sys::TRITONSERVER_InferenceRequestAppendInputData(
                self.ptr,
                cstr_name.as_ptr(),
                c_data_ptr,
                data.len(),
                triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                data_memory_id,
            )
        })?;
        Ok(())
    }

    #[cfg(feature = "ndarray")]
    pub fn add_input_array<T>(&mut self, name: &str, array: Array<T, IxDyn>,
    ) -> Result<(), Box<dyn std::error::Error>>
    where T: Copy + crate::data_type::SupportedTypes {
        let shape: Vec<i64> = array.shape().iter().map(|x| *x as i64).collect();
        let data_type = <T as crate::data_type::SupportedTypes>::of();
        assert_eq!(data_type.byte_size() as usize, std::mem::size_of::<T>());
        self.add_input(name, data_type, &shape)?;
        let is_empty = array.is_empty();
        let (vec, Some(offset)) = array.into_raw_vec_and_offset() else {
            assert!(is_empty);
            return Ok(()); // no data needs be sent
        };
        assert_eq!(offset, 0);
        let vec: Vec<u8> = get_raw_bytes(vec);
        self.datas.push(vec);
        let slice: &[u8] = self.datas.last().unwrap();
        self.set_input_data(name, slice)
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
    if !request.is_null() {
        let result = check_err(unsafe {
            triton_sys::TRITONSERVER_InferenceRequestDelete(request)
        });
        if let Err(error) = result {
            println!("Failed to delete InferenceRequest: {error:?}");
        }
    }
}

#[cfg(feature="ndarray")]
fn get_raw_bytes<T>(mut vec: Vec<T>) -> Vec<u8> {
    let ratio = std::mem::size_of::<T>() / std::mem::size_of::<u8>();
    let length = vec.len() * ratio;
    let capacity = vec.capacity() * ratio;
    let data = vec.as_mut_ptr() as *mut u8;
    std::mem::forget(vec); // don't run destructor
    let vec: Vec<u8> = unsafe { Vec::from_raw_parts(data, length, capacity) };
    vec
}

#[test]
#[cfg(feature="ndarray")]
fn test_get_raw_bytes() {
    let data: Vec<u8> = vec![1, 2, 3, 4];
    assert_eq!(data, get_raw_bytes(data.clone()));

    let data: Vec<i16> = vec![-1, 0, 100];
    #[cfg(target_endian="little")]
    assert_eq!(vec![255, 255, 0, 0, 100, 0], get_raw_bytes(data));
    #[cfg(target_endian="big")]
    assert_eq!(vec![255, 255, 0, 0, 0, 100], get_raw_bytes(data));

    let data: Vec<f32> = vec![0.0, 1.234, f32::NAN];
    #[cfg(target_endian="little")]
    assert_eq!(vec![0,0,0,0, 182,243,157,63, 0,0,192,127], get_raw_bytes(data));
    #[cfg(target_endian="big")]
    assert_eq!(vec![0,0,0,0, 63,157,243,182, 127,192,0,0], get_raw_bytes(data));
}
