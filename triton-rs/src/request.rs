use crate::{check_err, DataType, decode_string, Error, data_type::SupportedTypes};
use libc::c_void;
use ndarray::{ArrayView, IxDyn, IntoDimension};
use std::ffi::CStr;
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;
use std::slice;

pub struct Request {
    ptr: *mut triton_sys::TRITONBACKEND_Request,
}

impl Request {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Request) -> Self {
        Self { ptr }
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_sys::TRITONBACKEND_Request {
        self.ptr
    }

    pub fn get_input(&self, name: &str) -> Result<Input, Error> {
        let name = CString::new(name).expect("CString::new failed");

        let mut input: *mut triton_sys::TRITONBACKEND_Input = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_RequestInput(self.ptr, name.as_ptr(), &mut input)
        })?;

        Ok(Input::from_ptr(input))
    }

    pub fn get_request_id(&self
    ) -> Result<String, Error> {
        let mut id_ptr: *const c_char = std::ptr::null();

        check_err(unsafe {
            triton_sys::TRITONBACKEND_RequestId(self.ptr, &mut id_ptr)
        })?;

        unsafe {
            Ok(CStr::from_ptr(id_ptr).to_string_lossy().into_owned())
        }
    }

    pub fn get_correlation_id(&self) -> Result<u64, Error> {
        let mut id: u64 = 0;
        check_err(unsafe {
            triton_sys::TRITONBACKEND_RequestCorrelationId(self.ptr, &mut id)
        })?;
        Ok(id)
    }

    pub fn get_flags(&self) -> Result<RequestFlags, Error> {
        let mut flags: u32 = 0;
        check_err(unsafe {
            triton_sys::TRITONBACKEND_RequestFlags(self.ptr, &mut flags)
        })?;
        Ok(RequestFlags::from(flags))
    }
}

pub struct Input {
    ptr: *mut triton_sys::TRITONBACKEND_Input,
}
impl Input {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Input) -> Self {
        Self { ptr }
    }

    pub fn slice<T>(&self) -> Result<&[T], Error> {
        let mut buffer: *const c_void = ptr::null_mut();
        let index = 0;
        let mut memory_type: triton_sys::TRITONSERVER_MemoryType = 0;
        let mut memory_type_id = 0;
        let mut buffer_byte_size = 0;
        check_err(unsafe {
            triton_sys::TRITONBACKEND_InputBuffer(
                self.ptr,
                index,
                &mut buffer,
                &mut buffer_byte_size,
                &mut memory_type,
                &mut memory_type_id,
            )
        })?;

        println!("buffer: {:?}, byte_size: {:?}", buffer, buffer_byte_size);
        println!("memory_type: {:?}, memory_type_id: {:?}", memory_type, memory_type_id);

        let element_len = buffer_byte_size as usize / std::mem::size_of::<T>();

        let mem: &[T] =
            unsafe { slice::from_raw_parts(buffer as *mut T, element_len) };
        Ok(mem)
    }

    pub fn as_string(&self) -> Result<String, Error> {
        let properties = self.properties()?;
        if properties.datatype != DataType::BYTES {
            return Err(format!("DataType does not match String {properties:?}").into());
        }
        let buffer = self.slice::<u8>()?;
        let strings = decode_string(buffer)?;
        Ok(strings.first().unwrap().clone())
    }

    pub fn as_u64(&self) -> Result<u64, Error> {
        let properties = self.properties()?;
        if properties.datatype != DataType::UINT64 {
            return Err(format!("DataType does not match u64 {properties:?}").into());
        }
        if properties.byte_size < 8 {
            return Err("Buffer too small".into())
        }
        let buffer = self.slice::<u8>()?;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(buffer);
        Ok(u64::from_le_bytes(bytes))
    }

    pub fn as_array<T, const N: usize>(&self) -> Result<ArrayView<T, IxDyn>, Error>
            where T: SupportedTypes {
        let properties = self.properties()?;
        assert!(<T as SupportedTypes>::of() == properties.datatype);
        assert!(N == properties.shape.len());
        const { assert!( N < 6) };
        let shape: Vec<usize> = properties.shape.iter().map(|&x| x as usize).collect();
        let shape = shape.into_dimension();
        let data = self.slice::<T>()?;
        let array: ArrayView<T, IxDyn> = ArrayView::from_shape(shape, data)?;
        Ok(array)
    }

    pub fn properties(&self) -> Result<InputProperties, Error> {
        let mut name = ptr::null();
        let mut datatype = 0u32;
        let mut shape = ptr::null();
        let mut dims_count = 0u32;
        let mut byte_size = 0u64;
        let mut buffer_count = 0u32;

        check_err(unsafe {
            triton_sys::TRITONBACKEND_InputProperties(
                self.ptr,
                &mut name,
                &mut datatype,
                &mut shape,
                &mut dims_count,
                &mut byte_size,
                &mut buffer_count,
            )
        })?;

        let name: &CStr = unsafe { CStr::from_ptr(name) };
        let name = name.to_string_lossy().to_string();
        let datatype: DataType = datatype.into();
        let dims_count = dims_count as usize;
        let shape = unsafe { slice::from_raw_parts(shape, dims_count) };
        let shape = Vec::from(shape);

        Ok(InputProperties {
            name,
            datatype,
            shape,
            byte_size,
            buffer_count,
        })
    }
}


#[derive(Debug)]
pub struct InputProperties {
    pub name: String,
    pub datatype: DataType,
    pub shape: Vec<i64>,
    pub byte_size: u64,
    pub buffer_count: u32,
}

#[derive(Debug, PartialEq)]
pub struct RequestFlags(u32);

impl RequestFlags {
    pub fn is_start(&self) -> bool {
        self.0 & triton_sys::tritonserver_requestflag_enum_TRITONSERVER_REQUEST_FLAG_SEQUENCE_START != 0
    }

    pub fn is_end(&self) -> bool {
        self.0 & triton_sys::tritonserver_requestflag_enum_TRITONSERVER_REQUEST_FLAG_SEQUENCE_END != 0
    }

}

impl From<u32> for RequestFlags {
    fn from(v: u32) -> RequestFlags {
        RequestFlags(v)
    }
}
