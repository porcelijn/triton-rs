use std::ffi::{CStr, CString};

#[derive(Debug,Clone,Copy,PartialEq)]
#[repr(u32)]
pub enum DataType {
    INVALID = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INVALID,
    BOOL = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL,
    UINT8 = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8,
    UINT16 = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT16,
    UINT32 = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT32,
    UINT64 = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64,
    INT8 = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT8,
    INT16 = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT16,
    INT32 = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32,
    INT64 = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT64,
    FP16 = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP16,
    FP32 = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32,
    FP64 = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP64,
    BYTES = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES,
    BF16 = triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BF16,
}

impl DataType {
    pub fn byte_size(&self) -> u32 {
        unsafe { triton_sys::TRITONSERVER_DataTypeByteSize(self.into()) }
    }
}

impl From<u32> for DataType {
    fn from(v: u32) -> DataType {
        match v {
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL => Self::BOOL,
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8 => Self::UINT8,
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT16 => Self::UINT16,
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT32 => Self::UINT32,
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64 => Self::UINT64,
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT8 => Self::INT8,
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT16 => Self::INT16,
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32 => Self::INT32,
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT64 => Self::INT64,
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP16 => Self::FP16,
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32 => Self::FP32,
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP64 => Self::FP64,
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES => Self::BYTES,
            triton_sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BF16 => Self::BF16,
            _ => Self::INVALID,
        }
    }
}

impl From<&str> for DataType {
    fn from(v: &str) -> DataType {
        let v = CString::new(v).expect("malformed DataType str");
        let data_type = unsafe {
            triton_sys::TRITONSERVER_StringToDataType(v.as_ptr())
        };
        data_type.into()
    }
}

impl From<&DataType> for u32 { fn from(v: &DataType) -> u32 { *v as u32 } }
impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cstr = unsafe {
            // The returned string is not owned by the caller and so should not be
            // modified or freed.
            let char_ptr = triton_sys::TRITONSERVER_DataTypeString(self.into());
            CStr::from_ptr(char_ptr)
        };
        f.write_str(cstr.to_str().unwrap())
    }
}

pub trait SupportedTypes { fn of() -> DataType { DataType::INVALID } }

impl SupportedTypes for bool { fn of() -> DataType { DataType::BOOL } }
//impl SupportedTypes for u8 { fn of() -> DataType { DataType::UINT8 } }
// ...
impl SupportedTypes for u64 { fn of() -> DataType { DataType::UINT64 } }
impl SupportedTypes for i64 { fn of() -> DataType { DataType::INT64 } }
impl SupportedTypes for f32 { fn of() -> DataType { DataType::FP32 } }
// ...
//impl SupportedTypes for &str { fn of() -> DataType { DataType::BYTES } }
impl SupportedTypes for u8 { fn of() -> DataType { DataType::BYTES } } // fixme
