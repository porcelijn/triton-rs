mod backend;
mod data_type;
mod inference_request;
mod inference_response;
mod model;
mod model_executor;
mod model_instance;
mod request;
mod response;
mod server;

pub use backend::Backend;
pub use data_type::DataType;
pub use inference_request::InferenceRequest;
pub use inference_response::InferenceResponse;
pub use model_executor::ModelExecutor;
pub use model_instance::ModelInstance;
pub use model_instance::ModelInstanceImpl;
pub use model::Model;
pub use model::ModelImpl;
pub use request::Request;
pub use request::RequestFlags;
pub use request::RequestReleaseFlags;
pub use response::Response;
pub use response::ResponseFactory;
pub use response::ResponseFlags;
pub use server::Server;
pub use triton_sys as sys;

pub type Error = Box<dyn std::error::Error>;

#[allow(non_snake_case)]
pub fn to_TRITONSERVER_Error(err: Error) -> *const triton_sys::TRITONSERVER_Error {
    let err = std::ffi::CString::new(err.to_string()).expect("CString::new failed");
    unsafe {
        triton_sys::TRITONSERVER_ErrorNew(
            triton_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
            err.as_ptr(),
        )
    }
}

pub(crate) fn check_err(err: *mut triton_sys::TRITONSERVER_Error) -> Result<(), Error> {
    if !err.is_null() {
        let error = into_error(err);
        eprintln!("check err: {error}");
        Err(error)
    } else {
        Ok(())
    }
}

pub fn decode_string(data: &[u8]) -> Result<Vec<String>, Error> {
    let mut strings = vec![];
    let mut i = 0;

    while i < data.len() {
        // l = struct.unpack_from("<I", val_buf, offset)[0]
        // offset += 4
        let wide = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        i += 4;

        // sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
        // offset += l
        // strs.append(sb)
        let bytes = &data[i..i + wide];
        let string = String::from_utf8_lossy(bytes).to_string();
        i += wide;

        strings.push(string);
    }

    Ok(strings)
}

pub fn encode_string(value: &str) -> Vec<u8> {
    let mut bytes = vec![];

    let value: Vec<u8> = value.bytes().collect();

    // l = struct.unpack_from("<I", val_buf, offset)[0]
    // offset += 4
    let len = (value.len() as u32).to_le_bytes();
    bytes.extend_from_slice(&len);

    // sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
    // offset += l
    // strs.append(sb)
    bytes.extend_from_slice(&value);

    bytes
}

fn into_error(err: *mut triton_sys::TRITONSERVER_Error) -> Error {
    // extract code and null terminated description from TRITONSERVER_Error
    let code = unsafe { triton_sys::TRITONSERVER_ErrorCode(err) };
    let c_msg_ptr = unsafe { triton_sys::TRITONSERVER_ErrorCodeString(err) };
    let msg = &unsafe { std::ffi::CStr::from_ptr(c_msg_ptr) };
    // Clean up heap-allocated TRITONSERVER_Error object (created by callee)
    unsafe {
        triton_sys::TRITONSERVER_ErrorDelete(err);
    }
    format!("TRITONSERVER_Error: {msg:?} ({code})").into()
}

