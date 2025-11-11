use crate::{check_err, Error, Server};
use libc::{c_char, size_t};
use std::ffi::{c_void, CStr};
use std::fs::File;
use std::io::prelude::*;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::ptr;

pub struct Model<S = ()> {
    ptr: *mut triton_sys::TRITONBACKEND_Model,
    _state: PhantomData<S>,
}

impl<S> Model<S> {

    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Model) -> Self {
        Self { ptr, _state: PhantomData }
    }

    pub fn server(&self) -> Result<Server, Error> {
        let mut server: *mut triton_sys::TRITONSERVER_Server = ptr::null_mut();
        check_err(unsafe { triton_sys::TRITONBACKEND_ModelServer(self.ptr, &mut server) })?;
        Ok(Server::from_ptr(server))
    }

    pub fn name(&self) -> Result<String, Error> {
        let mut model_name: *const c_char = ptr::null_mut();
        check_err(unsafe { triton_sys::TRITONBACKEND_ModelName(self.ptr, &mut model_name) })?;

        let c_str = unsafe { CStr::from_ptr(model_name) };
        Ok(c_str.to_string_lossy().to_string())
    }

    pub fn version(&self) -> Result<u64, Error> {
        let mut version = 0u64;
        check_err(unsafe { triton_sys::TRITONBACKEND_ModelVersion(self.ptr, &mut version) })?;
        Ok(version)
    }

    pub fn location(&self) -> Result<String, Error> {
        let mut artifact_type: triton_sys::TRITONBACKEND_ArtifactType = 0u32;
        let mut location: *const c_char = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelRepository(self.ptr, &mut artifact_type, &mut location)
        })?;

        let c_str = unsafe { CStr::from_ptr(location) };
        Ok(c_str.to_string_lossy().to_string())
    }

    pub fn path(&self, filename: &str) -> Result<PathBuf, Error> {
        Ok(PathBuf::from(format!(
            "{}/{}/{}",
            self.location()?,
            self.version()?,
            filename
        )))
    }

    pub fn load_file(&self, filename: &str) -> Result<Vec<u8>, Error> {
        let path = self.path(filename)?;
        let mut f = File::open(path)?;

        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;

        Ok(buffer)
    }

    pub fn model_config(&self) -> Result<String, Error> {
        // first step: call TRITONBACKEND_ModelConfig c func
        let config_version = 1;
        let mut msg : *mut triton_sys::TRITONSERVER_Message = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelConfig(self.ptr, config_version, &mut msg)
        })?;

        if msg.is_null() {
            return Err("Failed to get the message pointer".into());
        }

        // second step: call TRITONSERVER_MessageSerializeToJson c func, get json string from base and byte_size
        let mut base: *const libc::c_char = std::ptr::null();
        let mut byte_size: size_t = 0;
        check_err(unsafe {
            triton_sys::TRITONSERVER_MessageSerializeToJson(msg, &mut base, &mut byte_size)
        })?;

        if base.is_null() || byte_size == 0 {
            return Err("Failed to serialize the message to JSON".into());
        }

        // Convert C char array to Rust String
        let json_str = unsafe {
            std::ffi::CStr::from_ptr(base).to_string_lossy().into_owned()
        };

        Ok(json_str)
    }

    pub fn state(&self) -> Result<&mut S, Error> {
        let state = self.raw_state()?;
        if state.is_null() {
            return Err("Failed to get the state pointer".into());
        }

        let state: *mut S = state as _;
        let state: &mut S = unsafe { state.as_mut() }.unwrap();

        Ok(state)
    }

    pub fn replace_state(&self, new_state: Option<S>) -> Result<Option<S>, Error> {
        let old_state = self.raw_state()?;

        let new_state = match new_state {
            Some(new_state) => {
                let new_state = Box::new(new_state);
                Box::<S>::into_raw(new_state) as *mut c_void
            },
            None => ptr::null_mut()
        };
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelSetState(self.ptr, new_state)
        })?;

        if old_state.is_null() {
            return Ok(None);
        }

        let old_state = unsafe { Box::<S>::from_raw(old_state) };
        Ok(Some(*old_state))
    }

    fn raw_state(&self) -> Result<*mut S, Error> {
        let mut state : *mut c_void = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelState(self.ptr, &mut state)
        })?;
        Ok(state as *mut S)
    }
}
