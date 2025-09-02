use crate::{check_err, Error};
use crate::model::Model;
use std::ptr;

pub struct ModelInstance {
    ptr: *mut triton_sys::TRITONBACKEND_ModelInstance,
}

impl ModelInstance {
    
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_ModelInstance) -> Self {
        Self { ptr }
    }

    pub fn model(&self) -> Result<Model, Error> {
        let mut model : *mut triton_sys::TRITONBACKEND_Model = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelInstanceModel(self.ptr, &mut model)
        })?;
    
        if model.is_null() {
            return Err("Failed to get the model pointer".into());
        }

        Ok(Model::from_ptr(model))
    }
}
