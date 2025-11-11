use crate::{check_err, Error};
use crate::model::Model;
use std::{ffi::c_void, marker::PhantomData, ptr};

pub struct ModelInstance<S = ()> {
    ptr: *mut triton_sys::TRITONBACKEND_ModelInstance,
    _state: PhantomData<S>,
}

impl<S> ModelInstance<S> {

    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_ModelInstance) -> Self {
        Self { ptr, _state: PhantomData }
    }

    pub fn model<T>(&self) -> Result<Model<T>, Error> {
        let mut model : *mut triton_sys::TRITONBACKEND_Model = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelInstanceModel(self.ptr, &mut model)
        })?;

        if model.is_null() {
            return Err("Failed to get the model pointer".into());
        }

        Ok(Model::from_ptr(model))
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
            triton_sys::TRITONBACKEND_ModelInstanceSetState(self.ptr, new_state)
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
            triton_sys::TRITONBACKEND_ModelInstanceState(self.ptr, &mut state)
        })?;
        Ok(state as *mut S)
    }
}

