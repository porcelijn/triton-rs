use crate::{check_err, Error};
use crate::model::ModelImpl;
use std::{ffi::c_void, marker::PhantomData, ptr};

pub trait ModelInstance {
    type S;
    fn state(&self) -> Result<&mut Self::S, Error>;
    fn replace_state(&self, new_state: Option<Self::S>) -> Result<Option<Self::S>, Error>;
}

pub struct ModelInstanceImpl<ModelInstanceState, ModelState> {
    ptr: *mut triton_sys::TRITONBACKEND_ModelInstance,
    _model_instance_state: PhantomData<ModelInstanceState>,
    _model_state: PhantomData<ModelState>,
}

impl<ModelInstanceState, ModelState> ModelInstance
        for ModelInstanceImpl<ModelInstanceState, ModelState> {
    type S = ModelInstanceState;

    fn state(&self) -> Result<&mut Self::S, Error> {
        let state = self.raw_state()?;
        if state.is_null() {
            return Err("Failed to get the state pointer".into());
        }

        let state: *mut Self::S = state as _;
        let state: &mut Self::S = unsafe { state.as_mut() }.unwrap();

        Ok(state)
    }

    fn replace_state(&self, new_state: Option<Self::S>)
            -> Result<Option<Self::S>, Error> {
        let old_state = self.raw_state()?;

        let new_state = match new_state {
            Some(new_state) => {
                let new_state = Box::new(new_state);
                Box::<Self::S>::into_raw(new_state) as *mut c_void
            },
            None => ptr::null_mut()
        };
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelInstanceSetState(self.ptr, new_state)
        })?;

        if old_state.is_null() {
            return Ok(None);
        }

        let old_state = unsafe { Box::<Self::S>::from_raw(old_state) };
        Ok(Some(*old_state))
    }
}

impl<ModelInstanceState, ModelState> ModelInstanceImpl<ModelInstanceState, ModelState> {

    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_ModelInstance) -> Self {
        Self { ptr, _model_instance_state: PhantomData, _model_state: PhantomData }
    }

    pub fn model(&self) -> Result<ModelImpl<ModelState>, Error> {
        let mut model : *mut triton_sys::TRITONBACKEND_Model = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelInstanceModel(self.ptr, &mut model)
        })?;

        if model.is_null() {
            return Err("Failed to get the model pointer".into());
        }

        Ok(ModelImpl::from_ptr(model))
    }

    fn raw_state(&self) -> Result<*mut ModelInstanceState, Error> {
        let mut state : *mut c_void = ptr::null_mut();
        check_err(unsafe {
            triton_sys::TRITONBACKEND_ModelInstanceState(self.ptr, &mut state)
        })?;
        Ok(state as *mut ModelInstanceState)
    }
}

