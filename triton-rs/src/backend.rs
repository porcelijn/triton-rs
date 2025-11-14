use super::Error;
use super::Model;
use super::ModelInstance;

pub trait Backend {
    type ModelInstanceState;
    type ModelState;

    /// Initialize a backend. This function is optional, a backend is not
    /// required to implement it. This function is called once when a
    /// backend is loaded to allow the backend to initialize any state
    /// associated with the backend. A backend has a single state that is
    /// shared across all models that use the backend.
    ///
    /// Corresponds to TRITONBACKEND_Initialize.
    fn initialize() -> Result<(), Error> {
        Ok(())
    }

    /// Finalize for a backend. This function is optional, a backend is
    /// not required to implement it. This function is called once, just
    /// before the backend is unloaded. All state associated with the
    /// backend should be freed and any threads created for the backend
    /// should be exited/joined before returning from this function.
    /// Corresponds to TRITONBACKEND_Finalize.
    fn finalize() -> Result<(), Error> {
        Ok(())
    }

    /// Initialize for a model instance. This function is optional, a
    /// backend is not required to implement it. This function is called
    /// once when a model instance is created to allow the backend to
    /// initialize any state associated with the instance.
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceInitialize.
    fn model_instance_initialize(_model_instance: super::ModelInstanceImpl<Self::ModelInstanceState, Self::ModelState>) -> Result<(), Error> {
        Ok(())
    }

    /// Finalize for a model instance. This function is optional, a
    /// backend is not required to implement it. This function is called
    /// once for an instance, just before the corresponding model is
    /// unloaded from Triton. All state associated with the instance
    /// should be freed and any threads created for the instance should be
    /// exited/joined before returning from this function.
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceFinalize.
    fn model_instance_finalize(model_instance: super::ModelInstanceImpl<Self::ModelInstanceState, Self::ModelState>) -> Result<(), Error> {
        let _previous = model_instance.replace_state(None)?;
        Ok(())
    }

    fn model_initialize(_model: super::ModelImpl<Self::ModelState>) -> Result<(), Error> {
        Ok(())
    }

    fn model_finalize(model: super::ModelImpl<Self::ModelState>) -> Result<(), Error> {
        let _previous = model.replace_state(None)?;
        Ok(())
    }

    /// Execute a batch of one or more requests on a model instance. This
    /// function is required. Triton will not perform multiple
    /// simultaneous calls to this function for a given model 'instance';
    /// however, there may be simultaneous calls for different model
    /// instances (for the same or different models).
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceExecute.
    fn model_instance_execute(
        model_instance: super::ModelInstanceImpl<Self::ModelInstanceState, Self::ModelState>,
        requests: &[super::Request],
    ) -> Result<(), Error>;
}

#[macro_export]
macro_rules! call_checked {
    ($res:expr) => {
        match $res {
            Err(err) => triton_rs::to_TRITONSERVER_Error(err),
            Ok(ok) => ptr::null(),
        }
    };
}

#[macro_export]
macro_rules! declare_backend {
    ($class:ident) => {
        #[no_mangle]
        extern "C" fn TRITONBACKEND_Initialize(
            backend: *const triton_rs::sys::TRITONBACKEND_Backend,
        ) -> *const triton_rs::sys::TRITONSERVER_Error {
            triton_rs::call_checked!($class::initialize())
        }

        #[no_mangle]
        extern "C" fn TRITONBACKEND_Finalize(
            backend: *const triton_rs::sys::TRITONBACKEND_Backend,
        ) -> *const triton_rs::sys::TRITONSERVER_Error {
            triton_rs::call_checked!($class::finalize())
        }

        #[no_mangle]
        extern "C" fn TRITONBACKEND_ModelInitialize(
            model: *mut triton_rs::sys::TRITONBACKEND_Model,
        ) -> *const triton_rs::sys::TRITONSERVER_Error {
            let mut model = triton_rs::ModelImpl::from_ptr(model);
            triton_rs::call_checked!($class::model_initialize(model))
        }

        #[no_mangle]
        extern "C" fn TRITONBACKEND_ModelFinalize(
            model: *mut triton_rs::sys::TRITONBACKEND_Model,
        ) -> *const triton_rs::sys::TRITONSERVER_Error {
            let mut model = triton_rs::ModelImpl::from_ptr(model);
            triton_rs::call_checked!($class::model_finalize(model))
        }

        #[no_mangle]
        extern "C" fn TRITONBACKEND_ModelInstanceInitialize(
            instance: *mut triton_rs::sys::TRITONBACKEND_ModelInstance,
        ) -> *const triton_rs::sys::TRITONSERVER_Error {
            let mut instance = triton_rs::ModelInstanceImpl::from_ptr(instance);
            triton_rs::call_checked!($class::model_instance_initialize(instance))
        }

        #[no_mangle]
        extern "C" fn TRITONBACKEND_ModelInstanceFinalize(
            instance: *mut triton_rs::sys::TRITONBACKEND_ModelInstance,
        ) -> *const triton_rs::sys::TRITONSERVER_Error {
            let mut instance = triton_rs::ModelInstanceImpl::from_ptr(instance);
            triton_rs::call_checked!($class::model_instance_finalize(instance))
        }

        #[no_mangle]
        extern "C" fn TRITONBACKEND_ModelInstanceExecute(
            instance: *mut triton_rs::sys::TRITONBACKEND_ModelInstance,
            requests: *const *mut triton_rs::sys::TRITONBACKEND_Request,
            request_count: u32,
        ) -> *const triton_rs::sys::TRITONSERVER_Error {
                let instance = triton_rs::ModelInstanceImpl::from_ptr(instance);
                let requests = unsafe { slice::from_raw_parts(requests, request_count as usize) };
                let requests = requests
                    .iter()
                    .map(|req| triton_rs::Request::from_ptr(*req))
                    .collect::<Vec<triton_rs::Request>>();

            triton_rs::call_checked!($class::model_instance_execute(instance, &requests))
        }
    };
}
