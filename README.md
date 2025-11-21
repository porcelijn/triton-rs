# Rust bindings to the Triton Inference Server backend

This a fork of [xtuc/triton-rs](https://github.com/xtuc/triton-rs) that includes
BLS extensions made by [KenForever1](https://github.com/KenForever1/triton-rs) for
asynchronous sub-inferences (upstream).
It was further extended to support (stateful) *decoupled* `model_transaction_policy`,
using *response factory*.

## Triton Rust API

See [triton_rs documentation].

## Implementing a backend

```rust
use triton_rs::{Backend, Response};

struct ExampleBackend;

impl Backend for ExampleBackend {
    type ModelInstanceState = ();
    type ModelState = ();

    fn model_instance_execute(
        model_instance: triton_rs::ModelInstance,
        requests: &[triton_rs::Request],
    ) -> Result<(), triton_rs::Error> {

        for request in requests {
            let input = request.get_input("input")?;
            let data: &[f32] = input.slice()?;
            let shape = input.properties()?.shape;
            // or, alternatively with "ndarray" feature enabled:
            // let tensor = input.as_array::<f32, 2>()?;

            // Handle inference request here
            todo!();

            let response = Response::from_request(request)?;
            // or, alternatively:
            // let factory = ResponseFactory::from_request(request)?;
            // let response = Response::from_factory(factory)?;
            // request.release();
            response.add_output("output", &shape, data);
            // or: response.add_output("output", tensor);
            response.send();
        }

        Ok(())
    }
}

// Register the backend with Triton
triton_rs::declare_backend!(ExampleBackend);
```

See [example-backend] for full example.

[example-backend]: ./example-backend
[triton_rs documentation]: https://docs.rs/triton-rs

For more details, see NVIDIA's header files:
[tritonbackend.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonbackend.h) and
[tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).
