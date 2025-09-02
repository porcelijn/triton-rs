# Rust bindings to the Triton Inference Server 

## Triton Rust API

See [triton_rs documentation].

## Implementing a backend

```rust
use triton_rs::Backend;

struct ExampleBackend;

impl Backend for ExampleBackend {
    fn model_instance_execute(
        model_instance: triton_rs::ModelInstance,
        requests: &[triton_rs::Request],
    ) -> Result<(), triton_rs::Error> {

        for request in requests {
            // Handle inference request here
            todo!();
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
