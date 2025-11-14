//! https://github.com/triton-inference-server/backend/blob/main/README.md#triton-backend-api

use futures::executor::block_on;
use std::ptr;
use std::slice;
use triton_rs::Backend;
use triton_rs::DataType;
use triton_rs::Model;
use triton_rs::ModelInstance;
use triton_rs::Response;
use triton_rs::ResponseFlags;

#[derive(Debug, Default)]
struct InstanceState(usize);

impl InstanceState {
    fn change(&mut self) {
         self.0 += 1; // do stuff
    }
}

#[derive(Debug, Default)]
struct ModelState;

impl ModelState {
    fn read_setting(&self) -> &'static str { "foo" }
}

struct ExampleBackend;

impl Backend for ExampleBackend {
    type ModelInstanceState = InstanceState;
    type ModelState = ModelState;

    fn model_initialize(
            model: triton_rs::ModelImpl<ModelState>
    ) -> Result<(), triton_rs::Error> {
        let previous = model.replace_state(Some(ModelState::default()))?;
        assert!(previous.is_none());
        Ok(())
    }

    fn model_instance_initialize(
            model_instance: triton_rs::ModelInstanceImpl<InstanceState, ModelState>
    ) -> Result<(), triton_rs::Error> {
        let previous = model_instance.replace_state(Some(InstanceState::default()))?;
        assert!(previous.is_none());
        Ok(())
    }

    fn model_instance_execute(
        model_instance: triton_rs::ModelInstanceImpl<InstanceState, ModelState>,
        requests: &[triton_rs::Request],
    ) -> Result<(), triton_rs::Error> {
        let state = model_instance.state()?;
        state.change();
        println!("[EXAMPLE] model_instance_execute ({state:?}");

        let model: triton_rs::ModelImpl<ModelState> = model_instance.model()?;

        println!("[EXAMPLE] model config: {:?}", model.model_config()?);
        let model_setting = model.state()?.read_setting();

        println!(
            "[EXAMPLE] request for model {} {} {} {model_setting}",
            model.name()?,
            model.version()?,
            model.location()?
        );

        for request in requests {
            let prompt = request.get_input("prompt")?;
            let floats = prompt.slice::<f32>()?;
            println!("[EXAMPLE] prompt as f32: {}, len={}", floats[0], floats.len());
            let array =  prompt.as_array::<f32, 2>()?;
            println!("[EXAMPLE] prompt as ndarray: {array}");
            let prompt = prompt.as_string()?;
            println!("[EXAMPLE] prompt as_string: {prompt}");

            // model_excutor
            let request_id = request.get_request_id()?;
            println!("[EXAMPLE] request_id: {}", request_id);
            let correlation_id = request.get_correlation_id()?;
            println!("[EXAMPLE] correlation_id: {}", correlation_id);
            let model_name = "test";
            let model_version = 1;
            let input1_name = "prompt";
            let output1_name = "output";

            let server = model.server()?;
            let executor = triton_rs::TritonModelExecuter::new(&server)?;

            let inference_request =
                triton_rs::InferenceRequest::new(&server, model_name, model_version)?;

            inference_request.set_request_id(request_id.as_str())?;
            inference_request.set_correlation_id(correlation_id)?;
            inference_request.set_release_callback()?;

            println!("[EXAMPLE] set request id and correlation id finish");

            inference_request.add_input(input1_name, DataType::BYTES, &[1],)?;
            println!("add input finish");

            let input1_data = vec![b"test_bls_infer_req".as_ptr() as u8; 18];
            inference_request.set_input_data(input1_name, &input1_data)?;

            inference_request.add_output(output1_name)?;

            println!("set input data finish");

            let infer_response = block_on(executor.execute(&inference_request))?;

            // executor.execute(&inference_request).await.map_err(|e| {
            //     // Custom error handling logic
            //     triton_rs::ModelExecuterError::ExecutionError(e)
            // })?;

            println!(
                "[EXAMPLE] inference_response output : {:?}",
                infer_response.get_output_count()
            );

            for out_idx in 0..infer_response.get_output_count().unwrap() {
                let output = infer_response.get_output_data(out_idx).unwrap();
                println!("[EXAMPLE] output: {:?}", out_idx);
                output.print_info();
            }

            // let mut response = Response::from_request(request)?;
            let factory = &triton_rs::ResponseFactory::from_request(request)?;
            let mut response = Response::from_factory(factory)?;


            response.add_output(output1_name, array)?;
            response.send(ResponseFlags::NONE, None)?;
            // ... send more responses, then:
            factory.send_flags(ResponseFlags::FINAL)?;
        }

        Ok(())
    }
}

triton_rs::declare_backend!(ExampleBackend);
