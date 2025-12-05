//! https://github.com/triton-inference-server/backend/blob/main/README.md#triton-backend-api

use futures::executor::block_on;
use triton_rs::Backend;
use triton_rs::Model;
use triton_rs::ModelInstance;
use triton_rs::RequestReleaseFlags;
use triton_rs::Response;
use triton_rs::ResponseFlags;

#[derive(Debug, Default)]
struct InstanceState(usize);

impl InstanceState {
    fn change(&mut self) {
         self.0 += 1; // do stuff
    }
}

#[derive(Debug)]
struct SubModelExecutor(triton_rs::ModelExecutor);

impl SubModelExecutor {
    fn new(model: &triton_rs::ModelImpl<SubModelExecutor>,
           model_name: &str,
           model_version: i64) -> Result<Self, triton_rs::Error> {
        let server = model.server()?;
        let executor = triton_rs::ModelExecutor::new(
            server, model_name, model_version)?;
        Ok(Self(executor))
    }
}

struct ExampleBackend;

impl Backend for ExampleBackend {
    type ModelInstanceState = InstanceState;
    type ModelState = SubModelExecutor;

    fn model_initialize(
            model: triton_rs::ModelImpl<SubModelExecutor>
    ) -> Result<(), triton_rs::Error> {
        let sub_model_executor = SubModelExecutor::new(&model, "passthrough", 1)?;
        let previous = model.replace_state(Some(sub_model_executor))?;
        assert!(previous.is_none());
        Ok(())
    }

    fn model_instance_initialize(
            model_instance: triton_rs::ModelInstanceImpl<InstanceState, SubModelExecutor>
    ) -> Result<(), triton_rs::Error> {
        let previous = model_instance.replace_state(Some(InstanceState::default()))?;
        assert!(previous.is_none());
        Ok(())
    }

    fn model_instance_execute(
        model_instance: triton_rs::ModelInstanceImpl<InstanceState, SubModelExecutor>,
        requests: &[triton_rs::Request],
    ) -> Result<(), triton_rs::Error> {
        let state = model_instance.state()?;
        state.change();
        println!("[EXAMPLE] model_instance_execute ({state:?}");

        let model: triton_rs::ModelImpl<SubModelExecutor> = model_instance.model()?;

        println!("[EXAMPLE] model config: {:?}", model.model_config()?);
        let executor = &model.state()?.0;

        println!(
            "[EXAMPLE] request for model {} {} {} {executor:?}",
            model.name()?,
            model.version()?,
            model.location()?
        );

        for request in requests {
            let prompt = request.get_input("prompt")?;
            let floats = prompt.slice::<f32>()?;
            println!("[EXAMPLE] prompt as f32: {}, len={}", floats[0], floats.len());
            let array =  prompt.as_array::<f32, 1>()?.to_owned();
            println!("[EXAMPLE] prompt as ndarray: {array}");
//          let prompt = prompt.as_string()?;
//          println!("[EXAMPLE] prompt as_string: {prompt}");

            // model_excutor
            let request_id = request.get_request_id()?;
            println!("[EXAMPLE] request_id: {}", request_id);
            let correlation_id = request.get_correlation_id()?;
            println!("[EXAMPLE] correlation_id: {}", correlation_id);
            let input1_name = "INPUT";
            let output1_name = "OUTPUT";
            let inference_request = triton_rs::InferenceRequest::new(executor)?;

            inference_request.set_request_id(request_id.as_str())?;
            inference_request.set_correlation_id(correlation_id)?;
            inference_request.set_release_callback()?;

            println!("[EXAMPLE] set request id and correlation id finish");
            inference_request.add_input_array(input1_name, &array)?;
            inference_request.add_output(output1_name)?;

            let infer_response = block_on(executor.execute(&inference_request))?;

            // executor.execute(&inference_request).await.map_err(|e| {
            //     // Custom error handling logic
            //     triton_rs::ModelExecutorError::ExecutionError(e)
            // })?;

            println!(
                "[EXAMPLE] inference_response output : {}",
                infer_response.get_output_count()
            );

            let output1 = infer_response.get_output_data(output1_name)?;
            println!("[EXAMPLE] sub-model returned: {output1:?}");
            let output1 = output1.as_array::<f32, 1>();
            println!("[EXAMPLE] {output1_name} as Array1<f32>: {output1:?}");
            let output1 = output1?.to_owned();

            // let mut response = Response::from_request(request)?;
            let factory = &triton_rs::ResponseFactory::from_request(request)?;
            // beyond here, we no longer need request
            request.release(RequestReleaseFlags::ALL)?;

            let mut response = Response::from_factory(factory)?;
            response.add_output_array("output", output1)?;
            response.send(ResponseFlags::NONE, None)?;
            // ... send more responses, then:
            factory.send_flags(ResponseFlags::FINAL)?;
        }

        Ok(())
    }
}

triton_rs::declare_backend!(ExampleBackend);
