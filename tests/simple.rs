use emu_core::prelude::*;

pub struct Environment {}

#[tokio::test]
pub async fn basic() -> Result<(), Box<dyn std::error::Error>> {
    // Wait for device pool to be initialized.
    assert_device_pool_initialized().await;

    // Check that the device info exists.
    assert!(take()?.lock().unwrap().info.is_some());

    // compile GslKernel to SPIR-V
    // then, we can either inspect the SPIR-V or finish the compilation by generating a DeviceFnMut
    // then, run the DeviceFnMut
    let c = compile::<GlslKernel, GlslKernelCompile, Vec<u32>, GlobalCache>(
        GlslKernel::new()
            .spawn(64)
            .param::<i32, _>("int origin")
            .param_mut::<[u32], _>("uint[] data")
            .with_kernel_code(
                r#"
            uint id = gl_GlobalInvocationID.x;
            data[id] = data[id] + 1;
                "#,
            ),
    )?
    .finish()?;

    let mut x: DeviceBox<[u32]> = (0..4096).collect::<Vec<u32>>().as_device_boxed_mut()?;
    unsafe {
        spawn(4096 / 64).launch(call!(c, &DeviceBox::new(0)?, &mut x))?;
    }

    // Retrieve data from GPU.
    let output = x.get().await?;
    assert_eq!(&*output, (1..4097).collect::<Vec<u32>>().as_slice());
    Ok(())
}