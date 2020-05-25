use emu_core::prelude::*;
use toil::Array;

pub struct Environment {}

#[tokio::test]
pub async fn clone_array() -> Result<(), Box<dyn std::error::Error>> {
    // Wait for device pool to be initialized.
    assert_device_pool_initialized().await;

    // Check that the device info exists.
    assert!(take()?.lock().unwrap().info.is_some());

    let input: Array<u32, 1> = Array::new([4096], (0..4096).collect::<Vec<u32>>());
    let output = input.clone();

    // Retrieve data from GPU.
    let input = input.to_vec().await;
    let output = output.to_vec().await;
    assert_eq!(input, output);
    Ok(())
}
