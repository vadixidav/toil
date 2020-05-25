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

    // Retrieve data from GPU and compare.
    assert_eq!(input.to_vec().await, output.to_vec().await);
    assert_eq!(input.dims(), output.dims());
    assert_eq!(input.strides(), output.strides());
    Ok(())
}

#[tokio::test]
pub async fn add_assign_array() -> Result<(), Box<dyn std::error::Error>> {
    // Wait for device pool to be initialized.
    assert_device_pool_initialized().await;

    // Check that the device info exists.
    assert!(take()?.lock().unwrap().info.is_some());

    let mut a: Array<u32, 1> = Array::new([4096], (0..4096).collect::<Vec<u32>>());
    let b: Array<u32, 1> = Array::new([4096], std::iter::repeat(1).take(4096).collect::<Vec<u32>>());
    a += b;

    // Retrieve data from GPU and compare.
    assert_eq!(a.to_vec().await, (1..4097).collect::<Vec<u32>>());
    Ok(())
}
