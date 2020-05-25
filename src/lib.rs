#![feature(const_generics)]

use emu_core::prelude::*;
use zerocopy::*;
use std::mem::size_of;
use duplicate::duplicate;

pub struct Environment {}

pub struct Array<T, const N: usize> {
    strides: [isize; N],
    dims: [usize; N],
    data: DeviceBox<[T]>,
}

fn next_64_multiple(n: usize) -> usize {
    (n + 63) / 64 * 64
}

impl<T: AsBytes + FromBytes + Default + Copy, const N: usize> Array<T, N> {
    pub fn try_new(dims: [usize; N], mut data: Vec<T>) -> Result<Self, NoDeviceError> {
        let mut strides = [0; N];
        // The buffer must be a multple of 64, so resize everything appropriately.
        let len = dims.iter().copied().product::<usize>();
        let buffer_len = next_64_multiple(len);
        data.resize_with(buffer_len, Default::default);
        // Compute the strides assuming a dense packing.
        for i in 0..N {
            strides[i] = dims[0..i].iter().copied().product::<usize>() as isize;
        }
        // Copy the now-resized data to the GPU.
        let data = data.as_device_boxed_mut()?;
        Ok(Self {
            strides,
            dims,
            data,
        })
    }

    pub fn new(dims: [usize; N], data: Vec<T>) -> Self {
        Self::try_new(dims, data).expect("tried to create toil Array with no device")
    }

    fn len(&self) -> usize {
        self.dims.iter().copied().product::<usize>()
    }

    fn batches(&self) -> usize {
        (self.len() + 63) / 64
    }

    fn buffer_len(&self) -> usize {
        self.batches() * 64
    }

    pub async fn to_vec(&self) -> Vec<T> {
        self.try_to_vec().await.expect("failed to get Vec from GPU")
    }
    
    pub async fn try_to_vec(&self) -> Result<Vec<T>, GetError> {
        let mut data = self.data.get().await?.into_vec();
        data.resize_with(self.len(), || unreachable!("this resize should only shrink the vector"));
        Ok(data)
    }
}
#[duplicate(
    RustType glsl_type;
    [u32] ["uint"];
    [i32] ["int"];
    [f32] ["float"];
)]
impl<const N: usize> Clone for Array<RustType, N> {
    fn clone(&self) -> Self {
        let mut data: DeviceBox<[RustType]> = DeviceBox::with_size_mut(self.buffer_len() * size_of::<RustType>()).expect("tried to clone toil Array with no device");
        let c = compile::<GlslKernel, GlslKernelCompile, Vec<u32>, GlobalCache>(
            GlslKernel::new()
                .spawn(64)
                .param::<[RustType], _>(format!("{}[] source", glsl_type))
                .param_mut::<[RustType], _>(format!("{}[] dest", glsl_type))
                .with_kernel_code(
                    r#"
                        uint id = gl_GlobalInvocationID.x;
                        dest[id] = source[id];
                    "#,
                ),
        ).expect("failed to compile copy kernel")
        .finish().expect("failed to finish copy kernel");
        unsafe {
            spawn(self.batches() as u32).launch(call!(c, &self.data, &mut data)).expect("failed to copy data in toil::Array::clone");
        }
        Self {
            strides: self.strides,
            dims: self.dims,
            data,
        }
    }
}
