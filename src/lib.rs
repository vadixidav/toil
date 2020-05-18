use emu_core::prelude::*;
use emu_glsl::*;
use zerocopy::*;

pub struct Environment {}

#[repr(C)]
#[derive(AsBytes, FromBytes, Copy, Clone, Default, Debug, GlslStruct)]
struct Array2 {
    strides: [i32; 2],
    dims: [u32; 2],
}
