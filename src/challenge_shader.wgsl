// Vertex shader

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec2<f32>,
};

@vertex // This is called "vs_main" for "vertex shader"
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput; 
    // Variables define with "var" can be modified but must specify their type.
    // Variables created with "let" can have their types inferred
    // but their value cannot be changed during the shader.

    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;

    out.position = vec2<f32>(x, y);
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);

    return out;
}

// Fragment shader

@fragment // This is called "fs_main" for "fragment shader"
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // The "@location(0)" bit tells WGPU to store the vec4 value returned
    // by this function into the first color target.
    return vec4<f32>(in.position, 0.1, 1.0);
}