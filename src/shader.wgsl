// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex // This is called "vs_main" for "vertex shader"
fn vs_main(
    model: VertexInput
) -> VertexOutput {
    var out: VertexOutput; 
    // Variables define with "var" can be modified but must specify their type.
    // Variables created with "let" can have their types inferred
    // but their value cannot be changed during the shader.
    out.color = model.color;
    out.clip_position = vec4<f32>(model.position, 1.0);

    return out;
}

// Fragment shader

@fragment // This is called "fs_main" for "fragment shader"
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // The "@location(0)" bit tells WGPU to store the vec4 value returned
    // by this function into the first color target.
    return vec4<f32>(in.color, 1.0);
}