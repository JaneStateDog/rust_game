use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder}
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)] 
// The struct needs to be "Copy" so we can create a buffer with it
// Pod indicates that our Vertex is "Plain Old Data" and thus can be interpreted as a "&[u8]".
// Zeroable indicates that we can use std::mem::zeroed().
struct Vertex {
    position: [f32; 3], // X, Y, Z
    color: [f32; 3], // R, G, B
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        /* // This way of of specifying the attributes is quite verbose.
        // We can use a different method (like we did below) to do it much cleaner.
        // I'm just keeping this here for note-taking-reasons
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            // This defines how wide a vertex is. When the shader goes to read the
            // next vertex, it will skip over "array_stride" number of bytes. In our case, "array_stride"
            // will probably be 24 bytes.
            step_mode: wgpu::VertexStepMode::Vertex,
            // This tells the pipeline whether each element of the array in this buffer represents
            // per-vertex data or per-instance data. We can specify "wgpu::VertexStepMode::Instance"
            // if we only want to change vertices when we start drawing a new instance.
            attributes: &[ // This describes the individual parts of the vertex.
            // Generally, this is a 1:1 mapping with a struct's fields, which is true in our case.
                wgpu::VertexAttribute {
                    offset: 0, // This defines the offset in bytes until the attribute starts.
                    // For the first attribute, the offset is usually zero. For any later attributes,
                    // the offset is the sum over "size_of" of the previous attributes' data.
                    shader_location: 0, // This tells the shader what location to store this attribute at.
                    // For example "@location(0) x: vec3<f32>" in the vertex shader would correspond to the
                    // position field of the "Vertex" struct, while "@location(1) x: vec3<f32>" would be the color field.
                    format: wgpu::VertexFormat::Float32x3, // This tells the shader the shape of the attribute.
                    // "Float32x3" corresponds to "vec3<f32>" in shader code. The max value we can store in
                    // an attribute is "Float32x4" ("Uint32x4" and "Sint32x4" work as well.) We'll keep this in
                    // mind for when we have to store things that are bigger than "Float32x4".
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ]
        } 
        */

        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// We've got cooler vertices now *puts on sunglasses*
/*
const VERTICES: &[Vertex] = &[ // We arrange the vertices in counter-clockwise order:
// top, bottom left, bottom right. We do it this way partially out of tradition, but mostly
// because we specified in the "primitive" of the "render_pipeline" that we want the
// "front_face" of our triangle to be "wgpu::FrontFace::Ccw" so that we cull the back face.
// This means that any triangle that should be facing us should have its vertices in counter-clockwise order.
    Vertex { position: [0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] },
    Vertex { position: [0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0] },
];
*/

const VERTICES: &[Vertex] = &[
    Vertex { position: [-0.0868241, 0.49240386, 0.0], color: [0.5, 0.0, 0.5] }, // A
    Vertex { position: [-0.49513406, 0.06958647, 0.0], color: [0.5, 0.0, 0.5] }, // B
    Vertex { position: [-0.21918549, -0.44939706, 0.0], color: [0.5, 0.0, 0.5] }, // C
    Vertex { position: [0.35966998, -0.3473291, 0.0], color: [0.5, 0.0, 0.5] }, // D
    Vertex { position: [0.44147372, 0.2347359, 0.0], color: [0.5, 0.0, 0.5] }, // E
];

const INDICES: &[u16] = &[
    0, 1, 4,
    1, 2, 4,
    2, 3, 4
];

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    clear_color: wgpu::Color,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    render_pipeline: wgpu::RenderPipeline,
    challenge_render_pipeline: wgpu::RenderPipeline,
    use_color: bool,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // Safety
        // The surface needs to live as long as the window that created it
        // State owns the window so this should be safe
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                // WebGL doesn't support all of wpgu's features, so if
                // we're building for the web we'll have to disable some
                limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
            },
            None, // Trace path
        ).await.unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Sahder code in this tutorial assumes an sRGB surface texture.
        // Using a different one will result all the colors coming out darker.
        // If you want to support non sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
        let diffuse_rgba = diffuse_image.to_rgba8();

        use image::GenericImageView;
        let dimensions = diffuse_image.dimensions();
        // Here we get the bytes of our image file, load them into an image, which is
        // then converted into a "Vec" of rgba bytes. We also save the image's dimensions
        // for when we create the actual "Texture".

        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let diffuse_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                // All textures are stored as 3D, we represent our 2D texture
                // by setting depth to 1.
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // Most images are stored using sRGB so we need to reflect that here.
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders.
                // COPY_DST means that we want to copy data to this texture.
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: Some("diffuse_texture"),
                // This is the same as with the SurfaceConfig. It specifies what
                // texture formats can be used to create TextureViews for this texture.
                // The base texture format (Rgba8UnormSrgb in this case) is always supported.
                // Note that using a different texture format is not supported on the WebGL2 backend.
                view_formats: &[],
            }
        );

        queue.write_texture(
            // Tells wgpu where to copy the pixel data
            wgpu::ImageCopyTexture {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            // The actual pixel data
            &diffuse_rgba,
            // The layout of the texture
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            texture_size,
        );

        let clear_color = wgpu::Color::BLACK;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { 
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[], 
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main", // Specify which function inside the shader should be the "entry_point".
                // The "entry_point"s are the functions we marked with @vertex and @fragment.
                buffers: &[
                    Vertex::desc(),
                ], // This field tells wgpu what type of vertices we want to pass to the vertex shader.
            },
            fragment: Some(wgpu::FragmentState { 
            // The fragment shader is technically optional, so we have to wrap it in "Some()".
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { // This tells wgpu what color outputs to set up.
                    // Currently, we only need one for the surface. We use the surface's format so that copying
                    // to it is easy, and we specify that the blending should just replace old pixel data with new data.
                    // We also tell wgpu to write to all colors: red, blue, green, and alpha.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { 
            // The primitive field describes how to interpret our vertices when converting them into triangles.
                topology: wgpu::PrimitiveTopology::TriangleList, 
                // ^ Using PrimitiveTopology::TriangleList means that every three vertices will correspond to one triangle.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back), // The "front_face" and "cull_mode" fields tell wgpu 
                // how to determine whether a given triangle is facing forward or not. 
                // "FrontFace::Ccw" means that a triangle is facing forward if the vertices are arranged in a counter-clockwise direction.
                // Triangles that are not considered facing forward are culled (not included in the render) as specified by CullMode::Back.
                polygon_mode: wgpu::PolygonMode::Fill, // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                unclipped_depth: false, // Requires Features::DEPTH_CLIP_CONTROL
                conservative: false, // Requires Features::CONSERVATIVE_RASTERIZATION
            },
            depth_stencil: None, // We're not using a depth/stencil buffer currently,
            // so we leave this as None. We will change this later.
            multisample: wgpu::MultisampleState {
                count: 1, // This determines how many samples the pipeline will use.
                // Multisampling is a complex topic, soooooo uhhhh *shrugs*
                mask: !0, // This specifies which samples should be active. In this case, we're using all of them.
                alpha_to_coverage_enabled: false, // This has to do with anti-aliasing. So, same as count, uhhhh *shrugs*
            },
            multiview: None, // This indicates how many array layers the render attachments can have.
            // We won't be rendering to array textures so we can set this to None.
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Challenge Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("challenge_shader.wgsl").into()),
        });

        let challenge_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Challenge Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState { 
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { 
                topology: wgpu::PrimitiveTopology::TriangleList, 
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let use_color = true;

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        let num_indices = INDICES.len() as u32;

        Self {
            window,
            surface,
            device,
            queue,
            config,
            clear_color,
            size,
            render_pipeline,
            challenge_render_pipeline,
            use_color,
            vertex_buffer,
            index_buffer,
            num_indices,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.clear_color = wgpu::Color {
                    r: position.x as f64 / self.size.width as f64,
                    g: position.y as f64 / self.size.height as f64,
                    b: 1.0,
                    a: 1.0,
                };
                true
            },
            WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    state, 
                    virtual_keycode: Some(VirtualKeyCode::Space),
                    ..
                },
                ..
            } => {
                self.use_color = *state == ElementState::Released;
                true
            },
            _ => false,
        }
    }

    fn update(&mut self) {
        // We'll add some code here later when we have things to update.
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.clear_color),
                            store: true,
                        },
                    })
                ],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(if self.use_color {
                &self.render_pipeline
            } else {
                &self.challenge_render_pipeline
            });
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..)); // This takes two parameters.
            // The first is what buffer slot to use for this vertex buffer (you can have multiple vertex buffers at the same time.)
            // The second is the slice of the buffer to use. You can store as many objects in a buffer as your hardware allows,
            // so "slice" allows us to specify which portion of the buffer to use. We use ".." to specify the entire buffer.
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            
            if self.use_color {
                render_pass.draw_indexed(0..self.num_indices, 0, 0..1); // When using an index buffer, we need to use "draw_indexed" instead of just "draw".
                // The "draw" method ignores the index buffer. Also make sure you use the number of indices ("num_indices"),
                // not vertices as your model will either draw wrong, or the method will "panic" becasuse there are not enough indices.
            } else {
                render_pass.draw(0..self.num_indices, 0..1); // We'll tell wgpu to draw something with our number of vertices and 1 instance
                // This is where @builtin(vertex_index) comes from.
            }
        } 
        // The reason this is in an extra block ("{}") is because something something scope
        // and we need rust to drop any variables within this block when the code leaves that scope
        // thus releasing the mutable borrow on "encoder" and allowing us to ".finish()" it.

        // Submit will accept anything that implements IntoIter.
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    // Window setup...

    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = State::new(window).await;

    // Event loop...

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested | WindowEvent::KeyboardInput { 
                        input: KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    },
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        // new_inner_size is &&mut so we have to dereference it twice.
                        state.resize(**new_inner_size);
                    },
                    _ => {}
                }
            },
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                state.update();
                match state.render() {
                    Ok(_) => {},
                    // Reconfigure the surface if lost.
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit.
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame.
                    Err(e) => eprintln!("{:?}", e),
                }
            },
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually request it.
                state.window().request_redraw();
            },
            _ => {}
        }
    });
}