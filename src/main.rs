//// 此项目仅用于个人学习使用，可以自行拉取项目并使用。
//// 此项目参考自 https://doodlewind.github.io/learn-wgpu-cn/
//// 此项目部分内容与源代码和教程有所不同，加上了本人自行修改的一部分内容。
//// 此项目仅供参考，若侵权请私信或者在问题区提出，本人将删除此仓库。

mod texture;
mod model;
mod resources;

use wgpu::*;
use wgpu::util::DeviceExt;
use cgmath::prelude::*;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{WindowBuilder,Window}
};
use crate::model::{DrawModel, Vertex};

// 定义每行实例的数量
const NUM_INSTANCES_PER_ROW: u32 = 10;
// 定义实例的位移 上面定义为10，则实例化为10x10。
const INSTANCE_DISPLACEMENT: cgmath::Vector3<f32> = cgmath::Vector3::new(NUM_INSTANCES_PER_ROW as f32 * 0.5, 0.0, NUM_INSTANCES_PER_ROW as f32 * 0.5);


//将OPENGL矩阵转换为WGPU矩阵
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

// 我们需要这个标注来让 Rust 正确存储用于着色器的数据
#[repr(C)]
// 这样配置可以让我们将其存储在缓冲区之中
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // 我们不能将 bytemuck 与 cgmath 直接一起使用
    // 因此需要先将 Matrix4 矩阵转为一个 4x4 的 f32 数组
    view_proj: [[f32; 4]; 4],
}

///相机控制部分
impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}
//增加相机
struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
    // 创建一个视图矩阵，该矩阵表示从self.eye看向self.target，并且self.up表示向上的方向
    let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
    // 创建一个透视投影矩阵，该矩阵表示从self.eye看向self.target，并且self.up表示向上的方向
    let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
    // 将视图矩阵和透视投影矩阵相乘，得到最终的变换矩阵
    OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

//相机位置控制
struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
        }
    }

    //此部分用于处理相机控制事件
    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    state,
                    virtual_keycode: Some(keycode),
                    ..
                },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    //这边增加Q和E用来控制上下移动。
                    VirtualKeyCode::Q | VirtualKeyCode::Right => {
                        self.is_up_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::E | VirtualKeyCode::Right => {
                        self.is_down_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }


    //此部分用于更新相机视角
    fn update_camera(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;
        // 计算摄像机的前向向量
        let forward = camera.target - camera.eye;
        // 归一化前向向量
        let forward_norm = forward.normalize();
        // 计算前向向量的长度
        let forward_mag = forward.magnitude();

        // 防止摄像机离场景中心太近时出现故障
        if self.is_forward_pressed && forward_mag > self.speed {
            // 如果按下前进键，则摄像机向前移动
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            // 如果按下后退键，则摄像机向后移动
            camera.eye -= forward_norm * self.speed;
        }

        // 计算摄像机的右向量
        let right = forward_norm.cross(camera.up);

        // 在按下前进或后退键时重做半径计算
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // 重新调整目标与眼睛之间的距离，以使其不发生变化
            // 因此，眼睛仍位于由目标和眼睛所组成的圆上。
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
        if self.is_up_pressed {
            // 摄像机向上移动
            camera.eye += camera.up * self.speed;
        }
        if self.is_down_pressed {
            // 摄像机向下移动
            camera.eye -= camera.up * self.speed;
        }
    }
}

//创建实例缓冲区
struct Instance{
    position : cgmath::Vector3<f32>,
                //这里为四元数
    rotation : cgmath::Quaternion<f32>,
}
//将实例化缓冲区转换为矩阵
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw{
    model : [[f32 ; 4]; 4],
}
//增加方法 实现矩阵转换
impl Instance{
    fn to_raw(&self) -> InstanceRaw{
        InstanceRaw{
            // 计算模型的变换矩阵
            model: (cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation)).into(),
        }
    }
}

impl InstanceRaw{
    fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: size_of::<InstanceRaw>() as BufferAddress,
            // 我们需要从把 Vertex 的 step mode 切换为 Instance
            // 这样着色器只有在开始处理一次新实例化绘制时，才会接受下一份实例
            step_mode: VertexStepMode::Instance,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    // 虽然顶点着色器现在只使用位置 0 和 1，但在后面的教程中，我们将对 Vertex 使用位置 2、3 和 4
                    // 因此我们将从 5 号 slot 开始，以免在后面导致冲突
                    shader_location: 5,
                    format: VertexFormat::Float32x4,
                },
                // 一个 mat4 需要占用 4 个顶点 slot，因为严格来说它是 4 个vec4
                // 我们需要为每个 vec4 定义一个 slot，并在着色器中重新组装出 mat4
                VertexAttribute {
                    offset: size_of::<[f32; 4]>() as BufferAddress,
                    shader_location: 6,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: size_of::<[f32; 8]>() as BufferAddress,
                    shader_location: 7,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: size_of::<[f32; 12]>() as BufferAddress,
                    shader_location: 8,
                    format: VertexFormat::Float32x4,
                },
            ],
        }
    }
}

struct State {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: RenderPipeline,
    diffuse_bind_group: BindGroup,
    diffuse_texture: texture::Texture,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: Buffer,
    camera_bind_group: BindGroup,
    camera_controller: CameraController,
    instances: Vec<Instance>,
    instance_buffer: Buffer,
    depth_texture: texture::Texture,
    obj_model: model::Model,
}

impl State {
    // 某些 wgpu 类型需要使用异步代码才能创建
    //此方法调用规则:
    // let mut state = pollster::block_on(State::new(&window));
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // instance 变量是到 GPU 的 handle,用于创建Adapter 和 Surface
        // Backends::all 对应 Vulkan + Metal + DX12 + 浏览器的 WebGPU
        let instance = wgpu::Instance::new(Backends::all());

        //用于绘制窗口，将内容绘制到屏幕,同时我们还需要用 surface 来请求 adapter
        let surface = unsafe {
            instance.create_surface(window)
        };

        //adapter(适配器)是指向显卡的一个handle.
        //它可以用来获取显卡信息
        let adapter = instance.request_adapter(
            &RequestAdapterOptions{
                //power_preference 参数有两个可选项：LowPower 和 HighPerformance
                //使用LowPower 时将对应一个有利于电池续航的适配器（如集成显卡）。
                // 相应地，HighPerformance 对应的适配器将指向独立显卡这样更耗电但性能更强的 GPU。
                // 如果不存在符合 HighPerformance 选项的适配器，wgpu 将选择 LowPower
                power_preference: PowerPreference::default(),

                //compatible_surface 字段要求 wgpu 所找到的适配器应当能与此处所传入的 surface 兼容
                compatible_surface: Some(&surface),

                //force_fallback_adapter 强制 wgpu 选择一个能在所有硬件上工作的适配器。
                // 这通常表明渲染后端将使用一个「软渲染」系统，而非 GPU 这样的硬件。
                force_fallback_adapter:false,
            },
        ).await.unwrap();

        //使用adapter创建device和queue
        let (device, queue) = adapter.request_device(
            &DeviceDescriptor{
                //features 字段允许我们指定我们想要的额外特性
                //使用adapter.features() 或 device.features() 获得设备所支持特性的列表
                features: Features::empty(),

                //limits 字段描述了对我们所能创建的某些资源类型的限制
                //使用默认值以支持大多数设备
                limits: Limits::default(),

                label: None,
            },
            None,
        ).await.unwrap();

        //为surface定义一份配置,用于确定 surface 如何创建其底层的 SurfaceTexture
        let config = SurfaceConfiguration{
            //usage 字段用于定义应如何使用 SurfaceTextures
            usage : TextureUsages::RENDER_ATTACHMENT,

            //format 字段定义了 SurfaceTexture 在 GPU 上的存储方式
            format : surface.get_preferred_format(&adapter).unwrap(),

            //设置窗口的分辨率
            //请确保 SurfaceTexture 的宽高不为 0，否则可能导致应用崩溃。
            width: size.width,
            height: size.height,

            //present_mode 使用 wgpu::PresentMode 枚举值来确定应如何将 surface 同步到显示器上
            //使用FIFO将显示速率限制为显示器的帧速率（仅Fifo可适配移动设备）
            present_mode: PresentMode::Fifo,
        };
        surface.configure(&device,&config);

        //读取纹理
        let diffuse_bytes = include_bytes!("img/happy_tree.png");
        let diffuse_texture = texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png").unwrap();


        //创建BindGroup
        // 创建一个纹理绑定的布局
        let texture_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor {
                // 定义绑定的入口
                entries: &[
                    // 第一个入口，绑定纹理
                    BindGroupLayoutEntry {
                        binding: 0, // 绑定索引
                        visibility: ShaderStages::FRAGMENT, // 可见性，片段着色器
                        ty: BindingType::Texture { // 绑定类型为纹理
                            multisampled: false, // 非多重采样
                            view_dimension: TextureViewDimension::D2, // 纹理视图维度为2D
                            sample_type: TextureSampleType::Float { filterable: true }, // 样本类型为浮点数，可过滤
                        },
                        count: None, // 无数量限制
                    },
                    // 第二个入口，绑定采样器
                    BindGroupLayoutEntry {
                        binding: 1, // 绑定索引
                        visibility: ShaderStages::FRAGMENT, // 可见性，片段着色器
                        ty: BindingType::Sampler( // 绑定类型为采样器
                            // SamplerBindingType::Comparison 仅可供 TextureSampleType::Depth 使用
                            // 如果纹理的 sample_type 是 TextureSampleType::Float { filterable: true }
                            // 那么就应当使用 SamplerBindingType::Filtering
                            // 否则会报错
                            SamplerBindingType::Filtering, // 采样器类型为过滤
                        ),
                        count: None, // 无数量限制
                    },
                ],
                label: Some("texture_bind_group_layout"), // 标签
            }
        );
        // 创建一个diffuse_bind_group，用于绑定纹理和采样器
        let diffuse_bind_group = device.create_bind_group(
            &BindGroupDescriptor {
                // 指定绑定的布局
                layout: &texture_bind_group_layout,
                // 指定绑定的条目
                entries: &[
                    BindGroupEntry {
                        // 指定绑定的索引
                        binding: 0,
                        // 指定绑定的资源，这里是一个纹理视图
                        resource: BindingResource::TextureView(&diffuse_texture.view),
                    },
                    BindGroupEntry {
                        // 指定绑定的索引
                        binding: 1,
                        // 指定绑定的资源，这里是一个采样器
                        resource: BindingResource::Sampler(&diffuse_texture.sampler),
                    }
                ],
                // 指定绑定的标签
                label: Some("diffuse_bind_group"),
            }
        );

        //创建着色器并载入着色器文件
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Shader"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        //相机控制
        let camera = Camera {
            // 将相机向上移动 1 个单位，向后移动 2 个单位
            // +z 对应屏幕外侧方向
            eye: (0.0, 1.0, 2.0).into(),
            // 将相机朝向原点
            target: (0.0, 0.0, 0.0).into(),
            // 定义哪个方向朝上
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(
            &util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("camera_bind_group_layout"),
        });

        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("camera_bind_group"),
        });

        //创建渲染管线布局
        let render_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                //定义管线中使用的绑定组布局
                //绑定组布局描述了管线中使用的资源集合的布局
                bind_group_layouts: &[
                    &texture_bind_group_layout ,
                    &camera_bind_group_layout ,
                ],
                //定义管线中使用到的推送常量（Push Constants）的范围
                //推送常量是一种在绘制调用中快速传递少量数据的机制
                push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            //指定渲染管线的布局
            layout: Some(&render_pipeline_layout),
            //设置顶点着色器状态
            vertex: VertexState {
                module: &shader,
                //指定顶点着色器的入口点函数名
                entry_point: "vs_main",
                //buffers 字段用于告知 wgpu 我们要传递给顶点着色器的顶点类型
                buffers: &[
                    model::ModelVertex::desc(),
                    InstanceRaw::desc(),
                ],
            },
            //设置片段着色器
            fragment: Some(FragmentState {
                module: &shader,
                //指定片段着色器的入口点函数名
                entry_point: "fs_main",
                //片段着色器的输出颜色目标状态
                targets: &[ColorTargetState {
                    format: config.format,
                    //混合模式为 REPLACE
                    blend: Some(BlendState::REPLACE),
                    //写掩码为 ColorWrites::ALL
                    write_mask: ColorWrites::ALL,
                }],
            }),
            //设置图元状态
            primitive: PrimitiveState {
                //图元的拓扑类型,使用TriangleList指定为三角形
                topology: PrimitiveTopology::TriangleList,
                //不使用索引缓冲区
                strip_index_format: None,
                //使用逆时针
                front_face: FrontFace::Ccw,
                //剔除背面
                cull_mode: None,
                //多边形模式设置为填充
                // 如果将该字段设置为除了 Fill 之外的任何值，都需要 Features::NON_FILL_POLYGON_MODE
                polygon_mode: PolygonMode::Fill,
                //WebGPU特性
                unclipped_depth: false,
                conservative: false,
            },
            //深度状态和模板测试
            depth_stencil: Some(DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            //重采样的状态
            multisample: MultisampleState {
                //当为1时不使用多重采样
                count: 1,
                //多重采样掩码，这里为 !0，表示所有样本都有效
                mask: !0,
                //不启用 alpha 到覆盖（coverage）的转换
                alpha_to_coverage_enabled: false,
            },
            //不使用多视图渲染
            multiview: None,
        });

        //相机控制
        let camera_controller = CameraController::new(0.02);

        //实例化创建
        const SPACE_BETWEEN: f32 = 3.0;
        // 创建一个包含NUM_INSTANCES_PER_ROW个实例的向量
        let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                let position = cgmath::Vector3 { x, y: 0.0, z };

                let rotation = if position.is_zero() {
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                } else {
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(0.0))
                };

                Instance {
                    position, rotation,
                }
            })
        }).collect::<Vec<_>>();

        //创建instance_buffer
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(
            &util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: BufferUsages::VERTEX,
            }
        );

        //创建深度
        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        //纹理材质
        log::warn!("Load model");
        let obj_model =
            resources::load_model("cube.obj", &device, &queue, &texture_bind_group_layout)
                .await
                .unwrap();



        //将以上配置返回
        Self{
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            diffuse_bind_group,
            diffuse_texture,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            instances,
            instance_buffer,
            depth_texture,
            obj_model,
        }
    }

    //此方法用于重新设置窗口大小
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        //我们需要在每次窗口尺寸改变时重新配置 surface
        //同时存储了物理 size 和用于配置 surface 的 config

        //检查是否大于0如果宽高不大于0则方法不适用。
        if new_size.width > 0 && new_size.height > 0{
            //重新设置size
            self.size = new_size;
            //重新设置宽高
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            //重新设置surface
            self.surface.configure(&self.device,&self.config);
            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        //get_current_texture 函数会等待 surface 提供一个新的 SurfaceTexture 以用于渲染
        let output = self.surface.get_current_texture()?;

        //创建了一个使用默认配置的 TextureView 用于控制渲染代码与纹理之间的交互
        let view = output.texture.create_view(&TextureViewDescriptor::default());

        //创建一个 CommandEncoder 来创建实际发送到 GPU 上的命令
        //使用 encoder 充当命令缓冲区
        let mut encoder = self.device.create_command_encoder(
            &CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            }
        );

        //使用 encoder 来创建 RenderPass 以执行清屏
        //由于 begin_render_pass() 是以可变方式借用了 encoder , 因此在我们释放这个可变的借用之前，我们都不能调用 encoder.finish()
        //所以需要释放 encoder 上的可变借用，从而使得我们能使用 finish()
        {
            //使用作用域以释放内存
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor{
                //开启渲染通道
                label: Some("Render Pass"),
                //片元着色器中 [[location(0)]] 对应的目标
                //color_attachments 定义颜色附件,用于存储颜色数据的纹理
                color_attachments: &[RenderPassColorAttachment{
                    //iew 变量作为颜色附件的视图
                    view: &view,
                    //resolve_target 用于指定颜色附件的解析目标
                    resolve_target: None,
                    //Operations 用于描述颜色附件的操作
                    ops: Operations{
                        //load 字段来指定颜色附件的加载,其中Clear用来清除颜色附件
                        //此处可以设置背景颜色
                        load: LoadOp::Clear(wgpu::Color{
                            r : 0.0,
                            g : 0.8,
                            b : 0.8,
                            a : 1.0  //此处为不透度
                        }),
                        //store 用于指定颜色附件的存储操作
                        store: true,
                    }
                }],
                //用于定义深度和模板附件
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_pipeline(&self.render_pipeline);
            //纹理绑定
            render_pass.draw_model_instanced(
                &self.obj_model,
                0..self.instances.len() as u32,
                &self.camera_bind_group,
            );
        }
        // submit 方法能传入任何实现了 IntoIter 的参数
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    //初始化窗口代码
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        //此处设置窗口标题
        .with_title("Learning Rust WGPU")
        .build(&event_loop)
        .unwrap();
    let mut state = pollster::block_on(State::new(&window));

    // let mut surface_configured = false;  这边忘记干什么用的了 先注释掉

    //初始化窗口大小
    state.resize(winit::dpi::PhysicalSize::new(800, 600));

    event_loop.run(move |event, _, control_flow|
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            }
            if window_id == window.id() => if !state.input(event){
                match event {
                    WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                        input: KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                        ..
                    } => {
                        *control_flow = ControlFlow::Exit
                    },

                    //此处用于窗口改变大小后重绘
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        // new_inner_size 是 &&mut 类型，因此需要解引用两次
                        state.resize(**new_inner_size);
                    },

                    _ => {}
                }
            },
            //使用事件更新调用render方法
            Event::RedrawRequested(window_id ) if window_id == window.id() => {
                //调用state.upadte 以更新state
                state.update();
                match state.render() {
                    Ok(_) => {},
                    //如果发生上下文丢失，则重新配置surface
                    Err(SurfaceError::Lost) => state.resize(state.size),
                    //内存不足时退出
                    Err(SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    //若出现其他错误则应在下一帧处理
                    //打印输出错误
                    Err(e) => eprintln!("{:?}",e),
                }
            },
            Event::MainEventsCleared => {
                // 除非手动请求，否则 RedrawRequested 只会触发一次
                window.request_redraw();
            },
            _ => {}
        }
    );
}