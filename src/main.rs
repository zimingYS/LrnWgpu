use wgpu::*;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{WindowBuilder,Window}
};

struct State {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
}

impl State {
    // 某些 wgpu 类型需要使用异步代码才能创建
    //此方法调用规则:
    // let mut state = pollster::block_on(State::new(&window));
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // instance 变量是到 GPU 的 handle,用于创建Adapter 和 Surface
        // Backends::all 对应 Vulkan + Metal + DX12 + 浏览器的 WebGPU
        let instance = Instance::new(wgpu::Backends::all());

        //用于绘制窗口，将内容绘制到屏幕,同时我们还需要用 surface 来请求 adapter
        let surface = unsafe {
            instance.create_surface(window)
        };

        //adapter(适配器)是指向显卡的一个handle.
        //它可以用来获取显卡信息
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions{
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
            &wgpu::DeviceDescriptor{
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

        //将以上配置返回
        Self{
            surface,
            device,
            queue,
            config,
            size,
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
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        todo!()
    }

    fn update(&mut self) {
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
            let _render_pass = encoder.begin_render_pass(&RenderPassDescriptor{
                //开启渲染通道
                label: Some("Render Pass"),
                //color_attachments 定义颜色附件,用于存储颜色数据的纹理
                color_attachments: &[RenderPassColorAttachment{
                    //iew 变量作为颜色附件的视图
                    view: &view,
                    //resolve_target 用于指定颜色附件的解析目标
                    resolve_target: None,
                    //Operations 用于描述颜色附件的操作
                    ops: Operations{
                        //load 字段来指定颜色附件的加载,其中Clear用来清除颜色附件
                        load: LoadOp::Clear(wgpu::Color{
                            r : 0.1,
                            g : 0.2,
                            b : 0.3,
                            a : 1.0  //此处为不透度
                        }),
                        //store 用于指定颜色附件的存储操作
                        store: true,
                    }
                }],
                //用于定义深度和模板附件,此处没有定义深度和模板附件，所以置为None
                depth_stencil_attachment: None,
            });
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
        .with_title("learning Rust WGPU")
        .build(&event_loop)
        .unwrap();
    let mut state = pollster::block_on(State::new(&window));

    //初始化窗口大小
    state.resize(winit::dpi::PhysicalSize::new(800, 600));

    event_loop.run(move |event, _, control_flow|
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            }
            if window_id == window.id() => match event {
                WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                    input: KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                    ..
                } => *control_flow = ControlFlow::Exit,

                //此处用于窗口改变大小后重绘
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    // new_inner_size 是 &&mut 类型，因此需要解引用两次
                    state.resize(**new_inner_size);
                },

                _ => {}
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
    });
}