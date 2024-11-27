use wgpu::{Features, Instance, Limits, PowerPreference, SurfaceConfiguration, TextureUsages};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{WindowBuilder,Window}
};

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
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
            present_mode: wgpu::PresentMode::Fifo,
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
        todo!()
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        todo!()
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

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                KeyboardInput {
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
        _ => {}
    });
}