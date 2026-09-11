# 第一种安装方式 使用UP主提供的一点通仓库公开镜像为基础
FROM docker.cnb.cool/skdzss90/fenxiang/3lian_guan_zhu:0531-v.0.3.39-n120

# 设置插件管理器工作区（我的旧链接全都不要，替换成你要安装的链接，安装特点自动安装依赖）
WORKDIR /opt/ComfyUI/custom_nodes/comfyui-manager
RUN ./cm-cli.sh install \
    https://github.com/chengzeyi/Comfy-WaveSpeed.git \
    https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux.git \
    https://github.com/huchenlei/ComfyUI-openpose-editor.git \
    https://github.com/1038lab/ComfyUI-RMBG.git --mode remote
