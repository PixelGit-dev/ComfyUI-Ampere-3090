#!/bin/bash

# ======================================================================
# MiniMax-H3 完整模型下载脚本（含扩散模型、文本编码器、VAE、LoRA、潜空间放大）
# ======================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 计数器
TOTAL_MODELS=0
SUCCESS_MODELS=0
FAILED_MODELS=0

# 模型列表数组
declare -a MODEL_LIST=()

# ======================================================================
# 创建目录结构
# ======================================================================
echo -e "${BLUE}正在创建目录结构...${NC}"
mkdir -p "/workspace/models/vae"
mkdir -p "/workspace/models/text_encoders"
mkdir -p "/workspace/models/diffusion_models"
mkdir -p "/workspace/models/loras"
mkdir -p "/workspace/models/latent_upscale_models"
echo -e "${GREEN}目录创建完成！${NC}\n"

# ======================================================================
# 下载函数
# ======================================================================
download_model() {
    local url=$1
    local filename=$2
    local directory=$3
    local description=$4
    
    echo -e "${YELLOW}正在下载 ${description} (${filename})...${NC}"
    
    aria2c -c -x 4 -s 4 \
        "$url" \
        -o "$filename" \
        -d "$directory"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}${description} 下载完成！${NC}"
        MODEL_LIST+=("✅ $filename - $description")
        ((SUCCESS_MODELS++))
        return 0
    else
        echo -e "${RED}${description} 下载失败！${NC}"
        MODEL_LIST+=("❌ $filename - $description")
        ((FAILED_MODELS++))
        return 1
    fi
}

# ======================================================================
# MiniMax-H3 VAE 模型
# ======================================================================
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}开始下载 VAE 模型${NC}"
echo -e "${BLUE}==================================================${NC}\n"

# minimax_h3_audio_vae_fp32.safetensors
download_model \
    "https://cnb.cool/ai-models/Comfy-Org/MiniMax-H3/-/lfs/8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48?name=minimax_h3_audio_vae_fp32.safetensors" \
    "minimax_h3_audio_vae_fp32.safetensors" \
    "/workspace/models/vae" \
    "MiniMax-H3 音频VAE (FP32)"

((TOTAL_MODELS++))

# minimax_h3_video_vae_fp16.safetensors
download_model \
    "https://cnb.cool/ai-models/Comfy-Org/MiniMax-H3/-/lfs/7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522?name=minimax_h3_video_vae_fp16.safetensors" \
    "minimax_h3_video_vae_fp16.safetensors" \
    "/workspace/models/vae" \
    "MiniMax-H3 视频VAE (FP16)"

((TOTAL_MODELS++))

# ======================================================================
# MiniMax-H3 文本编码器
# ======================================================================
echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}开始下载文本编码器${NC}"
echo -e "${BLUE}==================================================${NC}\n"

# qwen3vl_32b_minimax_h3_int8_convrot.safetensors
download_model \
    "https://cnb.cool/ai-models/Comfy-Org/MiniMax-H3/-/lfs/bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923?name=qwen3vl_32b_minimax_h3_int8_convrot.safetensors" \
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors" \
    "/workspace/models/text_encoders" \
    "Qwen3VL 32B 文本编码器 (INT8 ConvRot)"

((TOTAL_MODELS++))

# qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
download_model \
    "https://cnb.cool/ai-models/Comfy-Org/MiniMax-H3/-/lfs/35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6?name=qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
    "/workspace/models/text_encoders" \
    "Qwen3VL 32B 文本编码器 (NVFP4 AWQ)"

((TOTAL_MODELS++))

# ======================================================================
# MiniMax-H3 扩散模型
# ======================================================================
echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}开始下载扩散模型${NC}"
echo -e "${BLUE}==================================================${NC}\n"

# minimax_h3_fl2va_int8_convrot.safetensors
download_model \
    "https://cnb.cool/ai-models/Comfy-Org/MiniMax-H3/-/lfs/7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5?name=minimax_h3_fl2va_int8_convrot.safetensors" \
    "minimax_h3_fl2va_int8_convrot.safetensors" \
    "/workspace/models/diffusion_models" \
    "MiniMax-H3 FL2VA 扩散模型 (INT8 ConvRot)"

((TOTAL_MODELS++))

# minimax_h3_ref2va_int8_convrot.safetensors
download_model \
    "https://cnb.cool/ai-models/Comfy-Org/MiniMax-H3/-/lfs/9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9?name=minimax_h3_ref2va_int8_convrot.safetensors" \
    "minimax_h3_ref2va_int8_convrot.safetensors" \
    "/workspace/models/diffusion_models" \
    "MiniMax-H3 REF2VA 扩散模型 (INT8 ConvRot)"

((TOTAL_MODELS++))

# ======================================================================
# MiniMax-H3 LoRA 模型
# ======================================================================
echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}开始下载 LoRA 模型${NC}"
echo -e "${BLUE}==================================================${NC}\n"

# 1. KungFu_MiniMaxH3_LoRA_step4500.safetensors
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/301fe157279dbbe04915e3ec6b3f13ad9e14ede33a6c007c7dc1d33152c0064f" \
    "KungFu_MiniMaxH3_LoRA_step4500.safetensors" \
    "/workspace/models/loras" \
    "KungFu MiniMaxH3 LoRA (step4500)"

((TOTAL_MODELS++))

# 2. minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/5cae6df40a06ea825f85fc8876c9ea1c9692c833a9af07bb8b3bac9ce2a71bac" \
    "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors" \
    "/workspace/models/loras" \
    "MiniMax-H3 FL2V Turbo 4步 v0.1 (768p SLA)"

((TOTAL_MODELS++))

# 3. minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert.safetensors
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/35a4465a7911c8917c8ba3cfd2991270184474dc2fa9122bbdecaff15d7c1b39" \
    "minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert.safetensors" \
    "/workspace/models/loras" \
    "MiniMax-H3 FL2V Turbo 4步 v0.1 (alpha8-T8)"

((TOTAL_MODELS++))

# 4. minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc" \
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors" \
    "/workspace/models/loras" \
    "MiniMax-H3 FL2V Turbo 4步 v1.0 (768p)"

((TOTAL_MODELS++))

# 5. minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e" \
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" \
    "/workspace/models/loras" \
    "MiniMax-H3 FL2V Turbo 8步 v1.0"

((TOTAL_MODELS++))

# 6. minimax_h3_turbo_4步加速_comfyui.safetensors
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/35946f9f2957c2766e28b627c88169535249dd07a3040ce3c2c8c99951fdbc7b" \
    "minimax_h3_turbo_4步加速_comfyui.safetensors" \
    "/workspace/models/loras" \
    "MiniMax-H3 Turbo 4步加速"

((TOTAL_MODELS++))

# 7. wushu_action_h3_lora_v4_2000_full.safetensors
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/1e3367aadc25f9b06f7014bb80707ca5844615c3ec2c6c520a6190915db97ac5" \
    "wushu_action_h3_lora_v4_2000_full.safetensors" \
    "/workspace/models/loras" \
    "武术动作 LoRA v4 (2000 full)"

((TOTAL_MODELS++))

# 8. wushu_action_h3_lora_v5_3000_pruned.safetensors
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/136c16924f7413f0edb1c439f29c1e4c9da13d2cf07117c4e46556691193aa16" \
    "wushu_action_h3_lora_v5_3000_pruned.safetensors" \
    "/workspace/models/loras" \
    "武术动作 LoRA v5 (3000 pruned)"

((TOTAL_MODELS++))

# 9. minimax_h3_turbo_4step.safetensors
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/c468c61ebf715699b5a710fed016654e4a244195ce2e1c4f9f84dc63f82da905?name=minimax_h3_turbo_4step.safetensors" \
    "minimax_h3_turbo_4step.safetensors" \
    "/workspace/models/loras" \
    "MiniMax-H3 Turbo 4步"

((TOTAL_MODELS++))

# 10. minimax_h3_fl2v_lightx2v_turbo_4step_v1.0_768p_resized_avg_rank_31_bf16.safetensors
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/9515eee9f642aa0e7fcc401f56d408ef2d6388f81881fe50bddded8220870a4d?name=minimax_h3_fl2v_lightx2v_turbo_4step_v1.0_768p_resized_avg_rank_31_bf16.safetensors" \
    "minimax_h3_fl2v_lightx2v_turbo_4step_v1.0_768p_resized_avg_rank_31_bf16.safetensors" \
    "/workspace/models/loras" \
    "MiniMax-H3 FL2V LightX2V Turbo 4步 v1.0 (768p, rank31)"

((TOTAL_MODELS++))

# 11. minimax_h3_fl2v_lightx2v_turbo_8step_v1.0_resized_avg_rank_24_bf16.safetensors
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/8e05b7b982c3aff7deb692a188c8a8d8acaeff8a12abfe1aeac822fb8ee3f0b7?name=minimax_h3_fl2v_lightx2v_turbo_8step_v1.0_resized_avg_rank_24_bf16.safetensors" \
    "minimax_h3_fl2v_lightx2v_turbo_8step_v1.0_resized_avg_rank_24_bf16.safetensors" \
    "/workspace/models/loras" \
    "MiniMax-H3 FL2V LightX2V Turbo 8步 v1.0 (rank24)"

((TOTAL_MODELS++))

# 12. minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/9ea3bd3a6aac22994153e294cf1ecab0a8766fc0f8d056ace645a01d1a6a4daf?name=minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors" \
    "minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors" \
    "/workspace/models/loras" \
    "MiniMax-H3 REF2V LightX2V Turbo 4步 v0.1 (rank20)"

((TOTAL_MODELS++))

# 13. minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c?name=minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" \
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" \
    "/workspace/models/loras" \
    "MiniMax-H3 REF2V Turbo 4步 v0.1 (ComfyUI)"

((TOTAL_MODELS++))

# ======================================================================
# Latent Upscale 模型（潜空间放大）
# ======================================================================
echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}开始下载 Latent Upscale 模型${NC}"
echo -e "${BLUE}==================================================${NC}\n"

# 14. Latent Upscale Model (FP16)
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2?name=minimax_h3_latent_upscaler_3d_fp16.safetensors" \
    "minimax_h3_latent_upscaler_3d_fp16.safetensors" \
    "/workspace/models/latent_upscale_models" \
    "MiniMax-H3 Latent Upscaler 3D (FP16)"

((TOTAL_MODELS++))

# 15. Latent Upscale Model (FP32)
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/d54716fe021a0b653ef07943e3ea4f704a26f3f60a23f905cf45dc87de149f68?name=minimax_h3_latent_upscaler_3d_fp32.pth" \
    "minimax_h3_latent_upscaler_3d_fp32.pth" \
    "/workspace/models/latent_upscale_models" \
    "MiniMax-H3 Latent Upscaler 3D (FP32)"

((TOTAL_MODELS++))

# ======================================================================
# 真实电影质感 LoRA
# ======================================================================
echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}开始下载真实电影质感 LoRA${NC}"
echo -e "${BLUE}==================================================${NC}\n"

# 16. LoRA (真实电影质感)
download_model \
    "https://cnb.cool/wenxinqiling/H3-lora/-/lfs/51dda79218ea126cbb2e08f3a6d9cc595e2224f4977d7618061954043a8bafcf?name=Minimax%20H3%E7%9C%9F%E5%AE%9E%E7%94%B5%E5%BD%B1%E8%B4%A8%E6%84%9FV0.1.safetensors" \
    "Minimax H3真实电影质感V0.1.safetensors" \
    "/workspace/models/loras" \
    "MiniMax-H3 真实电影质感 LoRA (V0.1)"

((TOTAL_MODELS++))

# ======================================================================
# 下载完成统计
# ======================================================================
echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}下载完成！统计信息：${NC}"
echo -e "${BLUE}==================================================${NC}"

echo -e "\n${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}📊 下载统计 ${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "📦 总计模型数: ${TOTAL_MODELS}"
echo -e "✅ 成功下载: ${SUCCESS_MODELS}"
echo -e "❌ 失败下载: ${FAILED_MODELS}"

if [ $FAILED_MODELS -eq 0 ]; then
    echo -e "\n🎉 ${GREEN}所有模型下载成功！${NC}"
else
    echo -e "\n⚠️ ${YELLOW}有 ${FAILED_MODELS} 个模型下载失败${NC}"
fi

echo -e "\n${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}📁 模型目录结构 ${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "🎨 VAE: /workspace/models/vae"
echo -e "📝 Text Encoders: /workspace/models/text_encoders"
echo -e "🧠 Diffusion Models: /workspace/models/diffusion_models"
echo -e "🔧 LoRAs: /workspace/models/loras"
echo -e "📐 Latent Upscale Models: /workspace/models/latent_upscale_models"

echo -e "\n${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}📋 下载模型列表 ${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"

for model_info in "${MODEL_LIST[@]}"; do
    echo -e "$model_info"
done

echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}脚本执行完成！${NC}"
echo -e "${BLUE}==================================================${NC}"

# 如果有失败的下载，以错误状态退出
if [ $FAILED_MODELS -gt 0 ]; then
    exit 1
else
    exit 0
fi