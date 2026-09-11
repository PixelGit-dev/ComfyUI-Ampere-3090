#!/bin/bash

# ======================================================================
# LTX-2.3 模型下载脚本
# 功能：下载所有 LTX-2.3 相关模型文件
# ======================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TOTAL=0
SUCCESS=0
FAILED=0
declare -a STATUS_LIST=()

echo -e "${BLUE}正在创建目录结构...${NC}"
mkdir -p "/workspace/models/checkpoints"
mkdir -p "/workspace/models/latent_upscale_models"
mkdir -p "/workspace/models/loras"
echo -e "${GREEN}目录创建完成！${NC}\n"

# -------------------- Checkpoints --------------------
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}开始下载 Checkpoints${NC}"
echo -e "${BLUE}==================================================${NC}\n"

cmd="aria2c -x 4 -s 4 -c -d \"/workspace/models/checkpoints\" -o \"ltx-2.3-22b-dev.safetensors\" \"https://cnb.cool/ai-models/Lightricks/LTX-2.3/-/lfs/7ab7225325bc403448ea84b6db2269811a880e5118cd2ee2b6282a93d585016f\""
echo -e "${YELLOW}下载: ltx-2.3-22b-dev.safetensors${NC}"
eval $cmd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}成功${NC}"; ((SUCCESS++)); STATUS_LIST+=("✅ ltx-2.3-22b-dev.safetensors")
else
    echo -e "${RED}失败${NC}"; ((FAILED++)); STATUS_LIST+=("❌ ltx-2.3-22b-dev.safetensors")
fi
((TOTAL++))

cmd="aria2c -x 4 -s 4 -c -d \"/workspace/models/checkpoints\" -o \"ltx-2.3-22b-distilled-1.1.safetensors\" \"https://cnb.cool/ai-models/Lightricks/LTX-2.3/-/lfs/b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc\""
echo -e "${YELLOW}下载: ltx-2.3-22b-distilled-1.1.safetensors${NC}"
eval $cmd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}成功${NC}"; ((SUCCESS++)); STATUS_LIST+=("✅ ltx-2.3-22b-distilled-1.1.safetensors")
else
    echo -e "${RED}失败${NC}"; ((FAILED++)); STATUS_LIST+=("❌ ltx-2.3-22b-distilled-1.1.safetensors")
fi
((TOTAL++))

cmd="aria2c -x 4 -s 4 -c -d \"/workspace/models/checkpoints\" -o \"ltx-2.3-22b-distilled.safetensors\" \"https://cnb.cool/ai-models/Lightricks/LTX-2.3/-/lfs/14409a4d1337a8ded02fa87fb895b17a91ab2c6588f7cc3352e624ff18a689bf\""
echo -e "${YELLOW}下载: ltx-2.3-22b-distilled.safetensors${NC}"
eval $cmd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}成功${NC}"; ((SUCCESS++)); STATUS_LIST+=("✅ ltx-2.3-22b-distilled.safetensors")
else
    echo -e "${RED}失败${NC}"; ((FAILED++)); STATUS_LIST+=("❌ ltx-2.3-22b-distilled.safetensors")
fi
((TOTAL++))

cmd="aria2c -x 4 -s 4 -c -d \"/workspace/models/checkpoints\" -o \"ltx-2.3-22b-dev-fp8.safetensors\" \"https://cnb.cool/ai-models/Lightricks/LTX-2.3-fp8/-/lfs/28606c5b5a06ce56f896d4dfcb20f212739e07a68fbe48e53638188449d26450\""
echo -e "${YELLOW}下载: ltx-2.3-22b-dev-fp8.safetensors${NC}"
eval $cmd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}成功${NC}"; ((SUCCESS++)); STATUS_LIST+=("✅ ltx-2.3-22b-dev-fp8.safetensors")
else
    echo -e "${RED}失败${NC}"; ((FAILED++)); STATUS_LIST+=("❌ ltx-2.3-22b-dev-fp8.safetensors")
fi
((TOTAL++))

# -------------------- Latent Upscale Models --------------------
echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}开始下载 Latent Upscale Models${NC}"
echo -e "${BLUE}==================================================${NC}\n"

cmd="aria2c -x 4 -s 4 -c -d \"/workspace/models/latent_upscale_models\" -o \"ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors\" \"https://cnb.cool/ai-models/Lightricks/LTX-2.3/-/lfs/b2b3193e68cb04b4701e1d59bec4ed5d5e3e84506d9a42d5a129d37d39823df7\""
echo -e "${YELLOW}下载: ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors${NC}"
eval $cmd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}成功${NC}"; ((SUCCESS++)); STATUS_LIST+=("✅ ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors")
else
    echo -e "${RED}失败${NC}"; ((FAILED++)); STATUS_LIST+=("❌ ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors")
fi
((TOTAL++))

cmd="aria2c -x 4 -s 4 -c -d \"/workspace/models/latent_upscale_models\" -o \"ltx-2.3-spatial-upscaler-x2-1.0.safetensors\" \"https://cnb.cool/ai-models/Lightricks/LTX-2.3/-/lfs/93800de87dbc448b5b31f3c5c3a1579ba6335151de061a564f6f026b0fc770ad\""
echo -e "${YELLOW}下载: ltx-2.3-spatial-upscaler-x2-1.0.safetensors${NC}"
eval $cmd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}成功${NC}"; ((SUCCESS++)); STATUS_LIST+=("✅ ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
else
    echo -e "${RED}失败${NC}"; ((FAILED++)); STATUS_LIST+=("❌ ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
fi
((TOTAL++))

cmd="aria2c -x 4 -s 4 -c -d \"/workspace/models/latent_upscale_models\" -o \"ltx-2.3-spatial-upscaler-x2-1.1.safetensors\" \"https://cnb.cool/ai-models/Lightricks/LTX-2.3/-/lfs/5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed\""
echo -e "${YELLOW}下载: ltx-2.3-spatial-upscaler-x2-1.1.safetensors${NC}"
eval $cmd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}成功${NC}"; ((SUCCESS++)); STATUS_LIST+=("✅ ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
else
    echo -e "${RED}失败${NC}"; ((FAILED++)); STATUS_LIST+=("❌ ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
fi
((TOTAL++))

cmd="aria2c -x 4 -s 4 -c -d \"/workspace/models/latent_upscale_models\" -o \"ltx-2.3-temporal-upscaler-x2-1.0.safetensors\" \"https://cnb.cool/ai-models/Lightricks/LTX-2.3/-/lfs/2bc3300f2b3c3c1834d72164fbf13a3b9fd73e5a741e8a2c3f4035f89a75c3fe\""
echo -e "${YELLOW}下载: ltx-2.3-temporal-upscaler-x2-1.0.safetensors${NC}"
eval $cmd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}成功${NC}"; ((SUCCESS++)); STATUS_LIST+=("✅ ltx-2.3-temporal-upscaler-x2-1.0.safetensors")
else
    echo -e "${RED}失败${NC}"; ((FAILED++)); STATUS_LIST+=("❌ ltx-2.3-temporal-upscaler-x2-1.0.safetensors")
fi
((TOTAL++))

# -------------------- LoRAs --------------------
echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}开始下载 LoRAs${NC}"
echo -e "${BLUE}==================================================${NC}\n"

cmd="aria2c -x 4 -s 4 -c -d \"/workspace/models/loras\" -o \"ltx-2.3-22b-distilled-lora-384-1.1.safetensors\" \"https://cnb.cool/ai-models/Lightricks/LTX-2.3/-/lfs/f5d4953f3386197a4b4f5abdb17616ff256171e8075c111d6e7d2dfa6e823b3a\""
echo -e "${YELLOW}下载: ltx-2.3-22b-distilled-lora-384-1.1.safetensors${NC}"
eval $cmd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}成功${NC}"; ((SUCCESS++)); STATUS_LIST+=("✅ ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
else
    echo -e "${RED}失败${NC}"; ((FAILED++)); STATUS_LIST+=("❌ ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
fi
((TOTAL++))

cmd="aria2c -x 4 -s 4 -c -d \"/workspace/models/loras\" -o \"ltx-2.3-22b-distilled-lora-384.safetensors\" \"https://cnb.cool/ai-models/Lightricks/LTX-2.3/-/lfs/2943ab994f3c9d88052e5a2a34cca14e4a2dfc36b1d8c407931d52d5c25dd72b\""
echo -e "${YELLOW}下载: ltx-2.3-22b-distilled-lora-384.safetensors${NC}"
eval $cmd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}成功${NC}"; ((SUCCESS++)); STATUS_LIST+=("✅ ltx-2.3-22b-distilled-lora-384.safetensors")
else
    echo -e "${RED}失败${NC}"; ((FAILED++)); STATUS_LIST+=("❌ ltx-2.3-22b-distilled-lora-384.safetensors")
fi
((TOTAL++))

cmd="aria2c -x 4 -s 4 -c -d \"/workspace/models/loras\" -o \"ltx-2.3-id-lora-celebvhq-3k.safetensors\" \"https://cnb.cool/ai-models/Comfy-Org/ltx-2.3/-/lfs/12e6be9c52c83047cb71470c1c63b64ebc8b13dc0dfa34400f6576ba2cd88ecf\""
echo -e "${YELLOW}下载: ltx-2.3-id-lora-celebvhq-3k.safetensors${NC}"
eval $cmd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}成功${NC}"; ((SUCCESS++)); STATUS_LIST+=("✅ ltx-2.3-id-lora-celebvhq-3k.safetensors")
else
    echo -e "${RED}失败${NC}"; ((FAILED++)); STATUS_LIST+=("❌ ltx-2.3-id-lora-celebvhq-3k.safetensors")
fi
((TOTAL++))

cmd="aria2c -x 4 -s 4 -c -d \"/workspace/models/loras\" -o \"ltx-2.3-id-lora-talkvid-3k.safetensors\" \"https://cnb.cool/ai-models/Comfy-Org/ltx-2.3/-/lfs/e5af73441743b4852f228b03e444888dff3da80d2666033af2367ab7bda6d8b9\""
echo -e "${YELLOW}下载: ltx-2.3-id-lora-talkvid-3k.safetensors${NC}"
eval $cmd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}成功${NC}"; ((SUCCESS++)); STATUS_LIST+=("✅ ltx-2.3-id-lora-talkvid-3k.safetensors")
else
    echo -e "${RED}失败${NC}"; ((FAILED++)); STATUS_LIST+=("❌ ltx-2.3-id-lora-talkvid-3k.safetensors")
fi
((TOTAL++))

cmd="aria2c -x 4 -s 4 -c -d \"/workspace/models/loras\" -o \"ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors\" \"https://cnb.cool/ai-models/Comfy-Org/ltx-2.3/-/lfs/31e0c0195fb841bf31af78e8b60858f489e87ddcea4a5239abc80943da65e3ac\""
echo -e "${YELLOW}下载: ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors${NC}"
eval $cmd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}成功${NC}"; ((SUCCESS++)); STATUS_LIST+=("✅ ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors")
else
    echo -e "${RED}失败${NC}"; ((FAILED++)); STATUS_LIST+=("❌ ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors")
fi
((TOTAL++))

# -------------------- 统计信息 --------------------
echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}下载完成！统计信息：${NC}"
echo -e "${BLUE}==================================================${NC}"
echo -e "\n${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}📊 下载统计${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "📦 总计模型数: ${TOTAL}"
echo -e "✅ 成功下载: ${SUCCESS}"
echo -e "❌ 失败下载: ${FAILED}"
if [ $FAILED -eq 0 ]; then
    echo -e "\n🎉 ${GREEN}所有模型下载成功！${NC}"
else
    echo -e "\n⚠️ ${YELLOW}有 ${FAILED} 个模型下载失败${NC}"
fi
echo -e "\n${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}📁 模型目录结构${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "🏷️  Checkpoints: /workspace/models/checkpoints"
echo -e "🔼  Upscale: /workspace/models/latent_upscale_models"
echo -e "🎭  LoRAs: /workspace/models/loras"
echo -e "\n${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}📋 下载模型列表${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
for item in "${STATUS_LIST[@]}"; do
    echo -e "$item"
done
echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}脚本执行完成！${NC}"
echo -e "${BLUE}==================================================${NC}"

if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi