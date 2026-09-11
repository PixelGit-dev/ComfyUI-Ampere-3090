#!/bin/bash

# ======================================================================
# LTX-2.3 附加 LoRA 模型下载脚本（独立版）
# 功能：下载您最新提供的所有 LTX-2.3 LoRA 模型
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

echo -e "${BLUE}正在创建目录 /workspace/models/loras ...${NC}"
mkdir -p "/workspace/models/loras"
echo -e "${GREEN}目录创建完成！${NC}\n"

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}开始下载 LTX-2.3 LoRA 模型（共 27 个）${NC}"
echo -e "${BLUE}==================================================${NC}\n"

# 下载函数
download_lora() {
    local url=$1
    local filename=$2
    echo -e "${YELLOW}正在下载 ${filename} ...${NC}"
    aria2c -x 4 -s 4 -c -d "/workspace/models/loras" -o "$filename" "$url"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ ${filename} 下载成功${NC}"
        STATUS_LIST+=("✅ $filename")
        ((SUCCESS++))
    else
        echo -e "${RED}❌ ${filename} 下载失败${NC}"
        STATUS_LIST+=("❌ $filename")
        ((FAILED++))
    fi
    ((TOTAL++))
}

# -------------------- 开始下载 --------------------
download_lora \
    "https://cnb.cool/ai-models/Kijai/LTX2.3_comfy/-/lfs/31e0c0195fb841bf31af78e8b60858f489e87ddcea4a5239abc80943da65e3ac" \
    "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors"

download_lora \
    "https://cnb.cool/ai-models/TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments/-/lfs/e383540ea8aa5b03dddc45023b87a4466278b6ba89f0a511dc3d93b6464cb0b9" \
    "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil36.safetensors"

download_lora \
    "https://cnb.cool/ai-models/SulphurAI/Sulphur-2-base/-/lfs/e970f64a2ce5469491fb1714a3fa72c8b606fa82affff0531e836dbc91d31f34" \
    "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/LTX-2.3/-/lfs/f5d4953f3386197a4b4f5abdb17616ff256171e8075c111d6e7d2dfa6e823b3a" \
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/LTX-2.3/-/lfs/2943ab994f3c9d88052e5a2a34cca14e4a2dfc36b1d8c407931d52d5c25dd72b" \
    "ltx-2.3-22b-distilled-lora-384.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Kijai/LTX2.3_comfy/-/lfs/289441f530ca30520bd5e1d4f34b8437f75d6c66f9b40e32b3824e225dd96325" \
    "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"

download_lora \
    "https://cnb.cool/ai-models/TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments/-/lfs/10184866c06bec45b724135699af0bf526c8582d66b01eb44721a4e20ce780f5" \
    "ltx-2.3-22b-distilled-lora-fro90_ceil72.safetensors"

download_lora \
    "https://cnb.cool/ai-models/drbaph/LTX-2.3-FP8/-/lfs/edc0d231800ef5988e09477ccb953aef336abe8d53336d3353a2631562c4f456" \
    "ltx-2.3-22b-distilled-lora-resized_dynamic_rank_159_fro09_bf16.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/ltx-2.3-22b-ic-lora-day-to-night/-/lfs/984771a6aae278c42538984a22a74b4a57549463b47483e8ebd0d40d02aa9f3a" \
    "ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors"

download_lora \
    "https://cnb.cool/ai-models/drbaph/LTX-2.3-FP8/-/lfs/47c68fc5cc48e926756ee7846e8337b41a7bd28fffdc56251eeac802c5c56c53" \
    "ltx-2.3-22b-distilled-lora-resized_dynamic_rank_208_fro095_bf16.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/ltx-2.3-22b-ic-lora-colorization/-/lfs/092fb0e1f65f6a750586f1c0994a89c4fd85fe43fecb823f7f59463338da4e13" \
    "ltx-2.3-22b-ic-lora-colorization-0.9.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/ltx-2.3-22b-ic-lora-cross-eyed/-/lfs/afce58f60fb7e47d56a85443b0aa5c09e02c3b10e847310af52935827b727cc8" \
    "ltx-2.3-22b-ic-lora-cross-eyed-0.9.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/ltx-2.3-22b-ic-lora-deblur/-/lfs/dcdd73b57c2c4d5f5bc6535e825f4758b654a583bc991caa50c6809b6990b4ab" \
    "ltx-2.3-22b-ic-lora-deblur-0.9.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/ltx-2.3-22b-ic-lora-decompression/-/lfs/cf76f7cb10e9605dd2322ae0b0eba68438837017a614ef22818891e87f255d0d" \
    "ltx-2.3-22b-ic-lora-decompression-0.9.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/-/lfs/c56bfa0f2e4461a8b2f318f494c61c5bf97f462f2220e31ece93ea7851ca871e" \
    "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/-/lfs/78bffa6049bae2649a4365ec8769db88052c21348d643e8fc1ce6d483d994c5b" \
    "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/ltx-2.3-22b-ic-lora-in-outpainting/-/lfs/73dd0841c0d4f0eb26fb1f017781b841b2752021944ac5ecefe57917f6dae6b5" \
    "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/ltx-2.3-22b-ic-lora-ingredients/-/lfs/515e4e139001ac6282357a5b35372e42e98b3affd5fcc886a52242abeed19559" \
    "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/ltx-2.3-22b-ic-lora-instant-shave/-/lfs/04231f1befeda653ab98081dd0f58114b9bc71782cdc687239a1610a39a9b0a2" \
    "ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub/-/lfs/fc415b12cb639e78511bc264f85080c2f7b188e334c1d9fade76b310e2bc419c" \
    "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control/-/lfs/ee256c5da0850f574dd1da1694b40363860a1b76746c5ff660ee64190891bed0" \
    "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"

download_lora \
    "https://cnb.cool/ai-models/oumoumad-ai/LTX-2.3-22b-IC-LoRA-Outpaint/-/lfs/32c5d3e0649aa4e89b192319f3c79460dfd2319d2859ca11fa6f88e983a81665" \
    "ltx-2.3-22b-ic-lora-outpaint.safetensors"

download_lora \
    "https://cnb.cool/ai-models/oumoumad-ai/LTX-2.3-22b-IC-LoRA-ReFocus/-/lfs/e9123df9b3ca887355e10b7bca81bbe9141b060c9e953503b02c5f77d4bedfc9" \
    "ltx-2.3-22b-ic-lora-refocus.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/-/lfs/a1b888a87f661d27f08b394ae559e8e1050be33900bcc36a5cdf659e48f88d18" \
    "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Lightricks/ltx-2.3-22b-ic-lora-water-simulation/-/lfs/19ad9007d78d88d19623fdeb9245298565bb4f30aebe829ad5b65fa7f8e52e87" \
    "ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Comfy-Org/ltx-2.3/-/lfs/12e6be9c52c83047cb71470c1c63b64ebc8b13dc0dfa34400f6576ba2cd88ecf" \
    "ltx-2.3-id-lora-celebvhq-3k.safetensors"

download_lora \
    "https://cnb.cool/ai-models/Comfy-Org/ltx-2.3/-/lfs/e5af73441743b4852f228b03e444888dff3da80d2666033af2367ab7bda6d8b9" \
    "ltx-2.3-id-lora-talkvid-3k.safetensors"

# -------------------- 统计信息 --------------------
echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}下载完成！统计信息：${NC}"
echo -e "${BLUE}==================================================${NC}"
echo -e "\n${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}📊 下载统计${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "📦 总计 LoRA 数: ${TOTAL}"
echo -e "✅ 成功下载: ${SUCCESS}"
echo -e "❌ 失败下载: ${FAILED}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n🎉 ${GREEN}所有 LoRA 下载成功！${NC}"
else
    echo -e "\n⚠️ ${YELLOW}有 ${FAILED} 个 LoRA 下载失败${NC}"
fi

echo -e "\n${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}📁 存放目录${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "🎭  LoRAs: /workspace/models/loras"

echo -e "\n${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}📋 下载列表${NC}"
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