//
// Copyright (c) 2009-2013 Mikko Mononen memon@inside.org
// Port of _gl.h to _DE.cpp by dtcxzyw
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

// SHA=2bead03bea43b2418060aaa154f972829995e663

#include "nanovg_DE.hpp"
#include <DiligentCore/Graphics/GraphicsTools/interface/GraphicsUtilities.h>
#include <DiligentCore/Graphics/GraphicsTools/interface/MapHelper.hpp>
#include <DiligentCore/Graphics/GraphicsTools/interface/ShaderMacroHelper.hpp>
#include <algorithm>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <vector>

struct Uniform final {
    float scissorMat[12];  // matrices are actually 3 vec4s
    float paintMat[12];
    struct NVGcolor innerCol;
    struct NVGcolor outerCol;
    float scissorExt[2];
    float scissorScale[2];
    float extent[2];
    float radius;
    float feather;
    float strokeMult;
    float strokeThr;
    int texType;
    int type;
};

static_assert(sizeof(Uniform) == 11 * 4 * sizeof(float));

struct NVGDEPipelineState final {
    DE::RefCntAutoPtr<DE::IPipelineState> PSO;
    DE::RefCntAutoPtr<DE::IShaderResourceBinding> SRB;
};

struct NVGDETexture final {
    DE::RefCntAutoPtr<DE::ITexture> tex;
    DE::RefCntAutoPtr<DE::ITextureView> texView;
    int flags, type;
};

template <typename T>
class ResourceManager final {
private:
    std::vector<T> mSlots;
    std::vector<int> mFree;

public:
    int alloc(const T& val) {
        int id = -1;
        if(mFree.size()) {
            id = mFree.back();
            mFree.pop_back();
            mSlots[id] = val;
        } else {
            id = static_cast<int>(mSlots.size());
            mSlots.push_back(val);
        }
        return id;
    }
    T& get(int id) {
        return mSlots[id];
    }
    void free(int id) {
        T empty;
        std::swap(empty, mSlots[id]);
        mFree.push_back(id);
    }
};

template <typename Key, typename T, typename Hasher = std::hash<Key>>
class StateCache final {
private:
    std::unordered_map<size_t, T> mMap;
    std::function<T(const Key&)> mGenerator;
    Hasher mHasher;

public:
    void setGenerator(const std::function<T(const Key&)>& gen) {
        mGenerator = gen;
    }
    T& get(const Key& key) {
        size_t hv = mHasher(key);
        auto iter = mMap.find(hv);
        if(iter != mMap.end())
            return iter->second;
        T& res = mMap[hv];
        res = mGenerator(key);
        return res;
    }
};

enum class Type { fill, convexFill, stroke, triangles };
enum class ShaderType {
    NSVG_SHADER_FILLGRAD,
    NSVG_SHADER_FILLIMG,
    NSVG_SHADER_SIMPLE,
    NSVG_SHADER_IMG
};

struct BlendFactor final {
    DE::BLEND_FACTOR srcRGB, dstRGB, srcAlpha, dstAlpha;
};

struct NVGDEPath {
    int fillOffset;
    int fillCount;
    int strokeOffset;
    int strokeCount;
};

struct NVGDECall final {
    Type type;
    int image, triangleOffset, triangleCount, pathOffset, pathCount;
    int uniform[2];
    BlendFactor blendFactor;
};

struct StateHasher final {
    size_t operator()(const DE::PipelineStateDesc& state) const {
        // FNV-1a
        size_t res = static_cast<size_t>(
            sizeof(size_t) == 4 ? 2166136261ULL : 14695981039346656037ULL);
        auto push = [&res](const auto& x) {
            constexpr size_t prime = static_cast<size_t>(
                sizeof(size_t) == 4 ? 16777619ULL : 1099511628211ULL);
            auto ptr = reinterpret_cast<const unsigned char*>(&x);
            auto end = ptr + sizeof(x);
            while(ptr != end) {
                res = (res ^ (*ptr)) * prime;
                ++ptr;
            }
        };
        auto&& rasterizer = state.GraphicsPipeline.RasterizerDesc;
        push(rasterizer.CullMode);

        auto&& blend = state.GraphicsPipeline.BlendDesc.RenderTargets[0];
        if(blend.BlendEnable) {
            push(blend.SrcBlend);
            push(blend.DestBlend);
            push(blend.SrcBlendAlpha);
            push(blend.DestBlendAlpha);
            push(blend.RenderTargetWriteMask);
        }
        auto&& depthStencil = state.GraphicsPipeline.DepthStencilDesc;
        if(depthStencil.StencilEnable) {
            push(depthStencil.StencilReadMask);
            // push(depthStencil.StencilWriteMask); //read==write
            push(depthStencil.FrontFace);
            push(depthStencil.BackFace);
        }
        push(state.GraphicsPipeline.PrimitiveTopology);

        // sampler's flag
        push(state.ResourceLayout.NumStaticSamplers);
        return res;
    }
};

struct NVGDEContext final {
    // Interface and Configuration
    DE::IRenderDevice* device;
    DE::IDeviceContext* context;
    DE::SampleDesc MSAA;
    int flags;
    DE::TEXTURE_FORMAT colorFormat, depthFormat;

    // Pipeline static
    DE::RefCntAutoPtr<DE::IShader> pVS, pPS;
    DE::PipelineStateDesc defaultState;

    // Pipeline and SRV dynamic
    StateCache<DE::PipelineStateDesc, NVGDEPipelineState, StateHasher> pipeline;
    ResourceManager<NVGDETexture> texture;
    DE::RefCntAutoPtr<DE::IBuffer> viewConstant;
    DE::RefCntAutoPtr<DE::IBuffer> triangleFansIndex;
    DE::RefCntAutoPtr<DE::IBuffer> vertBuffer;
    DE::RefCntAutoPtr<DE::IBuffer> uniformBuffer, uidBuffer;
    DE::RefCntAutoPtr<DE::IBufferView> uniformSRV;

    // Host Buffer
    std::vector<NVGvertex> vertBufferHost;
    std::vector<NVGDEPath> pathBuffer;
    std::vector<Uniform> uniformBufferHost;
    std::vector<NVGDECall> calls;

    // IndirectDraw
    bool useIndirectDraw;
    DE::RefCntAutoPtr<DE::IBuffer> indirectCall;
};

static const char* vertShader = R"(
cbuffer VSConstants {
    float2 viewSize;
};

struct PSInput {
    float4 pos : SV_POSITION;
    float4 arg : ARG;//pos texCoord
    int uid : UID;
};

struct VSInput {
    float4 arg : ATTRIB0;
    int uid : ATTRIB1;
};

void VS(in VSInput vin,out PSInput vout) {
    vout.pos = float4(2.0f*vin.arg.x/viewSize.x - 1.0f, 1.0f - 2.0f*vin.arg.y/viewSize.y, 0.0f, 1.0f);
    vout.arg = vin.arg;
    vout.uid = vin.uid;
}
)";

static const char* fragShader = R"(
Texture2D gTexture;
SamplerState gTextureSampler;
struct Uniform {
    float4 scissorMat[3];  // matrices are actually 3 vec4s
    float4 paintMat[3];
    float4 innerCol;
    float4 outerCol;
    float2 scissorExt;
    float2 scissorScale;
    float2 extent;
    float radius;
    float feather;
    float strokeMult;
    float strokeThr;
    int texType;
    int type;
};

StructuredBuffer<Uniform> gUniform;

struct PSInput {
    float4 pos : SV_POSITION;
    float4 arg : ARG;  // pos texCoord
    int uid : UID;
};
float3 mulx(float3 r0,float3 r1,float3 r2,float3 x) {
    return float3(dot(float3(r0.x,r1.x,r2.x),x),dot(float3(r0.y,r1.y,r2.y),x),dot(float3(r0.z,r1.z,r2.z),x));
}
float sdroundrect(float2 pt, float2 ext, float rad) {
    float2 ext2 = ext - float2(rad, rad);
    float2 d = abs(pt) - ext2;
    return min(max(d.x, d.y), 0.0f) + length(max(d, 0.0f)) - rad;
}
// Scissoring
float scissorMask(float3 sm1,float3 sm2,float3 sm3,float2 scissorExt,float2 scissorScale,float2 p) {
    float2 sc = (abs(mulx(sm1,sm2,sm3, float3(p, 1.0f)).xy) - scissorExt);
    sc = float2(0.5f, 0.5f) - sc * scissorScale;
    return clamp(sc.x, 0.0f, 1.0f) * clamp(sc.y, 0.0f, 1.0f);
}
#ifdef EDGE_AA
// Stroke - from [0..1] to clipped pyramid, where the slope is 1px.
float strokeMask(float strokeMult,float2 ftcoord) {
    return min(1.0f, (1.0f - abs(ftcoord.x * 2.0f - 1.0f)) * strokeMult) *
        min(1.0f, ftcoord.y);
}
#endif
float4 mix(float4 a, float4 b, float d) {
    return a * (1.0f - d) + b * d;
}
void PS(in PSInput pin, out float4 result : SV_TARGET) {
    float2 fpos = pin.arg.xy;
    float2 ftcoord = pin.arg.zw;    

    Uniform uni = gUniform[pin.uid];
    int type=uni.type,texType=uni.texType;

    float scissor = scissorMask(uni.scissorMat[0].xyz,uni.scissorMat[1].xyz,uni.scissorMat[2].xyz,uni.scissorExt,uni.scissorScale,fpos);

#ifdef EDGE_AA
    float strokeAlpha = strokeMask(uni.strokeMult,ftcoord);
    if(strokeAlpha < uni.strokeThr)
        discard;
#else
    float strokeAlpha = 1.0f;
#endif
    if(type == 0) {  // Gradient
        // Calculate gradient color using box gradient
        float2 pt = mulx(uni.paintMat[0].xyz,uni.paintMat[1].xyz,uni.paintMat[2].xyz, float3(fpos, 1.0f)).xy;
        float d =
            clamp(sdroundrect(pt, uni.extent, uni.radius) / uni.feather+0.5f,
                  0.0f, 1.0f);
        float4 color = mix(uni.innerCol, uni.outerCol, d);
        // Combine alpha
        color *= strokeAlpha * scissor;
        result = color;
    } else if(type == 1) {  // Image
        // Calculate color fron texture
        float2 pt = mulx(uni.paintMat[0].xyz,uni.paintMat[1].xyz,uni.paintMat[2].xyz , float3(fpos, 1.0f)).xy / uni.extent;
        float4 color = gTexture.Sample(gTextureSampler, pt);
        if(texType == 1)
            color = float4(color.xyz * color.w, color.w);
        else if(texType == 2)
            color = color.xxxx;
        // Apply color tint and alpha.
        color *= uni.innerCol;
        // Combine alpha
        color *= strokeAlpha * scissor;
        result = color;
    } else if(type == 2) {  // Stencil fill
        result = float4(1, 1, 1, 1);
    } else if (type == 3) {  // Textured tris
        float4 color = gTexture.Sample(gTextureSampler, ftcoord);
        if(texType == 1)
            color = float4(color.xyz * color.w, color.w);
        else if(texType == 2)
            color = color.xxxx;
        color *= scissor;
        result = color * uni.innerCol;
    }
}
)";
static int nvgde_renderCreateTexture(void* uptr, int type, int w, int h,
                                     int imageFlags, const unsigned char* data);
static int nearestPowerOf2(int siz) {
    int res = 1;
    while(res < siz)
        res *= 2;
    return res;
};
static void prepareTriangleFansIndexBuffer(NVGDEContext* context, int siz) {
    int oldSiz = context->triangleFansIndex ?
        context->triangleFansIndex->GetDesc().uiSizeInBytes /
            (3 * sizeof(int)) :
        0;
    if(siz <= oldSiz)
        return;
    siz = nearestPowerOf2(siz);
    context->triangleFansIndex.Release();
    DE::BufferDesc bufferDesc = {};
    bufferDesc.BindFlags = DE::BIND_INDEX_BUFFER;
    bufferDesc.Name = "NanoVG Triangle Fans Index Buffer";
    bufferDesc.uiSizeInBytes = siz * 3 * sizeof(int);
    bufferDesc.Usage = DE::USAGE_STATIC;

    std::vector<int> index(3 * siz);
    int* ptr = index.data();
    for(int i = 0; i < siz; ++i) {
        ++ptr;
        *ptr = i + 1;
        ++ptr;
        *ptr = i + 2;
        ++ptr;
    }

    DE::BufferData bufferData = {};
    bufferData.DataSize = bufferDesc.uiSizeInBytes;
    bufferData.pData = index.data();
    context->device->CreateBuffer(bufferDesc, &bufferData,
                                  &(context->triangleFansIndex));
}
static void prepareVertexBuffer(NVGDEContext* context, int siz) {
    int oldSiz = context->vertBuffer ?
        context->vertBuffer->GetDesc().uiSizeInBytes / sizeof(NVGvertex) :
        0;
    if(siz <= oldSiz)
        return;
    siz = nearestPowerOf2(siz);

    context->vertBuffer.Release();
    DE::BufferDesc bufferDesc = {};
    bufferDesc.BindFlags = DE::BIND_VERTEX_BUFFER;
    bufferDesc.Name = "NanoVG Vertex Buffer";
    bufferDesc.uiSizeInBytes = siz * sizeof(NVGvertex);
    bufferDesc.Usage = DE::USAGE_DEFAULT;
    bufferDesc.CPUAccessFlags = DE::CPU_ACCESS_NONE;
    // bufferDesc.Usage = DE::USAGE_DYNAMIC;
    // bufferDesc.CPUAccessFlags = DE::CPU_ACCESS_WRITE;

    context->device->CreateBuffer(bufferDesc, nullptr, &(context->vertBuffer));
}
static void prepareUniformBuffer(NVGDEContext* context, int siz) {
    int oldSiz = context->uniformBuffer ?
        context->uniformBuffer->GetDesc().uiSizeInBytes / sizeof(Uniform) :
        0;
    if(siz <= oldSiz)
        return;
    siz = nearestPowerOf2(siz);
    {
        context->uniformBuffer.Release();
        context->uniformSRV.Release();
        DE::BufferDesc bufferDesc = {};
        bufferDesc.BindFlags = DE::BIND_SHADER_RESOURCE;
        bufferDesc.Name = "NanoVG Uniform Array";
        bufferDesc.uiSizeInBytes = siz * sizeof(Uniform);
        bufferDesc.Usage = DE::USAGE_DEFAULT;
        bufferDesc.CPUAccessFlags = DE::CPU_ACCESS_NONE;
        bufferDesc.Mode = DE::BUFFER_MODE_STRUCTURED;
        bufferDesc.ElementByteStride = sizeof(Uniform);

        context->device->CreateBuffer(bufferDesc, nullptr,
                                      &(context->uniformBuffer));
        context->uniformSRV = context->uniformBuffer->GetDefaultView(
            DE::BUFFER_VIEW_SHADER_RESOURCE);
    }
    {
        context->uidBuffer.Release();
        DE::BufferDesc bufferDesc = {};
        bufferDesc.BindFlags = DE::BIND_VERTEX_BUFFER;
        bufferDesc.Name = "NanoVG Uniform Offset";
        bufferDesc.uiSizeInBytes = siz * sizeof(int);
        bufferDesc.Usage = DE::USAGE_STATIC;

        std::vector<int> offset(siz);
        std::iota(offset.begin(), offset.end(), 0);
        DE::BufferData data = {};
        data.DataSize = bufferDesc.uiSizeInBytes;
        data.pData = offset.data();

        context->device->CreateBuffer(bufferDesc, &data, &(context->uidBuffer));
    }
}
static DE::StaticSamplerDesc setSampler(int imageFlags) {
    DE::StaticSamplerDesc desc = {};
    desc.SamplerOrTextureName = "gTexture";
    desc.ShaderStages = DE::SHADER_TYPE_PIXEL;
    auto&& sd = desc.Desc;
    sd.Name = "NanoVG Sampler";
    sd.AddressU = (imageFlags & NVG_IMAGE_REPEATX ? DE::TEXTURE_ADDRESS_WRAP :
                                                    DE::TEXTURE_ADDRESS_CLAMP);
    sd.AddressV = (imageFlags & NVG_IMAGE_REPEATY ? DE::TEXTURE_ADDRESS_WRAP :
                                                    DE::TEXTURE_ADDRESS_CLAMP);
    sd.AddressW = DE::TEXTURE_ADDRESS_WRAP;

    if(imageFlags & NVG_IMAGE_NEAREST) {
        sd.MinFilter = sd.MagFilter = sd.MipFilter = DE::FILTER_TYPE_POINT;
    } else {
        sd.MinFilter = sd.MagFilter = sd.MipFilter = DE::FILTER_TYPE_LINEAR;
    }
    return desc;
}

static int nvgde_renderCreate(void* uptr) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);

    DE::PipelineStateDesc& PSODesc = context->defaultState;

    PSODesc.Name = "NanoVG Pipeline";
    PSODesc.IsComputePipeline = false;
    PSODesc.GraphicsPipeline.NumRenderTargets = 1;
    PSODesc.GraphicsPipeline.RTVFormats[0] = context->colorFormat;
    PSODesc.GraphicsPipeline.DSVFormat = context->depthFormat;

    PSODesc.GraphicsPipeline.PrimitiveTopology =
        DE::PRIMITIVE_TOPOLOGY_UNDEFINED;
    PSODesc.GraphicsPipeline.RasterizerDesc.CullMode = DE::CULL_MODE_BACK;
    PSODesc.GraphicsPipeline.RasterizerDesc.FrontCounterClockwise = true;
    PSODesc.GraphicsPipeline.RasterizerDesc.ScissorEnable = false;

    DE::RenderTargetBlendDesc& blendDesc =
        PSODesc.GraphicsPipeline.BlendDesc.RenderTargets[0];
    blendDesc.BlendEnable = true;
    blendDesc.RenderTargetWriteMask = DE::COLOR_MASK_ALL;

    DE::DepthStencilStateDesc& SDesc =
        PSODesc.GraphicsPipeline.DepthStencilDesc;
    SDesc.StencilEnable = false;
    SDesc.FrontFace.StencilFunc = SDesc.BackFace.StencilFunc =
        DE::COMPARISON_FUNC_ALWAYS;
    SDesc.FrontFace.StencilDepthFailOp = SDesc.FrontFace.StencilFailOp =
        SDesc.FrontFace.StencilPassOp = SDesc.BackFace.StencilDepthFailOp =
            SDesc.BackFace.StencilFailOp = SDesc.BackFace.StencilPassOp =
                DE::STENCIL_OP_KEEP;
    SDesc.StencilReadMask = SDesc.StencilWriteMask = 0xff;
    context->context->SetStencilRef(0);

    SDesc.DepthEnable = false;

    DE::ShaderCreateInfo shaderCI = {};
    shaderCI.SourceLanguage = DE::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCI.UseCombinedTextureSamplers = true;
    shaderCI.CombinedSamplerSuffix = "Sampler";

    {
        shaderCI.Desc.ShaderType = DE::SHADER_TYPE_VERTEX;
        shaderCI.EntryPoint = "VS";
        shaderCI.Desc.Name = "NanoVG VS";
        shaderCI.Source = vertShader;
        context->device->CreateShader(shaderCI, &(context->pVS));
    }

    DE::RefCntAutoPtr<DE::IShader> pPS;
    {
        shaderCI.Desc.ShaderType = DE::SHADER_TYPE_PIXEL;
        shaderCI.EntryPoint = "PS";
        shaderCI.Desc.Name = "NanoVG PS";
        shaderCI.Source = fragShader;
        DE::ShaderMacroHelper macros;
        if(context->flags & NVGCreateFlags::NVG_ANTIALIAS) {
            macros.AddShaderMacro("EDGE_AA", true);
            shaderCI.Macros = macros;
        }
        context->device->CreateShader(shaderCI, &(context->pPS));
    }

    PSODesc.GraphicsPipeline.SmplDesc = context->MSAA;

    PSODesc.GraphicsPipeline.pVS = context->pVS;
    PSODesc.GraphicsPipeline.pPS = context->pPS;
    static DE::LayoutElement layout[] = {
        DE::LayoutElement{ 0, 0, 4, DE::VT_FLOAT32, false },
        DE::LayoutElement{ 1, 1, 1, DE::VT_INT32, false,
                           DE::INPUT_ELEMENT_FREQUENCY_PER_INSTANCE }
    };
    PSODesc.GraphicsPipeline.InputLayout.LayoutElements = layout;
    PSODesc.GraphicsPipeline.InputLayout.NumElements =
        static_cast<DE::Uint32>(std::size(layout));

    static DE::ShaderResourceVariableDesc VDesc[] = {
        DE::ShaderResourceVariableDesc{
            DE::SHADER_TYPE_PIXEL, "gTexture",
            DE::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC },
        DE::ShaderResourceVariableDesc{
            DE::SHADER_TYPE_PIXEL, "gUniform",
            DE::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE }
    };
    PSODesc.ResourceLayout.NumVariables =
        static_cast<DE::Uint32>(std::size(VDesc));
    PSODesc.ResourceLayout.Variables = VDesc;
    PSODesc.ResourceLayout.DefaultVariableType =
        DE::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

    context->pipeline.setGenerator(
        [context](const DE::PipelineStateDesc& PSODesc) {
            DE::PipelineStateCreateInfo PSOCI;
            PSOCI.PSODesc = PSODesc;

            DE::StaticSamplerDesc SSDesc =
                setSampler(PSODesc.ResourceLayout.NumStaticSamplers);
            PSOCI.PSODesc.ResourceLayout.NumStaticSamplers = 1U;
            PSOCI.PSODesc.ResourceLayout.StaticSamplers = &SSDesc;

            NVGDEPipelineState pipeline;
            context->device->CreatePipelineState(PSOCI, &pipeline.PSO);
            pipeline.PSO
                ->GetStaticVariableByName(DE::SHADER_TYPE_VERTEX, "VSConstants")
                ->Set(context->viewConstant);
            pipeline.PSO->CreateShaderResourceBinding(&pipeline.SRB, true);
            return pipeline;
        });

    DE::CreateUniformBuffer(context->device, sizeof(float) * 2,
                            "NanoVG Constant viewSize",
                            &(context->viewConstant));

    // dummyTex 0
    unsigned char black = 0;
    nvgde_renderCreateTexture(uptr, NVG_TEXTURE_ALPHA, 1, 1, 0, &black);

    prepareTriangleFansIndexBuffer(context, 1024);
    prepareVertexBuffer(context, 16384);
    prepareUniformBuffer(context, 1024);

    context->pathBuffer.reserve(4096);
    context->calls.reserve(1024);
    context->vertBufferHost.reserve(16384);
    context->uniformBufferHost.reserve(1024);
    return 1;
}
static int nvgde_renderCreateTexture(void* uptr, int type, int w, int h,
                                     int imageFlags,
                                     const unsigned char* data) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);

    DE::TextureDesc desc = {};
    desc.BindFlags = DE::BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = DE::CPU_ACCESS_NONE;  // DE::CPU_ACCESS_WRITE
    desc.Format = (type == NVG_TEXTURE_RGBA ? DE::TEX_FORMAT_RGBA8_UNORM :
                                              DE::TEX_FORMAT_R8_UNORM);
    desc.Width = w;
    desc.Height = h;
    desc.MipLevels = 1;
    desc.MiscFlags = (imageFlags & NVG_IMAGE_GENERATE_MIPMAPS ?
                          DE::MISC_TEXTURE_FLAG_GENERATE_MIPS :
                          DE::MISC_TEXTURE_FLAG_NONE);
    desc.Name = "NanoVG Texture";
    desc.SampleCount = 1;
    desc.Type = DE::RESOURCE_DIM_TEX_2D;
    desc.Usage = DE::USAGE_DEFAULT;

    DE::TextureData tdata = {};
    tdata.NumSubresources = 1;
    DE::TextureSubResData sdata = {};
    sdata.pData = data;
    sdata.Stride = w * (type == NVG_TEXTURE_RGBA ? 4 : 1);

    tdata.pSubResources = &sdata;

    NVGDETexture tex;
    context->device->CreateTexture(desc, data ? &tdata : nullptr, &tex.tex);

    tex.texView = tex.tex->GetDefaultView(DE::TEXTURE_VIEW_SHADER_RESOURCE);
    if(imageFlags & DE::MISC_TEXTURE_FLAG_GENERATE_MIPS)
        context->context->GenerateMips(tex.texView);

    tex.flags = imageFlags;
    tex.type = type;

    return context->texture.alloc(tex);
}
static int nvgde_renderDeleteTexture(void* uptr, int image) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    context->texture.free(image);
    return 1;
}
static int nvgde_renderUpdateTexture(void* uptr, int image, int x, int y, int w,
                                     int h, const unsigned char* data) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    NVGDETexture& tex = context->texture.get(image);

    DE::Box rect = {};
    rect.MinX = x;
    rect.MinY = y;
    rect.MaxX = x + w;
    rect.MaxY = y + h;

    DE::TextureSubResData sdata = {};
    sdata.Stride =
        tex.tex->GetDesc().Width * (tex.type == NVG_TEXTURE_RGBA ? 4 : 1);
    sdata.pData = data + sdata.Stride * y + x;
    context->context->UpdateTexture(
        tex.tex, 0U, 0U, rect, sdata,
        DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
        DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    return 1;
}
static int nvgde_renderGetTextureSize(void* uptr, int image, int* w, int* h) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    const auto& desc = context->texture.get(image).tex->GetDesc();
    *w = desc.Width, *h = desc.Height;
    return 1;
}
static void nvgde_renderViewport(void* uptr, float width, float height,
                                 float devicePixelRatio) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    DE::MapHelper<float> guard(context->context, context->viewConstant,
                               DE::MAP_WRITE, DE::MAP_FLAG_DISCARD);
    auto ptr = static_cast<float*>(guard);
    ptr[0] = width;
    ptr[1] = height;
}
static void nvgde_renderCancel(void* uptr) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    context->calls.clear();
    context->pathBuffer.clear();
    context->vertBufferHost.clear();
    context->uniformBufferHost.clear();
}

static void nvgde_renderFlush(void* uptr) {
    // TODO:batch/indirect draw/Multithreading

    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    auto immediateContext = context->context;

    DE::PipelineStateDesc curState = context->defaultState;
    auto&& rasterizer = curState.GraphicsPipeline.RasterizerDesc;
    auto&& blend = curState.GraphicsPipeline.BlendDesc.RenderTargets[0];
    auto&& depthStencil = curState.GraphicsPipeline.DepthStencilDesc;
    auto&& stencilFront = depthStencil.FrontFace;
    auto&& stencilBack = depthStencil.BackFace;
    auto&& primitiveTopology = curState.GraphicsPipeline.PrimitiveTopology;

    {
        auto& vertBuf = context->vertBufferHost;
        prepareVertexBuffer(context, static_cast<int>(vertBuf.size()));
        immediateContext->UpdateBuffer(
            context->vertBuffer, 0U,
            static_cast<int>(vertBuf.size() * sizeof(NVGvertex)),
            vertBuf.data(), DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    {
        auto& uniBuf = context->uniformBufferHost;
        prepareUniformBuffer(context, static_cast<int>(uniBuf.size()));
        immediateContext->UpdateBuffer(
            context->uniformBuffer, 0U,
            static_cast<int>(uniBuf.size() * sizeof(Uniform)), uniBuf.data(),
            DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    {
        int maxFillCount = 0;
        for(const auto& path : context->pathBuffer) {
            maxFillCount = std::max(path.fillCount, maxFillCount);
        }
        prepareTriangleFansIndexBuffer(context, maxFillCount - 2);
    }

    auto setStencil = [&](DE::COMPARISON_FUNCTION func, DE::STENCIL_OP sfail,
                          DE::STENCIL_OP zfail, DE::STENCIL_OP zpass) {
        stencilFront.StencilFunc = stencilBack.StencilFunc = func;
        stencilFront.StencilFailOp = stencilBack.StencilFailOp = sfail;
        stencilFront.StencilDepthFailOp = stencilBack.StencilDepthFailOp =
            zfail;
        stencilFront.StencilPassOp = stencilBack.StencilPassOp = zpass;
    };

    auto prepareRendering = [&](int image) {
        auto&& tex = context->texture.get(image);

        curState.ResourceLayout.NumStaticSamplers = tex.flags;

        NVGDEPipelineState& pipeline = context->pipeline.get(curState);

        pipeline.SRB->GetVariableByName(DE::SHADER_TYPE_PIXEL, "gTexture")
            ->Set(tex.texView);
        pipeline.SRB->GetVariableByName(DE::SHADER_TYPE_PIXEL, "gUniform")
            ->Set(context->uniformSRV);

        immediateContext->SetPipelineState(pipeline.PSO);
        immediateContext->CommitShaderResources(
            pipeline.SRB, DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    };

    auto drawStroke = [&](const NVGDECall& call, int image, int uniform) {
        if(std::none_of(
               context->pathBuffer.cbegin() + call.pathOffset,
               context->pathBuffer.cbegin() + call.pathOffset + call.pathCount,
               [](const NVGDEPath& path) { return path.strokeCount > 0; }))
            return;
        primitiveTopology = DE::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        prepareRendering(image);
        bool used = false;
        for(int i = 0; i < call.pathCount; ++i) {
            const auto& path = context->pathBuffer[call.pathOffset + i];
            if(path.strokeCount > 0) {
                DE::DrawAttribs DA = {};
                DA.NumVertices = path.strokeCount;
                DA.StartVertexLocation = path.strokeOffset;
                DA.FirstInstanceLocation = uniform;
                if(context->flags & NVGCreateFlags::NVG_DEBUG)
                    DA.Flags |= DE::DRAW_FLAG_VERIFY_ALL;
                if(used)
                    DA.Flags |= DE::DRAW_FLAG_DYNAMIC_RESOURCE_BUFFERS_INTACT;
                else
                    used = true;
                immediateContext->Draw(DA);
            }
        }
    };

    auto drawFans = [&](const NVGDECall& call, int image, int uniform) {
        if(std::none_of(
               context->pathBuffer.cbegin() + call.pathOffset,
               context->pathBuffer.cbegin() + call.pathOffset + call.pathCount,
               [](const NVGDEPath& path) { return path.fillCount > 2; }))
            return;
        primitiveTopology = DE::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        immediateContext->SetIndexBuffer(
            context->triangleFansIndex, 0U,
            DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        prepareRendering(image);
        bool used = false;
        for(int i = 0; i < call.pathCount; ++i) {
            const auto& path = context->pathBuffer[call.pathOffset + i];
            if(path.fillCount > 2) {
                DE::DrawIndexedAttribs DIA = {};
                DIA.FirstInstanceLocation = uniform;
                DIA.BaseVertex = path.fillOffset;
                DIA.IndexType = DE::VT_UINT32;
                DIA.NumIndices =
                    3U * static_cast<DE::Uint32>(path.fillCount - 2);
                if(context->flags & NVGCreateFlags::NVG_DEBUG)
                    DIA.Flags |= DE::DRAW_FLAG_VERIFY_ALL;
                if(used)
                    DIA.Flags |= DE::DRAW_FLAG_DYNAMIC_RESOURCE_BUFFERS_INTACT;
                else
                    used = true;
                immediateContext->DrawIndexed(DIA);
            }
        }
    };

    auto drawTriangle = [&](const NVGDECall& call, int image,
                            DE::PRIMITIVE_TOPOLOGY pt, int uniform) {
        if(call.triangleCount == 0)
            return;
        primitiveTopology = pt;
        prepareRendering(image);
        DE::DrawAttribs DA = {};
        DA.FirstInstanceLocation = uniform;
        DA.NumVertices = call.triangleCount;
        DA.StartVertexLocation = call.triangleOffset;
        if(context->flags & NVGCreateFlags::NVG_DEBUG)
            DA.Flags |= DE::DRAW_FLAG_VERIFY_ALL;
        immediateContext->Draw(DA);
    };

    immediateContext->SetStencilRef(0);
    {
        DE::IBuffer* buf[2] = { context->vertBuffer, context->uidBuffer };
        DE::Uint32 voff[2] = { 0, 0 };
        immediateContext->SetVertexBuffers(
            0, 2, buf, voff, DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
            DE::SET_VERTEX_BUFFERS_FLAG_RESET);
    }

    for(const auto& call : context->calls) {
        // setBlend
        {
            auto&& bf = call.blendFactor;
            blend.SrcBlend = bf.srcRGB;
            blend.SrcBlendAlpha = bf.srcAlpha;
            blend.DestBlend = bf.dstRGB;
            blend.DestBlendAlpha = bf.dstAlpha;
        }
        switch(call.type) {
            case Type::fill: {
                // Draw shapes
                depthStencil.StencilEnable = true;
                setStencil(DE::COMPARISON_FUNC_ALWAYS, DE::STENCIL_OP_KEEP,
                           DE::STENCIL_OP_KEEP, DE::STENCIL_OP_KEEP);
                blend.RenderTargetWriteMask = 0;

                // set bindpoint for solid loc

                stencilFront.StencilPassOp = DE::STENCIL_OP_INCR_WRAP;
                stencilBack.StencilPassOp = DE::STENCIL_OP_DECR_WRAP;
                rasterizer.CullMode = DE::CULL_MODE_NONE;

                drawFans(call, 0, call.uniform[0]);
                rasterizer.CullMode = DE::CULL_MODE_BACK;

                // Draw anti-aliased pixels
                blend.RenderTargetWriteMask = DE::COLOR_MASK_ALL;

                if(context->flags & NVGCreateFlags::NVG_ANTIALIAS) {
                    setStencil(DE::COMPARISON_FUNC_EQUAL, DE::STENCIL_OP_KEEP,
                               DE::STENCIL_OP_KEEP, DE::STENCIL_OP_KEEP);
                    // Draw fringes
                    drawStroke(call, call.image, call.uniform[1]);
                }

                // Draw fill
                setStencil(DE::COMPARISON_FUNC_NOT_EQUAL, DE::STENCIL_OP_ZERO,
                           DE::STENCIL_OP_ZERO, DE::STENCIL_OP_ZERO);
                drawTriangle(call, call.image,
                             DE::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                             call.uniform[1]);

                depthStencil.StencilEnable = false;
            } break;
            case Type::convexFill: {

                // Notice:draw order
                drawFans(call, call.image, call.uniform[0]);

                drawStroke(call, call.image, call.uniform[0]);
            } break;
            case Type::stroke: {
                if(context->flags & NVG_STENCIL_STROKES) {
                    depthStencil.StencilEnable = true;

                    // Fill the stroke base without overlap
                    setStencil(DE::COMPARISON_FUNC_EQUAL, DE::STENCIL_OP_KEEP,
                               DE::STENCIL_OP_KEEP, DE::STENCIL_OP_INCR_SAT);

                    drawStroke(call, call.image, call.uniform[1]);

                    // Draw anti-aliased pixels.

                    setStencil(DE::COMPARISON_FUNC_EQUAL, DE::STENCIL_OP_KEEP,
                               DE::STENCIL_OP_KEEP, DE::STENCIL_OP_KEEP);

                    drawStroke(call, call.image, call.uniform[0]);

                    // Clear stencil buffer.
                    blend.RenderTargetWriteMask = 0;

                    setStencil(DE::COMPARISON_FUNC_ALWAYS, DE::STENCIL_OP_ZERO,
                               DE::STENCIL_OP_ZERO, DE::STENCIL_OP_ZERO);
                    drawStroke(call, call.image, call.uniform[0]);

                    blend.RenderTargetWriteMask = DE::COLOR_MASK_ALL;

                    depthStencil.StencilEnable = false;
                } else {
                    // Draw Strokes
                    drawStroke(call, call.image, call.uniform[0]);
                }
            } break;
            case Type::triangles: {
                drawTriangle(call, call.image,
                             DE::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                             call.uniform[0]);
            } break;
            default:
                break;
        }
    }

    nvgde_renderCancel(uptr);
}

static DE::BLEND_FACTOR castBlendFactor(int op) {
    switch(op) {
        case NVG_ZERO:
            return DE::BLEND_FACTOR_ZERO;
            break;
        case NVG_ONE:
            return DE::BLEND_FACTOR_ONE;
            break;
        case NVG_SRC_COLOR:
            return DE::BLEND_FACTOR_SRC_COLOR;
            break;
        case NVG_ONE_MINUS_SRC_COLOR:
            return DE::BLEND_FACTOR_INV_SRC_COLOR;
            break;
        case NVG_DST_COLOR:
            return DE::BLEND_FACTOR_DEST_COLOR;
            break;
        case NVG_ONE_MINUS_DST_COLOR:
            return DE::BLEND_FACTOR_INV_DEST_COLOR;
            break;
        case NVG_SRC_ALPHA:
            return DE::BLEND_FACTOR_SRC_ALPHA;
            break;
        case NVG_ONE_MINUS_SRC_ALPHA:
            return DE::BLEND_FACTOR_INV_SRC_ALPHA;
            break;
        case NVG_DST_ALPHA:
            return DE::BLEND_FACTOR_DEST_ALPHA;
            break;
        case NVG_ONE_MINUS_DST_ALPHA:
            return DE::BLEND_FACTOR_INV_DEST_ALPHA;
            break;
        case NVG_SRC_ALPHA_SATURATE:
            return DE::BLEND_FACTOR_SRC_ALPHA_SAT;
            break;
        default:
            return DE::BLEND_FACTOR_UNDEFINED;
            break;
    }
}

static BlendFactor castBlendFactor(const NVGcompositeOperationState& op) {
    BlendFactor res = {};
    res.srcRGB = castBlendFactor(op.srcRGB);
    res.dstRGB = castBlendFactor(op.dstRGB);
    res.srcAlpha = castBlendFactor(op.srcAlpha);
    res.dstAlpha = castBlendFactor(op.dstAlpha);
    if(res.srcRGB == DE::BLEND_FACTOR_UNDEFINED ||
       res.dstRGB == DE::BLEND_FACTOR_UNDEFINED ||
       res.srcAlpha == DE::BLEND_FACTOR_UNDEFINED ||
       res.dstAlpha == DE::BLEND_FACTOR_UNDEFINED) {
        res.srcRGB = DE::BLEND_FACTOR_ONE;
        res.dstRGB = DE::BLEND_FACTOR_INV_SRC_ALPHA;
        res.srcAlpha = DE::BLEND_FACTOR_ONE;
        res.dstAlpha = DE::BLEND_FACTOR_INV_SRC_ALPHA;
    }
    return res;
}

static void xformToMat3x4(float* m3, float* t) {
    m3[0] = t[0];
    m3[1] = t[1];
    m3[2] = 0.0f;
    m3[3] = 0.0f;
    m3[4] = t[2];
    m3[5] = t[3];
    m3[6] = 0.0f;
    m3[7] = 0.0f;
    m3[8] = t[4];
    m3[9] = t[5];
    m3[10] = 1.0f;
    m3[11] = 0.0f;
}

static NVGcolor premulColor(NVGcolor c) {
    c.r *= c.a;
    c.g *= c.a;
    c.b *= c.a;
    return c;
}

static Uniform convertPaint(NVGDEContext* context, NVGpaint* paint,
                            NVGscissor* scissor, float width, float fringe,
                            float strokeThr) {
    NVGDETexture* tex = nullptr;
    float invxform[6];

    Uniform frag;
    memset(&frag, 0, sizeof(frag));

    frag.innerCol = premulColor(paint->innerColor);
    frag.outerCol = premulColor(paint->outerColor);

    if(scissor->extent[0] < -0.5f || scissor->extent[1] < -0.5f) {
        frag.scissorExt[0] = 1.0f;
        frag.scissorExt[1] = 1.0f;
        frag.scissorScale[0] = 1.0f;
        frag.scissorScale[1] = 1.0f;
    } else {
        nvgTransformInverse(invxform, scissor->xform);
        xformToMat3x4(frag.scissorMat, invxform);
        frag.scissorExt[0] = scissor->extent[0];
        frag.scissorExt[1] = scissor->extent[1];
        frag.scissorScale[0] = sqrtf(scissor->xform[0] * scissor->xform[0] +
                                     scissor->xform[2] * scissor->xform[2]) /
            fringe;
        frag.scissorScale[1] = sqrtf(scissor->xform[1] * scissor->xform[1] +
                                     scissor->xform[3] * scissor->xform[3]) /
            fringe;
    }

    memcpy(frag.extent, paint->extent, sizeof(frag.extent));
    frag.strokeMult = (width * 0.5f + fringe * 0.5f) / fringe;
    frag.strokeThr = strokeThr;

    if(paint->image != 0) {
        tex = &(context->texture.get(paint->image));
        if((tex->flags & NVG_IMAGE_FLIPY) != 0) {
            float m1[6], m2[6];
            nvgTransformTranslate(m1, 0.0f, frag.extent[1] * 0.5f);
            nvgTransformMultiply(m1, paint->xform);
            nvgTransformScale(m2, 1.0f, -1.0f);
            nvgTransformMultiply(m2, m1);
            nvgTransformTranslate(m1, 0.0f, -frag.extent[1] * 0.5f);
            nvgTransformMultiply(m1, m2);
            nvgTransformInverse(invxform, m1);
        } else {
            nvgTransformInverse(invxform, paint->xform);
        }
        frag.type = static_cast<int>(ShaderType::NSVG_SHADER_FILLIMG);

        if(tex->type == NVG_TEXTURE_RGBA)
            frag.texType = (tex->flags & NVG_IMAGE_PREMULTIPLIED) ? 0 : 1;
        else
            frag.texType = 2;
    } else {
        frag.type = static_cast<int>(ShaderType::NSVG_SHADER_FILLGRAD);
        frag.radius = paint->radius;
        frag.feather = paint->feather;
        nvgTransformInverse(invxform, paint->xform);
    }

    xformToMat3x4(frag.paintMat, invxform);
    return frag;
}

static int pushVert(NVGDEContext* context, const NVGvertex* vert, int siz) {
    auto& vertBuf = context->vertBufferHost;
    int res = static_cast<int>(vertBuf.size());
    vertBuf.insert(vertBuf.cend(), vert, vert + siz);
    return res;
}

static int pushUniform(NVGDEContext* context, const Uniform& uni) {
    auto& uniBuf = context->uniformBufferHost;
    int res = static_cast<int>(uniBuf.size());
    uniBuf.push_back(uni);
    return res;
}

static void nvgde_renderFill(void* uptr, NVGpaint* paint,
                             NVGcompositeOperationState compositeOperation,
                             NVGscissor* scissor, float fringe,
                             const float* bounds, const NVGpath* paths,
                             int npaths) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);

    NVGDECall call = {};

    call.type = Type::fill;
    call.triangleCount = 4;
    call.image = paint->image;
    call.blendFactor = castBlendFactor(compositeOperation);

    if(npaths == 1 && paths[0].convex) {
        call.type = Type::convexFill;
        call.triangleCount = 0;
        // Bounding box fill quad not needed for convex fill
    }

    call.pathCount = npaths;
    call.pathOffset = static_cast<int>(context->pathBuffer.size());

    for(int i = 0; i < npaths; i++) {
        NVGDEPath dpath = {};
        const NVGpath& path = paths[i];
        if(path.nfill > 0) {
            dpath.fillOffset = pushVert(context, path.fill, path.nfill);
            dpath.fillCount = path.nfill;
        }
        if(path.nstroke > 0) {
            dpath.strokeOffset = pushVert(context, path.stroke, path.nstroke);
            dpath.strokeCount = path.nstroke;
        }
        context->pathBuffer.push_back(dpath);
    }

    // Setup uniforms for draw calls
    if(call.type == Type::fill) {
        // Quad
        auto& vertBuf = context->vertBufferHost;
        call.triangleOffset = static_cast<int>(vertBuf.size());
        vertBuf.push_back(NVGvertex{ bounds[2], bounds[3], 0.5f, 1.0f });
        vertBuf.push_back(NVGvertex{ bounds[2], bounds[1], 0.5f, 1.0f });
        vertBuf.push_back(NVGvertex{ bounds[0], bounds[3], 0.5f, 1.0f });
        vertBuf.push_back(NVGvertex{ bounds[0], bounds[1], 0.5f, 1.0f });

        // Simple shader for stencil
        Uniform frag;
        memset(&frag, 0, sizeof(frag));
        frag.strokeThr = -1.0f;
        frag.type = static_cast<int>(ShaderType::NSVG_SHADER_SIMPLE);
        call.uniform[0] = pushUniform(context, frag);
        // Fill shader
        call.uniform[1] = pushUniform(
            context,
            convertPaint(context, paint, scissor, fringe, fringe, -1.0f));
    } else {
        // Fill shader
        call.uniform[0] = pushUniform(
            context,
            convertPaint(context, paint, scissor, fringe, fringe, -1.0f));
    }

    context->calls.push_back(call);
}
static void nvgde_renderStroke(void* uptr, NVGpaint* paint,
                               NVGcompositeOperationState compositeOperation,
                               NVGscissor* scissor, float fringe,
                               float strokeWidth, const NVGpath* paths,
                               int npaths) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);

    NVGDECall call = {};

    call.type = Type::stroke;
    call.image = paint->image;
    call.blendFactor = castBlendFactor(compositeOperation);

    call.pathCount = npaths;
    call.pathOffset = static_cast<int>(context->pathBuffer.size());

    for(int i = 0; i < npaths; i++) {
        NVGDEPath dpath = {};
        const NVGpath& path = paths[i];
        if(path.nstroke > 0) {
            dpath.strokeOffset = pushVert(context, path.stroke, path.nstroke);
            dpath.strokeCount = path.nstroke;
        }
        context->pathBuffer.push_back(dpath);
    }

    if(context->flags & NVG_STENCIL_STROKES) {
        // Fill shader
        call.uniform[0] = pushUniform(
            context,
            convertPaint(context, paint, scissor, strokeWidth, fringe, -1.0f));
        call.uniform[1] =
            pushUniform(context,
                        convertPaint(context, paint, scissor, strokeWidth,
                                     fringe, 1.0f - 0.5f / 255.0f));
    } else {
        // Fill shader
        call.uniform[0] = pushUniform(
            context,
            convertPaint(context, paint, scissor, strokeWidth, fringe, -1.0f));
    }

    context->calls.push_back(call);
}
static void nvgde_renderTriangles(void* uptr, NVGpaint* paint,
                                  NVGcompositeOperationState compositeOperation,
                                  NVGscissor* scissor, const NVGvertex* verts,
                                  int nverts, float fringe) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);

    NVGDECall call = {};

    call.type = Type::triangles;
    call.image = paint->image;
    call.blendFactor = castBlendFactor(compositeOperation);

    // Allocate vertices for all the paths.
    call.triangleCount = nverts;
    call.triangleOffset = pushVert(context, verts, nverts);

    // Fill shader
    Uniform frag = convertPaint(context, paint, scissor, 1.0f, fringe, -1.0f);
    frag.type = static_cast<int>(ShaderType::NSVG_SHADER_IMG);
    call.uniform[0] = pushUniform(context, frag);

    context->calls.push_back(call);
}

static void nvgde_renderDelete(void* uptr) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    delete context;
}

NVGcontext* nvgCreateDE(DE::IRenderDevice* device, DE::IDeviceContext* context,
                        const DE::SampleDesc& MSAA,
                        DE::TEXTURE_FORMAT colorFormat,
                        DE::TEXTURE_FORMAT depthFormat, int flags) {
    CHECK_ERR(depthFormat == DE::TEX_FORMAT_D24_UNORM_S8_UINT,
              "Need stencil buffer.");
    CHECK_WARN(
        MSAA.Count == 1 || !(flags & NVG_ANTIALIAS),
        "Geometry based anti-aliasing may not be needed when using MSAA.");
    auto ptr = new NVGDEContext;

    ptr->device = device;
    ptr->context = context;
    ptr->MSAA = MSAA;
    ptr->colorFormat = colorFormat;
    ptr->depthFormat = depthFormat;
    ptr->flags = flags;

    ptr->useIndirectDraw = device->GetDeviceCaps().Features.IndirectRendering;

    NVGparams params = {};
    params.edgeAntiAlias = (flags & NVGCreateFlags::NVG_ANTIALIAS);
    params.renderCancel = nvgde_renderCancel;
    params.renderCreate = nvgde_renderCreate;
    params.renderCreateTexture = nvgde_renderCreateTexture;
    params.renderDelete = nvgde_renderDelete;
    params.renderDeleteTexture = nvgde_renderDeleteTexture;
    params.renderFill = nvgde_renderFill;
    params.renderFlush = nvgde_renderFlush;
    params.renderGetTextureSize = nvgde_renderGetTextureSize;
    params.renderStroke = nvgde_renderStroke;
    params.renderTriangles = nvgde_renderTriangles;
    params.renderUpdateTexture = nvgde_renderUpdateTexture;
    params.renderViewport = nvgde_renderViewport;
    params.userPtr = ptr;
    return nvgCreateInternal(&params);
}

void nvgDeleteDE(NVGcontext* ctx) {
    nvgDeleteInternal(ctx);
}
