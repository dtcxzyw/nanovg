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
#include <functional>
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
    float texType;
    float type;
};

struct NVGDEPipelineState final {
    DE::RefCntAutoPtr<DE::IPipelineState> PSO;
    DE::RefCntAutoPtr<DE::IShaderResourceBinding> SRB;
};

struct NVGDETexture final {
    DE::RefCntAutoPtr<DE::ITexture> tex;
    DE::RefCntAutoPtr<DE::ITextureView> texView;
    DE::Uint32 stride;
    int flags, type;
};

template <typename T>
class ResourceManager final {
private:
    std::vector<T> mSlots;
    std::vector<size_t> mFree;

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

struct BlendFunc final {
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
    int image, vertOffset, triangleOffset, triangleCount;
    std::vector<NVGvertex> vert;
    std::vector<NVGDEPath> path;
    Uniform uniform[2];
    BlendFunc blendFunc;
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
            while(ptr < end) {
                res = (res ^ (*ptr)) * prime;
                ++ptr;
            }
        };
        auto&& rasterizer = state.GraphicsPipeline.RasterizerDesc;
        push(rasterizer.CullMode);
        push(rasterizer.ScissorEnable);

        auto&& blend = state.GraphicsPipeline.BlendDesc.RenderTargets[0];
        if(blend.BlendEnable) {
            push(blend.SrcBlend);
            push(blend.DestBlend);
            push(blend.SrcBlendAlpha);
            push(blend.DestBlendAlpha);
            push(blend.RenderTargetWriteMask);
        }
        auto&& depthStencil = state.GraphicsPipeline.DepthStencilDesc;
        if(depthStencil.DepthEnable)
            push(depthStencil.DepthEnable);
        if(depthStencil.StencilEnable) {
            push(depthStencil.FrontFace);
            push(depthStencil.BackFace);
        }
        // push(state.GraphicsPipeline.PrimitiveTopology);
        return res;
    }
};

struct NVGDEContext final {
    DE::IRenderDevice* device;
    DE::IDeviceContext* context;
    int MSAA, flags;

    DE::RefCntAutoPtr<DE::IShader> pVS, pPS;
    DE::PipelineStateDesc defaultState;
    StateCache<DE::PipelineStateDesc, NVGDEPipelineState, StateHasher> pipeline;

    ResourceManager<NVGDETexture> texture;
    StateCache<int, DE::RefCntAutoPtr<DE::ISampler>> sampler;

    std::vector<NVGDECall> calls;

    DE::TEXTURE_FORMAT colorFormat, depthFormat;
    DE::RefCntAutoPtr<DE::IBuffer> viewConstant, uniform;

    DE::RefCntAutoPtr<DE::IBuffer> triangleFansIndex;
    int indexSize;
};

static const char* vertShader = R"(
cbuffer VSConstants {
    float2 viewSize;
};

struct PSInput {
    float4 pos:SV_POSITION;
    float4 arg:ARG;//pos texCoord
};

void VS(in float4 vin,out PSInput vout) {
    vout.arg=vin;
    vout.pos = float4(2.0f*arg.x/viewSize.x - 1.0f, 1.0f - 2.0f*arg.y/viewSize.y, 0.0f, 1.0f);
}
)";

static const char* fragShader = R"(
Texture2D gTexture;
SamplerState gTextureSampler;
cbuffer PSConstants {
    float4 frag[11];
};
struct PSInput {
    float4 pos:SV_POSITION;
    float4 arg:ARG;//pos texCoord
};
#define scissorMat mat3(frag[0].xyz, frag[1].xyz, frag[2].xyz)
#define paintMat mat3(frag[3].xyz, frag[4].xyz, frag[5].xyz)
#define innerCol frag[6]
#define outerCol frag[7]
#define scissorExt frag[8].xy
#define scissorScale frag[8].zw
#define extent frag[9].xy
#define radius frag[9].z
#define feather frag[9].w
#define strokeMult frag[10].x
#define strokeThr frag[10].y
#define texType int(frag[10].z)
#define type int(frag[10].w)
float sdroundrect(float2 pt, float2 ext, float rad) {
    float2 ext2 = ext - float2(rad,rad);
    float2 d = abs(pt) - ext2;
    return min(max(d.x,d.y),0.0f) + length(max(d,0.0f)) - rad;
}
// Scissoring
float scissorMask(float2 p) {
    float2 sc = (abs((scissorMat * float3(p,1.0f)).xy) - scissorExt);
    sc = float2(0.5f,0.5f) - sc * scissorScale;
    return clamp(sc.x,0.0f,1.0f) * clamp(sc.y,0.0f,1.0f);
}
#ifdef EDGE_AA
// Stroke - from [0..1] to clipped pyramid, where the slope is 1px.
float strokeMask() {
    return min(1.0f, (1.0f-abs(ftcoord.x*2.0f-1.0f))*strokeMult) * min(1.0f, ftcoord.y);
}
#endif
void PS(in PSInput pin,out float4 result:SV_TARGET) {
    float2 fpos=pin.arg.xy;
    float2 ftcoord=pin.arg.zw;
    float scissor = scissorMask(fpos);
    #ifdef EDGE_AA
    float strokeAlpha = strokeMask();
    if (strokeAlpha < strokeThr) discard;
    #else
    float strokeAlpha = 1.0f;
    #endif
    if (type == 0) {			// Gradient
        // Calculate gradient color using box gradient
        float2 pt = (paintMat * float3(fpos,1.0f)).xy;
        float d = clamp((sdroundrect(pt, extent, radius) + 
        feather*0.5f) / feather, 0.0f, 1.0f);
        float4 color = mix(innerCol,outerCol,d);
        // Combine alpha
        color *= strokeAlpha * scissor;
        result = color;
    } else if (type == 1) {		// Image
        // Calculate color fron texture
        float2 pt = (paintMat * float3(fpos,1.0f)).xy / extent;
        float4 color = gTexture.Sample(gTextureSampler, fptcoord);
        if (texType == 1) color = float4(color.xyz*color.w,color.w);
        else if(texType == 2)color = float4(color.x);
        // Apply color tint and alpha.
        color *= innerCol;
        // Combine alpha
        color *= strokeAlpha * scissor;
        result = color;
    } else if (type == 2) {		// Stencil fill
        result = float4(1,1,1,1);
    } else {		// Textured tris
        float4 color = gTexture.Sample(gTextureSampler, fptcoord);
        if (texType == 1) color = float4(color.xyz*color.w,color.w);
        else if(texType == 2) color = float4(color.x);
        color *= scissor;
        result = color * innerCol;
    }
}
)";
static int nvgde_renderCreateTexture(void* uptr, int type, int w, int h,
                                     int imageFlags, const unsigned char* data);
static int nvgde_renderCreate(void* uptr) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);

    DE::PipelineStateDesc& PSODesc = context->defaultState;

    PSODesc.Name = "NanoVG Pipeline";
    PSODesc.IsComputePipeline = false;
    PSODesc.GraphicsPipeline.NumRenderTargets = 1;
    PSODesc.GraphicsPipeline.RTVFormats[0] = context->colorFormat;
    PSODesc.GraphicsPipeline.DSVFormat = context->depthFormat;

    PSODesc.GraphicsPipeline.PrimitiveTopology =
        DE::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    PSODesc.GraphicsPipeline.RasterizerDesc.CullMode = DE::CULL_MODE_BACK;
    PSODesc.GraphicsPipeline.RasterizerDesc.FrontCounterClockwise = true;

    DE::RenderTargetBlendDesc& blendDesc =
        PSODesc.GraphicsPipeline.BlendDesc.RenderTargets[0];
    blendDesc.BlendEnable = true;

    PSODesc.GraphicsPipeline.RasterizerDesc.ScissorEnable = false;
    PSODesc.GraphicsPipeline.DepthStencilDesc.StencilEnable = true;
    DE::StencilOpDesc& SOpDesc =
        PSODesc.GraphicsPipeline.DepthStencilDesc.FrontFace;
    SOpDesc.StencilFunc = DE::COMPARISON_FUNC_ALWAYS;
    SOpDesc.StencilDepthFailOp = SOpDesc.StencilFailOp = SOpDesc.StencilPassOp =
        DE::STENCIL_OP_KEEP;
    PSODesc.GraphicsPipeline.DepthStencilDesc.DepthEnable = false;

    DE::ShaderCreateInfo shaderCI = {};
    shaderCI.SourceLanguage = DE::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCI.UseCombinedTextureSamplers = true;
    shaderCI.CombinedSamplerSuffix = "Sampler";

    DE::RefCntAutoPtr<DE::IShader> pVS;
    {
        shaderCI.Desc.ShaderType = DE::SHADER_TYPE_VERTEX;
        shaderCI.EntryPoint = "VS";
        shaderCI.Desc.Name = "NanoVG VS";
        shaderCI.Source = vertShader;
        context->device->CreateShader(shaderCI, &pVS);
    }

    DE::RefCntAutoPtr<DE::IShader> pPS;
    {
        shaderCI.Desc.ShaderType = DE::SHADER_TYPE_PIXEL;
        shaderCI.EntryPoint = "PS";
        shaderCI.Desc.Name = "NanoVG PS";
        shaderCI.Source = fragShader;
        DE::ShaderMacroHelper macros;
        if(context->MSAA == 1) {
            macros.AddShaderMacro("EDGE_AA", true);
            shaderCI.Macros = macros;
        }
        context->device->CreateShader(shaderCI, &pPS);
    }

    PSODesc.GraphicsPipeline.SmplDesc.Count = context->MSAA;
    PSODesc.GraphicsPipeline.pVS = pVS;
    PSODesc.GraphicsPipeline.pPS = pPS;
    static DE::LayoutElement layout[] = {
        DE::LayoutElement{ 0, 0, 4, DE::VT_FLOAT32, false },
    };
    PSODesc.GraphicsPipeline.InputLayout.LayoutElements = layout;
    PSODesc.GraphicsPipeline.InputLayout.NumElements = std::size(layout);

    DE::CreateUniformBuffer(context->device, sizeof(float) * 2,
                            "NanoVG Constant viewSize",
                            &(context->viewConstant));
    DE::CreateUniformBuffer(context->device, sizeof(float) * 44,
                            "NanoVG Constant frag", &(context->uniform));

    context->pipeline.setGenerator(
        [context](const DE::PipelineStateDesc& PSODesc) {
            DE::PipelineStateCreateInfo PSOCI;
            PSOCI.PSODesc = PSODesc;
            // TODO:ResourceLayout StaticSampler
            throw;
            NVGDEPipelineState pipeline;
            context->device->CreatePipelineState(PSOCI, &pipeline.PSO);
            pipeline.PSO
                ->GetStaticVariableByName(DE::SHADER_TYPE_VERTEX, "viewSize")
                ->Set(context->viewConstant);
            pipeline.PSO->GetStaticVariableByName(DE::SHADER_TYPE_PIXEL, "frag")
                ->Set(context->uniform);
            pipeline.PSO->CreateShaderResourceBinding(&pipeline.SRB, true);
            return pipeline;
        });

    // dummyTex 0
    unsigned char black = 0;
    nvgde_renderCreateTexture(uptr, NVG_TEXTURE_ALPHA, 1, 1, 0, &black);
    return 1;
}
static int nvgde_renderCreateTexture(void* uptr, int type, int w, int h,
                                     int imageFlags,
                                     const unsigned char* data) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);

    DE::TextureDesc desc = {};
    desc.BindFlags = DE::BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = DE::CPU_ACCESS_NONE;  // DE::CPU_ACCESS_WRITE
    desc.Format = (type == NVG_TEXTURE_RGBA ? DE::TEX_FORMAT_RGBA8_SINT :
                                              DE::TEX_FORMAT_R8_SINT);
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
    context->device->CreateTexture(desc, &tdata, &tex.tex);

    tex.texView = tex.tex->GetDefaultView(DE::TEXTURE_VIEW_SHADER_RESOURCE);
    tex.texView->SetSampler(context->sampler.get(imageFlags));

    tex.flags = imageFlags;
    tex.stride = sdata.Stride;
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
    rect.MaxX = rect.MinX + w;
    rect.MaxY = rect.MinY + h;

    DE::TextureSubResData sdata;
    sdata.Stride = tex.stride;
    sdata.pData = data;
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
    DE::MapHelper<float> guard(context->context, context->uniform,
                               DE::MAP_WRITE, DE::MAP_FLAG_DISCARD);
    auto ptr = static_cast<float*>(guard);
    ptr[0] = width;
    ptr[1] = height;
}
static void nvgde_renderCancel(void* uptr) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    context->calls.clear();
}
static void bindUniform(NVGDEContext* context, const Uniform& uni) {
    DE::MapHelper<Uniform> guard(context->context, context->uniform,
                                 DE::MAP_WRITE, DE::MAP_FLAG_DISCARD);
    *guard = uni;
}
static void bindTex(DE::IShaderResourceBinding* SRB, NVGDETexture& tex) {
    SRB->GetVariableByName(DE::SHADER_TYPE_PIXEL, "gTextureSampler")
        ->Set(tex.texView);
}
static void prepareTriangleFansIndexBuffer(NVGDEContext* context, int siz) {
    if(siz <= context->indexSize)
        return;
    auto nearestPowerOf2 = [](int siz) {
        int res = 1;
        while(res < siz)
            res <<= 1;
        return res;
    };
    context->indexSize = nearestPowerOf2(siz);
    context->triangleFansIndex.Release();
    DE::BufferDesc bufferDesc = {};
    bufferDesc.BindFlags = DE::BIND_INDEX_BUFFER;
    bufferDesc.CPUAccessFlags = DE::CPU_ACCESS_NONE;
    bufferDesc.Mode = DE::BUFFER_MODE_RAW;
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
    bufferData.DataSize = siz * 3 * sizeof(int);
    bufferData.pData = index.data();
    context->device->CreateBuffer(bufferDesc, &bufferData,
                                  &(context->triangleFansIndex));
    context->context->SetIndexBuffer(
        context->triangleFansIndex, 0U,
        DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}
static void nvgde_renderFlush(void* uptr) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    auto immediateContext = context->context;

    DE::PipelineStateDesc curState = context->defaultState;
    auto&& rasterizer = curState.GraphicsPipeline.RasterizerDesc;
    auto&& blend = curState.GraphicsPipeline.BlendDesc.RenderTargets[0];
    auto&& depthStencil = curState.GraphicsPipeline.DepthStencilDesc;
    auto&& stencilFront = depthStencil.FrontFace;
    auto&& stencilBack = depthStencil.BackFace;
    auto&& primitiveTopology = curState.GraphicsPipeline.PrimitiveTopology;

    DE::RefCntAutoPtr<DE::IBuffer> vertBuffer;
    {
        size_t siz = 0;
        for(auto& call : context->calls) {
            call.vertOffset = siz;
            siz += static_cast<int>(call.vert.size());
        }

        std::vector<NVGvertex> verts(siz);
        for(auto&& call : context->calls) {
            memcpy(verts.data() + call.vertOffset, call.vert.data(),
                   call.vert.size() * sizeof(NVGvertex));
        }
        DE::BufferDesc bufDesc = {};
        bufDesc.BindFlags = DE::BIND_VERTEX_BUFFER;
        bufDesc.CPUAccessFlags = DE::CPU_ACCESS_NONE;
        bufDesc.Mode = DE::BUFFER_MODE_RAW;
        bufDesc.Name = "NanoVG Vertex Buffer";
        bufDesc.uiSizeInBytes =
            static_cast<DE::Uint32>(siz * sizeof(NVGvertex));
        bufDesc.Usage = DE::USAGE_STATIC;
        DE::BufferData bufData = {};
        bufData.DataSize = bufDesc.uiSizeInBytes;
        bufData.pData = verts.data();
        context->device->CreateBuffer(bufDesc, &bufData, &vertBuffer);
        DE::IBuffer* buf = vertBuffer;
        DE::Uint32 voff = 0;
        immediateContext->SetVertexBuffers(
            0, 1, &buf, &voff, DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
            DE::SET_VERTEX_BUFFERS_FLAG_RESET);
    }

    auto setStencil = [&](DE::Uint32 ref, DE::Uint32 mask,
                          DE::COMPARISON_FUNCTION func, DE::STENCIL_OP sfail,
                          DE::STENCIL_OP zfail, DE::STENCIL_OP zpass) {
        immediateContext->SetStencilRef(ref);
        depthStencil.StencilReadMask = depthStencil.StencilWriteMask = mask;
        stencilFront.StencilFunc = stencilBack.StencilFunc = func;
        stencilFront.StencilFailOp = stencilBack.StencilFailOp = sfail;
        stencilFront.StencilDepthFailOp = stencilBack.StencilDepthFailOp =
            zfail;
        stencilFront.StencilPassOp = stencilBack.StencilPassOp = zpass;
    };

    auto prepareRendering = [&](int image) {
        NVGDEPipelineState& pipeline = context->pipeline.get(curState);
        immediateContext->SetPipelineState(pipeline.PSO);
        bindTex(pipeline.SRB, context->texture.get(image));
        immediateContext->CommitShaderResources(
            pipeline.SRB, DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    };

    auto drawStroke = [&](const NVGDECall& call, int image) {
        prepareRendering(image);
        for(const auto& path : call.path)
            if(path.strokeCount > 0) {
                DE::DrawAttribs DA = {};
                DA.NumVertices = path.strokeCount;
                DA.StartVertexLocation = call.vertOffset + path.strokeOffset;
                immediateContext->Draw(DA);
            }
    };

    auto drawFans = [&](const NVGDECall& call, int image) {
        prepareRendering(image);
        for(const auto& path : call.path) {
            DE::DrawIndexedAttribs DIA = {};
            DIA.BaseVertex = call.vertOffset + path.fillOffset;
            DIA.IndexType = DE::VT_UINT32;
            DIA.NumIndices = 3U * static_cast<DE::Uint32>(path.fillCount - 2);
            immediateContext->DrawIndexed(DIA);
        }
    };

    auto drawTriangle = [&](const NVGDECall& call, int image) {
        prepareRendering(image);
        DE::DrawAttribs DA = {};
        DA.NumVertices = call.triangleCount;
        DA.StartVertexLocation = call.vertOffset + call.triangleOffset;
        immediateContext->Draw(DA);
    };

    for(const auto& call : context->calls) {
        // setBlend
        {
            auto&& bf = call.blendFunc;
            blend.SrcBlend = bf.srcRGB;
            blend.SrcBlendAlpha = bf.srcAlpha;
            blend.DestBlend = bf.dstRGB;
            blend.DestBlendAlpha = bf.dstAlpha;
        }
        switch(call.type) {
            case Type::fill: {
                // Draw shapes
                depthStencil.StencilEnable = true;
                setStencil(0, 0xff, DE::COMPARISON_FUNC_ALWAYS,
                           DE::STENCIL_OP_KEEP, DE::STENCIL_OP_KEEP,
                           DE::STENCIL_OP_KEEP);
                blend.RenderTargetWriteMask = 0;

                // set bindpoint for solid loc
                bindUniform(context, call.uniform[0]);

                stencilFront.StencilPassOp = DE::STENCIL_OP_INCR_WRAP;
                stencilBack.StencilPassOp = DE::STENCIL_OP_DECR_WRAP;
                rasterizer.CullMode = DE::CULL_MODE_NONE;
                primitiveTopology = DE::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

                for(const auto& path : call.path)
                    prepareTriangleFansIndexBuffer(context, path.fillCount - 2);

                drawFans(call, 0);
                rasterizer.CullMode = DE::CULL_MODE_BACK;

                // Draw anti-aliased pixels
                blend.RenderTargetWriteMask = DE::COLOR_MASK_ALL;
                bindUniform(context, call.uniform[1]);

                if(context->MSAA == 1) {
                    setStencil(0x00, 0xff, DE::COMPARISON_FUNC_EQUAL,
                               DE::STENCIL_OP_KEEP, DE::STENCIL_OP_KEEP,
                               DE::STENCIL_OP_KEEP);
                    primitiveTopology = DE::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
                    // Draw fringes
                    drawStroke(call, call.image);
                }

                // Draw fill
                setStencil(0x00, 0xff, DE::COMPARISON_FUNC_NOT_EQUAL,
                           DE::STENCIL_OP_ZERO, DE::STENCIL_OP_ZERO,
                           DE::STENCIL_OP_ZERO);
                primitiveTopology = DE::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
                drawTriangle(call, call.image);

                depthStencil.StencilEnable = false;
            } break;
            case Type::convexFill: {
                bindUniform(context, call.uniform[0]);
                primitiveTopology = DE::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
                for(const auto& path : call.path)
                    prepareTriangleFansIndexBuffer(context, path.fillCount - 2);

                drawFans(call, call.image);

                primitiveTopology = DE::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

                drawStroke(call, call.image);
            } break;
            case Type::stroke: {
                if(context->flags & NVG_STENCIL_STROKES) {
                    depthStencil.StencilEnable = true;

                    // Fill the stroke base without overlap
                    setStencil(0x00, 0xff, DE::COMPARISON_FUNC_EQUAL,
                               DE::STENCIL_OP_KEEP, DE::STENCIL_OP_KEEP,
                               DE::STENCIL_OP_INCR_SAT);

                    bindUniform(context, call.uniform[1]);

                    primitiveTopology = DE::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

                    drawStroke(call, call.image);

                    // Draw anti-aliased pixels.
                    bindUniform(context, call.uniform[0]);

                    setStencil(0x00, 0xff, DE::COMPARISON_FUNC_EQUAL,
                               DE::STENCIL_OP_KEEP, DE::STENCIL_OP_KEEP,
                               DE::STENCIL_OP_KEEP);

                    drawStroke(call, call.image);

                    // Clear stencil buffer.
                    blend.RenderTargetWriteMask = 0;

                    setStencil(0x00, 0xff, DE::COMPARISON_FUNC_ALWAYS,
                               DE::STENCIL_OP_ZERO, DE::STENCIL_OP_ZERO,
                               DE::STENCIL_OP_ZERO);
                    drawStroke(call, call.image);

                    blend.RenderTargetWriteMask = DE::COLOR_MASK_ALL;

                    depthStencil.StencilEnable = false;

                    //		glnvg__convertPaint(gl, nvg__fragUniformPtr(gl,
                    // call->uniformOffset + gl->fragSize), paint, scissor,
                    // strokeWidth, fringe, 1.0f - 0.5f/255.0f);

                } else {
                    bindUniform(context, call.uniform[0]);
                    // Draw Strokes
                    drawStroke(call, call.image);
                }
            } break;
            case Type::triangles: {
                bindUniform(context, call.uniform[0]);
                drawTriangle(call, call.image);
            } break;
            default:
                break;
        }
    }

    context->calls.clear();
}

static DE::BLEND_FACTOR castBlendFunc(int op) {
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

static BlendFunc castBlendFunc(const NVGcompositeOperationState& op) {
    BlendFunc res = {};
    res.srcRGB = castBlendFunc(op.srcRGB);
    res.dstRGB = castBlendFunc(op.dstRGB);
    res.srcAlpha = castBlendFunc(op.srcAlpha);
    res.dstAlpha = castBlendFunc(op.dstAlpha);
    return res;
}

static int maxVertCount(const NVGpath* paths, int npaths) {
    int count = 0;
    for(int i = 0; i < npaths; i++) {
        count += paths[i].nfill;
        count += paths[i].nstroke;
    }
    return count;
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

static void convertPaint(NVGDEContext* context, Uniform& frag, NVGpaint* paint,
                         NVGscissor* scissor, float width, float fringe,
                         float strokeThr) {
    NVGDETexture* tex = nullptr;
    float invxform[6];

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
            frag.texType = (tex->flags & NVG_IMAGE_PREMULTIPLIED) ? 0.0f : 1.0f;
        else
            frag.texType = 2.0f;
    } else {
        frag.type = static_cast<int>(ShaderType::NSVG_SHADER_FILLGRAD);
        frag.radius = paint->radius;
        frag.feather = paint->feather;
        nvgTransformInverse(invxform, paint->xform);
    }

    xformToMat3x4(frag.paintMat, invxform);
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
    call.blendFunc = castBlendFunc(compositeOperation);

    if(npaths == 1 && paths[0].convex) {
        call.type = Type::convexFill;
        call.triangleCount =
            0;  // Bounding box fill quad not needed for convex fill
    }

    int maxVerts = maxVertCount(paths, npaths) + call.triangleCount;

    call.vert.reserve(maxVerts);
    call.path.reserve(npaths);

    int offset = 0;

    for(int i = 0; i < npaths; i++) {
        NVGDEPath dpath = {};
        const NVGpath* path = &paths[i];
        if(path->nfill > 0) {
            dpath.fillOffset = offset;
            dpath.fillCount = path->nfill;
            call.vert.insert(call.vert.cend(), path->fill,
                             path->fill + path->nfill);
            offset += path->nfill;
        }
        if(path->nstroke > 0) {
            dpath.strokeOffset = offset;
            dpath.strokeCount = path->nstroke;
            call.vert.insert(call.vert.cend(), path->stroke,
                             path->stroke + path->nstroke);
            offset += path->nstroke;
        }
        call.path.push_back(dpath);
    }

    // Setup uniforms for draw calls
    if(call.type == Type::fill) {
        // Quad
        call.triangleOffset = offset;
        call.vert.push_back(NVGvertex{ bounds[2], bounds[3], 0.5f, 1.0f });
        call.vert.push_back(NVGvertex{ bounds[2], bounds[1], 0.5f, 1.0f });
        call.vert.push_back(NVGvertex{ bounds[0], bounds[3], 0.5f, 1.0f });
        call.vert.push_back(NVGvertex{ bounds[0], bounds[1], 0.5f, 1.0f });

        // Simple shader for stencil
        Uniform& frag = call.uniform[0];
        memset(&frag, 0, sizeof(frag));
        frag.strokeThr = -1.0f;
        frag.type = static_cast<int>(ShaderType::NSVG_SHADER_SIMPLE);
        // Fill shader
        convertPaint(context, call.uniform[1], paint, scissor, fringe, fringe,
                     -1.0f);
    } else {
        Uniform& frag = call.uniform[0];
        // Fill shader
        convertPaint(context, frag, paint, scissor, fringe, fringe, -1.0f);
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
    call.blendFunc = castBlendFunc(compositeOperation);

    // Allocate vertices for all the paths.
    int maxverts = maxVertCount(paths, npaths);
    call.vert.reserve(maxverts);
    call.path.reserve(npaths);

    int offset = 0;
    for(int i = 0; i < npaths; i++) {
        NVGDEPath dpath = {};
        const NVGpath* path = &paths[i];
        if(path->nstroke) {
            dpath.strokeOffset = offset;
            dpath.strokeCount = path->nstroke;
            call.vert.insert(call.vert.cend(), path->stroke,
                             path->stroke + path->nstroke);
            offset += path->nstroke;
        }
        call.path.push_back(dpath);
    }

    if(context->flags & NVG_STENCIL_STROKES) {
        // Fill shader
        convertPaint(context, call.uniform[0], paint, scissor, strokeWidth,
                     fringe, -1.0f);
        convertPaint(context, call.uniform[1], paint, scissor, strokeWidth,
                     fringe, 1.0f - 0.5f / 255.0f);
    } else {
        // Fill shader
        Uniform& uni = call.uniform[0];
        convertPaint(context, uni, paint, scissor, strokeWidth, fringe, -1.0f);
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
    call.blendFunc = castBlendFunc(compositeOperation);

    // Allocate vertices for all the paths.
    call.vert.assign(verts, verts + nverts);
    call.triangleCount = nverts;
    call.triangleOffset = 0;

    // Fill shader
    Uniform& frag = call.uniform[0];
    convertPaint(context, frag, paint, scissor, 1.0f, fringe, -1.0f);
    frag.type = static_cast<int>(ShaderType::NSVG_SHADER_IMG);

    context->calls.push_back(call);
}
static void nvgde_renderDelete(void* uptr) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    delete reinterpret_cast<NVGDEContext*>(context);
}
static auto generateSampler(DE::IRenderDevice* device, int imageFlags) {
    DE::SamplerDesc sd = {};
    sd.Name = "NanoVG Sampler";
    sd.AddressU = (imageFlags & NVG_IMAGE_REPEATX ? DE::TEXTURE_ADDRESS_WRAP :
                                                    DE::TEXTURE_ADDRESS_CLAMP);
    sd.AddressV = (imageFlags & NVG_IMAGE_REPEATY ? DE::TEXTURE_ADDRESS_WRAP :
                                                    DE::TEXTURE_ADDRESS_CLAMP);
    if(imageFlags & NVG_IMAGE_NEAREST)
        sd.MinFilter = sd.MagFilter = DE::FILTER_TYPE_POINT;
    else
        sd.MinFilter = sd.MagFilter = DE::FILTER_TYPE_LINEAR;
    DE::RefCntAutoPtr<DE::ISampler> sampler;
    device->CreateSampler(sd, &sampler);
    return sampler;
}

NVGcontext* nvgCreateDE(DE::IRenderDevice* device, DE::IDeviceContext* context,
                        int MSAA, DE::TEXTURE_FORMAT colorFormat,
                        DE::TEXTURE_FORMAT depthFormat, int flags) {
    auto ptr = new NVGDEContext;

    ptr->device = device;
    ptr->context = context;
    ptr->MSAA = MSAA;
    ptr->colorFormat = colorFormat;
    ptr->depthFormat = depthFormat;
    ptr->flags = flags;
    ptr->indexSize = 0;
    ptr->sampler.setGenerator(
        [device](int flags) { return generateSampler(device, flags); });

    NVGparams params = {};
    params.edgeAntiAlias = (MSAA != 1);
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
