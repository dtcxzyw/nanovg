//
// Copyright (c) 2009-2013 Mikko Mononen memon@inside.org
// Port of _gl.h to _DE.hpp by dtcxzyw
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
#include <functional>
#include <unordered_map>
#include <vector>

struct Uniform final {
    float scissorMat[12];  // matrices are actually 3 vec4s
    float paintMat[12];
    NVGcolor innerCol;
    NVGcolor outerCol;
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

struct NVGDEPipelineState final {
    DE::RefCntAutoPtr<DE::IPipelineState> PSO;
    DE::RefCntAutoPtr<DE::IShaderResourceBinding> SRB;
};

struct NVGDETexture final {
    DE::RefCntAutoPtr<DE::ITexture> tex;
    DE::RefCntAutoPtr<DE::ITextureView> texView;
    DE::Uint32 stride;
    bool flipY;
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
class StateManager final {
private:
    std::unordered_map<Key, T, Hasher> mMap;
    std::function<T(const Key&)> mGenerator;

public:
    void setGenerator(const std::function<T(const Key&)>& gen) {
        mGenerator = gen;
    }
    T& get(const Key& key) {
        auto iter = mMap.find(key);
        if(iter != mMap.end())
            return iter->second;
        T& res = mMap[key];
        res = gen(key);
        return res;
    }
};

enum class Type { none, fill, convexFill, stroke, triangles };

struct BlendFunc final {
    DE::BLEND_OPERATION srcRGB, dstRGB, srcAlpha, dstAlpha;
};

struct NVGDEPath {
    int fillOffset;
    int fillCount;
    int strokeOffset;
    int strokeCount;
};

struct NVGDECall final {
    Type type;
    int image;
    std::vector<NVGDEPath> path;
    std::vector<NVGvertex> tri;
    Uniform uniform;
    BlendFunc blendFunc;
};

struct NVGDEContext final {
    DE::IRenderDevice* device;
    DE::IDeviceContext* context;
    int MSAA;
    NVGDEPipelineState pipeline;

    ResourceManager<NVGDETexture> texture;
    StateManager<int, DE::RefCntAutoPtr<DE::ISampler>> sampler;

    std::vector<NVGDECall> calls;

    float viewSize[2];
    DE::TEXTURE_FORMAT colorFormat, depthFormat;
    DE::RefCntAutoPtr<DE::IBuffer> viewConstant, uniform;
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
        else color = float4(color.x);
        // Apply color tint and alpha.
        color *= innerCol;
        // Combine alpha
        color *= strokeAlpha * scissor;
        result = color;
    } else if (type == 2) {		// Stencil fill
        result = float4(1,1,1,1);
    } else if (type == 3) {		// Textured tris
        float4 color = gTexture.Sample(gTextureSampler, fptcoord);
        if (texType == 1) color = float4(color.xyz*color.w,color.w);
        else color = float4(color.x);
        color *= scissor;
        result = color * innerCol;
    }
}
)";

static void nvgde_renderCancel(void* uptr) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
}
static int nvgde_renderCreate(void* uptr) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);

    DE::PipelineStateCreateInfo PSOCreateInfo;
    DE::PipelineStateDesc& PSODesc = PSOCreateInfo.PSODesc;

    PSODesc.Name = "NanoVG Pipeline";
    PSODesc.IsComputePipeline = false;
    PSODesc.GraphicsPipeline.NumRenderTargets = 1;
    auto SCDesc = mEngine.swapChain->GetDesc();
    PSODesc.GraphicsPipeline.RTVFormats[0] = context->colorFormat;
    PSODesc.GraphicsPipeline.DSVFormat = context->depthFormat;
    // TODO:Primitive,cull,depthTest
    PSODesc.GraphicsPipeline.PrimitiveTopology =
        DE::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    PSODesc.GraphicsPipeline.RasterizerDesc.CullMode = DE::CULL_MODE_NONE;
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
        mEngine.device->CreateShader(shaderCI, &pVS);
    }

    DE::RefCntAutoPtr<DE::IShader> pPS;
    {
        shaderCI.Desc.ShaderType = DE::SHADER_TYPE_PIXEL;
        shaderCI.EntryPoint = "PS";
        shaderCI.Desc.Name = "NanoVG PS";
        shaderCI.Source = fragShader;
        DE::ShaderMacro macro = {};
        DE::ShaderMacro* macros[2] = { &macro, nullptr };
        if(context->MSAA == 1) {
            macro.Definition = "EDGE_AA";
            macro.Name = "EDGE_AA";
            shaderCI.Macros = macros;
        }
        mEngine.device->CreateShader(shaderCI, &pPS);
    }

    DE::PipelineStateDesc PSDesc = {};
    PSDesc.GraphicsPipeline.SmplDesc.Count = context->MSAA;
    PSODesc.GraphicsPipeline.pVS = pVS;
    PSODesc.GraphicsPipeline.pPS = pPS;
    DE::LayoutElement layout[] = {
        DE::LayoutElement{ 0, 0, 4, DE::VT_FLOAT32, false },
    };
    PSODesc.GraphicsPipeline.InputLayout.LayoutElements = layout;
    PSODesc.GraphicsPipeline.InputLayout.NumElements = std::size(layout);

    NVGDEPipelineState pipeline;
    context->device->CreatePipelineState(&PSOCreateInfo, &pipeline.PSO);

    DE::CreateUniformBuffer(context->device, sizeof(float) * 2,
                            "NanoVG Constant viewSize",
                            &(context->viewConstant));
    DE::CreateUniformBuffer(context->device, sizeof(float) * 44,
                            "NanoVG Constant frag", &(context->uniform));
    pipeline.PSO->GetStaticVariableByName(DE::SHADER_TYPE_VERTEX, "viewSize")
        ->Set(context->viewConstant);
    pipeline.PSO->GetStaticVariableByName(DE::SHADER_TYPE_PIXEL, "frag")
        ->Set(context->uniform);

    pipeline.PSO->CreateShaderResourceBinding(&pipeline.SRB, true);
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

    tex.flipY = (imageFlags & NVG_IMAGE_FLIPY);
    tex.stride = sdata.Stride;

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
    rect.MaxX = UpdateBox.MinX + w;
    rect.MaxY = UpdateBox.MinY + h;

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
    context->viewSize[0] = width;
    context->viewSize[1] = height;
}
static void nvgde_renderCancel(void* uptr) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    context->calls.clear();
}
static void nvgde_renderFlush(void* uptr) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    throw;
}

static DE::BLEND_FACTOR castBlendFunc(NVGblendFactor op) {
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

    !!!

    // Allocate vertices for all the paths.
    maxverts = glnvg__maxVertCount(paths, npaths) + call->triangleCount;
    offset = glnvg__allocVerts(gl, maxverts);
    if(offset == -1)
        goto error;

    for(i = 0; i < npaths; i++) {
        GLNVGpath* copy = &gl->paths[call->pathOffset + i];
        const NVGpath* path = &paths[i];
        memset(copy, 0, sizeof(GLNVGpath));
        if(path->nfill > 0) {
            copy->fillOffset = offset;
            copy->fillCount = path->nfill;
            memcpy(&gl->verts[offset], path->fill,
                   sizeof(NVGvertex) * path->nfill);
            offset += path->nfill;
        }
        if(path->nstroke > 0) {
            copy->strokeOffset = offset;
            copy->strokeCount = path->nstroke;
            memcpy(&gl->verts[offset], path->stroke,
                   sizeof(NVGvertex) * path->nstroke);
            offset += path->nstroke;
        }
    }

    // Setup uniforms for draw calls
    if(call->type == GLNVG_FILL) {
        // Quad
        call->triangleOffset = offset;
        quad = &gl->verts[call->triangleOffset];
        glnvg__vset(&quad[0], bounds[2], bounds[3], 0.5f, 1.0f);
        glnvg__vset(&quad[1], bounds[2], bounds[1], 0.5f, 1.0f);
        glnvg__vset(&quad[2], bounds[0], bounds[3], 0.5f, 1.0f);
        glnvg__vset(&quad[3], bounds[0], bounds[1], 0.5f, 1.0f);

        call->uniformOffset = glnvg__allocFragUniforms(gl, 2);
        if(call->uniformOffset == -1)
            goto error;
        // Simple shader for stencil
        frag = nvg__fragUniformPtr(gl, call->uniformOffset);
        memset(frag, 0, sizeof(*frag));
        frag->strokeThr = -1.0f;
        frag->type = NSVG_SHADER_SIMPLE;
        // Fill shader
        glnvg__convertPaint(
            gl, nvg__fragUniformPtr(gl, call->uniformOffset + gl->fragSize),
            paint, scissor, fringe, fringe, -1.0f);
    } else {
        call->uniformOffset = glnvg__allocFragUniforms(gl, 1);
        // Fill shader
        glnvg__convertPaint(gl, nvg__fragUniformPtr(gl, call->uniformOffset),
                            paint, scissor, fringe, fringe, -1.0f);
    }
}
static void nvgde_renderStroke(void* uptr, NVGpaint* paint,
                               NVGcompositeOperationState compositeOperation,
                               NVGscissor* scissor, float fringe,
                               float strokeWidth, const NVGpath* paths,
                               int npaths) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    throw;
}
static void nvgde_renderTriangles(void* uptr, NVGpaint* paint,
                                  NVGcompositeOperationState compositeOperation,
                                  NVGscissor* scissor, const NVGvertex* verts,
                                  int nverts, float fringe) {
    auto context = reinterpret_cast<NVGDEContext*>(uptr);
    throw;
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
    device->CreateSampler(&sd, &sampler);
    return sampler;
}

NVGcontext* nvgCreateDE(DE::IRenderDevice* device, DE::IDeviceContext* context,
                        int MSAA, DE::TEXTURE_FORMAT colorFormat,
                        DE::TEXTURE_FORMAT depthFormat) {
    auto ptr = new NVGDEContext;

    ptr->device = device;
    ptr->context = context;
    ptr->MSAA = MSAA;
    ptr->colorFormat = colorFormat;
    ptr->depthFormat = depthFormat;
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
