//
// Copyright (c) 2013 Mikko Mononen memon@inside.org
// Port of _gl3.c to _DE.cpp by dtcxzyw
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

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "demo.h"
#include "perf.h"
#include <Windows.h>
#include <nanovg.h>
#include <nanovg_DE.hpp>

#define ENGINE_DLL 1
#define D3D11_SUPPORTED 1
#define D3D12_SUPPORTED 1
#define GL_SUPPORTED 1
#define VULKAN_SUPPORTED 1

#include <DiligentCore/Common/interface/FileWrapper.hpp>
#include <DiligentCore/Graphics/GraphicsEngineD3D11/interface/EngineFactoryD3D11.h>
#include <DiligentCore/Graphics/GraphicsEngineD3D12/interface/EngineFactoryD3D12.h>
#include <DiligentCore/Graphics/GraphicsEngineOpenGL/interface/EngineFactoryOpenGL.h>
#include <DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h>
#include <DiligentCore/Graphics/GraphicsTools/interface/DurationQueryHelper.hpp>
#include <DiligentCore/Graphics/GraphicsTools/interface/ScreenCapture.hpp>
#include <DiligentTools/TextureLoader/interface/Image.h>
#include <chrono>
#include <sstream>

using Clock = std::chrono::high_resolution_clock;

#ifdef _DEBUG
#define DILIGENT_DEBUG
#endif  // _DEBUG

#define NANOVG_MSAA

static void callback(DE::DEBUG_MESSAGE_SEVERITY severity, const char* message,
                     const char* function, const char* file, int line) {
    if(severity != DE::DEBUG_MESSAGE_SEVERITY_INFO) {
        DebugBreak();
    } else
        OutputDebugStringA(message);
}

class Engine final {
public:
    Engine(void* hWnd, DE::RENDER_DEVICE_TYPE type) {
        DE::SwapChainDesc SCDesc;
        SCDesc.DefaultDepthValue = 1.0f;
        SCDesc.DefaultStencilValue = 0;
        SCDesc.ColorBufferFormat = DE::TEX_FORMAT_RGBA8_UNORM;
        SCDesc.DepthBufferFormat = DE::TEX_FORMAT_D24_UNORM_S8_UINT;
        SCDesc.Usage = DE::SWAP_CHAIN_USAGE_RENDER_TARGET |
            DE::SWAP_CHAIN_USAGE_COPY_SOURCE;

        switch(type) {
#if D3D11_SUPPORTED
            case DE::RENDER_DEVICE_TYPE_D3D11: {
                DE::EngineD3D11CreateInfo EngineCI;
                EngineCI.DebugMessageCallback = callback;
#ifdef DILIGENT_DEBUG
                EngineCI.DebugFlags |=
                    DE::D3D11_DEBUG_FLAG_CREATE_DEBUG_DEVICE |
                    DE::D3D11_DEBUG_FLAG_VERIFY_COMMITTED_SHADER_RESOURCES;
#endif
#if ENGINE_DLL
                // Load the dll and import GetEngineFactoryD3D11() function
                auto GetEngineFactoryD3D11 = DE::LoadGraphicsEngineD3D11();
#endif
                auto* pFactoryD3D11 = GetEngineFactoryD3D11();
                pFactoryD3D11->CreateDeviceAndContextsD3D11(EngineCI, &device,
                                                            &immediateContext);
                DE::Win32NativeWindow window{ hWnd };
                pFactoryD3D11->CreateSwapChainD3D11(
                    device, immediateContext, SCDesc, DE::FullScreenModeDesc{},
                    window, &swapChain);
            } break;
#endif

#if D3D12_SUPPORTED
            case DE::RENDER_DEVICE_TYPE_D3D12: {
#if ENGINE_DLL
                // Load the dll and import GetEngineFactoryD3D12() function
                auto GetEngineFactoryD3D12 = DE::LoadGraphicsEngineD3D12();
#endif
                DE::EngineD3D12CreateInfo EngineCI;
                EngineCI.DebugMessageCallback = callback;
#ifdef DILIGENT_DEBUG
                EngineCI.EnableDebugLayer = true;
#endif
                auto* pFactoryD3D12 = GetEngineFactoryD3D12();
                pFactoryD3D12->CreateDeviceAndContextsD3D12(EngineCI, &device,
                                                            &immediateContext);
                DE::Win32NativeWindow window{ hWnd };
                pFactoryD3D12->CreateSwapChainD3D12(
                    device, immediateContext, SCDesc, DE::FullScreenModeDesc{},
                    window, &swapChain);
            } break;
#endif

#if GL_SUPPORTED
            case DE::RENDER_DEVICE_TYPE_GL: {

#if EXPLICITLY_LOAD_ENGINE_GL_DLL
                // Load the dll and import GetEngineFactoryOpenGL() function
                auto GetEngineFactoryOpenGL = DE::LoadGraphicsEngineOpenGL();
#endif
                auto* pFactoryOpenGL = GetEngineFactoryOpenGL();

                DE::EngineGLCreateInfo EngineCI;
                EngineCI.Window.hWnd = hWnd;
                EngineCI.DebugMessageCallback = callback;
#ifdef DILIGENT_DEBUG
                EngineCI.CreateDebugContext = true;
#endif
                pFactoryOpenGL->CreateDeviceAndSwapChainGL(
                    EngineCI, &device, &immediateContext, SCDesc, &swapChain);
            } break;
#endif

#if VULKAN_SUPPORTED
            case DE::RENDER_DEVICE_TYPE_VULKAN: {
#if EXPLICITLY_LOAD_ENGINE_VK_DLL
                // Load the dll and import GetEngineFactoryVk() function
                auto GetEngineFactoryVk = DE::LoadGraphicsEngineVk();
#endif
                DE::EngineVkCreateInfo EngineCI;
                EngineCI.DebugMessageCallback = callback;
#ifdef DILIGENT_DEBUG
                EngineCI.EnableValidation = true;
#endif
                auto* pFactoryVk = GetEngineFactoryVk();
                pFactoryVk->CreateDeviceAndContextsVk(EngineCI, &device,
                                                      &immediateContext);

                if(!swapChain && hWnd != nullptr) {
                    DE::Win32NativeWindow window{ hWnd };
                    pFactoryVk->CreateSwapChainVk(device, immediateContext,
                                                  SCDesc, window, &swapChain);
                }
            } break;
#endif

            default:
                throw std::logic_error("Unknown/unsupported device type");
                break;
        }
    }

    ~Engine() {
        immediateContext->Flush();
    }

    void updateTarget(const float* clearColor) {
        auto* pRTV = swapChain->GetCurrentBackBufferRTV();
        auto* pDSV = swapChain->GetDepthBufferDSV();
        if(sampleCount) {
            pRTV = msaaColorRTV;
            pDSV = msaaDepthDSV;
        }

        immediateContext->SetRenderTargets(
            1, &pRTV, pDSV, DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        // Clear the back buffer
        // Let the engine perform required state transitions
        immediateContext->ClearRenderTarget(
            pRTV, clearColor, DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        immediateContext->ClearDepthStencil(
            pDSV, DE::CLEAR_DEPTH_FLAG | DE::CLEAR_STENCIL_FLAG, 1.0f, 0,
            DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    void resetRenderTarget() {
        if(sampleCount == 1)
            return;

        const auto& SCDesc = swapChain->GetDesc();
        // Create window-size multi-sampled offscreen render target
        DE::TextureDesc colTexDesc = {};
        colTexDesc.Name = "Color RTV";
        colTexDesc.Type = DE::RESOURCE_DIM_TEX_2D;
        colTexDesc.BindFlags = DE::BIND_RENDER_TARGET;
        colTexDesc.Width = SCDesc.Width;
        colTexDesc.Height = SCDesc.Height;
        colTexDesc.MipLevels = 1;
        colTexDesc.Format = SCDesc.ColorBufferFormat;
        bool needSRGBConversion = device->GetDeviceCaps().IsD3DDevice() &&
            (colTexDesc.Format == DE::TEX_FORMAT_RGBA8_UNORM_SRGB ||
             colTexDesc.Format == DE::TEX_FORMAT_BGRA8_UNORM_SRGB);
        if(needSRGBConversion) {
            // Internally Direct3D swap chain images are not SRGB, and
            // ResolveSubresource requires source and destination formats to
            // match exactly or be typeless. So we will have to create a
            // typeless texture and use SRGB render target view with it.
            colTexDesc.Format =
                colTexDesc.Format == DE::TEX_FORMAT_RGBA8_UNORM_SRGB ?
                DE::TEX_FORMAT_RGBA8_TYPELESS :
                DE::TEX_FORMAT_BGRA8_TYPELESS;
        }

        // Set the desired number of samples
        colTexDesc.SampleCount = sampleCount;
        // Define optimal clear value
        float col[4] = { 0.3f, 0.3f, 0.32f, 1.0f };
        memcpy(colTexDesc.ClearValue.Color, col, sizeof(col));
        colTexDesc.ClearValue.Format = SCDesc.ColorBufferFormat;
        DE::RefCntAutoPtr<DE::ITexture> pColor;
        device->CreateTexture(colTexDesc, nullptr, &pColor);

        // Store the render target view
        msaaColorRTV.Release();
        if(needSRGBConversion) {
            DE::TextureViewDesc RTVDesc;
            RTVDesc.ViewType = DE::TEXTURE_VIEW_RENDER_TARGET;
            RTVDesc.Format = SCDesc.ColorBufferFormat;
            pColor->CreateView(RTVDesc, &msaaColorRTV);
        } else {
            msaaColorRTV =
                pColor->GetDefaultView(DE::TEXTURE_VIEW_RENDER_TARGET);
        }

        // Create window-size multi-sampled depth buffer
        DE::TextureDesc depthDesc = colTexDesc;
        depthDesc.Name = "depth DSV";
        depthDesc.Format = SCDesc.DepthBufferFormat;
        depthDesc.BindFlags = DE::BIND_DEPTH_STENCIL;
        // Define optimal clear value
        depthDesc.ClearValue.Format = depthDesc.Format;

        DE::RefCntAutoPtr<DE::ITexture> pDepth;
        device->CreateTexture(depthDesc, nullptr, &pDepth);
        // Store the depth-stencil view
        msaaDepthDSV = pDepth->GetDefaultView(DE::TEXTURE_VIEW_DEPTH_STENCIL);
    }

    DE::RefCntAutoPtr<DE::IRenderDevice> device;
    DE::RefCntAutoPtr<DE::IDeviceContext> immediateContext;
    DE::RefCntAutoPtr<DE::ISwapChain> swapChain;
    DE::Uint32 sampleCount = 1;
    DE::RefCntAutoPtr<DE::ITextureView> msaaColorRTV;
    DE::RefCntAutoPtr<DE::ITextureView> msaaDepthDSV;
};

std::unique_ptr<Engine> gEngine;
bool blowup = false, screenshot = false, premult = false, escape = false;

void SaveScreenCapture(const std::string& FileName,
                       DE::ScreenCapture::CaptureInfo& Capture) {
    DE::MappedTextureSubresource texData;
    gEngine->immediateContext->MapTextureSubresource(
        Capture.pTexture, 0, 0, DE::MAP_READ, DE::MAP_FLAG_DO_NOT_WAIT, nullptr,
        texData);
    const auto& texDesc = Capture.pTexture->GetDesc();

    DE::Image::EncodeInfo Info;
    Info.Width = texDesc.Width;
    Info.Height = texDesc.Height;
    Info.TexFormat = texDesc.Format;
    Info.KeepAlpha = !premult;
    Info.pData = texData.pData;
    Info.Stride = texData.Stride;
    Info.FileFormat = DE::IMAGE_FILE_FORMAT_PNG;

    DE::RefCntAutoPtr<DE::IDataBlob> pEncodedImage;
    DE::Image::Encode(Info, &pEncodedImage);
    gEngine->immediateContext->UnmapTextureSubresource(Capture.pTexture, 0, 0);

    DE::FileWrapper pFile(FileName.c_str(), EFileAccessMode::Overwrite);
    if(pFile) {
        auto res =
            pFile->Write(pEncodedImage->GetDataPtr(), pEncodedImage->GetSize());
        pFile.Close();
    }
}

void saveScreenShot() {
    gEngine->immediateContext->SetRenderTargets(
        0, nullptr, nullptr, DE::RESOURCE_STATE_TRANSITION_MODE_NONE);
    DE::ScreenCapture sc(gEngine->device);
    sc.Capture(gEngine->swapChain, gEngine->immediateContext, 0);
    while(!sc.HasCapture())
        gEngine->device->IdleGPU();
    auto cap = sc.GetCapture();
    SaveScreenCapture("dump.png", cap);
}

std::unique_ptr<DE::DurationQueryHelper> query;

void initGPUTimer(GPUtimer* timer) {
    memset(timer, 0, sizeof(GPUtimer));
    timer->supported =
        gEngine->device->GetDeviceCaps().Features.TimestampQueries;
    if(timer->supported)
        query = std::make_unique<DE::DurationQueryHelper>(gEngine->device, 100);
}
void startGPUTimer(GPUtimer* timer) {
    if(!timer->supported)
        return;
    query->Begin(gEngine->immediateContext);
}
bool stopGPUTimer(GPUtimer* timer, float* time) {
    if(!timer->supported)
        return 0;
    double res;
    if(query->End(gEngine->immediateContext, res)) {
        *time = static_cast<float>(res);
        return true;
    }
    return false;
}

#ifdef D3D11_SUPPORTED
#include <d3d11.h>
#endif  // D3D11_SUPPORTED
#ifdef D3D12_SUPPORTED
#include <d3d12.h>
#endif  // D3D12_SUPPORTED

DE::Uint8 getQualityLevel() {
    auto dev = gEngine->device->GetDeviceCaps().DevType;
    void* nativeHandle = gEngine->swapChain->GetCurrentBackBufferRTV()
                             ->GetTexture()
                             ->GetNativeHandle();
#ifdef D3D11_SUPPORTED
    if(dev == DE::RENDER_DEVICE_TYPE_D3D11) {
        auto res = reinterpret_cast<ID3D11Resource*>(nativeHandle);
        ID3D11Device* device = nullptr;
        res->GetDevice(&device);
        UINT level = 0;
        device->CheckMultisampleQualityLevels(DXGI_FORMAT_R8G8B8A8_UNORM,
                                              gEngine->sampleCount, &level);
        return static_cast<DE::Uint8>(level - 1);
    }
#endif  // D3D11_SUPPORTED
#ifdef D3D12_SUPPORTED
    if(dev == DE::RENDER_DEVICE_TYPE_D3D12) {
        auto res = reinterpret_cast<ID3D12Resource*>(nativeHandle);
        void* device = nullptr;
        res->GetDevice(__uuidof(ID3D12Device), &device);
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS
        data = {};
        data.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        data.SampleCount = gEngine->sampleCount;
        data.Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;
        reinterpret_cast<ID3D12Device*>(device)->CheckFeatureSupport(
            D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &data, sizeof(data));
        return static_cast<DE::Uint8>(data.NumQualityLevels - 1);
    }
#endif  // D3D12_SUPPORTED
    return 0;
}

LRESULT CALLBACK MessageProc(HWND wnd, UINT message, WPARAM wParam,
                             LPARAM lParam);

int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int cmdShow) {
    WNDCLASSEX wcex = { sizeof(WNDCLASSEX),
                        CS_HREDRAW | CS_VREDRAW,
                        MessageProc,
                        0L,
                        0L,
                        instance,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        L"NanoVG",
                        NULL };
    RegisterClassExW(&wcex);

    RECT rc = { 0, 0, 1000, 600 };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(L"NanoVG", L"NanoVG", WS_OVERLAPPEDWINDOW,
                              CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left,
                              rc.bottom - rc.top, NULL, NULL, instance, NULL);
    if(!hwnd) {
        return -1;
    }
    ShowWindow(hwnd, cmdShow);
    UpdateWindow(hwnd);

    gEngine = std::make_unique<Engine>(hwnd, DE::RENDER_DEVICE_TYPE_GL);

    DemoData data;
    NVGcontext* vg = NULL;
    GPUtimer gpuTimer;
    PerfGraph fps, cpuGraph, gpuGraph;
    float cpuTime = 0;
    initGraph(&fps, GRAPH_RENDER_FPS, "Frame Time");
    initGraph(&cpuGraph, GRAPH_RENDER_MS, "CPU Time");
    initGraph(&gpuGraph, GRAPH_RENDER_MS, "GPU Time");

    auto&& SDesc = gEngine->swapChain->GetDesc();

#ifdef NANOVG_MSAA
    const auto& colorFmtInfo =
        gEngine->device->GetTextureFormatInfoExt(SDesc.ColorBufferFormat);
    const auto& depthFmtInfo =
        gEngine->device->GetTextureFormatInfoExt(SDesc.DepthBufferFormat);
    DE::Uint32 supportedSampleCounts =
        colorFmtInfo.SampleCounts & depthFmtInfo.SampleCounts;
    while(supportedSampleCounts & (gEngine->sampleCount << 1))
        gEngine->sampleCount <<= 1;
#endif  // NANOVG_MSAA

    if(gEngine->sampleCount > 1)
        gEngine->resetRenderTarget();

    DE::SampleDesc msaa = {};
    msaa.Count = gEngine->sampleCount;
    msaa.Quality = getQualityLevel();

    vg = nvgCreateDE(
        gEngine->device, gEngine->immediateContext, msaa,
        SDesc.ColorBufferFormat, SDesc.DepthBufferFormat,
        static_cast<int>((msaa.Count == 1 ? NVGCreateFlags::NVG_ANTIALIAS : 0) |
                         NVG_ALLOW_INDIRECT_RENDERING
    // | NVGCreateFlags::NVG_STENCIL_STROKES
#ifdef _DEBUG
                         | NVGCreateFlags::NVG_DEBUG
#endif
                         ));

    if(loadDemoData(vg, &data) == -1)
        return -1;

    initGPUTimer(&gpuTimer);

    auto lastTime = Clock::now();
    float sum = 0.0f;

    MSG msg = { 0 };
    while(WM_QUIT != msg.message && !escape) {
        if(PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        } else {
            auto cur = Clock::now();
            constexpr auto den = Clock::period::den;
            float delta = static_cast<float>((cur - lastTime).count()) / den;
            sum += delta;
            lastTime = cur;

            int winWidth, winHeight, mx, my, fbWidth, fbHeight;
            {
                RECT rect;
                GetClientRect(hwnd, &rect);
                winWidth = rect.right - rect.left;
                winHeight = rect.bottom - rect.top;
            }
            {
                POINT point;
                GetCursorPos(&point);
                ScreenToClient(hwnd, &point);
                mx = point.x;
                my = point.y;
            }
            {
                auto&& SDesc = gEngine->swapChain->GetDesc();
                fbWidth = SDesc.Width;
                fbHeight = SDesc.Height;
            }

            float pxRatio = static_cast<float>(fbWidth) / winWidth;

            startGPUTimer(&gpuTimer);

            const float clearA[] = { 0.0f, 0.0f, 0.0f, 1.0f };
            const float clearB[] = { 0.3f, 0.3f, 0.32f, 1.0f };
            gEngine->updateTarget(premult ? clearA : clearB);

            nvgBeginFrame(vg, static_cast<float>(winWidth),
                          static_cast<float>(winHeight), pxRatio);

            renderDemo(vg, static_cast<float>(mx), static_cast<float>(my),
                       static_cast<float>(winWidth),
                       static_cast<float>(winHeight), sum, blowup, &data);

            renderGraph(vg, 5, 5, &fps);
            renderGraph(vg, 5 + 200 + 5, 5, &cpuGraph);
            if(gpuTimer.supported)
                renderGraph(vg, 5 + 200 + 5 + 200 + 5, 5, &gpuGraph);

            nvgEndFrame(vg);

            auto ct = Clock::now();
            cpuTime = static_cast<float>((ct - cur).count()) / den;

            updateGraph(&fps, delta);
            updateGraph(&cpuGraph, cpuTime);

            float gpuTime;
            if(stopGPUTimer(&gpuTimer, &gpuTime))
                updateGraph(&gpuGraph, gpuTime);

            if(gEngine->sampleCount > 1) {
                // Resolve multi-sampled render taget into the current swap
                // chain back buffer.
                auto pCurrentBackBuffer =
                    gEngine->swapChain->GetCurrentBackBufferRTV()->GetTexture();

                DE::ResolveTextureSubresourceAttribs RA = {};
                RA.SrcTextureTransitionMode =
                    DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
                RA.DstTextureTransitionMode =
                    DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
                gEngine->immediateContext->ResolveTextureSubresource(
                    gEngine->msaaColorRTV->GetTexture(), pCurrentBackBuffer,
                    RA);
            }

            if(screenshot) {
                screenshot = false;
                saveScreenShot();
            }

            gEngine->swapChain->Present(0U);
        }
    }

    freeDemoData(vg, &data);
    nvgDeleteDE(vg);

    {
        std::stringstream ss;
        ss.precision(5);
        ss << "Average Frame Time: " << (getGraphAverage(&fps) * 1000.0f)
           << " ms\n";
        ss << "          CPU Time: " << (getGraphAverage(&cpuGraph) * 1000.0f)
           << " ms\n";
        ss << "          GPU Time: " << (getGraphAverage(&gpuGraph) * 1000.0f)
           << " ms\n";
        std::string str = ss.str();
        OutputDebugStringA(str.c_str());
    }

    gEngine.reset();
    DestroyWindow(hwnd);
    UnregisterClassW(wcex.lpszClassName, wcex.hInstance);
    return 0;
}

LRESULT CALLBACK MessageProc(HWND wnd, UINT message, WPARAM wParam,
                             LPARAM lParam) {
    switch(message) {
        case WM_KEYDOWN: {
            if(wParam == VK_ESCAPE)
                escape = true;
            if(wParam == VK_SPACE) {
                blowup = !blowup;
                return true;
            }
            if(wParam == 'S') {
                screenshot = 1;
                return true;
            }
            if(wParam == 'P') {
                premult = !premult;
                return true;
            }
            return false;
        }
        case WM_PAINT: {
            PAINTSTRUCT ps;
            BeginPaint(wnd, &ps);
            EndPaint(wnd, &ps);
            return 0;
        }
        case WM_SIZE:
            if(gEngine) {
                gEngine->swapChain->Resize(LOWORD(lParam), HIWORD(lParam));
                gEngine->resetRenderTarget();
            }
            return 0;

        case WM_CHAR:
            if(wParam == VK_ESCAPE)
                PostQuitMessage(0);
            return 0;

        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;

        case WM_GETMINMAXINFO: {
            LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam;

            lpMMI->ptMinTrackSize.x = 320;
            lpMMI->ptMinTrackSize.y = 240;
            return 0;
        }

        default:
            return DefWindowProcW(wnd, message, wParam, lParam);
    }
}
