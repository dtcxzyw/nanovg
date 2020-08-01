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


#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <Windows.h>
#include <nanovg.h>
#include "demo.h"
#include "perf.h"
#include <nanovg_DE.hpp>

#define ENGINE_DLL 1
#define D3D11_SUPPORTED 1
#define D3D12_SUPPORTED 1
#define GL_SUPPORTED 1
#define VULKAN_SUPPORTED 1

#include "engine.hpp"
#include <DiligentCore/Graphics/GraphicsEngineD3D11/interface/EngineFactoryD3D11.h>
#include <DiligentCore/Graphics/GraphicsEngineD3D12/interface/EngineFactoryD3D12.h>
#include <DiligentCore/Graphics/GraphicsEngineOpenGL/interface/EngineFactoryOpenGL.h>
#include <DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h>

class Engine final :private Unmovable {
public:
    Engine(void *hWnd, DE::RENDER_DEVICE_TYPE type) {
        DE::SwapChainDesc SCDesc;
        switch (type) {
#if D3D11_SUPPORTED
            case DE::RENDER_DEVICE_TYPE_D3D11:
            {
                DE::EngineD3D11CreateInfo EngineCI;
#ifdef DILIGENT_DEBUG
                EngineCI.DebugFlags |= DE::D3D11_DEBUG_FLAG_CREATE_DEBUG_DEVICE |
                    DE::D3D11_DEBUG_FLAG_VERIFY_COMMITTED_SHADER_RESOURCES;
#endif
#if ENGINE_DLL
            // Load the dll and import GetEngineFactoryD3D11() function
                auto GetEngineFactoryD3D11 = DE::LoadGraphicsEngineD3D11();
#endif
                auto *pFactoryD3D11 = GetEngineFactoryD3D11();
                pFactoryD3D11->CreateDeviceAndContextsD3D11(EngineCI, &device,
                    &immediateContext);
                DE::Win32NativeWindow window{ hWnd };
                pFactoryD3D11->CreateSwapChainD3D11(
                    device, immediateContext, SCDesc, DE::FullScreenModeDesc{},
                    window, &swapChain);
            } break;
#endif

#if D3D12_SUPPORTED
            case DE::RENDER_DEVICE_TYPE_D3D12:
            {
#if ENGINE_DLL
            // Load the dll and import GetEngineFactoryD3D12() function
                auto GetEngineFactoryD3D12 = DE::LoadGraphicsEngineD3D12();
#endif
                DE::EngineD3D12CreateInfo EngineCI;
#ifdef DILIGENT_DEBUG
            // There is a bug in D3D12 debug layer that causes memory leaks in
            // this tutorial
            // EngineCI.EnableDebugLayer = true;
#endif
                auto *pFactoryD3D12 = GetEngineFactoryD3D12();
                pFactoryD3D12->CreateDeviceAndContextsD3D12(EngineCI, &device,
                    &immediateContext);
                DE::Win32NativeWindow window{ hWnd };
                pFactoryD3D12->CreateSwapChainD3D12(
                    device, immediateContext, SCDesc, DE::FullScreenModeDesc{},
                    window, &swapChain);
            } break;
#endif

#if GL_SUPPORTED
            case DE::RENDER_DEVICE_TYPE_GL:
            {

#if EXPLICITLY_LOAD_ENGINE_GL_DLL
            // Load the dll and import GetEngineFactoryOpenGL() function
                auto GetEngineFactoryOpenGL = DE::LoadGraphicsEngineOpenGL();
#endif
                auto *pFactoryOpenGL = GetEngineFactoryOpenGL();

                DE::EngineGLCreateInfo EngineCI;
                EngineCI.Window.hWnd = hWnd;
#ifdef DILIGENT_DEBUG
                EngineCI.CreateDebugContext = true;
#endif
                pFactoryOpenGL->CreateDeviceAndSwapChainGL(
                    EngineCI, &device, &immediateContext, SCDesc, &swapChain);
            } break;
#endif

#if VULKAN_SUPPORTED
            case DE::RENDER_DEVICE_TYPE_VULKAN:
            {
#if EXPLICITLY_LOAD_ENGINE_VK_DLL
            // Load the dll and import GetEngineFactoryVk() function
                auto GetEngineFactoryVk = DE::LoadGraphicsEngineVk();
#endif
                DE::EngineVkCreateInfo EngineCI;
#ifdef DILIGENT_DEBUG
                EngineCI.EnableValidation = true;
#endif
                auto *pFactoryVk = GetEngineFactoryVk();
                pFactoryVk->CreateDeviceAndContextsVk(EngineCI, &device,
                    &immediateContext);

                if (!swapChain && hWnd != nullptr) {
                    DE::Win32NativeWindow window{ hWnd };
                    pFactoryVk->CreateSwapChainVk(device, immediateContext, SCDesc,
                        window, &swapChain);
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

    void updateTarget() {
        auto *pRTV = swapChain->GetCurrentBackBufferRTV();
        auto *pDSV = swapChain->GetDepthBufferDSV();
        immediateContext->SetRenderTargets(
            1, &pRTV, pDSV, DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        // Clear the back buffer
        const float ClearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };
        // Let the engine perform required state transitions
        immediateContext->ClearRenderTarget(
            pRTV, ClearColor, DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        immediateContext->ClearDepthStencil(
            pDSV, DE::CLEAR_DEPTH_FLAG, 1.f, 0,
            DE::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    DE::RefCntAutoPtr<DE::IRenderDevice>  device;
    DE::RefCntAutoPtr<DE::IDeviceContext> immediateContext;
    DE::RefCntAutoPtr<DE::ISwapChain>     swapChain;
};

std::unique_ptr<Engine> gEngine;

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
    if (!hwnd) {
        return -1;
    }
    ShowWindow(hwnd, cmdShow);
    UpdateWindow(hwnd);

    gEngine = std::make_unique<Engine>(hwnd, DE::RENDER_DEVICE_TYPE_VULKAN);

    DemoData data;
    NVGcontext *vg = NULL;
    GPUtimer gpuTimer;
    PerfGraph fps, cpuGraph, gpuGraph;
    double prevt = 0, cpuTime = 0;
    initGraph(&fps, GRAPH_RENDER_FPS, "Frame Time");
    initGraph(&cpuGraph, GRAPH_RENDER_MS, "CPU Time");
    initGraph(&gpuGraph, GRAPH_RENDER_MS, "GPU Time");

    auto &&SDesc = gEngine->swapChain->GetDesc();

    vg = nvgCreateDE(gEngine->device, gEngine->immediateContext, 1, SDesc.ColorBufferFormat, SDesc.DepthBufferFormat,
        static_cast<int> (NVGCreateFlags::NVG_STENCIL_STROKES)
    );

    if (loadDemoData(vg, &data) == -1)
        return -1;

    initGPUTimer(&gpuTimer);

    MSG msg = { 0 };
    while (WM_QUIT != msg.message) {
        if (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        else {
            gEngine->updateTarget();

            // TODO:MSAA
            gEngine->swapChain->Present(0U);
        }
    }

    gEngine.reset();
    DestroyWindow(hwnd);
    UnregisterClassW(wcex.lpszClassName, wcex.hInstance);
    return 0;
}

LRESULT CALLBACK MessageProc(HWND wnd, UINT message, WPARAM wParam,
    LPARAM lParam) {
    switch (message) {
        case WM_PAINT:
        {
            PAINTSTRUCT ps;
            BeginPaint(wnd, &ps);
            EndPaint(wnd, &ps);
            return 0;
        }
        case WM_SIZE:
            if (gEngine)
                gEngine->swapChain->Resize(LOWORD(lParam), HIWORD(lParam));
            return 0;

        case WM_CHAR:
            if (wParam == VK_ESCAPE)
                PostQuitMessage(0);
            return 0;

        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;

        case WM_GETMINMAXINFO:
        {
            LPMINMAXINFO lpMMI = (LPMINMAXINFO) lParam;

            lpMMI->ptMinTrackSize.x = 320;
            lpMMI->ptMinTrackSize.y = 240;
            return 0;
        }

        default:
            return DefWindowProcW(wnd, message, wParam, lParam);
    }
}
