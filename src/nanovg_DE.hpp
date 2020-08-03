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

#pragma once
#define PLATFORM_WIN32 1
#include <DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h>
#include <DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h>

#include <DiligentCore/Common/interface/RefCntAutoPtr.hpp>
#include <memory>

namespace DE = Diligent;

#include "nanovg.h"

enum NVGCreateFlags {
    NVG_ANTIALIAS = 1 << 0,
    NVG_STENCIL_STROKES = 1 << 1,
    NVG_DEBUG = 1 << 2
};

NVGcontext* nvgCreateDE(DE::IRenderDevice* device, DE::IDeviceContext* context,
                        const DE::SampleDesc& MSAA,
                        DE::TEXTURE_FORMAT colorFormat,
                        DE::TEXTURE_FORMAT depthFormat, int flags);
void nvgDeleteDE(NVGcontext* ctx);
