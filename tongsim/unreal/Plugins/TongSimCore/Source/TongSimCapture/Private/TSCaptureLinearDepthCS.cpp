#include "TSCaptureLinearDepthCS.h"
#include "RenderGraphUtils.h"

IMPLEMENT_GLOBAL_SHADER(FTSCaptureLinearDepthCS,
                        "/Plugin/TongSimCapture/Private/TSCaptureLinearDepthCS.usf",
                        "MainCS", SF_Compute);
