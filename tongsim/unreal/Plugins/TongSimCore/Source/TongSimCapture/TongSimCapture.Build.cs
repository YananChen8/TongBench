// Copyright (c) TongSim. All rights reserved.
using System.IO;
using UnrealBuildTool;

public class TongSimCapture : ModuleRules
{
    public TongSimCapture(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "RenderCore",
            "RHI",
        });

        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "Projects",
            "Slate",
            "SlateCore",
            "Renderer",
            "ImageWrapper",
        });

        PublicIncludePaths.AddRange(new string[]
        {
            ModuleDirectory + "/Public"
        });

        PrivateIncludePaths.AddRange(new string[]
        {
            ModuleDirectory + "/Private",
            Path.Combine(EngineDirectory, "Source/Runtime/Renderer/Private"),
            Path.Combine(EngineDirectory, "Source/Runtime/Renderer/Internal")
        });
    }
}
