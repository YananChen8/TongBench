// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.Collections.Generic;

public class TongSim_LiteTarget : TargetRules
{
	public TongSim_LiteTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.V5;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_6;
		ExtraModuleNames.Add("TongSim_Lite");

        // grpc插件里的dll使用的是ANSI内存分配器，UE默认使用的是FMalloc分配器，
        // 使用这个宏会改为ANSI内存分配器，否则grpc插件和UE间数据交互时会出现内存分配不匹配的错误
        // 只有Windows上有这个问题，Linux上没有这个问题
        GlobalDefinitions.Add("FORCE_ANSI_ALLOCATOR=1");
        GlobalDefinitions.Add("UE_USE_MALLOC_FILL_BYTES=0");
    }
}
