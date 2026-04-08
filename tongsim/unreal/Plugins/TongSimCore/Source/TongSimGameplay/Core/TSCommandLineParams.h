#pragma once
#pragma once

#include "CoreMinimal.h"

// TODO: move form TongSimGameplay to other module
struct TONGSIMGAMEPLAY_API FTSCommandLineParams
{
public:
	static const FTSCommandLineParams& Get() { return CommandLineParams;}

	static void InitializeCommandLineParams();

	/**/
	FString DefaultPixelStreamerName;
	FString MiniMapCameraPixelStreamerName;

	/**/
	bool bIsTongSimServer = false;
	bool bIsTongSimClient = false;

	int TongSimDistributionGroupID = -1;
	int HeartBeatToUEProxyInterval = -1;

	bool bDisablePreLoadedAsset = false;
	bool bForceEnableAgentCapturePixelStreaming = false;

	bool bIsVisionMoCapEnable = false;

	FString TongOS_U_HttpURL;

private:
	void ParseCommandLines();
	static FTSCommandLineParams CommandLineParams;
};
