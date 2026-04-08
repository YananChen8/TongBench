#include "TSCommandLineParams.h"
#include "TSLogChannels.h"

FTSCommandLineParams FTSCommandLineParams::CommandLineParams;

namespace TongSimCommandLineHelper
{
	FORCEINLINE void LogParseResult(const TCHAR* Param, const FString& Result)
	{
		UE_LOG(LogTongSimCore, Log, TEXT("Parse command-line %s, result is %s"), Param, *Result);
	}

	FORCEINLINE void LogParseResult(const TCHAR* Param, const int Result)
	{
		UE_LOG(LogTongSimCore, Log, TEXT("Parse command-line %s, result is %d"), Param, Result);
	}

	FORCEINLINE void LogParseResult(const TCHAR* Param, const bool Result)
	{
		UE_LOG(LogTongSimCore, Log, TEXT("Parse command-line %s, result is %s"), Param, Result ? TEXT("true") : TEXT("false"));
	}

	void ParseParam(const TCHAR* Param, bool& Out)
	{
		Out = FParse::Param(FCommandLine::Get(), Param);
		LogParseResult(Param, Out);
	}

	template <typename T>
	void ParseValue(const TCHAR* Param, T& Out, const T& DefaultValue)
	{
		if (FParse::Value(FCommandLine::Get(), Param, Out))
		{
			LogParseResult(Param, Out);
		}
		else
		{
			Out = DefaultValue;
			UE_LOG(LogTongSimCore, Log, TEXT("Parse command-line %s, not found"), Param);
		}
	}
}


void FTSCommandLineParams::InitializeCommandLineParams()
{
	CommandLineParams.ParseCommandLines();
}

void FTSCommandLineParams::ParseCommandLines()
{
	using namespace TongSimCommandLineHelper;

	// Distribution
	ParseParam(TEXT("TongSimClient"), bIsTongSimClient);
	ParseParam(TEXT("TongSimServer"), bIsTongSimServer);

	check(!(bIsTongSimClient && bIsTongSimServer))

	if (bIsTongSimServer)
	{
		ParseValue(TEXT("TongSimServer"), TongSimDistributionGroupID, -1);
	} else if (bIsTongSimClient)
	{
		ParseValue(TEXT("TongSimClient"), TongSimDistributionGroupID, -1);
	}

	// Pixel Streaming
	ParseValue(TEXT("DefaultStreamer="), DefaultPixelStreamerName, FString("DefaultStreamer"));
	ParseValue(TEXT("MinimapCamera="), MiniMapCameraPixelStreamerName, FString("MiniMapCamera"));

	ParseValue(TEXT("HeartBeatInterval="), HeartBeatToUEProxyInterval, -1);

	ParseParam(TEXT("DisablePreLoadedAsset"), bDisablePreLoadedAsset);
	ParseParam(TEXT("ForceEnableAgentCapturePixelStreaming"), bForceEnableAgentCapturePixelStreaming);


	ParseParam(TEXT("TSVisionMoCap"), bIsVisionMoCapEnable);

	// TTS and Avatar
	ParseValue(TEXT("TongOSHttpURL="), TongOS_U_HttpURL, FString("http://10.2.161.4/tongos_u"));
}
