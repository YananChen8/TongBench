// Fill out your copyright notice in the Description page of Project Settings.


#include "TSPerformanceStatSubSystem.h"

#include "Common/TSGameplaySettings.h"
#include "GameFramework/PlayerState.h"
#include "GameModes/TSGameStateBase.h"
#include "TSLogChannels.h"

UTSPerformanceStatSubSystem* UTSPerformanceStatSubSystem::Instance = nullptr;

void FTSPerformanceStatCache::StartCharting()
{
	if (const UTSGameplaySettings* GameSettings = GetDefault<UTSGameplaySettings>())
	{
		bLogPerformance = GameSettings->bLogPerformance;
		LogIntervalInSec  = GameSettings->LogPerformanceIntervalInSec;
		TotalTime = 0.f;
	}
}

void FTSPerformanceStatCache::ProcessFrame(const FFrameData& FrameData)
{
	CachedData = FrameData;
	CachedServerFPS = 0.0f;
	CachedPingMS = 0.0f;
	CachedPacketLossIncomingPercent = 0.0f;
	CachedPacketLossOutgoingPercent = 0.0f;
	CachedPacketRateIncoming = 0.0f;
	CachedPacketRateOutgoing = 0.0f;
	CachedPacketSizeIncoming = 0.0f;
	CachedPacketSizeOutgoing = 0.0f;

	if (UWorld* World = MySubsystem->GetGameInstance()->GetWorld())
	{
		if (const ATSGameStateBase* GameState = World->GetGameState<ATSGameStateBase>())
		{
			CachedServerFPS = GameState->GetServerFPS();
		}

		// Net stat:
		if (APlayerController* LocalPC = GEngine->GetFirstLocalPlayerController(World))
		{
			if (APlayerState* PS = LocalPC->GetPlayerState<APlayerState>())
			{
				CachedPingMS = PS->GetPingInMilliseconds();
			}

			if (UNetConnection* NetConnection = LocalPC->GetNetConnection())
			{
				const UNetConnection::FNetConnectionPacketLoss& InLoss = NetConnection->GetInLossPercentage();
				CachedPacketLossIncomingPercent = InLoss.GetAvgLossPercentage();
				const UNetConnection::FNetConnectionPacketLoss& OutLoss = NetConnection->GetOutLossPercentage();
				CachedPacketLossOutgoingPercent = OutLoss.GetAvgLossPercentage();

				CachedPacketRateIncoming = NetConnection->InPacketsPerSecond;
				CachedPacketRateOutgoing = NetConnection->OutPacketsPerSecond;

				CachedPacketSizeIncoming = (NetConnection->InPacketsPerSecond != 0) ? (NetConnection->InBytesPerSecond / (float)NetConnection->InPacketsPerSecond) : 0.0f;
				CachedPacketSizeOutgoing = (NetConnection->OutPacketsPerSecond != 0) ? (NetConnection->OutBytesPerSecond / (float)NetConnection->OutPacketsPerSecond) : 0.0f;
			}
		}

		// TODO: Move this to Debugger by wukunlun
		if (bLogPerformance)
		{
			TotalTime += FrameData.TrueDeltaSeconds;
			if (TotalTime > LogIntervalInSec)
			{
				for(ETongSimPerformanceStat Stat : TEnumRange<ETongSimPerformanceStat>())
				{
					const FString StatName = StaticEnum<ETongSimPerformanceStat>()->GetNameStringByIndex(static_cast<int32>(Stat));
					UE_LOG(LogTongSimCore, Log, TEXT("[TongSim Performance %lld] %s : %.4lf"), GFrameCounter, *StatName, GetCachedStat(Stat));
				}
				TotalTime = 0.f;
			}
		}
	}
}

void FTSPerformanceStatCache::StopCharting()
{
	bLogPerformance  = false;
	TotalTime = 0.f;
}

double FTSPerformanceStatCache::GetCachedStat(ETongSimPerformanceStat Stat) const
{
	static_assert((int32)ETongSimPerformanceStat::Count == 15, "Need to update this function to deal with new performance stats");
	switch (Stat)
	{
	case ETongSimPerformanceStat::ClientFPS:
		return (CachedData.TrueDeltaSeconds != 0.0) ? (1.0 / CachedData.TrueDeltaSeconds) : 0.0;
	case ETongSimPerformanceStat::ServerFPS:
		return CachedServerFPS;
	case ETongSimPerformanceStat::IdleTime:
		return CachedData.IdleSeconds;
	case ETongSimPerformanceStat::FrameTime:
		return CachedData.TrueDeltaSeconds;
	case ETongSimPerformanceStat::FrameTime_GameThread:
		return CachedData.GameThreadTimeSeconds;
	case ETongSimPerformanceStat::FrameTime_RenderThread:
		return CachedData.RenderThreadTimeSeconds;
	case ETongSimPerformanceStat::FrameTime_RHIThread:
		return CachedData.RHIThreadTimeSeconds;
	case ETongSimPerformanceStat::FrameTime_GPU:
		return CachedData.GPUTimeSeconds;
	case ETongSimPerformanceStat::Ping:
		return CachedPingMS;
	case ETongSimPerformanceStat::PacketLoss_Incoming:
		return CachedPacketLossIncomingPercent;
	case ETongSimPerformanceStat::PacketLoss_Outgoing:
		return CachedPacketLossOutgoingPercent;
	case ETongSimPerformanceStat::PacketRate_Incoming:
		return CachedPacketRateIncoming;
	case ETongSimPerformanceStat::PacketRate_Outgoing:
		return CachedPacketRateOutgoing;
	case ETongSimPerformanceStat::PacketSize_Incoming:
		return CachedPacketSizeIncoming;
	case ETongSimPerformanceStat::PacketSize_Outgoing:
		return CachedPacketSizeOutgoing;
	}

	return 0.0f;
}

double UTSPerformanceStatSubSystem::GetCachedStat(ETongSimPerformanceStat Stat) const
{
	return Tracker->GetCachedStat(Stat);
}

void UTSPerformanceStatSubSystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	Instance = this;
	AddToRoot();
	Tracker = MakeShared<FTSPerformanceStatCache>(this);
	GEngine->AddPerformanceDataConsumer(Tracker);
}

void UTSPerformanceStatSubSystem::Deinitialize()
{
	GEngine->RemovePerformanceDataConsumer(Tracker);
	Tracker.Reset();
	Instance = nullptr;
	RemoveFromRoot();
	Super::Deinitialize();
}

UTSPerformanceStatSubSystem* UTSPerformanceStatSubSystem::GetInstance()
{
	return Instance;
}
