// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "ChartCreation.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "TSPerformanceStatSubSystem.generated.h"

// Different kinds of stats that can be displayed on-screen
UENUM(BlueprintType)
enum class ETongSimPerformanceStat : uint8
{
	// stat fps (in Hz)
	ClientFPS,

	// server tick rate (in Hz)
	ServerFPS,

	// idle time spent waiting for vsync or frame rate limit (in seconds)
	IdleTime,

	// Stat unit overall (in seconds)
	FrameTime,

	// Stat unit (game thread, in seconds)
	FrameTime_GameThread,

	// Stat unit (render thread, in seconds)
	FrameTime_RenderThread,

	// Stat unit (RHI thread, in seconds)
	FrameTime_RHIThread,

	// Stat unit (inferred GPU time, in seconds)
	FrameTime_GPU,

	// Network ping (in ms)
	Ping,

	// The incoming packet loss percentage (%)
	PacketLoss_Incoming,

	// The outgoing packet loss percentage (%)
	PacketLoss_Outgoing,

	// The number of packets received in the last second
	PacketRate_Incoming,

	// The number of packets sent in the past second
	PacketRate_Outgoing,

	// The avg. size (in bytes) of packets received
	PacketSize_Incoming,

	// The avg. size (in bytes) of packets sent
	PacketSize_Outgoing,

	// New stats should go above here
	Count UMETA(Hidden)
};

ENUM_RANGE_BY_COUNT(ETongSimPerformanceStat, ETongSimPerformanceStat::Count);

class UTSPerformanceStatSubSystem;

struct FTSPerformanceStatCache : public IPerformanceDataConsumer
{
public:
	FTSPerformanceStatCache(UTSPerformanceStatSubSystem* InSubsystem)
		: MySubsystem(InSubsystem)
	{
	}

	//~IPerformanceDataConsumer interface
	virtual void StartCharting() override;
	virtual void ProcessFrame(const FFrameData& FrameData) override;
	virtual void StopCharting() override;
	//~End of IPerformanceDataConsumer interface

	double GetCachedStat(ETongSimPerformanceStat Stat) const;

protected:
	IPerformanceDataConsumer::FFrameData CachedData;
	UTSPerformanceStatSubSystem* MySubsystem;

	float CachedServerFPS = 0.0f;
	float CachedPingMS = 0.0f;
	float CachedPacketLossIncomingPercent = 0.0f;
	float CachedPacketLossOutgoingPercent = 0.0f;
	float CachedPacketRateIncoming = 0.0f;
	float CachedPacketRateOutgoing = 0.0f;
	float CachedPacketSizeIncoming = 0.0f;
	float CachedPacketSizeOutgoing = 0.0f;

private:
	bool bLogPerformance = false;
	float LogIntervalInSec = 10.f;
	float TotalTime = 0.f;
};

/**
 *
 */
UCLASS()
class TONGSIMGAMEPLAY_API UTSPerformanceStatSubSystem : public UGameInstanceSubsystem
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable)
	double GetCachedStat(ETongSimPerformanceStat Stat) const;

	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	static UTSPerformanceStatSubSystem* GetInstance();

protected:
	TSharedPtr<FTSPerformanceStatCache> Tracker;

private:
	static UTSPerformanceStatSubSystem* Instance;
};
