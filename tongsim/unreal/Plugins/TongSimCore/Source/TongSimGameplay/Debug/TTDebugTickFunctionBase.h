#pragma once
#include "CoreMinimal.h"

class ATTBaseDebugger;

struct FTTDebugTickFunc : public FTickFunction
{
	ATTBaseDebugger* DebuggerRawPtr = nullptr;

	FTTDebugTickFunc()
	{
		bTickEvenWhenPaused = true;
		bCanEverTick = true;
	}
};

struct FTTDebugTickFunc_PrePhysics : public FTTDebugTickFunc
{
	FTTDebugTickFunc_PrePhysics()
	{
		TickGroup = TG_PrePhysics;
	}

	virtual void ExecuteTick(float DeltaTime, ELevelTick TickType, ENamedThreads::Type CurrentThread, const FGraphEventRef& MyCompletionGraphEvent) override;

	virtual FString DiagnosticMessage() override
	{
		return FString(TEXT("FTTDebugTickFunc_PrePhysics"));
	}
};

struct FTTDebugTickFunc_DuringPhysics : public FTTDebugTickFunc
{
	FTTDebugTickFunc_DuringPhysics()
	{
		TickGroup = TG_DuringPhysics;
	}

	virtual void ExecuteTick(float DeltaTime, ELevelTick TickType, ENamedThreads::Type CurrentThread, const FGraphEventRef& MyCompletionGraphEvent) override;

	virtual FString DiagnosticMessage() override
	{
		return FString(TEXT("FTTDebugTickFunc_DuringPhysics"));
	}
};

struct FTTDebugTickFunc_PostPhysics : public FTTDebugTickFunc
{
	FTTDebugTickFunc_PostPhysics()
	{
		TickGroup = TG_PostPhysics;
	}

	virtual void ExecuteTick(float DeltaTime, ELevelTick TickType, ENamedThreads::Type CurrentThread, const FGraphEventRef& MyCompletionGraphEvent) override;

	virtual FString DiagnosticMessage() override
	{
		return FString(TEXT("FTTDebugTickFunc_PostPhysics"));
	}
};

struct FTTDebugTickFunc_PostUpdateWork : public FTTDebugTickFunc
{
	FTTDebugTickFunc_PostUpdateWork()
	{
		TickGroup = TG_PostUpdateWork;
	}
	virtual void ExecuteTick(float DeltaTime, ELevelTick TickType, ENamedThreads::Type CurrentThread, const FGraphEventRef& MyCompletionGraphEvent) override;

	virtual FString DiagnosticMessage() override
	{
		return FString(TEXT("FTTDebugTickFunc_PostUpdateWork"));
	}
};
