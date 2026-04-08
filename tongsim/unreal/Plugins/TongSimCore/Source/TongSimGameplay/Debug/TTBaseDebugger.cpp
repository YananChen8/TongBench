// Fill out your copyright notice in the Description page of Project Settings.


#include "TTBaseDebugger.h"


// Sets default values
ATTBaseDebugger::ATTBaseDebugger()
{
	PrimaryActorTick.bCanEverTick = true;
	bReplicates = false;
}

// Called when the game starts or when spawned
void ATTBaseDebugger::BeginPlay()
{
	Super::BeginPlay();
}

// Called every frame
void ATTBaseDebugger::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

void ATTBaseDebugger::RegisterActorTickFunctions(bool bRegister)
{
	if (IsTemplate())
	{
		return;
	}

	if (bRegister)
	{
		RegisterTickers(PrePhysicsTicker);
		RegisterTickers(DuringPhysicsTicker);
		RegisterTickers(PostPhysicsTicker);
		RegisterTickers(PostUpdateWorkTicker);
	}
	else
	{
		UnregisterTickers(PrePhysicsTicker);
		UnregisterTickers(DuringPhysicsTicker);
		UnregisterTickers(PostPhysicsTicker);
		UnregisterTickers(PostUpdateWorkTicker);
	}

	Super::RegisterActorTickFunctions(bRegister);
}

void ATTBaseDebugger::PrePhysicsTickActor(float DeltaTime, ELevelTick TickType, FTTDebugTickFunc& ThisTickFunction)
{
}

void ATTBaseDebugger::DuringPhysicsTickActor(float DeltaTime, ELevelTick TickType, FTTDebugTickFunc& ThisTickFunction)
{
}

void ATTBaseDebugger::PostPhysicsTickActor(float DeltaTime, ELevelTick TickType, FTTDebugTickFunc& ThisTickFunction)
{
}

void ATTBaseDebugger::PostUpdateWorkTickActor(float DeltaTime, ELevelTick TickType, FTTDebugTickFunc& ThisTickFunction)
{
}

void ATTBaseDebugger::RegisterTickers(FTTDebugTickFunc& Ticker)
{
	if (Ticker.bCanEverTick)
	{
		Ticker.DebuggerRawPtr = this;
		Ticker.SetTickFunctionEnable(true);
		Ticker.RegisterTickFunction(GetLevel());
	}
}

void ATTBaseDebugger::UnregisterTickers(FTTDebugTickFunc& Ticker)
{
	if (Ticker.IsTickFunctionRegistered())
	{
		Ticker.UnRegisterTickFunction();
	}
}
