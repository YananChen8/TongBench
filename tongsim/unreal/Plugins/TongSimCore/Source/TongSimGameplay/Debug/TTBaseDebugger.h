// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "TTDebugTickFunctionBase.h"

#include "TTBaseDebugger.generated.h"

UCLASS(Abstract)
class TONGSIMGAMEPLAY_API ATTBaseDebugger : public AActor
{
	GENERATED_BODY()

public:
	ATTBaseDebugger();

protected:
	virtual void BeginPlay() override;

public:
	virtual void Tick(float DeltaTime) override;

	virtual void RegisterActorTickFunctions(bool bRegister) override;

	/* Tick Functions */
public:
	FTTDebugTickFunc_PrePhysics PrePhysicsTicker;
	FTTDebugTickFunc_DuringPhysics DuringPhysicsTicker;
	FTTDebugTickFunc_PostPhysics PostPhysicsTicker;
	FTTDebugTickFunc_PostUpdateWork PostUpdateWorkTicker;

	virtual void PrePhysicsTickActor(float DeltaTime, enum ELevelTick TickType, FTTDebugTickFunc& ThisTickFunction);
	virtual void DuringPhysicsTickActor(float DeltaTime, enum ELevelTick TickType, FTTDebugTickFunc& ThisTickFunction);
	virtual void PostPhysicsTickActor(float DeltaTime, enum ELevelTick TickType, FTTDebugTickFunc& ThisTickFunction);
	virtual void PostUpdateWorkTickActor(float DeltaTime, enum ELevelTick TickType, FTTDebugTickFunc& ThisTickFunction);

private:
	void RegisterTickers(FTTDebugTickFunc& Ticker);
	void UnregisterTickers(FTTDebugTickFunc& Ticker);
};
