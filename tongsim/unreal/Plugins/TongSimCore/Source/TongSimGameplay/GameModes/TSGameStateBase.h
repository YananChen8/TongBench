// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameStateBase.h"
#include "TSGameStateBase.generated.h"

UCLASS()
class TONGSIMGAMEPLAY_API ATSGameStateBase : public AGameStateBase
{
	GENERATED_BODY()

public:
	ATSGameStateBase();
	virtual void Tick(float DeltaTime) override;

	FORCEINLINE float GetServerFPS() const { return ServerFPS; }

protected:
	UPROPERTY(Replicated)
	float ServerFPS;
};
