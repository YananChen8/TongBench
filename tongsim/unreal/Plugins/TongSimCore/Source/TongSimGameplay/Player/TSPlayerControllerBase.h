// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/PlayerController.h"
#include "TSPlayerControllerBase.generated.h"

class  UInputMappingContext;

UCLASS(Blueprintable)
class TONGSIMGAMEPLAY_API ATSPlayerControllerBase : public APlayerController
{
	GENERATED_BODY()

public:
	ATSPlayerControllerBase(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

	virtual void ReceivedPlayer() override;

	void SetPlayerMappableInputConfig(TSoftObjectPtr<UInputMappingContext> InputMapping) const;
};
