// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Engine/GameInstance.h"
#include "TongSimGameInstance.generated.h"

/**
 *
 */
UCLASS(Config = Game)
class TONGSIMGAMEPLAY_API UTongSimGameInstance : public UGameInstance
{
	GENERATED_BODY()

public:
	UTongSimGameInstance(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

protected:

	virtual void Init() override;
	virtual void Shutdown() override;
};
