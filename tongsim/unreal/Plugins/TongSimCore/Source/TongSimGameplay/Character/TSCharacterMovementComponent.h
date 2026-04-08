// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "TSCharacterMovementComponent.generated.h"


UCLASS()
class TONGSIMGAMEPLAY_API UTSCharacterMovementComponent : public UCharacterMovementComponent
{
	GENERATED_BODY()

public:
	UTSCharacterMovementComponent(const FObjectInitializer& ObjectInitializer);
};
