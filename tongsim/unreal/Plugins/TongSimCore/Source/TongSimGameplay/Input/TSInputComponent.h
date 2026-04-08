// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "EnhancedInputComponent.h"
#include "Input/TSInputConfig.h"
#include "TSInputComponent.generated.h"

/*
 * Component used to manage input mappings and bindings using an input config data asset.
 */
UCLASS(Config = Input)
class TONGSIMGAMEPLAY_API UTSInputComponent : public UEnhancedInputComponent
{
	GENERATED_BODY()

public:
	UTSInputComponent(const FObjectInitializer& ObjectInitializer);

	template<class UserClass, typename FuncType>
	static void BindNativeInput(UEnhancedInputComponent* PlayerInputComponent, const UTSInputConfig* InputConfig, const FGameplayTag& InputTag, ETriggerEvent TriggerEvent, UserClass* Object, FuncType Func);
};

template <class UserClass, typename FuncType>
void UTSInputComponent::BindNativeInput(UEnhancedInputComponent* PlayerInputComponent, const UTSInputConfig* InputConfig, const FGameplayTag& InputTag, ETriggerEvent TriggerEvent, UserClass* Object, FuncType Func)
{
	if (const UInputAction* IA = InputConfig->FindNativeInputActionForTag(InputTag))
	{
		PlayerInputComponent->BindAction(IA, TriggerEvent, Object, Func);
	}
}
