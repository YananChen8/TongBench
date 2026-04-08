// Fill out your copyright notice in the Description page of Project Settings.

#include "TSInputConfig.h"

#include "GameplayTagsManager.h"

UTSInputConfig::UTSInputConfig(const FObjectInitializer& ObjectInitializer)
{
}

const UInputAction* UTSInputConfig::FindNativeInputActionForTag(const FGameplayTag& InputTag) const
{
	for (const FTTInputAction& Action : NativeInputActions)
	{
		if (Action.InputAction && (Action.InputTag == InputTag))
		{
			return Action.InputAction;
		}
	}
	UE_LOG(LogTemp, Warning, TEXT("Can't find NativeInputAction for InputTag [%s] on InputConfig [%s]."), *InputTag.ToString(), *GetNameSafe(this));
	return nullptr;
}
