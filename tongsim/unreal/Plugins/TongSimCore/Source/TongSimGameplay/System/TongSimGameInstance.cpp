// Fill out your copyright notice in the Description page of Project Settings.


#include "TongSimGameInstance.h"

#include "TSGameplayTags.h"
#include "Components/GameFrameworkComponentManager.h"

UTongSimGameInstance::UTongSimGameInstance(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer)
{
}

void UTongSimGameInstance::Init()
{
	Super::Init();

	// Register our custom init states
	UGameFrameworkComponentManager* ComponentManager = GetSubsystem<UGameFrameworkComponentManager>(this);

	if (ensure(ComponentManager))
	{
		ComponentManager->RegisterInitState(TongSimGameplayTags::InitState_Spawned, false, FGameplayTag());
		ComponentManager->RegisterInitState(TongSimGameplayTags::InitState_DataAvailable, false, TongSimGameplayTags::InitState_Spawned);
		ComponentManager->RegisterInitState(TongSimGameplayTags::InitState_GameplayReady, false, TongSimGameplayTags::InitState_DataAvailable);
	}
}

void UTongSimGameInstance::Shutdown()
{
	Super::Shutdown();
}
