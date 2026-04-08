#pragma once

#include "CoreMinimal.h"
#include "GameplayTagContainer.h"
#include "NativeGameplayTags.h"

class UGameplayTagsManager;

namespace TongSimGameplayTags
{
	TONGSIMGAMEPLAY_API FGameplayTag FindTagByString(const FString& TagString, bool bMatchPartialString = false);

	// Declare the custom native tags
	TONGSIMGAMEPLAY_API UE_DECLARE_GAMEPLAY_TAG_EXTERN(InitState_Spawned)
	TONGSIMGAMEPLAY_API UE_DECLARE_GAMEPLAY_TAG_EXTERN(InitState_DataAvailable);
	TONGSIMGAMEPLAY_API UE_DECLARE_GAMEPLAY_TAG_EXTERN(InitState_GameplayReady);

	// UI Layer
	TONGSIMGAMEPLAY_API UE_DECLARE_GAMEPLAY_TAG_EXTERN(UI_Layer_Debug);
}
