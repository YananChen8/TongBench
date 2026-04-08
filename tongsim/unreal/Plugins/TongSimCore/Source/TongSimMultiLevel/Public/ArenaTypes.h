#pragma once
#include "CoreMinimal.h"
#include "ArenaTypes.generated.h"

// 可选软重置接口：关内 Actor 实现后可定制复位逻辑
UINTERFACE(BlueprintType)
class UArenaResettable : public UInterface
{
	GENERATED_BODY()
};
class IArenaResettable
{
	GENERATED_BODY()
public:
	UFUNCTION(BlueprintNativeEvent, BlueprintCallable)
	void OnArenaReset();
};

USTRUCT(BlueprintType)
struct FArenaDescriptor
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly) FGuid Id;
	UPROPERTY(EditAnywhere, BlueprintReadOnly) FString AssetPath;
	UPROPERTY(EditAnywhere, BlueprintReadOnly) FTransform Anchor;
	UPROPERTY(EditAnywhere, BlueprintReadOnly) bool bIsLoaded = false;
	UPROPERTY(EditAnywhere, BlueprintReadOnly) bool bIsVisible = false;
	UPROPERTY(EditAnywhere, BlueprintReadOnly) int32 NumActors = 0;
};
