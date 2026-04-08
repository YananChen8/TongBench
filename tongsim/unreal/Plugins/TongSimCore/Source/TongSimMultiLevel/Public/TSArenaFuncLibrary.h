#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "TSArenaFuncLibrary.generated.h"

class UTSArenaSubsystem;
class UArenaDebuggerWidget;

UCLASS()
class TONGSIMMULTILEVEL_API UTSArenaFuncLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable, Category="TongSim|Arena", meta=(WorldContext="WorldContextObject"))
	static FGuid LoadArena(UObject* WorldContextObject, const TSoftObjectPtr<UWorld>& LevelAsset, const FTransform& Anchor, bool bVisible);

	UFUNCTION(BlueprintCallable, Category="TongSim|Arena", meta=(WorldContext="WorldContextObject"))
	static bool DestroyArena(UObject* WorldContextObject, const FGuid& ArenaId);

	UFUNCTION(BlueprintCallable, Category="TongSim|Arena", meta=(WorldContext="WorldContextObject"))
	static bool ResetArena(UObject* WorldContextObject, const FGuid& ArenaId);

	UFUNCTION(BlueprintCallable, Category="TongSim|Arena", meta=(WorldContext="WorldContextObject"))
	static void GetArenas(UObject* WorldContextObject, TArray<FArenaDescriptor>& Out);

	// UFUNCTION(BlueprintCallable, Category="Arena|Debug", meta=(WorldContext="WorldContextObject"))
	// static UUserWidget* ShowArenaDebugger(UObject* WorldContextObject, TSubclassOf<UUserWidget> WidgetClass = nullptr);

private:
	static UTSArenaSubsystem* GetMgr(UObject* WorldContextObject);
};
