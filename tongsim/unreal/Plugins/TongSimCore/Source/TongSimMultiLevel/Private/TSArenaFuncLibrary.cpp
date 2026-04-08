#include "TSArenaFuncLibrary.h"

#include "Engine/World.h"
#include "Blueprint/UserWidget.h"
#include "Engine/Engine.h"
#include "GameFramework/PlayerController.h"
#include "TSArenaSubsystem.h"
// #include "ArenaDebuggerWidget.h"

UTSArenaSubsystem* UTSArenaFuncLibrary::GetMgr(UObject* WorldContextObject)
{
    if (!WorldContextObject) return nullptr;
    UWorld* World = GEngine ? GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::ReturnNull) : nullptr;
    return World ? World->GetSubsystem<UTSArenaSubsystem>() : nullptr;
}

FGuid UTSArenaFuncLibrary::LoadArena(UObject* WorldContextObject, const TSoftObjectPtr<UWorld>& LevelAsset, const FTransform& Anchor, bool bVisible)
{
    if (auto* M = GetMgr(WorldContextObject))
    {
        return M->LoadArena(LevelAsset, Anchor, bVisible);
    }
    return FGuid();
}

bool UTSArenaFuncLibrary::DestroyArena(UObject* WorldContextObject, const FGuid& ArenaId)
{
    if (auto* M = GetMgr(WorldContextObject))
    {
        return M->DestroyArena(ArenaId);
    }
    return false;
}

bool UTSArenaFuncLibrary::ResetArena(UObject* WorldContextObject, const FGuid& ArenaId)
{
    if (auto* M = GetMgr(WorldContextObject))
    {
        return M->ResetArena(ArenaId);
    }
    return false;
}

void UTSArenaFuncLibrary::GetArenas(UObject* WorldContextObject, TArray<FArenaDescriptor>& Out)
{
    if (auto* M = GetMgr(WorldContextObject))
    {
        M->GetArenas(Out);
    }
}

//
// UUserWidget* UTSArenaFuncLibrary::ShowArenaDebugger(UObject* WorldContextObject, TSubclassOf<UUserWidget> WidgetClass)
// {
//     if (!WorldContextObject) return nullptr;
//     UWorld* World = GEngine ? GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::ReturnNull) : nullptr;
//     if (!World) return nullptr;
//
//     APlayerController* PC = World->GetFirstPlayerController();
//     if (!PC) return nullptr;
//
//     if (!WidgetClass)
//     {
//         WidgetClass = UArenaDebuggerWidget::StaticClass();
//     }
//
//     UUserWidget* W = CreateWidget<UUserWidget>(PC, WidgetClass);
//     if (W)
//     {
//         W->AddToViewport(10000);
//     }
//     return W;
// }
