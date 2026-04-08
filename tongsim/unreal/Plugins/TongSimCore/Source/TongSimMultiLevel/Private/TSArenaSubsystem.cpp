// Fill out your copyright notice in the Description page of Project Settings.


#include "TSArenaSubsystem.h"
#include "Engine/World.h"
#include "Engine/Level.h"
#include "DrawDebugHelpers.h"
#include "Components/PrimitiveComponent.h"
#include "GameFramework/PlayerController.h"
#include "Camera/PlayerCameraManager.h"

static ULevel* GetLoadedLevel(ULevelStreamingDynamic* LSD)
{
	return (LSD && LSD->IsLevelLoaded()) ? LSD->GetLoadedLevel() : nullptr;
}

void UTSArenaSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
}

void UTSArenaSubsystem::Deinitialize()
{
	Arenas.Empty();
	Super::Deinitialize();
}

void UTSArenaSubsystem::OnStreamingLevelLoaded()
{
}

void UTSArenaSubsystem::OnStreamingLevelShown()
{
}

FGuid UTSArenaSubsystem::LoadArena(const TSoftObjectPtr<UWorld>& LevelAsset, const FTransform& Anchor, bool bMakeVisible)
{
	UWorld* World = GetWorld();
	// if (!World || !LevelAsset.IsValid())
	if (!World )
	{
		UE_LOG(LogTemp, Warning, TEXT("LoadArena failed: invalid world or asset"));
		return FGuid();
	}

	bool bSuccess = false;
	ULevelStreamingDynamic* LSD = ULevelStreamingDynamic::LoadLevelInstanceBySoftObjectPtr(
		World, LevelAsset, Anchor.GetLocation(), Anchor.Rotator(), bSuccess);

	if (!bSuccess || !LSD)
	{
		UE_LOG(LogTemp, Error, TEXT("LoadArena failed: LoadLevelInstanceBySoftObjectPtr returned null."));
		return FGuid();
	}

	// 设置可见与变换
	LSD->SetShouldBeLoaded(true);
	LSD->SetShouldBeVisible(bMakeVisible);

	// 生成锚点可视化（隐藏）
	FGuid Id = FGuid::NewGuid();

	FActorSpawnParameters Params;
	Params.Name = FName(*FString::Printf(TEXT("ArenaAnchor_%s"), *Id.ToString(EGuidFormats::Digits)));
	AActor* AnchorActor = World->SpawnActor<AActor>(AActor::StaticClass(), Anchor);
	if (AnchorActor) AnchorActor->SetActorHiddenInGame(true);

	FTSArenaInstance& Inst = Arenas.Add(Id);
	Inst.Id = Id;
	Inst.LevelAsset = LevelAsset;
	Inst.Streaming = LSD;
	Inst.Anchor = Anchor;
	Inst.AnchorActor = AnchorActor;

	// 绑定加载完成回调，记录初始状态
	LSD->LevelTransform = Anchor;
	LSD->OnLevelLoaded.AddDynamic(this, &UTSArenaSubsystem::OnStreamingLevelLoaded);
	LSD->OnLevelShown.AddDynamic(this, &UTSArenaSubsystem::OnStreamingLevelShown);

	return Id;
}

bool UTSArenaSubsystem::DestroyArena(const FGuid& ArenaId)
{
	if (FTSArenaInstance* Instance = Arenas.Find(ArenaId))
	{
		if (ULevelStreamingDynamic* LSD = Instance->Streaming.Get())
		{
			LSD->OnLevelLoaded.RemoveAll(this);
			LSD->OnLevelShown.RemoveAll(this);
			LSD->SetShouldBeVisible(false);
			LSD->SetShouldBeLoaded(false);
		}
		if (AActor* AnchorActor = Instance->AnchorActor.Get())
		{
			AnchorActor->Destroy();
		}
		Arenas.Remove(ArenaId);
		return true;
	}
	return false;
}

bool UTSArenaSubsystem::ResetArena(const FGuid& ArenaId)
{
    FTSArenaInstance* Instance = Arenas.Find(ArenaId);
    if (!Instance) return false;

    UWorld* World = GetWorld();
    if (!World) return false;

    // 1) 卸载旧 Streaming（不移除 Arenas 记录，不改 Id）
    if (ULevelStreamingDynamic* Old = Instance->Streaming.Get())
    {
        Old->OnLevelLoaded.RemoveAll(this);
        Old->OnLevelShown.RemoveAll(this);
        Old->SetShouldBeVisible(false);
        Old->SetShouldBeLoaded(false);
    }

    // 2) 以相同 Asset 与 Anchor 重建 Streaming
    bool bSuccess = false;
    ULevelStreamingDynamic* NewLSD = ULevelStreamingDynamic::LoadLevelInstanceBySoftObjectPtr(
        World, Instance->LevelAsset, Instance->Anchor.GetLocation(), Instance->Anchor.Rotator(), bSuccess);

    if (!bSuccess || !NewLSD)
    {
        UE_LOG(LogTemp, Error, TEXT("ResetArena failed: recreate streaming failed."));
        return false;
    }

    NewLSD->LevelTransform = Instance->Anchor;
    NewLSD->SetShouldBeLoaded(true);
    NewLSD->SetShouldBeVisible(true);
    NewLSD->OnLevelLoaded.AddDynamic(this, &UTSArenaSubsystem::OnStreamingLevelLoaded);
    NewLSD->OnLevelShown.AddDynamic(this, &UTSArenaSubsystem::OnStreamingLevelShown);

    Instance->Streaming = NewLSD;
    return true;
}

bool UTSArenaSubsystem::IsArenaReady(const FGuid& ArenaId, bool bRequireVisible) const
{
    const FTSArenaInstance* Instance = Arenas.Find(ArenaId);
    if (!Instance) return false;

    const ULevelStreamingDynamic* LSD = Instance->Streaming.Get();
    if (!LSD) return false;

    const bool bLoaded = LSD->HasLoadedLevel();
    const bool bVisible = LSD->IsLevelVisible();

    if (!bLoaded) return false;
    if (bRequireVisible && !bVisible) return false;

    return true;
}

bool UTSArenaSubsystem::IsActorInArena(const FGuid& ArenaId, const AActor* Actor) const
{
    if (!IsValid(Actor)) return false;
    const ULevel* L = GetArenaULevel(ArenaId);
    return L && (Actor->GetLevel() == L);
}

TWeakObjectPtr<ULevelStreamingDynamic> UTSArenaSubsystem::GetStreaming(const FGuid& ArenaId) const
{
    const FTSArenaInstance* Instance = Arenas.Find(ArenaId);
    return Instance ? Instance->Streaming : nullptr;
}

bool UTSArenaSubsystem::SetArenaVisible(const FGuid& ArenaId, bool bVisible)
{
	if (FTSArenaInstance* Instance = Arenas.Find(ArenaId))
	{
		if (ULevelStreamingDynamic* LSD = Instance->Streaming.Get())
		{
			LSD->SetShouldBeVisible(bVisible);
			return true;
		}
	}
	return false;
}

bool UTSArenaSubsystem::LocalToWorld(const FGuid& ArenaId, const FTransform& Local, FTransform& OutWorld) const
{
	if (const FTSArenaInstance* Instance = Arenas.Find(ArenaId))
	{
		OutWorld = Local * Instance->Anchor;
		return true;
	}
	return false;
}

bool UTSArenaSubsystem::WorldToLocal(const FGuid& ArenaId, const FTransform& WorldXf, FTransform& OutLocal) const
{
	if (const FTSArenaInstance* Instance = Arenas.Find(ArenaId))
	{
		OutLocal = WorldXf.GetRelativeTransform(Instance->Anchor);
		return true;
	}
	return false;
}

AActor* UTSArenaSubsystem::SpawnActorInArena(const FGuid& ArenaId, TSubclassOf<AActor> ActorClass, const FTransform& LocalTransform)
{
	UWorld* World = GetWorld();
	if (!World || !ActorClass) return nullptr;
	ULevel* Lvl = GetArenaULevel(ArenaId);
	if (!Lvl) return nullptr;

	FTransform WorldXf;
	if (!LocalToWorld(ArenaId, LocalTransform, WorldXf)) return nullptr;

	FActorSpawnParameters Params;
	Params.OverrideLevel = Lvl;
	Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AdjustIfPossibleButAlwaysSpawn;
	return World->SpawnActor<AActor>(ActorClass, WorldXf, Params);
}

AActor* UTSArenaSubsystem::SpawnActorInArenaByPath(const FGuid& ArenaId, const FSoftClassPath& ActorClassPath, const FTransform& LocalTransform)
{
	UClass* Cls = ActorClassPath.TryLoadClass<AActor>();
	if (!Cls) return nullptr;
	return SpawnActorInArena(ArenaId, Cls, LocalTransform);
}

bool UTSArenaSubsystem::SetActorPoseLocal(const FGuid& ArenaId, AActor* Actor, const FTransform& LocalTransform, bool bResetPhysics)
{
	if (!IsValid(Actor)) return false;

	FTransform WorldXf;
	if (!LocalToWorld(ArenaId, LocalTransform, WorldXf)) return false;

	Actor->SetActorTransform(WorldXf, false, nullptr, ETeleportType::TeleportPhysics);

	if (bResetPhysics)
	{
		TArray<UPrimitiveComponent*> Prims;
		Actor->GetComponents<UPrimitiveComponent>(Prims);
		for (UPrimitiveComponent* PrimitiveComponent : Prims)
		{
			if (PrimitiveComponent && PrimitiveComponent->IsSimulatingPhysics())
			{
				PrimitiveComponent->SetPhysicsLinearVelocity(FVector::ZeroVector);
				PrimitiveComponent->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
			}
		}
	}
	return true;
}

bool UTSArenaSubsystem::GetActorPoseLocal(const FGuid& ArenaId, const AActor* Actor, FTransform& OutLocalTransform) const
{
	if (!IsValid(Actor)) return false;
	return WorldToLocal(ArenaId, Actor->GetActorTransform(), OutLocalTransform);
}

void UTSArenaSubsystem::GetArenas(TArray<FArenaDescriptor>& Out) const
{
	Out.Reset();
	for (const auto& Kvp : Arenas)
	{
		const FTSArenaInstance& Instance = Kvp.Value;
		FArenaDescriptor Descriptor;
		Descriptor.Id = Instance.Id;
		Descriptor.AssetPath = Instance.LevelAsset.ToString();
		Descriptor.Anchor = Instance.Anchor;

		const ULevelStreamingDynamic* LSD = Instance.Streaming.Get();
		Descriptor.bIsLoaded = LSD ? LSD->IsLevelLoaded() : false;
		Descriptor.bIsVisible = LSD ? LSD->ShouldBeVisible() : false;

		int32 Num = 0;
		if (const ULevel* Lvl = LSD ? LSD->GetLoadedLevel() : nullptr)
		{
			for (AActor* Actor : Lvl->Actors) { if (IsValid(Actor)) ++Num; }
		}
		Descriptor.NumActors = Num;
		Out.Add(Descriptor);
	}
}

bool UTSArenaSubsystem::GetArenaAnchor(const FGuid& ArenaId, FTransform& OutAnchor) const
{
	if (const FTSArenaInstance* Instance = Arenas.Find(ArenaId))
	{
		OutAnchor = Instance->Anchor;
		return true;
	}
	return false;
}

ULevel* UTSArenaSubsystem::GetArenaULevel(const FGuid& ArenaId) const
{
	if (const FTSArenaInstance* Instance = Arenas.Find(ArenaId))
	{
		if (ULevelStreamingDynamic* LSD = Instance->Streaming.Get())
		{
			return GetLoadedLevel(LSD);
		}
	}
	return nullptr;
}
