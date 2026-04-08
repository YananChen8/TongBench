#pragma once
#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "Engine/LevelStreamingDynamic.h"
#include "UObject/SoftObjectPtr.h"
#include "ArenaTypes.h"
#include "TSArenaSubsystem.generated.h"

USTRUCT()
struct FTSArenaInstance
{
	GENERATED_BODY()

	UPROPERTY()
	FGuid Id;
	UPROPERTY()
	TSoftObjectPtr<UWorld> LevelAsset;
	UPROPERTY()
	TWeakObjectPtr<ULevelStreamingDynamic> Streaming;
	UPROPERTY()
	FTransform Anchor;
	UPROPERTY()
	TWeakObjectPtr<AActor> AnchorActor;

};

// 多 Level 运行时管理（单一职责）
UCLASS()
class TONGSIMMULTILEVEL_API UTSArenaSubsystem : public UWorldSubsystem
{
	GENERATED_BODY()

public:
	// === 生命周期 ===
	UFUNCTION(BlueprintCallable, Category="Arena")
	FGuid LoadArena(const TSoftObjectPtr<UWorld>& LevelAsset, const FTransform& Anchor, bool bMakeVisible = true);

	UFUNCTION(BlueprintCallable, Category="Arena")
	bool DestroyArena(const FGuid& ArenaId);

	UFUNCTION(BlueprintCallable, Category="Arena")
	bool ResetArena(const FGuid& ArenaId);

    // 查询某个 Arena 是否“已就绪”
    bool IsArenaReady(const FGuid& ArenaId, bool bRequireVisible = true) const;

    // 判断 Actor 是否属于该 Arena（通过 Level 判断）
    bool IsActorInArena(const FGuid& ArenaId, const AActor* Actor) const;

    // 若调用方需要拿到 Streaming 指针（用于绑定事件）
    TWeakObjectPtr<ULevelStreamingDynamic> GetStreaming(const FGuid& ArenaId) const;

	// === 可见性 ===
	UFUNCTION(BlueprintCallable, Category="Arena")
	bool SetArenaVisible(const FGuid& ArenaId, bool bVisible);

	// === 坐标变换（Python 全用相对锚点） ===
	UFUNCTION(BlueprintCallable, Category="Arena|Transform")
	bool LocalToWorld(const FGuid& ArenaId, const FTransform& Local, FTransform& OutWorld) const;

	UFUNCTION(BlueprintCallable, Category="Arena|Transform")
	bool WorldToLocal(const FGuid& ArenaId, const FTransform& WorldXf, FTransform& OutLocal) const;

	// === Spawn / Pose ===
	UFUNCTION(BlueprintCallable, Category="Arena|Spawn")
	AActor* SpawnActorInArena(const FGuid& ArenaId, TSubclassOf<AActor> ActorClass, const FTransform& LocalTransform);

	UFUNCTION(BlueprintCallable, Category="Arena|Spawn")
	AActor* SpawnActorInArenaByPath(const FGuid& ArenaId, const FSoftClassPath& ActorClassPath, const FTransform& LocalTransform);

	UFUNCTION(BlueprintCallable, Category="Arena|Pose")
	bool SetActorPoseLocal(const FGuid& ArenaId, AActor* Actor, const FTransform& LocalTransform, bool bResetPhysics = true);

	UFUNCTION(BlueprintCallable, Category="Arena|Pose")
	bool GetActorPoseLocal(const FGuid& ArenaId, const AActor* Actor, FTransform& OutLocalTransform) const;

	// === 查询 & 操作 ===
	UFUNCTION(BlueprintCallable, Category="Arena|Query")
	void GetArenas(TArray<FArenaDescriptor>& Out) const;

	UFUNCTION(BlueprintCallable, Category="Arena|Query")
	bool GetArenaAnchor(const FGuid& ArenaId, FTransform& OutAnchor) const;

	UFUNCTION(BlueprintCallable, Category="Arena|Query")
	ULevel* GetArenaULevel(const FGuid& ArenaId) const;


	// 初始化/销毁
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

private:
	UFUNCTION()
	void OnStreamingLevelLoaded();
	UFUNCTION()
	void OnStreamingLevelShown();

	TMap<FGuid, FTSArenaInstance> Arenas;
};
