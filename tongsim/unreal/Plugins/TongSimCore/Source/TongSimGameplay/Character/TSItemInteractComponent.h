// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Engine/DataAsset.h"

#include "TSItemInteractComponent.generated.h"

class UAnimMontage;
class UAnimInstance;
class ACharacter;
class USkeletalMeshComponent;

UENUM(BlueprintType)
enum class ETSHand : uint8
{
	Right UMETA(DisplayName = "Right"),
	Left  UMETA(DisplayName = "Left"),
};

UENUM(BlueprintType)
enum class ETSItemInteractState : uint8
{
	Empty UMETA(DisplayName = "Empty"),
	PickingUp UMETA(DisplayName = "PickingUp"),
	Holding UMETA(DisplayName = "Holding"),
};

USTRUCT(BlueprintType)
struct FTSItemInteractAnimEntry
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	ETSHand Hand = ETSHand::Right;

	// 该 Montage 覆盖的水平角度范围（绝对值，单位：度）
	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (ClampMin = "0.0", ClampMax = "180.0"))
	float YawMinDeg = 0.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (ClampMin = "0.0", ClampMax = "180.0"))
	float YawMaxDeg = 0.f;

	// 该 Montage 覆盖的高度范围（相对角色，单位：cm）
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float HeightMinCm = 0.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float HeightMaxCm = 0.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	TObjectPtr<UAnimMontage> Montage = nullptr;
};

UCLASS(BlueprintType)
class TONGSIMGAMEPLAY_API UTSItemInteractAnimDataAsset : public UDataAsset
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "TongSim|Interact")
	TArray<FTSItemInteractAnimEntry> Entries;
};

USTRUCT(BlueprintType)
struct FTSItemInteractResult
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly)
	bool bSuccess = false;

	UPROPERTY(BlueprintReadOnly)
	FString Message;
};

UCLASS(meta = (BlueprintSpawnableComponent))
class TONGSIMGAMEPLAY_API UTSItemInteractComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UTSItemInteractComponent(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

	virtual void BeginPlay() override;

	// ---- Public API (gRPC 解耦) ----

	// 开始抓取指定的 TargetActor（目标由外部显式传入，不在 UE 侧按距离查找）
	UFUNCTION(BlueprintCallable, Category = "TongSim|Interact")
	bool StartPickUpTargetActor(AActor* TargetActor, const FVector& TargetWorldLocationHint, ETSHand PreferredHand, FString& OutError);

	// 取消当前动作（会 Stop Montage；最终结果通过 ConsumeLastResult 获取）
	UFUNCTION(BlueprintCallable, Category = "TongSim|Interact")
	void CancelCurrentAction(const FString& Reason);

	// AnimNotify 回调：在抓取蒙太奇的关键帧 Attach 目标物体到手上
	UFUNCTION(BlueprintCallable, Category = "TongSim|Interact")
	void OnGrabAttachNotify();

	// 取走一次性结果（无结果返回 false）
	UFUNCTION(BlueprintCallable, Category = "TongSim|Interact")
	bool ConsumeLastResult(FTSItemInteractResult& OutResult);

	UFUNCTION(BlueprintPure, Category = "TongSim|Interact")
	bool IsBusy() const { return State == ETSItemInteractState::PickingUp; }

	UFUNCTION(BlueprintPure, Category = "TongSim|Interact")
	ETSItemInteractState GetState() const { return State; }

	UFUNCTION(BlueprintPure, Category = "TongSim|Interact")
	AActor* GetHeldActor(ETSHand Hand) const;

	// 动画蓝图可读取：当前抓取目标点（world）
	UPROPERTY(BlueprintReadOnly, VisibleInstanceOnly, Category = "TongSim|Interact")
	FVector CurrentHandTargetWorld = FVector::ZeroVector;

	UPROPERTY(BlueprintReadOnly, VisibleInstanceOnly, Category = "TongSim|Interact")
	ETSHand ActiveHand = ETSHand::Right;

	UPROPERTY(BlueprintReadOnly, VisibleInstanceOnly, Category = "TongSim|Interact")
	ETSItemInteractState State = ETSItemInteractState::Empty;
protected:
	UPROPERTY(EditAnywhere, Category = "TongSim|Interact|Animation")
	TObjectPtr<UTSItemInteractAnimDataAsset> PickUpAnimConfig;

	// 若目标物体上存在该 Socket，则优先用 Socket 作为抓取目标点；否则使用请求位置
	UPROPERTY(EditAnywhere, Category = "TongSim|Interact|Target")
	FName TargetSocketName = NAME_None;

	UPROPERTY(EditAnywhere, Category = "TongSim|Interact|Attach")
	FName RightHandSocketName = FName(TEXT("hand_r"));

	UPROPERTY(EditAnywhere, Category = "TongSim|Interact|Attach")
	FName LeftHandSocketName = FName(TEXT("hand_l"));

private:

	TWeakObjectPtr<AActor> PendingTargetActor;
	TWeakObjectPtr<AActor> HeldActorRight;
	TWeakObjectPtr<AActor> HeldActorLeft;

	TWeakObjectPtr<UAnimMontage> ActiveMontage;
	FString CancelReason;
	bool bAttachDone = false;

	bool bHasResult = false;
	FTSItemInteractResult LastResult;

	ACharacter* GetOwnerCharacter() const;
	USkeletalMeshComponent* GetOwnerMesh() const;
	UAnimInstance* GetOwnerAnimInstance() const;

	FVector ResolveHandTargetWorld(AActor* TargetActor, const FVector& FallbackWorldLocation) const;
	const FTSItemInteractAnimEntry* SelectPickUpAnim(ETSHand Hand, const FVector& TargetWorldLocation, FString& OutError) const;

	void OnPickUpMontageEnded(UAnimMontage* Montage, bool bInterrupted);
	void SetResult(bool bSuccess, const FString& Message);
	FName GetHandSocketName(ETSHand Hand) const;
};
