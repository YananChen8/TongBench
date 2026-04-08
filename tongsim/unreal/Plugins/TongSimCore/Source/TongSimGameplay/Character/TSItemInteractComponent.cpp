// Fill out your copyright notice in the Description page of Project Settings.


#include "TSItemInteractComponent.h"

#include "Animation/AnimInstance.h"
#include "GameFramework/Actor.h"
#include "GameFramework/Character.h"
#include "Components/PrimitiveComponent.h"
#include "Components/SkeletalMeshComponent.h"

UTSItemInteractComponent::UTSItemInteractComponent(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UTSItemInteractComponent::BeginPlay()
{
	Super::BeginPlay();
}

ACharacter* UTSItemInteractComponent::GetOwnerCharacter() const
{
	return Cast<ACharacter>(GetOwner());
}

USkeletalMeshComponent* UTSItemInteractComponent::GetOwnerMesh() const
{
	if (ACharacter* C = GetOwnerCharacter())
	{
		return C->GetMesh();
	}
	return nullptr;
}

UAnimInstance* UTSItemInteractComponent::GetOwnerAnimInstance() const
{
	if (USkeletalMeshComponent* Mesh = GetOwnerMesh())
	{
		return Mesh->GetAnimInstance();
	}
	return nullptr;
}

AActor* UTSItemInteractComponent::GetHeldActor(ETSHand Hand) const
{
	return (Hand == ETSHand::Right) ? HeldActorRight.Get() : HeldActorLeft.Get();
}

bool UTSItemInteractComponent::ConsumeLastResult(FTSItemInteractResult& OutResult)
{
	if (!bHasResult)
	{
		return false;
	}
	OutResult = LastResult;
	bHasResult = false;
	LastResult = FTSItemInteractResult{};
	return true;
}

bool UTSItemInteractComponent::StartPickUpTargetActor(AActor* TargetActor, const FVector& TargetWorldLocationHint, ETSHand PreferredHand, FString& OutError)
{
	OutError.Reset();

	if (State != ETSItemInteractState::Empty && State != ETSItemInteractState::Holding)
	{
		OutError = TEXT("ItemInteract is busy.");
		return false;
	}

	if (GetHeldActor(PreferredHand))
	{
		OutError = TEXT("Target hand is already holding an actor.");
		return false;
	}

	ACharacter* Character = GetOwnerCharacter();
	if (!IsValid(Character))
	{
		OutError = TEXT("Owner is not a Character.");
		return false;
	}
	if (!IsValid(GetOwnerAnimInstance()))
	{
		OutError = TEXT("No AnimInstance on Character mesh.");
		return false;
	}

	if (!IsValid(TargetActor))
	{
		OutError = TEXT("TargetActor is invalid.");
		return false;
	}
	if (TargetActor == Character)
	{
		OutError = TEXT("TargetActor must not be the same as owner.");
		return false;
	}
	if (TargetActor->IsA(ACharacter::StaticClass()))
	{
		OutError = TEXT("TargetActor must not be a Character.");
		return false;
	}

	const FVector HandTargetWorld = ResolveHandTargetWorld(TargetActor, TargetWorldLocationHint);
	const FTSItemInteractAnimEntry* Entry = SelectPickUpAnim(PreferredHand, HandTargetWorld, OutError);
	if (!Entry || !Entry->Montage)
	{
		if (OutError.IsEmpty())
		{
			OutError = TEXT("PickUp montage selection failed.");
		}
		return false;
	}

	// Reset runtime state
	bHasResult = false;
	LastResult = FTSItemInteractResult{};
	CancelReason.Reset();
	bAttachDone = false;
	ActiveMontage = Entry->Montage;
	PendingTargetActor = TargetActor;
	ActiveHand = PreferredHand;
	CurrentHandTargetWorld = HandTargetWorld;

	// Start montage
	UAnimInstance* AnimInst = GetOwnerAnimInstance();
	const float Duration = AnimInst->Montage_Play(Entry->Montage, 1.0f);
	if (Duration <= 0.f)
	{
		OutError = TEXT("Failed to play pickup montage.");
		ActiveMontage.Reset();
		PendingTargetActor.Reset();
		return false;
	}

	FOnMontageEnded EndDelegate;
	EndDelegate.BindUObject(this, &UTSItemInteractComponent::OnPickUpMontageEnded);
	AnimInst->Montage_SetEndDelegate(EndDelegate, Entry->Montage);

	State = ETSItemInteractState::PickingUp;
	return true;
}

void UTSItemInteractComponent::CancelCurrentAction(const FString& Reason)
{
	if (State != ETSItemInteractState::PickingUp)
	{
		return;
	}

	CancelReason = Reason;
	if (UAnimInstance* AnimInst = GetOwnerAnimInstance())
	{
		if (UAnimMontage* Montage = ActiveMontage.Get())
		{
			AnimInst->Montage_Stop(0.1f, Montage);
		}
		else
		{
			AnimInst->StopAllMontages(0.1f);
		}
	}
}

void UTSItemInteractComponent::OnGrabAttachNotify()
{
	if (State != ETSItemInteractState::PickingUp || bAttachDone)
	{
		return;
	}

	ACharacter* Character = GetOwnerCharacter();
	USkeletalMeshComponent* Mesh = GetOwnerMesh();
	AActor* TargetActor = PendingTargetActor.Get();
	if (!IsValid(Character) || !IsValid(Mesh) || !IsValid(TargetActor))
	{
		return;
	}

	// Disable physics if possible (all primitive components)
	TArray<UPrimitiveComponent*> PrimComps;
	TargetActor->GetComponents<UPrimitiveComponent>(PrimComps);
	for (UPrimitiveComponent* Prim : PrimComps)
	{
		if (IsValid(Prim) && Prim->IsSimulatingPhysics())
		{
			Prim->SetSimulatePhysics(false);
		}
	}

	const FName SocketName = GetHandSocketName(ActiveHand);
	TargetActor->AttachToComponent(Mesh, FAttachmentTransformRules::KeepWorldTransform, SocketName);

	if (ActiveHand == ETSHand::Right)
	{
		HeldActorRight = TargetActor;
	}
	else
	{
		HeldActorLeft = TargetActor;
	}

	bAttachDone = true;
}

FVector UTSItemInteractComponent::ResolveHandTargetWorld(AActor* TargetActor, const FVector& FallbackWorldLocation) const
{
	if (!IsValid(TargetActor))
	{
		return FallbackWorldLocation;
	}

	if (!TargetSocketName.IsNone())
	{
		TArray<USceneComponent*> Components;
		TargetActor->GetComponents(Components);
		for (USceneComponent* Comp : Components)
		{
			if (!IsValid(Comp))
			{
				continue;
			}
			if (Comp->DoesSocketExist(TargetSocketName))
			{
				return Comp->GetSocketTransform(TargetSocketName, RTS_World).GetLocation();
			}
		}
	}

	return FallbackWorldLocation;
}

const FTSItemInteractAnimEntry* UTSItemInteractComponent::SelectPickUpAnim(ETSHand Hand, const FVector& TargetWorldLocation, FString& OutError) const
{
	OutError.Reset();

	if (!PickUpAnimConfig)
	{
		OutError = TEXT("PickUpAnimConfig is not set on component.");
		return nullptr;
	}
	if (PickUpAnimConfig->Entries.Num() <= 0)
	{
		OutError = TEXT("PickUpAnimConfig has no entries.");
		return nullptr;
	}

	const ACharacter* Character = GetOwnerCharacter();
	if (!IsValid(Character))
	{
		OutError = TEXT("Owner is not a Character.");
		return nullptr;
	}

	const FVector Local = Character->GetActorTransform().InverseTransformPosition(TargetWorldLocation);
	const float HeightCm = Local.Z;
	const float YawDeg = FMath::RadiansToDegrees(FMath::Atan2(Local.Y, Local.X));
	const float AbsYaw = FMath::Abs(YawDeg);

	const FTSItemInteractAnimEntry* Best = nullptr;
	float BestCost = TNumericLimits<float>::Max();
	float BestRangeCost = TNumericLimits<float>::Max();

	for (const FTSItemInteractAnimEntry& E : PickUpAnimConfig->Entries)
	{
		if (E.Hand != Hand || !E.Montage)
		{
			continue;
		}

		const float YawMin = FMath::Min(E.YawMinDeg, E.YawMaxDeg);
		const float YawMax = FMath::Max(E.YawMinDeg, E.YawMaxDeg);
		const float HeightMin = FMath::Min(E.HeightMinCm, E.HeightMaxCm);
		const float HeightMax = FMath::Max(E.HeightMinCm, E.HeightMaxCm);

		if (AbsYaw < YawMin || AbsYaw > YawMax || HeightCm < HeightMin || HeightCm > HeightMax)
		{
			continue;
		}

		// 先保证落在范围内；在范围内则优先选择更“接近范围中心”的 Montage（若重叠则偏向更窄的覆盖范围）
		const float YawCenter = 0.5f * (YawMin + YawMax);
		const float HeightCenter = 0.5f * (HeightMin + HeightMax);
		const float Cost = FMath::Abs(AbsYaw - YawCenter) * 1000.f
		                 + FMath::Abs(HeightCm - HeightCenter);
		const float RangeCost = (YawMax - YawMin) * 1000.f + (HeightMax - HeightMin);

		if (!Best || Cost < BestCost || (FMath::IsNearlyEqual(Cost, BestCost) && RangeCost < BestRangeCost))
		{
			BestCost = Cost;
			BestRangeCost = RangeCost;
			Best = &E;
		}
	}

	if (!Best)
	{
		OutError = FString::Printf(
			TEXT("No pickup montage match (hand=%s yaw=%.1fdeg height=%.1fcm)."),
			Hand == ETSHand::Right ? TEXT("Right") : TEXT("Left"),
			AbsYaw,
			HeightCm);
		return nullptr;
	}

	return Best;
}

void UTSItemInteractComponent::OnPickUpMontageEnded(UAnimMontage* Montage, bool bInterrupted)
{
	(void)Montage;

	if (State != ETSItemInteractState::PickingUp)
	{
		return;
	}

	const bool bActiveHandHeld = (GetHeldActor(ActiveHand) != nullptr);
	const bool bAnyHeld = (HeldActorRight.IsValid() || HeldActorLeft.IsValid());
	if (bInterrupted)
	{
		const FString Msg = CancelReason.IsEmpty()
			                    ? TEXT("PickUp interrupted.")
			                    : FString::Printf(TEXT("PickUp cancelled: %s"), *CancelReason);
		SetResult(bActiveHandHeld, Msg);
		State = bAnyHeld ? ETSItemInteractState::Holding : ETSItemInteractState::Empty;
	}
	else
	{
		if (bActiveHandHeld)
		{
			SetResult(true, TEXT("OK"));
			State = bAnyHeld ? ETSItemInteractState::Holding : ETSItemInteractState::Empty;
		}
		else
		{
			SetResult(false, TEXT("PickUp finished but attach notify not triggered."));
			State = bAnyHeld ? ETSItemInteractState::Holding : ETSItemInteractState::Empty;
		}
	}

	PendingTargetActor.Reset();
	ActiveMontage.Reset();
	CancelReason.Reset();
	bAttachDone = false;
}

void UTSItemInteractComponent::SetResult(bool bSuccess, const FString& Message)
{
	LastResult.bSuccess = bSuccess;
	LastResult.Message = Message;
	bHasResult = true;
}

FName UTSItemInteractComponent::GetHandSocketName(ETSHand Hand) const
{
	return (Hand == ETSHand::Right) ? RightHandSocketName : LeftHandSocketName;
}
