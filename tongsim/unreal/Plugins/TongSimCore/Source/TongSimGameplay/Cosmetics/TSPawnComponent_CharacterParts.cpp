// Fill out your copyright notice in the Description page of Project Settings.


#include "TSPawnComponent_CharacterParts.h"

#include "Components/CapsuleComponent.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "Net/UnrealNetwork.h"

UTSPawnComponent_CharacterParts::UTSPawnComponent_CharacterParts(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer)
{
	SetIsReplicatedByDefault(true);
}

void UTSPawnComponent_CharacterParts::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const
{
	Super::GetLifetimeReplicatedProps(OutLifetimeProps);

	DOREPLIFETIME(ThisClass, CurrentCharacterTypeTag);
}

void UTSPawnComponent_CharacterParts::BeginPlay()
{
	Super::BeginPlay();
}

void UTSPawnComponent_CharacterParts::SetCurrentCharacterTypeTag(FGameplayTag NewTag)
{
	// should be called on the authority only
	if (GetOwnerRole() == ROLE_Authority)
	{
		CurrentCharacterTypeTag = NewTag;
		ChangeCharacterMesh();
	}
}

void UTSPawnComponent_CharacterParts::ChangeCharacterMesh()
{
	const bool bReinitPose = false;

	if (USkeletalMeshComponent* MeshComponent = GetParentMeshComponent())
	{
		// Determine the mesh to use based on cosmetic part tags

		const FGameplayTagContainer MergedTags = FGameplayTagContainer(CurrentCharacterTypeTag); // TODO: Get Tags form all parts
		const FTongSimAnimBodyStyleSelectionEntry& DesiredBodyStyle = BodyMeshes.SelectBestBodyStyle(MergedTags);

		if (DesiredBodyStyle.IsValid())
		{
			MeshComponent->SetSkeletalMesh(DesiredBodyStyle.Mesh, /*bReinitPose=*/ bReinitPose);
			MeshComponent->LinkAnimClassLayers(DesiredBodyStyle.AnimLayer);

			ACharacter* Character = GetParentCharacter(); // MeshComponent is valid, so as character
			Character->GetCapsuleComponent()->SetCapsuleHalfHeight(DesiredBodyStyle.CapsuleHalfHeight);
			Character->GetCharacterMovement()->MaxWalkSpeed = DesiredBodyStyle.MaxWalkSpeed;
		}

		// TODO: Physics Asset
		/*if (UPhysicsAsset* PhysicsAsset = BodyMeshes.ForcedPhysicsAsset)
		{
			MeshComponent->SetPhysicsAsset(PhysicsAsset, bForceReInit=#1# bReinitPose);
		}*/
	}
}

USkeletalMeshComponent* UTSPawnComponent_CharacterParts::GetParentMeshComponent() const
{
	if (ACharacter* OwningCharacter = GetParentCharacter())
	{
		if (USkeletalMeshComponent* MeshComponent = OwningCharacter->GetMesh())
		{
			return MeshComponent;
		}
	}
	return nullptr;
}

ACharacter* UTSPawnComponent_CharacterParts::GetParentCharacter() const
{
	if (AActor* OwnerActor = GetOwner())
	{
		if (ACharacter* OwningCharacter = Cast<ACharacter>(OwnerActor))
		{
			return OwningCharacter;
		}
	}
	return nullptr;
}

void UTSPawnComponent_CharacterParts::OnRep_CurrentCharacterTag()
{
	ChangeCharacterMesh();
}
