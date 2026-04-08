// Fill out your copyright notice in the Description page of Project Settings.


#include "TSCharacterBase.h"
#include "TSCharacterMovementComponent.h"
#include "TSPawnExtComponent.h"


ATSCharacterBase::ATSCharacterBase(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer.SetDefaultSubobjectClass<UTSCharacterMovementComponent>(ACharacter::CharacterMovementComponentName))
{
	// Avoid ticking characters if possible.
	PrimaryActorTick.bCanEverTick = false;
	PrimaryActorTick.bStartWithTickEnabled = false;

	// TODO:
	SetNetCullDistanceSquared(900000000.0f);

	USkeletalMeshComponent* MeshComp = GetMesh();
	check(MeshComp);
	MeshComp->SetRelativeRotation(FRotator(0.0f, -90.0f, 0.0f));

	bReplicates = true;

	// Pawn Ext Comp :
	PawnExtComponent = CreateDefaultSubobject<UTSPawnExtComponent>(TEXT("PawnExtensionComponent"));
}

void ATSCharacterBase::GetOwnedGameplayTags(FGameplayTagContainer& TagContainer) const
{
	return PawnExtComponent->GetOwnedGameplayTags(TagContainer);
}


void ATSCharacterBase::PossessedBy(AController* NewController)
{
	Super::PossessedBy(NewController);
	PawnExtComponent->HandleControllerChanged();
}

void ATSCharacterBase::UnPossessed()
{
	Super::UnPossessed();
	PawnExtComponent->HandleControllerChanged();
}

void ATSCharacterBase::OnRep_Controller()
{
	Super::OnRep_Controller();
	PawnExtComponent->HandleControllerChanged();
}

void ATSCharacterBase::OnRep_PlayerState()
{
	Super::OnRep_PlayerState();
	PawnExtComponent->HandlePlayerStateReplicated();
}

void ATSCharacterBase::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);
	PawnExtComponent->SetupPlayerInputComponent();
}
