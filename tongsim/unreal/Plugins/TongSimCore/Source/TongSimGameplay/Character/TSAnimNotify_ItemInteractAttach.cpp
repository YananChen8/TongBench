// Fill out your copyright notice in the Description page of Project Settings.


#include "TSAnimNotify_ItemInteractAttach.h"

#include "TSItemInteractComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "GameFramework/Actor.h"

void UTSAnimNotify_ItemInteractAttach::Notify(USkeletalMeshComponent* MeshComp, UAnimSequenceBase* /*Animation*/)
{
	if (!IsValid(MeshComp))
	{
		return;
	}

	AActor* Owner = MeshComp->GetOwner();
	if (!IsValid(Owner))
	{
		return;
	}

	if (UTSItemInteractComponent* InteractComp = Owner->FindComponentByClass<UTSItemInteractComponent>())
	{
		InteractComp->OnGrabAttachNotify();
	}
}
