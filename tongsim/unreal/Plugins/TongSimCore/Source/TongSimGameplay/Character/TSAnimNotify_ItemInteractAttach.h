// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Animation/AnimNotifies/AnimNotify.h"

#include "TSAnimNotify_ItemInteractAttach.generated.h"

/**
 * 在拾取蒙太奇的关键帧触发：通过角色身上的 UTSItemInteractComponent Attach 当前待抓取物体到手上。
 * 注意：该 AnimNotify 与 gRPC 解耦，可用于任意触发拾取的逻辑。
 */
UCLASS(meta = (DisplayName = "TongSim ItemInteract Attach"))
class TONGSIMGAMEPLAY_API UTSAnimNotify_ItemInteractAttach : public UAnimNotify
{
	GENERATED_BODY()

public:
	virtual void Notify(USkeletalMeshComponent* MeshComp, UAnimSequenceBase* Animation) override;
};
