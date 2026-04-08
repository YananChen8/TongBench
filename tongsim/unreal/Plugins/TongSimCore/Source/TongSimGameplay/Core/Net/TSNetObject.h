// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "TSNetObject.generated.h"

/**
 * Base UObject can be replicated in net.
 * Wiki: https://wiki.mybigai.ac.cn/wiki/#/team/LgsUbAUo/space/Fkw94gzW/page/55trXCwj/edit
 * Samples: https://docs.unrealengine.com/5.2/zh-CN/replicated-subobjects-in-unreal-engine/
 */
UCLASS()
class TONGSIMGAMEPLAY_API UTSNetObject : public UObject
{
	GENERATED_BODY()

public:
	virtual bool IsSupportedForNetworking() const override;

protected:
	/* Property Replication */
	virtual void GetLifetimeReplicatedProps( TArray< class FLifetimeProperty > & OutLifetimeProps ) const override;

	/* For RPC */
	virtual int32 GetFunctionCallspace(UFunction* Function, FFrame* Stack) override;
	virtual bool CallRemoteFunction(UFunction* Function, void* Parms, FOutParmRec* OutParms, FFrame* Stack) override;

protected:
	/* Net Owner */
	template<class T>
	T* GetNetOwner() const { return Cast<T>(GetNetOwnerActor()); };

	virtual AActor* GetNetOwnerActor() const;
};
