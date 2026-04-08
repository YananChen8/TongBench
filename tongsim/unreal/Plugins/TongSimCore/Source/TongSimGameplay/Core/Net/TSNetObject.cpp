// Fill out your copyright notice in the Description page of Project Settings.


#include "TSNetObject.h"

bool UTSNetObject::IsSupportedForNetworking() const
{
	// TODO: Check?
	return true;
}

void UTSNetObject::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const
{
	UObject::GetLifetimeReplicatedProps(OutLifetimeProps);
}

int32 UTSNetObject::GetFunctionCallspace(UFunction* Function, FFrame* Stack)
{
	if (IsValid(GetOuter()))
	{
		return GetOuter()->GetFunctionCallspace(Function, Stack);
	}
	return Super::GetFunctionCallspace(Function, Stack);
}

bool UTSNetObject::CallRemoteFunction(UFunction* Function, void* Parms, FOutParmRec* OutParms, FFrame* Stack)
{
	check(!HasAnyFlags(RF_ClassDefaultObject))
	if (AActor* Owner = GetNetOwnerActor())
	{
		if (UNetDriver* NetDriver = Owner->GetNetDriver())
		{
			NetDriver->ProcessRemoteFunction(Owner, Function, Parms, OutParms, Stack, this);
		}
	}

	return false;
}

AActor* UTSNetObject::GetNetOwnerActor() const
{
	return GetTypedOuter<AActor>();
}
