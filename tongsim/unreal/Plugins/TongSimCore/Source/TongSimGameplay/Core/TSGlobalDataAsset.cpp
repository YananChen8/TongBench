// Fill out your copyright notice in the Description page of Project Settings.


#include "TSGlobalDataAsset.h"
#include "TSAssetManager.h"

UTSGlobalDataAsset::UTSGlobalDataAsset()
{
}

const UTSGlobalDataAsset& UTSGlobalDataAsset::Get()
{
	return  UTSAssetManager::Get().GetGlobalDataAsset();
}

FPrimaryAssetId UTSGlobalDataAsset::GetPrimaryAssetId() const
{
	UPackage* Package = GetOutermost();

	if (!Package->HasAnyPackageFlags(PKG_PlayInEditor))
	{
		return FPrimaryAssetId(GetClass()->GetFName(), GetFName());
	}

	return FPrimaryAssetId();
}
