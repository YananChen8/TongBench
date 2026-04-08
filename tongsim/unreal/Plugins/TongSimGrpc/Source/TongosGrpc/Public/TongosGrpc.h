// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once
// #include "AsyncServer.h"
#include "Modules/ModuleManager.h"
DECLARE_LOG_CATEGORY_EXTERN(LogTongosGrpc, Log, All);

class FTongosGrpcModule : public IModuleInterface
{
public:

	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
private:
	// ServerImpl serverImpl;
};
