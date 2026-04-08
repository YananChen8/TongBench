// Fill out your copyright notice in the Description page of Project Settings.


#include "TestVoxel.h"

#include "TSVoxelGridFuncLib.h"
#include "Components/BoxComponent.h"
#include "Kismet/GameplayStatics.h"

ATestVoxel::ATestVoxel(){
	PrimaryActorTick.bCanEverTick = false;

	BoxComponent=CreateDefaultSubobject<UBoxComponent>(TEXT("Box Component"));
	BoxComponent->SetBoxExtent(FVector{BoxSize/2, BoxSize/2, BoxSize/2});
	SetRootComponent(BoxComponent);
}

void ATestVoxel::BeginPlay(){
	Super::BeginPlay();

	FVoxelGridQueryParam QueryParam{GetWorld()};
	QueryParam.IgnoredPrimitiveComponents.Add(BoxComponent);
	UGameplayStatics::GetAllActorsOfClass(this->GetWorld(), AActor::StaticClass(), QueryParam.Actors);
	QueryParam.Actors.Remove(this);
	QueryParam.GridBox = FVoxelBox{
		GetActorTransform(), 32, 32, 32, BoxComponent->GetScaledBoxExtent()*2
	};
	TArray<uint8> VoxelGrids;
	TSVoxelGridFuncLib::QueryVoxelGrids(QueryParam, VoxelGrids, GetWorld());

	TSVoxelGridFuncLib::DrawDebugGrids(this->GetWorld(), QueryParam.GridBox, 1000, FColor::Blue,
											   VoxelGrids);
}
